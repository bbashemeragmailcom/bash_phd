#!/usr/bin/env python3
"""Clean Labelbox exports, converting only BMP files to PNG. Run with --dry-run first. Requires Pillow."""
import argparse
import json
import os
import re
import shutil
from pathlib import Path
from PIL import Image

DEFAULT_ROOT = Path(__file__).resolve().parent / 'labelbox_export_multi' / 'projects copy'
EXT = {'.bmp', '.jpg', '.jpeg', '.png'}


def load(path, default):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def save(path, data):
    tmp = path.with_name(path.name + '.part')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def indexed(folder, overlay=False):
    result = {}
    for p in sorted(folder.iterdir(), key=lambda p: (p.name.casefold(), p.name)):
        if not p.is_file() or p.suffix.lower() not in EXT:
            continue
        stem = p.stem
        if overlay:
            if not stem.endswith('_overlay'):
                continue
            stem = stem[:-8]
        if stem.casefold() in result:
            raise ValueError(f'Duplicate stem in {folder}: {stem}')
        result[stem.casefold()] = p
    return result


def output_suffix(src):
    return '.png' if src.suffix.lower() == '.bmp' else src.suffix.lower()


def image_copy(src, dst):
    if src.suffix.lower() != '.bmp':
        with Image.open(src) as im:
            im.verify()
        shutil.copy2(src, dst)
    else:
        with Image.open(src) as im:
            im.load()
            # Preserve grayscale or RGB; no EXIF rotation or resampling.
            clean = im.copy() if im.mode in ('1', 'L', 'P', 'RGB', 'RGBA') else im.convert('RGB')
            clean.info.clear()
            clean.save(dst, format='PNG')
    with Image.open(dst) as im:
        im.verify()


def row_key(row):
    return (Path(row.get('image_filename') or '').stem or Path(row.get('mask_directory') or '').name).casefold()


def plan(project):
    images = indexed(project / 'images')
    overlays = indexed(project / 'overlays', True)
    masks = project / 'object_masks'
    if not masks.is_dir():
        raise FileNotFoundError(masks)
    if images and not overlays:
        raise ValueError('No recognized overlays in entire project; refusing mass removal')
    pattern = re.compile(re.escape(project.name) + r'-P-(\d{5})$')
    used = [int(m.group(1)) for p in images.values() if (m := pattern.fullmatch(p.stem))]
    mapping = load(project / 'rename_map.json', [])
    for item in mapping:
        if m := pattern.fullmatch(item.get('new_id', '')):
            used.append(int(m.group(1)))
    index = max(used, default=0) + 1
    assignments = []
    for key, src in images.items():
        ov = overlays.get(key)
        if ov is None:
            assignments.append((key, src, None, None))
            continue
        if pattern.fullmatch(src.stem):
            new = src.stem
        else:
            if index > 99999:
                raise ValueError('Five-digit ID capacity exceeded')
            new = f'{project.name}-P-{index:05d}'
            index += 1
        assignments.append((key, src, ov, new))
    return assignments, mapping


def process(project, args):
    # Build a complete cleaned sibling directory before changing the source.
    stage = project.with_name('.' + project.name + '.cleaning')
    backup = project.with_name('.' + project.name + '.original')
    if stage.exists() or backup.exists():
        raise RuntimeError(f'Previous staging/backup exists: {stage} or {backup}; inspect before rerunning')
    jobs, mapping = plan(project)
    rows = load(project / 'manifest.json', [])
    if not isinstance(rows, list) or not isinstance(mapping, list):
        raise ValueError('Expected JSON lists for manifest and rename map')
    by_key = {}
    for row in rows:
        key = row_key(row)
        if key in by_key:
            raise ValueError(f'Duplicate manifest entry: {key}')
        by_key[key] = row
    kept = sum(new is not None for _, _, _, new in jobs)
    removed = len(jobs) - kept
    converted = sum(p.suffix.lower() == '.bmp' for _, p, ov, new in jobs if new) + sum(ov.suffix.lower() == '.bmp' for _, p, ov, new in jobs if new)
    renamed = sum(p.stem != new for _, p, _, new in jobs if new)
    print(f'{project.name}: {len(jobs)} images | keep {kept} | remove {removed} | convert {converted} | rename {renamed}', flush=True)
    if args.dry_run:
        return kept, removed, converted, renamed
    if not jobs:
        return 0, 0, 0, 0
    stage.mkdir()
    for name in ('images', 'overlays', 'object_masks'):
        (stage / name).mkdir()
    quarantine = stage / '_quarantine'
    new_rows = []
    missing_masks = 0
    try:
        # Keep unrelated top-level project metadata unchanged.
        for p in project.iterdir():
            if p.name in {'images', 'overlays', 'object_masks', 'manifest.json', 'rename_map.json'}:
                continue
            if p.is_dir():
                shutil.copytree(p, stage / p.name)
            else:
                shutil.copy2(p, stage / p.name)
        for i, (key, src, ov, new) in enumerate(jobs, 1):
            mask = project / 'object_masks' / src.stem
            if new is None:
                if not args.delete_missing:
                    q = quarantine / src.stem
                    if q.exists():
                        raise FileExistsError(q)
                    q.mkdir(parents=True)
                    shutil.copy2(src, q / src.name)
                    if mask.is_dir():
                        shutil.copytree(mask, q / 'object_masks')
                    if key in by_key:
                        save(q / 'manifest_entry.json', by_key[key])
            else:
                image_filename = f'{new}{output_suffix(src)}'
                overlay_filename = f'{new}_overlay{output_suffix(ov)}'
                image_dest = stage / 'images' / image_filename
                overlay_dest = stage / 'overlays' / overlay_filename
                image_copy(src, image_dest)
                image_copy(ov, overlay_dest)
                with Image.open(image_dest) as a, Image.open(overlay_dest) as b:
                    if a.size != b.size:
                        raise ValueError(f'Image/overlay dimension mismatch: {src.name}')
                row = dict(by_key.get(key, {}))
                old_external = row.get('external_id')
                if mask.is_dir():
                    dest = stage / 'object_masks' / new
                    shutil.copytree(mask, dest)
                    info_path = dest / 'annotation_info.json'
                    if info_path.exists():
                        info = load(info_path, {})
                        old_external = info.get('external_id') or old_external
                        info.update(image_stem=new, external_id=new)
                        save(info_path, info)
                else:
                    missing_masks += 1
                    print(f'  WARNING: missing object-mask folder for {new}', flush=True)
                row.update(image_filename=image_filename, mask_directory=new if mask.is_dir() else None, external_id=new)
                new_rows.append(row)
                if src.stem != new or not any(x.get('new_id') == new for x in mapping):
                    mapping.append(dict(new_id=new, old_stem=src.stem, old_image_filename=src.name, old_external_id=old_external))
            if i % 100 == 0 or i == len(jobs):
                print(f'  {i}/{len(jobs)} processed', flush=True)
        # Keep unmatched assets outside the training folders rather than silently losing them.
        expected_masks = {p.stem for _, p, _, _ in jobs}
        matched_overlays = {ov.name for _, _, ov, _ in jobs if ov}
        matched_images = {p.name for _, p, _, _ in jobs}
        for name, expected in [('images', matched_images), ('overlays', matched_overlays), ('object_masks', expected_masks)]:
            for p in (project / name).iterdir():
                if p.name in expected:
                    continue
                dest = quarantine / '_unmatched' / name / p.name
                if dest.exists():
                    raise FileExistsError(dest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if p.is_dir():
                    shutil.copytree(p, dest)
                else:
                    shutil.copy2(p, dest)
        unmatched_rows = [r for r in rows if row_key(r) not in {j[0] for j in jobs}]
        if unmatched_rows:
            quarantine.mkdir(exist_ok=True)
            save(quarantine / 'unmatched_manifest.json', unmatched_rows)
        save(stage / 'manifest.json', new_rows)
        save(stage / 'rename_map.json', mapping)
        save(stage / 'cleaning_report.json', dict(kept=kept, removed=removed, converted=converted, renamed=renamed, missing_mask_folders=missing_masks))
        project.rename(backup)
        try:
            stage.rename(project)
        except BaseException:
            backup.rename(project)
            raise
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    if args.keep_backup:
        print(f'  Original backup: {backup}')
    else:
        shutil.rmtree(backup)
    return kept, removed, converted, renamed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--delete-missing', action='store_true', help='Permanently remove samples without overlays instead of quarantining')
    parser.add_argument('--keep-backup', action='store_true', help='Retain complete original project as hidden sibling')
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    totals = [0]*4
    failures = []
    for project in sorted(root.iterdir(), key=lambda p: p.name.casefold()):
        if not project.is_dir() or project.name.startswith('.') or not (project / 'images').is_dir():
            continue
        try:
            result = process(project, args)
            totals = [a+b for a,b in zip(totals, result)]
        except Exception as exc:
            failures.append(project.name)
            print(f'ERROR {project.name}: {exc}', flush=True)
    print(f'{"DRY RUN: " if args.dry_run else ""}Total kept={totals[0]}, removed={totals[1]}, converted={totals[2]}, renamed={totals[3]}; failed projects={len(failures)}')
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
