#!/usr/bin/env python3
"""Merge cleaned Labelbox projects without changing source files."""
import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parent / 'labelbox_export_multi' / 'projects copy'
EXT = {'.png', '.jpg', '.jpeg', '.bmp'}


def read_json(path):
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def write_json(path, value):
    with path.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write('\n')


def indexed(folder, overlay=False):
    result = {}
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.suffix.lower() not in EXT:
            continue
        stem = p.stem
        if overlay:
            if not stem.endswith('_overlay'):
                continue
            stem = stem[:-8]
        if stem in result:
            raise ValueError(f'Duplicate sample: {folder}/{stem}')
        result[stem] = p
    return result


def plan(root, output, allow_missing_masks):
    jobs, rows, report, seen = [], [], {}, set()
    for project in sorted(root.iterdir()):
        if not project.is_dir() or project.name.startswith('.') or project == output:
            continue
        if not (project / 'images').is_dir():
            continue
        images = indexed(project / 'images')
        overlays = indexed(project / 'overlays', True)
        masks = project / 'object_masks'
        if not masks.is_dir():
            raise FileNotFoundError(masks)
        manifest = read_json(project / 'manifest.json')
        if not isinstance(manifest, list):
            raise ValueError(f'Expected list in {project}/manifest.json')
        lookup = {}
        for row in manifest:
            name = Path(row.get('image_filename') or '').name
            if not name or name in lookup:
                raise ValueError(f'Invalid or duplicate manifest filename in {project}: {name}')
            lookup[name] = row
        stats = dict(images=0, missing_masks=0)
        for stem, img in images.items():
            if not re.fullmatch(re.escape(project.name) + r'-P-\d{5}', stem):
                raise ValueError(f'Uncleaned or unexpected sample ID: {img}')
            if stem.casefold() in seen:
                raise ValueError(f'Duplicate global ID: {stem}')
            seen.add(stem.casefold())
            if stem not in overlays:
                raise ValueError(f'Missing overlay: {img}')
            if img.name not in lookup:
                raise ValueError(f'Missing manifest entry: {img}')
            mask = masks / stem
            if not mask.is_dir():
                if not allow_missing_masks:
                    raise ValueError(f'Missing mask folder: {mask} (use --allow-missing-masks only if intentional)')
                mask = None
                stats['missing_masks'] += 1
            row = dict(lookup[img.name])
            row.update(image_filename=img.name, mask_directory=stem if mask else None,
                       external_id=stem, source_project=project.name,
                       overlay_filename=overlays[stem].name)
            rows.append(row)
            jobs.append((img, overlays[stem], mask))
            stats['images'] += 1
        stats['orphan_overlays_excluded'] = len(set(overlays) - set(images))
        stats['extra_manifest_rows_excluded'] = len(set(lookup) - {p.name for p in images.values()})
        stats['orphan_mask_folders_excluded'] = sum(p.is_dir() and p.name not in images for p in masks.iterdir())
        report[project.name] = stats
        print(f'{project.name}: {stats}', flush=True)
    if not jobs:
        raise ValueError('No samples found to merge')
    return jobs, rows, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path, help='New output directory; defaults to merged_dataset beside root')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--allow-missing-masks', action='store_true')
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else root.parent / 'merged_dataset'
    if not root.is_dir():
        raise FileNotFoundError(root)
    if output.exists() or output == root or root in output.parents:
        raise ValueError('Output must be a new directory outside the project root')
    jobs, rows, projects = plan(root, output, args.allow_missing_masks)
    print(f'Total: {len(jobs)} samples from {len(projects)} projects', flush=True)
    if args.dry_run:
        print(f'Would create: {output}')
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.merge-', dir=output.parent))
    try:
        for name in ('images', 'overlays', 'object_masks'):
            (stage / name).mkdir()
        for i, (img, overlay, mask) in enumerate(jobs, 1):
            shutil.copy2(img, stage / 'images' / img.name)
            shutil.copy2(overlay, stage / 'overlays' / overlay.name)
            if mask:
                shutil.copytree(mask, stage / 'object_masks' / mask.name)
            if i % 100 == 0 or i == len(jobs):
                print(f'Copied {i}/{len(jobs)} samples', flush=True)
        write_json(stage / 'manifest.json', rows)
        write_json(stage / 'merge_report.json', {'total_samples': len(jobs), 'projects': projects})
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(f'Finished: {output}')


if __name__ == '__main__':
    main()
