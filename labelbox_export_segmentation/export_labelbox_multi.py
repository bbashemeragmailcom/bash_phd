#!/usr/bin/env python3
"""Resumable Done-only Labelbox image segmentation exporter (one writer per output).
Run with PROJECT_IDS in .env or --projects ID ID. Use a new --output for a fresh snapshot.
Raw export and checkpoints contain sensitive URLs/metadata; keep output private.
"""
import argparse
import colorsys
import hashlib
import json
import logging
import os
import random
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
import fcntl  # macOS/Linux exclusive run lock

import certifi
import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

LOG = logging.getLogger('labelbox_export')
PARAMS = dict(attachments=True, data_row_details=True, project_details=True, label_details=True)
FILTERS = {'workflow_status': 'Done'}


def now():
    return datetime.now(timezone.utc).isoformat()


def safe(s):
    return re.sub(r'[^a-zA-Z0-9_-]+', '_', str(s)).strip('_')[:100] or 'unnamed'


def digest(s):
    return hashlib.sha256(str(s).encode()).hexdigest()[:20]


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.part')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_array(path, values):
    tmp = path.with_name(path.name + '.part')
    with tmp.open('w', encoding='utf-8') as f:
        f.write('[\n')
        for i, v in enumerate(values):
            if i: f.write(',\n')
            f.write(json.dumps(v, ensure_ascii=False))
        f.write('\n]\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def project_dir(root, pid):
    return root / 'projects' / (safe(pid) + '_' + digest(pid))


def db_open(folder):
    folder.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(folder / 'checkpoint.sqlite3')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    db.executescript('''CREATE TABLE IF NOT EXISTS rows (
        id TEXT PRIMARY KEY, raw TEXT NOT NULL, result TEXT, status TEXT DEFAULT 'pending');
        CREATE INDEX IF NOT EXISTS rows_status ON rows(status);
        CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);''')
    return db


def get_setting(db, key):
    row = db.execute('SELECT value FROM settings WHERE key=?', (key,)).fetchone()
    return row[0] if row else None


def set_setting(db, key, value):
    db.execute('INSERT OR REPLACE INTO settings VALUES (?,?)', (key, str(value)))


def objects(record, pid):
    p = record.get('projects', {}).get(pid)
    if p is None: raise ValueError('Requested project missing from export row')
    for label in p.get('labels', []):
        for obj in label.get('annotations', {}).get('objects', []):
            if obj.get('annotation_kind') == 'ImageSegmentationMask':
                yield obj


def snapshot(client, lb, pid, folder, db):
    if get_setting(db, 'snapshot_complete') == 'yes':
        return
    # Partial metadata exports are restarted; downloaded assets are never touched here.
    project = client.get_project(project_id=pid)
    task = project.export(params=PARAMS, filters=FILTERS)
    task.wait_till_done(timeout_seconds=14400)
    if task.has_errors():
        raise RuntimeError('Labelbox reported export errors; no partial snapshot accepted')
    with db:
        db.execute('DELETE FROM rows')
        if task.has_result():
            stream = task.get_buffered_stream(stream_type=lb.StreamType.RESULT)
            for item in tqdm(stream, desc=f'{pid}: export metadata', unit='row'):
                record = item.json
                if isinstance(record, str): record = json.loads(record)
                rid = record['data_row']['id']
                p = record.get('projects', {}).get(pid)
                if p is None: raise ValueError('Missing requested project')
                status = p.get('project_details', {}).get('workflow_status')
                if status is not None and str(status).lower() != 'done':
                    raise ValueError('Unexpected non-Done workflow status')
                db.execute('INSERT INTO rows(id,raw) VALUES (?,?)', (rid, json.dumps(record)))
        set_setting(db, 'snapshot_complete', 'yes')
        set_setting(db, 'snapshot_at', now())
        set_setting(db, 'project_name', getattr(project, 'name', pid))
    # Export is persisted in SQLite first, so interruption here is recoverable.


def raw_records(db):
    for (raw,) in db.execute('SELECT raw FROM rows ORDER BY id'):
        yield json.loads(raw)


def registry(root, prepared):
    path = root / 'metadata.json'
    meta = json.loads(path.read_text()) if path.exists() else {'background': {'id': 0, 'name': 'background'}, 'classes': {}}
    names = set()
    for pid, folder, db in prepared:
        for record in raw_records(db):
            for obj in objects(record, pid):
                name = obj.get('value') or obj.get('name')
                if isinstance(name, str) and name: names.add(name)
    classes = meta['classes']
    next_id = max((v['id'] for v in classes.values()), default=0) + 1
    for name in sorted(names - classes.keys()):
        rgb = colorsys.hsv_to_rgb((next_id * .61803398875) % 1, .8, 1)
        classes[name] = {'id': next_id, 'name': name, 'overlay_rgb': [round(x*255) for x in rgb]}
        next_id += 1
    meta['num_foreground_classes'] = len(classes)
    meta['mask_format'] = 'individual_binary_object_masks'
    atomic_json(path, meta)
    return meta


def inspect_image(path, size=None, mask=False):
    with Image.open(path) as im:
        im.verify()
    with Image.open(path) as im:
        im.load()
        if size is not None and im.size != size: raise ValueError('Mask dimensions mismatch')
        if mask:
            if im.format != 'PNG': raise ValueError('Expected PNG mask')
            a = binary(im)
            if not a.any(): raise ValueError('Empty foreground mask')
        return im.size


def binary(im):
    # Individual binary masks: black background, nonblack foreground.
    # For transparent masks, alpha defines the visible region if it varies.
    if 'A' in im.getbands() or 'transparency' in im.info:
        rgba = np.asarray(im.convert('RGBA'))
        alpha = rgba[..., 3]
        if np.any(alpha == 0) and np.any(alpha > 0): return alpha > 0
        return np.any(rgba[..., :3] > 0, axis=-1) & (alpha > 0)
    if im.mode == 'P': return np.any(np.asarray(im.convert('RGB')) > 0, axis=-1)
    a = np.asarray(im)
    return a > 0 if a.ndim == 2 else np.any(a[..., :3] > 0, axis=-1)


def download(session, url, path, args, headers=None, size=None, mask=False):
    if path.exists():
        try: return inspect_image(path, size, mask), 'existing'
        except (OSError, ValueError, SyntaxError): path.unlink()
    if not isinstance(url, str) or urlsplit(url).scheme != 'https':
        raise ValueError('Missing or non-HTTPS asset URL')
    # Never send the API key to an unrelated host. Requests strips Authorization on cross-host redirects.
    if headers and not (urlsplit(url).hostname or '').endswith('.labelbox.com') and urlsplit(url).hostname != 'labelbox.com':
        headers = None
    if headers:
        headers = {k:v for k,v in headers.items() if k.lower() == 'authorization'}
    tmp = path.with_name(path.name + '.part')
    for attempt in range(1, args.retries + 1):
        try:
            with session.get(url, headers=headers, stream=True, timeout=(15, args.read_timeout)) as r:
                r.raise_for_status()
                if urlsplit(r.url).scheme != 'https': raise ValueError('Non-HTTPS redirect')
                with tmp.open('wb') as f:
                    for chunk in r.iter_content(256 * 1024):
                        if chunk: f.write(chunk)
                    f.flush(); os.fsync(f.fileno())
            dimensions = inspect_image(tmp, size, mask)
            os.replace(tmp, path)
            return dimensions, 'downloaded'
        except (requests.RequestException, OSError, ValueError, SyntaxError) as exc:
            tmp.unlink(missing_ok=True)
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
            permanent = isinstance(exc, requests.exceptions.SSLError) or (status is not None and 400 <= status < 500 and status not in (408, 425, 429))
            reason = f'HTTP {status}' if status else type(exc).__name__
            if permanent or attempt == args.retries: raise RuntimeError(reason) from None
            delay = min(60, 2**attempt + random.random())
            retry_after = getattr(getattr(exc, 'response', None), 'headers', {}).get('Retry-After')
            if retry_after and retry_after.isdigit(): delay = min(300, max(delay, int(retry_after)))
            LOG.warning('Retry %s/%s for %s (%s) in %.1fs', attempt, args.retries, path.name, reason, delay)
            time.sleep(delay)


def process_row(record, pid, folder, meta, session, headers, args):
    row = record['data_row']; rid = row['id']; external = row.get('external_id') or rid
    stem = safe(Path(external).stem) + '_' + digest(rid)
    suffix = Path(external).suffix.lower()
    if suffix not in {'.png','.jpg','.jpeg','.tif','.tiff','.bmp','.webp'}: suffix = '.png'
    image_path = folder / 'images' / (stem + suffix)
    mask_dir = folder / 'object_masks' / stem; mask_dir.mkdir(exist_ok=True)
    info = {'external_id': external, 'image_stem': stem, 'objects': []}
    result = {'data_row_id': rid, 'external_id': external, 'image_filename': image_path.name,
              'mask_directory': stem, 'image_downloaded': False, 'overlay_created': False,
              'valid_objects': 0, 'invalid_objects': 0, 'errors': []}
    size = None
    try:
        size, _ = download(session, row.get('row_data'), image_path, args)
        result['image_downloaded'] = True
    except Exception as exc:
        result['errors'].append('image: ' + str(exc))
    # Keep each instance separate, including overlapping biomarkers.
    for index, obj in enumerate(objects(record, pid)):
        name = obj.get('value') or obj.get('name')
        fid = obj.get('feature_id')
        entry = {'class_name': name, 'class_id': meta['classes'].get(name, {}).get('id'),
                 'feature_id': fid, 'feature_schema_id': obj.get('feature_schema_id'),
                 'mask_filename': f'{safe(name)}_{index:04d}_{digest(fid or index)}.png'}
        try:
            if entry['class_id'] is None: raise ValueError('Missing class name')
            if size is None: raise ValueError('Source image unavailable; validation deferred')
            _, status = download(session, (obj.get('mask') or {}).get('url'), mask_dir / entry['mask_filename'], args, headers, size, True)
            entry['download_status'] = status
            result['valid_objects'] += 1
        except Exception as exc:
            entry.update(download_status='failed', error=str(exc))
            result['invalid_objects'] += 1
        info['objects'].append(entry)
    info['num_objects'] = len(info['objects']); result['num_objects'] = info['num_objects']
    atomic_json(mask_dir / 'annotation_info.json', info)
    overlay = folder / 'overlays' / (stem + '_overlay.png')
    if size and not args.skip_overlays and not result['invalid_objects']:
        try:
            with Image.open(image_path) as im: base = np.asarray(im.convert('RGB')).astype(np.float32)
            accum = np.zeros_like(base); count = np.zeros(base.shape[:2], dtype=np.uint32)
            # Average the colors of overlapping instances; visualization only.
            for entry in info['objects']:
                with Image.open(mask_dir / entry['mask_filename']) as im: b = binary(im)
                accum[b] += meta['classes'][entry['class_name']]['overlay_rgb']; count[b] += 1
            fg = count > 0
            base[fg] = .55 * base[fg] + .45 * (accum[fg] / count[fg, None])
            tmp = overlay.with_name(overlay.name + '.part')
            Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).save(tmp, format='PNG')
            os.replace(tmp, overlay); result['overlay_created'] = True
            result['foreground_pixels'] = int(fg.sum())
        except Exception as exc: result['errors'].append('overlay: ' + type(exc).__name__)
    result['complete'] = bool(size and not result['invalid_objects'] and not result['errors'] and (args.skip_overlays or result['overlay_created']))
    return result


def progress(db, pid, folder, state, error=None):
    counts = dict(db.execute('SELECT status, count(*) FROM rows GROUP BY status'))
    total = sum(counts.values()); complete = counts.get('complete', 0)
    p = dict(project_id=pid, project_name=get_setting(db,'project_name'), state=state,
             snapshot_at=get_setting(db,'snapshot_at'), total=total, completed=complete,
             failed=counts.get('failed',0), pending=counts.get('pending',0),
             percent=round(100*complete/total,2) if total else (100 if state=='complete' else 0), updated_at=now())
    if error: p['error'] = error
    atomic_json(folder/'progress.json', p)
    return p


def outputs(db, folder):
    atomic_array(folder/'manifest.json', (json.loads(r[0]) for r in db.execute('SELECT result FROM rows WHERE result IS NOT NULL ORDER BY id')))
    atomic_array(folder/'failed_rows.json', (json.loads(r[0]) for r in db.execute("SELECT result FROM rows WHERE status='failed' ORDER BY id")))


def run_project(pid, folder, db, meta, session, headers, args, report):
    for sub in ('images','object_masks','overlays'): (folder/sub).mkdir(exist_ok=True)
    atomic_json(folder/'metadata.json', dict(meta, project_id=pid))
    if not (folder/'export.json').exists(): atomic_array(folder/'export.json', raw_records(db))
    # Recheck completed rows if requested, or when overlays are newly requested.
    if args.verify_existing:
        with db: db.execute("UPDATE rows SET status='pending' WHERE status='complete'")
    elif not args.skip_overlays and get_setting(db,'overlay_mode') != 'yes':
        with db: db.execute("UPDATE rows SET status='pending' WHERE status='complete'")
    with db: set_setting(db,'overlay_mode','no' if args.skip_overlays else 'yes')
    total = db.execute('SELECT count(*) FROM rows').fetchone()[0]
    done = db.execute("SELECT count(*) FROM rows WHERE status='complete'").fetchone()[0]
    report(progress(db,pid,folder,'downloading'))
    try:
        # Keyset pagination avoids loading the dataset or keeping a read cursor open during writes.
        last = ''
        with tqdm(total=total, initial=done, desc=f'{pid}: rows', unit='row') as bar:
            while True:
                batch = db.execute("SELECT id,raw FROM rows WHERE id>? AND status!='complete' ORDER BY id LIMIT 100", (last,)).fetchall()
                if not batch: break
                for rid, raw in batch:
                    last = rid
                    try: result = process_row(json.loads(raw),pid,folder,meta,session,headers,args)
                    except Exception as exc:
                        result = {'data_row_id':rid,'complete':False,'errors':[type(exc).__name__]}
                    with db:
                        db.execute('UPDATE rows SET result=?,status=? WHERE id=?', (json.dumps(result),'complete' if result['complete'] else 'failed',rid))
                    bar.update(1)
                    p = progress(db,pid,folder,'downloading')
                    report(p)
                    bar.set_postfix(ok=p['completed'],failed=p['failed'],refresh=False)
    except KeyboardInterrupt:
        outputs(db,folder); progress(db,pid,folder,'interrupted'); raise
    outputs(db,folder)
    failed = db.execute("SELECT count(*) FROM rows WHERE status!='complete'").fetchone()[0]
    p = progress(db,pid,folder,'incomplete' if failed else 'complete')
    summary = dict(p)
    for (raw,) in db.execute('SELECT result FROM rows WHERE result IS NOT NULL'):
        item = json.loads(raw)
        for key in ('valid_objects','invalid_objects','num_objects','image_downloaded','overlay_created'):
            summary[key] = summary.get(key,0) + int(item.get(key,0))
    atomic_json(folder/'export_summary.json', summary)
    return p


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--projects', nargs='+', help='Project IDs (space or comma separated)')
    parser.add_argument('--output', type=Path, default=Path('labelbox_export_multi'))
    parser.add_argument('--skip-overlays', action='store_true')
    parser.add_argument('--verify-existing', action='store_true', help='Revalidate completed rows and repair corrupt assets')
    parser.add_argument('--retries', type=int, default=5)
    parser.add_argument('--read-timeout', type=int, default=120)
    args = parser.parse_args()
    ids = list(dict.fromkeys(re.split(r'[,\s]+', ' '.join(args.projects or [os.getenv('PROJECT_IDS') or os.getenv('PROJECT_ID','')]).strip())))
    ids = [p for p in ids if p]
    key = os.getenv('API_KEY')
    if not key or not ids: parser.error('Set API_KEY and PROJECT_IDS in .env, or use --projects')
    if args.retries < 1 or args.read_timeout < 1: parser.error('Retry and timeout must be positive')
    root = args.output; root.mkdir(parents=True,exist_ok=True)
    with (root/'.run.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: parser.error('Another exporter is using this output directory')
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=[logging.StreamHandler(),logging.FileHandler(root/'export.log')])
        import labelbox as lb
        client = lb.Client(key)
        prepared=[]; states={}
        for pid in ids:
            old = project_dir(root,pid)/'progress.json'
            states[pid] = json.loads(old.read_text()) if old.exists() else {'project_id':pid,'state':'queued'}
        def overall(): atomic_json(root/'project_progress.json', {'updated_at':now(),'projects':states})
        def report(p):
            states[p['project_id']] = p; overall()
        try:
            for pid in ids:
                folder=project_dir(root,pid); db=db_open(folder)
                try:
                    states[pid]=progress(db,pid,folder,'preparing'); overall()
                    snapshot(client,lb,pid,folder,db)
                    states[pid]=progress(db,pid,folder,'ready'); prepared.append((pid,folder,db))
                except Exception as exc:
                    states[pid]=progress(db,pid,folder,'export_failed',type(exc).__name__); db.close()
                    LOG.error('Project %s export failed: %s',pid,type(exc).__name__)
                except KeyboardInterrupt:
                    states[pid]=progress(db,pid,folder,'interrupted'); db.close(); raise
                overall()
            meta=registry(root,prepared)
            with requests.Session() as session:
                session.verify=os.getenv('REQUESTS_CA_BUNDLE') or certifi.where()
                for pid,folder,db in prepared:
                    try:
                        states[pid]=run_project(pid,folder,db,meta,session,client.headers,args,report)
                    except KeyboardInterrupt:
                        states[pid]=progress(db,pid,folder,'interrupted'); raise
                    except Exception as exc:
                        states[pid]=progress(db,pid,folder,'failed',type(exc).__name__)
                        LOG.error('Project %s failed: %s',pid,type(exc).__name__)
                    overall()
        except KeyboardInterrupt:
            LOG.warning('Interrupted; rerun the same command to resume'); return 130
        finally:
            overall()
            for _,_,db in prepared: db.close()
        return 0 if all(states.get(pid,{}).get('state')=='complete' for pid in ids) else 1


if __name__ == '__main__':
    raise SystemExit(main())
