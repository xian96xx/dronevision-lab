"""Download official-split VisDrone archives from the mirror used by Ultralytics.
Retain original archives/annotations; verify ZIP CRC and record SHA256.
"""
import argparse
import hashlib
import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def download(split):
    folder = ROOT / 'data' / 'archives'
    folder.mkdir(parents=True, exist_ok=True)
    name = f'VisDrone2019-DET-{split}.zip'
    dest = folder / name
    url = f'https://github.com/ultralytics/assets/releases/download/v0.0.0/{name}'
    if not dest.exists():
        if shutil.disk_usage(ROOT).free < 15 * 2**30:
            raise RuntimeError('Require 15 GiB free before downloading')
        tmp = dest.with_suffix('.part')
        for attempt in range(4):
            try:
                print(f'Downloading {url}, attempt {attempt+1}', flush=True)
                with urllib.request.urlopen(url, timeout=60) as src, tmp.open('wb') as out:
                    shutil.copyfileobj(src, out, 2**20)
                with zipfile.ZipFile(tmp) as z:
                    bad = z.testzip()
                    if bad:
                        raise RuntimeError(f'Bad ZIP member: {bad}')
                tmp.replace(dest)
                break
            except Exception as e:
                print(repr(e), flush=True)
                if attempt == 3:
                    raise
                time.sleep(3)
    digest = hashlib.file_digest(dest.open('rb'), 'sha256').hexdigest()
    out = ROOT / 'data' / 'raw'
    marker = out / f'.{split}.extracted'
    if not marker.exists():
        with zipfile.ZipFile(dest) as z:
            for entry in z.infolist():
                target = (out / entry.filename).resolve()
                if not target.is_relative_to(out.resolve()):
                    raise RuntimeError('Unsafe archive path')
            z.extractall(out)
        marker.write_text(digest)
    record = {'split':split, 'url':url, 'bytes':dest.stat().st_size, 'sha256':digest}
    (ROOT / 'reports' / f'download-{split}.json').write_text(json.dumps(record, indent=2))
    print(record, flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--splits', nargs='+', default=['val', 'train', 'test-dev'])
    for split in p.parse_args().splits:
        if split not in ['train','val','test-dev']:
            raise ValueError(split)
        download(split)
