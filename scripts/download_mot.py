"""Recover truncated official Google Drive MOT downloads using validated HTTP ranges."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import time
import subprocess
import sys
from fetch_file import fetch
ROOT=Path(__file__).resolve().parents[1]
IDS={'train':'1-qX2d-P1Xr64ke6nTdlm33om1VxCUTSh','val':'1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=IDS,default='train');p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    dest=ROOT/'data'/'archives'/f'VisDrone2019-MOT-{a.split}.zip'
    if dest.exists() and zipfile.is_zipfile(dest):
        with zipfile.ZipFile(dest) as z:
            if z.testzip() is None:
                print('Existing ZIP verified',flush=True);return
    prefix=dest.with_suffix('.truncated')
    if dest.exists():
        if not prefix.exists() or dest.stat().st_size>prefix.stat().st_size:dest.replace(prefix)
        else:dest.unlink()
    if prefix.exists():
        parts=dest.parent/(dest.name+'.parts');parts.mkdir(exist_ok=True)
        chunk=8*2**20
        # Preserve complete known prefix chunks; the final ZIP CRC validates the assembled archive.
        with prefix.open('rb') as f:
            for start in range(0,prefix.stat().st_size//chunk*chunk,chunk):
                data=f.read(chunk)
                cp=parts/str(start)
                if not cp.exists() or cp.stat().st_size!=chunk:cp.write_bytes(data)
    url=f'https://drive.usercontent.google.com/download?id={IDS[a.split]}&export=download&confirm=t'
    for attempt in range(4):
        try:
            fetch(url,dest,workers=a.workers)
            break
        except Exception as e:
            print(f'Archive resume attempt {attempt+1} failed: {e!r}',flush=True)
            if attempt==3:raise
            time.sleep(10)
    with zipfile.ZipFile(dest) as z:
        bad=z.testzip()
        if bad:raise ValueError(f'ZIP CRC error: {bad}')
        records={'split':a.split,'archive':str(dest),'source':'https://github.com/VisDrone/VisDrone-Dataset',
            'url':url,'bytes':dest.stat().st_size,'sha256':hashlib.file_digest(dest.open('rb'),'sha256').hexdigest(),
            'members':len(z.infolist()),'zip_crc':'all passed'}
    (ROOT/'reports'/f'mot-{a.split}-download.json').write_text(json.dumps(records,indent=2))
    print(records,flush=True)
    if a.split=='train':
        subprocess.run([sys.executable,str(ROOT/'scripts'/'audit_mot.py')],cwd=ROOT,check=True)

if __name__=='__main__':main()
