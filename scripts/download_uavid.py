"""Download official DANS UAVid images/masks, respecting archive split directories.
Validate each file against publisher-provided SHA-1 AND record local SHA256.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import requests

ROOT=Path(__file__).resolve().parents[1]
HOST='https://phys-techsciences.datastations.nl'

def main():
    meta=ROOT/'references'/'uavid-dans-metadata.json'
    if meta.exists():d=json.loads(meta.read_text())
    else:
        r=requests.get(HOST+'/api/datasets/:persistentId/',params={'persistentId':'doi:10.17026/dans-x9f-w9sa'},timeout=30)
        r.raise_for_status();d=r.json();meta.write_text(json.dumps(d,indent=2))
    files=[x for x in d['data']['latestVersion']['files'] if x['dataFile']['filename'].endswith('.png')]
    # Download official training and validation first. Test imagery is never used for fitting.
    files=[x for x in files if '/test/' not in x['directoryLabel']]
    def one(entry):
        if entry['restricted']:raise PermissionError('Restricted data file')
        f=entry['dataFile'];dest=ROOT/'data'/'uavid'/entry['directoryLabel']/f['filename']
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and dest.stat().st_size==f['filesize']:
            data=dest.read_bytes()
            if hashlib.sha1(data).hexdigest()==f['checksum']['value']:
                return {'path':str(dest.relative_to(ROOT)), 'id':f['id'], 'sha256':hashlib.sha256(data).hexdigest()}
        for attempt in range(8):
            try:
                r=requests.get(HOST+f'/api/access/datafile/{f["id"]}',timeout=(20,90));r.raise_for_status()
                data=r.content
                if len(data)!=f['filesize']:raise ValueError('Length mismatch')
                if hashlib.sha1(data).hexdigest()!=f['checksum']['value']:raise ValueError('Publisher checksum mismatch')
                tmp=dest.with_suffix('.part');tmp.write_bytes(data);tmp.replace(dest)
                print(str(dest.relative_to(ROOT)),flush=True)
                return {'path':str(dest.relative_to(ROOT)), 'id':f['id'], 'sha256':hashlib.sha256(data).hexdigest()}
            except Exception as e:
                print(f'retry {f["id"]} {attempt+1}: {e!r}',flush=True)
                if attempt==7:raise
                time.sleep(min(attempt+1,4))
    with ThreadPoolExecutor(max_workers=6) as pool:records=list(pool.map(one,files))
    report={'source':'doi:10.17026/dans-x9f-w9sa','downloaded_files':len(records),'files':records,
        'license_note':'Archive metadata says CC-BY-SA; embedded readme and official website say CC-BY-NC-SA. Follow stricter NC-SA terms.'}
    (ROOT/'reports'/'uavid-download.json').write_text(json.dumps(report,indent=2))
    print(f'Validated {len(records)} image/mask files',flush=True)

if __name__=='__main__':main()
