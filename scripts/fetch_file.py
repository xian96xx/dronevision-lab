"""Resumable concurrent HTTP range downloader with strict length/hash validation.
No TLS bypass. Useful for connections that silently truncate large responses.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import threading

def fetch(url,dest,expected_sha256=None,workers=6,chunk=8*2**20):
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    google='drive.usercontent.google.com/' in url
    sessions=threading.local()
    def session():
        if not hasattr(sessions,'client'):
            import requests
            sessions.client=requests.Session()
        return sessions.client
    if dest.exists() and expected_sha256:
        with dest.open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()==expected_sha256:
                return dest
    for i in range(6):
        try:
            if google:
                with session().head(url,headers={'Range':'bytes=0-0'},timeout=45) as r:
                    r.raise_for_status();size=int(r.headers['Content-Range'].split('/')[-1])
            else:
                with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=45) as r:
                    size=int(r.headers['Content-Length'])
                    if r.headers.get('Content-Range'):
                        size=int(r.headers['Content-Range'].split('/')[-1])
            if size<=0:
                probe=urllib.request.Request(url,headers={'Range':'bytes=0-0'})
                with urllib.request.urlopen(probe,timeout=45) as r:
                    if r.status!=206:raise RuntimeError('Cannot determine content size')
                    size=int(r.headers['Content-Range'].split('/')[-1])
                    assert len(r.read(2))==1
            if size<=0:raise RuntimeError('Empty download is not valid')
            break
        except Exception:
            if i==5:raise
            time.sleep(2)
    parts=dest.parent/(dest.name+'.parts');parts.mkdir(exist_ok=True)
    def part(start):
        end=min(start+chunk,size)-1
        out=parts/str(start)
        if out.exists() and out.stat().st_size==end-start+1:return
        for attempt in range(12):
            try:
                headers={'Range':f'bytes={start}-{end}','Cache-Control':'no-cache'}
                if google:
                    with session().get(url,headers=headers,stream=True,timeout=(20,60)) as r:
                        if r.status_code!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
                            raise RuntimeError(f'Invalid range response {r.status_code} {r.headers.get("Content-Range")}')
                        data=r.raw.read(end-start+2)
                else:
                    req=urllib.request.Request(url,headers=headers)
                    with urllib.request.urlopen(req,timeout=60) as r:
                        if r.status!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
                            raise RuntimeError(f'Invalid range response {r.status} {r.headers.get("Content-Range")}')
                        data=r.read(end-start+2)
                if len(data)!=end-start+1:raise RuntimeError(f'Truncated chunk {start}: {len(data)}')
                out.write_bytes(data)
                print(f'{dest.name}: {end+1}/{size}',flush=True)
                return
            except Exception as e:
                print(f'retry {start} #{attempt+1}: {e!r}',flush=True)
                if attempt==11:raise
                time.sleep(min(attempt+1,5))
    starts=list(range(0,size,chunk))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(part,starts))
    tmp=dest.with_suffix(dest.suffix+'.partial')
    digest=hashlib.sha256()
    with tmp.open('wb') as f:
        for start in starts:
            data=(parts/str(start)).read_bytes();f.write(data);digest.update(data)
    sha=digest.hexdigest()
    if expected_sha256 and sha!=expected_sha256:raise RuntimeError(f'SHA256 mismatch: {sha}')
    assert tmp.stat().st_size==size
    tmp.replace(dest)
    dest.with_suffix(dest.suffix+'.download.json').write_text(json.dumps({'url':url,'bytes':size,'sha256':sha},indent=2))
    # Delete only verified chunk files after assembly; retain directory for provenance.
    for start in starts:(parts/str(start)).unlink()
    print(f'Verified {dest.name}: {sha}',flush=True)
    return dest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('url');p.add_argument('dest');p.add_argument('--sha256');p.add_argument('--workers',type=int,default=6)
    a=p.parse_args();fetch(a.url,a.dest,a.sha256,a.workers)
