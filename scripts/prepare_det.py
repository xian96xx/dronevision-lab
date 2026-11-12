"""Immutable raw annotations, verified YOLO conversion and deterministic noise variants."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import numpy as np
from PIL import Image, ImageDraw
import yaml

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor']

def link(src, dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():
        try:
            os.link(src,dst)
        except OSError:
            shutil.copy2(src,dst)

def config(name, train, val):
    p = ROOT/'configs'/f'{name}.yaml'
    p.write_text(yaml.safe_dump({'path':str(ROOT/'data'/'det'), 'train':str(train), 'val':str(val),
        'names':dict(enumerate(NAMES))},sort_keys=False),encoding='utf8')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--subset',type=int,default=256)
    p.add_argument('--smoke',type=int,default=32)
    p.add_argument('--splits',nargs='+',choices=['train','val','test-dev'],default=['train','val','test-dev'])
    args=p.parse_args()
    previous=ROOT/'reports'/'data-quality.json'
    records=json.loads(previous.read_text()) if previous.exists() else {}
    for split in args.splits:
        raw=ROOT/'data'/'raw'/f'VisDrone2019-DET-{split}'
        if not (raw/'images').exists():
            continue
        stats=Counter(); classes=Counter(); sizes=[]; manifest=[]; anomalies=[]
        images=sorted((raw/'images').glob('*.jpg'))
        for img in images:
            try:
                with Image.open(img) as im:
                    im.verify()
                with Image.open(img) as im:
                    w,h=im.size
            except Exception as e:
                anomalies.append({'image':img.name,'error':str(e)})
                stats['corrupt_image']+=1
                continue
            ann=raw/'annotations'/f'{img.stem}.txt'
            if not ann.exists():
                raise FileNotFoundError(ann)
            rows=[]
            for lineno,line in enumerate(ann.read_text().splitlines(),1):
                fields=[float(x) for x in line.strip().strip(',').split(',')]
                if len(fields)<8 or not np.isfinite(fields).all():
                    raise ValueError(f'Malformed annotation {ann}:{lineno}')
                x,y,bw,bh,score,cat,trunc,occ=fields[:8]
                stats['raw_boxes']+=1
                if score==0 or not 1<=cat<=10:
                    stats['ignored_boxes']+=1
                    continue
                x1,y1=max(0,x),max(0,y)
                x2,y2=min(w,x+bw),min(h,y+bh)
                if x2<=x1 or y2<=y1:
                    stats['invalid_boxes']+=1
                    anomalies.append({'image':img.name,'line':lineno,'reason':'nonpositive/clipped area'})
                    continue
                if (x,y,x+bw,y+bh)!=(x1,y1,x2,y2):
                    stats['clipped_boxes']+=1
                bw,bh=x2-x1,y2-y1
                sizes.append([bw,bh,bw*bh,(bw*640/max(w,h))*(bh*640/max(w,h))])
                classes[NAMES[int(cat)-1]]+=1
                rows.append(f'{int(cat)-1} {(x1+x2)/2/w:.8f} {(y1+y2)/2/h:.8f} {bw/w:.8f} {bh/h:.8f}\n')
            dst=ROOT/'data'/'det'/'images'/split/img.name
            link(img,dst)
            label=ROOT/'data'/'det'/'labels'/split/f'{img.stem}.txt'
            label.parent.mkdir(parents=True,exist_ok=True)
            label.write_text(''.join(rows))
            manifest.append({'image':str(dst),'width':w,'height':h,'boxes':len(rows),
                'image_sha256':hashlib.file_digest(img.open('rb'),'sha256').hexdigest(),
                'annotation_sha256':hashlib.file_digest(ann.open('rb'),'sha256').hexdigest()})
            stats['images']+=1;stats['valid_boxes']+=len(rows);stats['empty_labels']+=not rows
        a=np.array(sizes)
        records[split]={'counts':dict(stats),'classes':dict(classes),
            'small_area_lt_32_squared_native_ratio':float((a[:,2]<1024).mean()),
            'small_area_lt_32_squared_at_640_ratio':float((a[:,3]<1024).mean()),
            'box_width_height_area_quantiles':np.quantile(a[:,:3],[0,.25,.5,.75,1],axis=0).tolist(),
            'anomalies':anomalies}
        (ROOT/'reports'/f'manifest-{split}.json').write_text(json.dumps(manifest,indent=2))
        print(split,records[split],flush=True)
        if split=='val':
            rng=random.Random(42)
            for item in rng.sample(manifest,min(6,len(manifest))):
                ip=Path(item['image'])
                im=Image.open(ip).convert('RGB');draw=ImageDraw.Draw(im);w,h=im.size
                for row in (ROOT/'data'/'det'/'labels'/split/f'{ip.stem}.txt').read_text().splitlines():
                    cat,cx,cy,bw,bh=map(float,row.split())
                    draw.rectangle([(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h],outline='lime',width=2)
                im.thumbnail((1280,960))
                im.save(ROOT/'reports'/f'labels-{ip.name}')
    (ROOT/'reports'/'data-quality.json').write_text(json.dumps(records,indent=2))
    if 'train' not in args.splits:
        print('Selected split audit completed; existing training variants unchanged.',flush=True)
        return
    base=ROOT/'data'/'det'; val=base/'images'/'val'
    config('det-full',base/'images'/'train',val)
    all_train=sorted((base/'images'/'train').glob('*.jpg'))
    if not all_train:
        print('Training archive not ready; validation audit saved. Rerun after download.',flush=True)
        return
    rng=random.Random(42); selected=rng.sample(all_train,min(args.subset,len(all_train)))
    noise_manifest=[]
    for variant in ['clean','shift','drop']:
        for ip in selected:
            out=ROOT/'data'/f'noise-{variant}'
            link(ip,out/'images'/'train'/ip.name)
            original=(base/'labels'/'train'/f'{ip.stem}.txt').read_text().splitlines()
            local=random.Random(int(hashlib.sha256((ip.stem+'42').encode()).hexdigest()[:16],16))
            lines=[]
            for i,row in enumerate(original):
                cat,x,y,w,h=map(float,row.split())
                if variant=='drop' and local.random()<.3:
                    noise_manifest.append({'variant':variant,'image':ip.name,'row':i,'action':'removed'})
                    continue
                if variant=='shift':
                    # Translate by independent uniform offsets up to 20% of bbox size.
                    x=float(np.clip(x+local.uniform(-.2,.2)*w,w/2,1-w/2))
                    y=float(np.clip(y+local.uniform(-.2,.2)*h,h/2,1-h/2))
                lines.append(f'{int(cat)} {x:.8f} {y:.8f} {w:.8f} {h:.8f}\n')
            lp=out/'labels'/'train'/f'{ip.stem}.txt';lp.parent.mkdir(parents=True,exist_ok=True)
            lp.write_text(''.join(lines))
        config(f'noise-{variant}',out/'images'/'train',val)
    for ip in selected[:args.smoke]:
        link(ip,ROOT/'data'/'smoke'/'images'/'train'/ip.name)
        link(base/'labels'/'train'/f'{ip.stem}.txt',ROOT/'data'/'smoke'/'labels'/'train'/f'{ip.stem}.txt')
    for ip in sorted(val.glob('*.jpg'))[:16]:
        link(ip,ROOT/'data'/'smoke'/'images'/'val'/ip.name)
        link(base/'labels'/'val'/f'{ip.stem}.txt',ROOT/'data'/'smoke'/'labels'/'val'/f'{ip.stem}.txt')
    config('smoke',ROOT/'data'/'smoke'/'images'/'train',ROOT/'data'/'smoke'/'images'/'val')
    experiment={'seed':42,'subset_size':len(selected),'images':[x.name for x in selected],
        'shift':'uniform independent x/y translation +-20% width/height, clipped to image',
        'drop':'Bernoulli 0.3 per box','validation':'official full 548 validation images, always clean',
        'noise_events':noise_manifest}
    (ROOT/'reports'/'noise-protocol.json').write_text(json.dumps(experiment,indent=2))
    assert not ({p.name for p in all_train}&{p.name for p in val.glob('*.jpg')})

if __name__=='__main__':main()
