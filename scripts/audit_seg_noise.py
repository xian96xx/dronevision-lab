"""Measure actual mask changes on the exact fixed segmentation-noise subset."""
import json
from pathlib import Path
import random
import numpy as np
from PIL import Image
from segmentation import ROOT,Crops,NAMES,PALETTE

def main():
    rows=json.loads((ROOT/'data'/'seg'/'patches.json').read_text())
    keys=sorted(set((r['source_sequence'],r['source_frame']) for r in rows))
    selected=set(random.Random(42).sample(keys,40))
    rows=[r for r in rows if (r['source_sequence'],r['source_frame']) in selected]
    result={};previews={}
    for variant in ['clean','boundary','wrong']:
        dataset=Crops(rows,variant);total=np.zeros(8,dtype=np.int64);changed=np.zeros(8,dtype=np.int64)
        for i,row in enumerate(rows):
            original=np.array(Image.open(row['mask']));_,label=dataset[i];noise=label.numpy();different=noise!=original
            total+=np.bincount(original.ravel(),minlength=8);changed+=np.bincount(original[different],minlength=8)
            if i<3:previews.setdefault(i,[]).append(PALETTE[noise.astype('uint8')])
        result[variant]={'patches':len(rows),'changed_pixels':int(changed.sum()),'total_pixels':int(total.sum()),
            'changed_fraction':float(changed.sum()/total.sum()),
            'changed_fraction_by_original_class':{n:float(c/max(t,1)) for n,c,t in zip(NAMES,changed,total)}}
    out=ROOT/'runs'/'seg-label-audit';out.mkdir(exist_ok=True)
    for i,panels in previews.items():
        source=np.array(Image.open(rows[i]['image']).convert('RGB'))
        Image.fromarray(np.concatenate([source,*panels],axis=1)).save(out/f'image-clean-boundary-wrong-{i}.jpg')
    report={'seed':42,'source_image_count':40,'patch_count':len(rows),'selected_sources':[list(x) for x in sorted(selected)],
        'variants':result,'visualization':'runs/seg-label-audit; source / clean / boundary / wrong. These are label variants, not model predictions.'}
    (ROOT/'reports'/'seg-noise-label-quality.json').write_text(json.dumps(report,indent=2));print(report)

if __name__=='__main__':main()
