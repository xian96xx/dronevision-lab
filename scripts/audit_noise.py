"""Measure the injected label interventions relative to unmodified public annotations."""
import json
from pathlib import Path
import numpy as np
from evaluate_det import overlap
ROOT=Path(__file__).resolve().parents[1]

def labels(p):
    return np.array([[float(x) for x in row.split()] for row in p.read_text().splitlines()],dtype=float).reshape(-1,5)

def boxes(a):return np.column_stack([a[:,1:3]-a[:,3:5]/2,a[:,1:3]+a[:,3:5]/2])

def main():
    protocol=json.loads((ROOT/'reports'/'noise-protocol.json').read_text())
    dims={Path(x['image']).name:(x['width'],x['height']) for x in json.loads((ROOT/'reports'/'manifest-train.json').read_text())}
    stats={}
    for variant in ['clean','shift','drop']:
        count=original=0;iou=[];center=[];classcounts=np.zeros(10,dtype=int)
        for name in protocol['images']:
            lp=Path(name).with_suffix('.txt').name
            clean=labels(ROOT/'data'/'noise-clean'/'labels'/'train'/lp)
            noise=labels(ROOT/'data'/f'noise-{variant}'/'labels'/'train'/lp)
            original+=len(clean);count+=len(noise)
            if len(noise):classcounts+=np.bincount(noise[:,0].astype(int),minlength=10)
            if variant!='drop':
                assert np.array_equal(clean[:,0],noise[:,0])
                # Pair by the original annotation row, not best-match IoU.
                iou.extend(np.diag(overlap(boxes(clean),boxes(noise))).tolist())
                w,h=dims[name]
                center.extend(np.linalg.norm((noise[:,1:3]-clean[:,1:3])*[w,h],axis=1).tolist())
        stats[variant]={'images':len(protocol['images']),'original_boxes':original,'retained_boxes':count,
            'removed_fraction':1-count/original,'class_counts':classcounts.tolist(),
            'paired_box_mean_IoU':float(np.mean(iou)) if iou else None,
            'paired_IoU_quantiles':np.quantile(iou,[0,.25,.5,.75,1]).tolist() if iou else None,
            'mean_center_error_pixels':float(np.mean(center)) if center else None}
    result={'reference':'original public labels, not newly audited perfect ground truth','noise_seed':42,
        'variants':stats,'interpretation':'Localization IoU and retained-label coverage are different quality dimensions. Do not collapse them into one unsupported quality score.'}
    (ROOT/'reports'/'noise-label-quality.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
