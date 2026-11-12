"""Actual keyframe-spacing sensitivity; exclude EVERY supplied keyframe from accuracy metrics."""
import json
from pathlib import Path
import numpy as np
from evaluate_det import overlap
ROOT=Path(__file__).resolve().parents[1]

def main():
    protocol=json.loads((ROOT/'runs'/'video'/'protocol.json').read_text());records=[]
    for gap in [10,30,60,119]:
        ious=[];centers=[];jitters=[];prompts=0;evaluated=0
        for subset in protocol['subsets']:
            n=len(subset['frames']);keys=sorted(set([*range(0,n,gap),n-1]));mask=np.ones(n,dtype=bool)
            # Common evaluation frames for EVERY spacing: exclude union of all methods' prompts.
            all_prompts=sorted(set([i for g in [10,30,60,119] for i in range(0,n,g)]+[n-1]))
            mask[all_prompts]=False
            for tid,track in subset['tracks'].items():
                gt=np.array([x['box'] for x in track]);pred=np.empty_like(gt,dtype=float)
                for start,end in zip(keys[:-1],keys[1:]):
                    alpha=np.linspace(0,1,end-start+1)[:,None]
                    pred[start:end+1]=(1-alpha)*gt[start]+alpha*gt[end]
                c=(pred[:,:2]+pred[:,2:])/2;gc=(gt[:,:2]+gt[:,2:])/2
                ious.extend(np.diag(overlap(pred[mask],gt[mask])).tolist())
                centers.extend(np.linalg.norm(c[mask]-gc[mask],axis=1).tolist())
                jitters.extend(np.linalg.norm(np.diff(c,n=2,axis=0),axis=1).tolist())
                prompts+=len(keys);evaluated+=int(mask.sum())
        records.append({'gap_frames':gap,'provided_GT_boxes':prompts,'evaluated_unprompted_boxes':evaluated,
            'mean_IoU':float(np.mean(ious)),'center_error_pixels':float(np.mean(centers)),
            'raw_predicted_center_acceleration_pixels':float(np.mean(jitters))})
    report={'protocol':'Same12 public MOT trajectories; piecewise linear xyxy interpolation. Fixed common evaluation frames exclude the union of ALL spacing conditions prompt frames. Raw acceleration is smoothness, not correctness. Input annotation budget differs between rows.', 'results':records}
    (ROOT/'reports'/'interpolation-spacing.json').write_text(json.dumps(report,indent=2));print(report)

if __name__=='__main__':main()
