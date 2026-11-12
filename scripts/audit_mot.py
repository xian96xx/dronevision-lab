"""Audit all official MOT training annotations; materialize a bounded GT video sample."""
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import zipfile
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    archive=ROOT/'data'/'archives'/'VisDrone2019-MOT-train.zip'
    assert (ROOT/'reports'/'mot-train-download.json').exists(),'Require verified complete archive first'
    counters=Counter();classes=Counter();occlusions=Counter();sequences=[]
    out=ROOT/'runs'/'mot-train-audit';out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        entries=z.namelist();image_names=[n for n in entries if n.lower().endswith('.jpg') and '/sequences/' in n]
        framecounts=Counter(n.split('/')[-2] for n in image_names)
        annotation_names=sorted(n for n in entries if '/annotations/' in n and n.endswith('.txt'))
        for name in annotation_names:
            data=z.read(name);rows=np.loadtxt(io.BytesIO(data),delimiter=',',ndmin=2)
            seq=Path(name).stem
            frames=rows[:,0].astype(int);ids=rows[:,1].astype(int)
            valid=(rows[:,6]==1)&np.isin(rows[:,7],[1,4,5,6,9])
            counters['annotation_rows']+=len(rows);counters['valid_target_rows']+=int(valid.sum())
            counters['illegal_boxes']+=int(((rows[:,4]<=0)|(rows[:,5]<=0)).sum())
            counters['images']+=framecounts[seq];counters['sequences']+=1
            classes.update(rows[valid,7].astype(int).tolist());occlusions.update(rows[valid,9].astype(int).tolist())
            tracklengths=[];gaps=0
            for tid in np.unique(ids[valid]):
                fs=np.unique(frames[valid&(ids==tid)]);tracklengths.append(len(fs));gaps+=int((np.diff(fs)>1).sum())
            sequences.append({'sequence':seq,'images':framecounts[seq],'annotation_rows':len(rows),
                'valid_tracks':len(tracklengths),'track_frame_count_quantiles':np.quantile(tracklengths,[0,.25,.5,.75,1]).tolist() if tracklengths else [],
                'trajectory_gaps':gaps,'annotation_sha256':hashlib.sha256(data).hexdigest()})
            if len(sequences)==1:
                sample_names=sorted(n for n in image_names if n.split('/')[-2]==seq)[:180]
                writer=cv2.VideoWriter(str(out/f'GT-{seq}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),15,(1280,720))
                assert writer.isOpened()
                for member in sample_names:
                    im=cv2.imdecode(np.frombuffer(z.read(member),dtype=np.uint8),cv2.IMREAD_COLOR)
                    frame=int(Path(member).stem)
                    for r in rows[(frames==frame)&valid]:
                        x,y,w,h=map(int,r[2:6]);cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
                        cv2.putText(im,str(int(r[1])),(x,y),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1)
                    im=cv2.resize(im,(1280,720));cv2.putText(im,'PUBLIC GROUND TRUTH - not model predictions',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
                    writer.write(im)
                writer.release()
    report={'source':'official VisDrone2019-MOT-train, separate from DET','counts':dict(counters),
        'class_counts':dict(classes),'occlusion_counts':dict(occlusions),'sequences':sequences,
        'storage':'full validated archive retained; bounded sample decoded for visualization, avoids full redundant extraction',
        'video':'runs/mot-train-audit/GT-<sequence>.mp4; ground truth visualization only'}
    (ROOT/'reports'/'mot-train-audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
