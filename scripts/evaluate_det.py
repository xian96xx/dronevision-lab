"""Shared clean-validation COCO-style evaluator with explicit ontology and ignore regions.
This is a documented local protocol, NOT the official VisDrone challenge evaluator.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
os.environ['YOLO_CONFIG_DIR']=str(Path(__file__).resolve().parents[1]/'work-config')
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO
from prepare_det import NAMES, ROOT

COMMON=['human','bicycle','car','truck','bus','motorcycle']
DET_MAP={0:0,1:0,2:1,3:2,5:3,8:4,9:5}
COCO_MAP={0:0,1:1,2:2,7:3,5:4,3:5}

def overlap(a,b,ioa=False):
    if len(b)==0:return np.zeros((len(a),0))
    a=np.asarray(a).reshape(-1,4);b=np.asarray(b).reshape(-1,4)
    lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,2:],b[None,:,2:])
    wh=np.maximum(hi-lo,0);inter=wh[:,:,0]*wh[:,:,1]
    area=np.prod(a[:,2:]-a[:,:2],axis=1)[:,None]
    den=area if ioa else area+np.prod(b[:,2:]-b[:,:2],axis=1)[None,:]-inter
    return inter/np.maximum(den,1e-9)

def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--name',required=True)
    p.add_argument('--ontology',choices=['visdrone','common'],default='visdrone')
    p.add_argument('--pretrained-coco',action='store_true')
    p.add_argument('--split',choices=['val','test-dev'],default='val');a=p.parse_args()
    names=COMMON if a.ontology=='common' else NAMES
    gtmap=DET_MAP if a.ontology=='common' else dict(enumerate(range(10)))
    predmap=COCO_MAP if a.pretrained_coco else gtmap
    if a.pretrained_coco and a.ontology!='common':raise ValueError('COCO baseline requires common ontology')
    raw=ROOT/'data'/'raw'/f'VisDrone2019-DET-{a.split}'
    paths=sorted((raw/'images').glob('*.jpg'))
    expected={'val':548,'test-dev':1610}[a.split]
    assert len(paths)==expected, f'Expected official {expected} {a.split} images, got {len(paths)}'
    manifest=json.loads((ROOT/'reports'/f'manifest-{a.split}.json').read_text())
    dims={Path(x['image']).name:(x['width'],x['height']) for x in manifest}
    images=[];anns=[];truth={};ignored={};idx=1
    for image_id,ip in enumerate(paths,1):
        w,h=dims[ip.name];images.append({'id':image_id,'file_name':ip.name,'width':w,'height':h})
        truth[image_id]=[];ignored[image_id]=[]
        for line in (raw/'annotations'/f'{ip.stem}.txt').read_text().splitlines():
            x,y,bw,bh,score,cat,*_=map(float,line.strip().strip(',').split(','))
            x1,y1=max(x,0),max(y,0);x2,y2=min(x+bw,w),min(y+bh,h)
            if x2<=x1 or y2<=y1:continue
            cls=gtmap.get(int(cat)-1) if score!=0 else None
            box=[x1,y1,x2-x1,y2-y1]
            classes=range(len(names)) if cls is None else [cls]
            for c in classes:
                anns.append({'id':idx,'image_id':image_id,'category_id':c+1,'bbox':box,
                    'area':box[2]*box[3],'iscrowd':int(cls is None)});idx+=1
            if cls is None:ignored[image_id].append([x1,y1,x2,y2])
            else:truth[image_id].append([cls,x1,y1,x2,y2])
    dataset={'images':images,'annotations':anns,'categories':[{'id':i+1,'name':n} for i,n in enumerate(names)],'info':{}}
    out=ROOT/'runs'/a.name;out.mkdir(parents=True,exist_ok=True)
    (out/'ground-truth.json').write_text(json.dumps(dataset))
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    model=YOLO(a.weights);predictions=[];counts=np.zeros((len(names),3),dtype=int);errors=[]
    # A Python list is treated as one in-memory batch by this Ultralytics release.
    # Directory source uses the bounded streaming loader; verify its image order.
    results=model.predict(source=str(raw/'images'),stream=True,batch=8,device=0,imgsz=640,
        conf=.001,iou=.7,max_det=500,verbose=False)
    for image_id,(ip,result) in enumerate(zip(paths,results),1):
        assert Path(result.path).name==ip.name,'Unexpected evaluation loader order'
        rows=[]
        for box,conf,cls in zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy(),result.boxes.cls.cpu().numpy()):
            c=predmap.get(int(cls))
            if c is None:continue
            x1,y1,x2,y2=map(float,box)
            predictions.append({'image_id':image_id,'category_id':c+1,'bbox':[x1,y1,x2-x1,y2-y1],'score':float(conf)})
            if conf>=.25:rows.append([c,float(conf),x1,y1,x2,y2])
        matched=set();tp=fp=0
        for row in sorted(rows,key=lambda x:-x[1]):
            c=row[0];candidates=[(i,t) for i,t in enumerate(truth[image_id]) if i not in matched and t[0]==c]
            ovs=overlap([row[2:]],[t[1:] for _,t in candidates])[0]
            if len(ovs) and ovs.max()>=.5:
                matched.add(candidates[int(ovs.argmax())][0]);counts[c,0]+=1;tp+=1
            elif len(ignored[image_id]) and overlap([row[2:]],ignored[image_id],True).max()>=.5:
                continue
            else:counts[c,1]+=1;fp+=1
        fn=len(truth[image_id])-len(matched)
        for i,t in enumerate(truth[image_id]):
            if i not in matched:counts[t[0],2]+=1
        errors.append({'image':ip.name,'tp':tp,'fp':fp,'fn':fn})
        if image_id<=8:
            result.update(boxes=result.boxes.data[result.boxes.conf>=.25])
            result.save(filename=str(out/f'prediction-{ip.name}'))
    (out/'predictions.json').write_text(json.dumps(predictions))
    coco=COCO(str(out/'ground-truth.json'))
    if not predictions:raise RuntimeError('No predictions')
    evaluator=COCOeval(coco,coco.loadRes(predictions),'bbox')
    evaluator.params.maxDets=[1,10,500]
    evaluator.evaluate();evaluator.accumulate()
    precision=evaluator.eval['precision'][:,:,:,0,2]
    valid=lambda x:float(x[x>=0].mean()) if np.any(x>=0) else None
    classes=[]
    for i,n in enumerate(names):
        tp,fp,fn=counts[i]
        classes.append({'class':n,'AP50':valid(precision[0,:,i]),'AP50_95':valid(precision[:,:,i]),
            'precision_conf025_iou050':float(tp/max(tp+fp,1)),
            'recall_conf025_iou050':float(tp/max(tp+fn,1)),'tp':int(tp),'fp':int(fp),'fn':int(fn)})
    totals=counts.sum(0);tp,fp,fn=totals
    with Path(a.weights).open('rb') as f:weight_hash=hashlib.file_digest(f,'sha256').hexdigest()
    report={'weights':a.weights,'weights_sha256':weight_hash,'ontology':a.ontology,
        'end_to_end_head':bool(getattr(model.model.model[-1],'end2end',False)),
        'prediction_batch_size':8,
        'split':a.split,'images':len(paths),'classes':names,'prediction_class_mapping':predmap,'gt_class_mapping':gtmap,
        'mAP50':valid(precision[0]),'mAP50_95':valid(precision),
        'AP_small_native_area_lt1024':valid(evaluator.eval['precision'][:,:,:,1,2]),
        'micro_precision_conf025_iou050':float(tp/max(tp+fp,1)),
        'micro_recall_conf025_iou050':float(tp/max(tp+fn,1)),'per_class':classes,
        'protocol':'local COCO-style 101-point AP; IoU .50:.95; maxDet=500; img=640; conf=.001; NMS iou=.7; ignored/unsupported GT represented as per-category crowd boxes; P/R greedy conf=.25 IoU=.5 with ignored-region IoA=.5; NOT official VisDrone score',
        'limitations':['COCO human merges pedestrian/people; shared ontology only, not a 10-class ranking','Excluded GT categories serve as crowd ignore boxes in common evaluation','AP crowd matching follows COCO IoA thresholds, unlike official VisDrone region logic']}
    (ROOT/'reports'/f'{a.name}-evaluation.json').write_text(json.dumps(report,indent=2))
    (out/'errors.json').write_text(json.dumps(sorted(errors,key=lambda x:x['fp']+x['fn'],reverse=True),indent=2))
    from case_analysis import build_cases
    report['error_cases']=build_cases(out,raw,truth,ignored,paths,names)
    report['evaluation_seconds']=time.perf_counter()-start
    report['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
    report['peak_cuda_reserved_bytes']=torch.cuda.max_memory_reserved()
    (ROOT/'reports'/f'{a.name}-evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
