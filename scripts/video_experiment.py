"""Compare endpoint interpolation, SAM2 propagation and trained YOLO trackers on one fixed MOT subset.
Box quality is measured on prompted tracks; MOT identity metrics are reported separately.
"""
import argparse
import gc
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import subprocess
import zipfile
ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HOME']=str(ROOT/'work'/'hf-cache')
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'work-config')
import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch
from evaluate_det import overlap

def extract():
    archive=ROOT/'data'/'archives'/'VisDrone2019-MOT-val.zip'
    target=ROOT/'data'/'mot'
    marker=target/'.validated-val'
    if not marker.exists():
        with zipfile.ZipFile(archive) as z:
            bad=z.testzip()
            if bad:raise ValueError(f'ZIP corrupt: {bad}')
            for item in z.infolist():
                if not (target/item.filename).resolve().is_relative_to(target.resolve()):raise ValueError('Unsafe ZIP path')
            z.extractall(target)
        marker.write_text(hashlib.file_digest(archive.open('rb'),'sha256').hexdigest())
    return target/'VisDrone2019-MOT-val'

def read_gt(path):
    result={}
    for row in np.loadtxt(path,delimiter=',',ndmin=2):
        frame,tid,x,y,w,h,score,cat,trunc,occ=row[:10]
        result.setdefault(int(frame),[]).append({'id':int(tid),'box':[x,y,x+w,y+h],
            'class':int(cat),'valid':bool(score==1 and int(cat) in [1,4,5,6,9]), 'occlusion':int(occ)})
    return result

def prepare_subset(frames=120,seq_count=3,track_count=4):
    root=extract();out=ROOT/'runs'/'video';out.mkdir(parents=True,exist_ok=True)
    subsets=[]
    for ap in sorted((root/'annotations').glob('*.txt'))[:seq_count]:
        gt=read_gt(ap);ips=sorted((root/'sequences'/ap.stem).glob('*.jpg'))[:frames]
        if len(ips)<frames:continue
        first=int(ips[0].stem);last=int(ips[-1].stem)
        byid={}
        for f in range(first,last+1):
            for t in gt.get(f,[]):
                if t['valid']:byid.setdefault(t['id'],{})[f]=t
        eligible=[(tid,rows) for tid,rows in byid.items() if all(f in rows for f in range(first,last+1))]
        # Coverage filter uses GT only to define a documented evaluation subset, never to correct outputs.
        # Include both occluded and visible trajectories, then deterministic ID order.
        eligible.sort(key=lambda x:(-max(t['occlusion'] for t in x[1].values()),x[0]))
        eligible=[(tid,r) for tid,r in eligible if np.prod(np.array(r[first]['box'][2:])-r[first]['box'][:2])>=100][:track_count]
        if not eligible:continue
        subset={'sequence':ap.stem,'frames':[str(p) for p in ips], 'first_frame':first,
            'target_ids':[tid for tid,_ in eligible],
            'tracks':{str(tid):[rows[f] for f in range(first,last+1)] for tid,rows in eligible},
            'all_gt':{str(f-first):gt.get(f,[]) for f in range(first,last+1)}}
        subsets.append(subset)
    protocol={'subsets':subsets,'frames':frames,'split':'VisDrone2019-MOT-val',
        'selection':'first 3 sequences lexicographically; continuous valid tracks for full clip; prioritize occlusion then ID; first-frame area>=100; max4 tracks',
        'inputs':{'linear':'GT first AND last frame; future endpoint provided',
            'log-size':'GT first AND last frame; future endpoint provided',
            'sam2':'GT first-frame boxes only; no later correction',
            'bytetrack/botsort':'trained detector; first-frame IoU association to prompted IDs only'},
        'evaluation':'interior frames only; output masks converted to tight xyxy boxes; bbox GT cannot validate mask boundaries; no task metric ranking across different outputs'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    return protocol

def interpolate(subset,logsize=False):
    pred={}
    n=len(subset['frames'])
    for tid,rows in subset['tracks'].items():
        endpoints=np.array([rows[0]['box'],rows[-1]['box']])
        centers=(endpoints[:,:2]+endpoints[:,2:])/2
        sizes=endpoints[:,2:]-endpoints[:,:2]
        t=np.linspace(0,1,n)[:,None]
        c=(1-t)*centers[0]+t*centers[1]
        s=np.exp((1-t)*np.log(sizes[0])+t*np.log(sizes[1])) if logsize else (1-t)*sizes[0]+t*sizes[1]
        pred[tid]=np.concatenate([c-s/2,c+s/2],axis=1).tolist()
    return pred

def sam2(subset):
    from transformers import Sam2VideoModel,Sam2VideoProcessor
    model=Sam2VideoModel.from_pretrained(ROOT/'models'/'sam2.1-tiny',local_files_only=True).to('cuda').eval()
    processor=Sam2VideoProcessor.from_pretrained(ROOT/'models'/'sam2.1-tiny',local_files_only=True)
    # Store frames on CPU to keep GPU memory bounded; only one visual feature cached.
    video=[Image.open(x).convert('RGB') for x in subset['frames']]
    ids=list(subset['target_ids'])
    predictions={str(tid):[None]*len(video) for tid in ids}
    writer=cv2.VideoWriter(str(ROOT/'runs'/'video'/f'SAM2-{subset["sequence"]}.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'),15,(1280,720))
    assert writer.isOpened(),'SAM2 video encoder unavailable'
    colors=np.array([[255,80,80],[80,255,80],[80,160,255],[255,200,60]],dtype=np.uint8)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        session=processor.init_video_session(video=video,inference_device='cuda',
            inference_state_device='cpu',video_storage_device='cpu',dtype=torch.bfloat16)
        boxes=[[subset['tracks'][str(tid)][0]['box'] for tid in ids]]
        processor.add_inputs_to_inference_session(session,frame_idx=0,obj_ids=ids,input_boxes=boxes)
        model(inference_session=session,frame_idx=0)
        for output in model.propagate_in_video_iterator(session):
            overlay=np.array(video[output.frame_idx]).copy()
            masks=processor.post_process_masks([output.pred_masks],
                original_sizes=[[session.video_height,session.video_width]],binarize=True)[0]
            for i,tid in enumerate(session.obj_ids):
                mask=masks[i].squeeze().cpu().numpy().astype(bool)
                yy,xx=np.where(mask)
                if len(xx):predictions[str(tid)][output.frame_idx]=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
                overlay[mask]=(overlay[mask]*.55+colors[i%len(colors)]*.45).astype('uint8')
                if output.frame_idx in [0,len(video)//2,len(video)-1]:
                    Image.fromarray(mask.astype('uint8')*255).save(ROOT/'runs'/'video'/f'sam2-{subset["sequence"]}-{tid}-{output.frame_idx}.png')
            shown=cv2.resize(cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR),(1280,720))
            cv2.rectangle(shown,(0,0),(1280,32),(0,0,0),-1)
            cv2.putText(shown,f'SAM2 predicted masks | frame {output.frame_idx} | frame0 box prompts, no later corrections',
                (8,23),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),1)
            writer.write(shown)
    writer.release()
    del model,session;gc.collect();torch.cuda.empty_cache()
    return predictions

def tracker(subset,weights,method):
    from ultralytics import YOLO
    model=YOLO(weights);pred={str(tid):[None]*len(subset['frames']) for tid in subset['target_ids']}
    mapping={};all_predictions=[]
    for f,ip in enumerate(subset['frames']):
        result=model.track(ip,persist=True,tracker=f'{method}.yaml',device=0,imgsz=640,
            conf=.1,iou=.7,max_det=500,verbose=False)[0]
        boxes=result.boxes
        detections=[]
        if boxes.id is not None:
            for b,tid,c,conf in zip(boxes.xyxy.cpu().numpy(),boxes.id.cpu().numpy(),boxes.cls.cpu().numpy(),boxes.conf.cpu().numpy()):
                if int(c)+1 in [1,4,5,6,9]:
                    detections.append({'id':int(tid),'box':b.tolist(),'class':int(c)+1,'confidence':float(conf)})
        all_predictions.append(detections)
        if f==0 and detections:
            truth=[subset['tracks'][str(tid)][0]['box'] for tid in subset['target_ids']]
            iou=overlap(truth,[x['box'] for x in detections])
            for r,tid in enumerate(subset['target_ids']):
                for c,d in enumerate(detections):
                    if subset['tracks'][str(tid)][0]['class']!=d['class']:iou[r,c]=0
            rr,cc=linear_sum_assignment(1-iou)
            for r,c in zip(rr,cc):
                if iou[r,c]>=.5:mapping[detections[c]['id']]=str(subset['target_ids'][r])
        for d in detections:
            if d['id'] in mapping:pred[mapping[d['id']]][f]=d['box']
    del model;gc.collect();torch.cuda.empty_cache()
    return pred,all_predictions

def mot_metrics(subset,predictions):
    import motmetrics as mm
    if not hasattr(np,'asfarray'):np.asfarray=lambda a:np.asarray(a,dtype=float)
    accumulator=mm.MOTAccumulator(auto_id=True)
    for f,ds in enumerate(predictions):
        gt=[x for x in subset['all_gt'][str(f)] if x['valid']]
        ignored=[x['box'] for x in subset['all_gt'][str(f)] if not x['valid']]
        kept=[]
        for d in ds:
            same_class=[g['box'] for g in gt if g['class']==d['class']]
            matches_valid=bool(same_class and overlap([d['box']],same_class).max()>=.5)
            if matches_valid or not ignored or overlap([d['box']],ignored,True).max()<.5:kept.append(d)
        ds=kept
        cost=1-overlap([x['box'] for x in gt],[x['box'] for x in ds])
        for i,g in enumerate(gt):
            for j,d in enumerate(ds):
                if g['class']!=d['class'] or cost[i,j]>.5:cost[i,j]=np.nan
        accumulator.update([x['id'] for x in gt],[x['id'] for x in ds],cost)
    metrics=['mota','idf1','num_switches','num_fragmentations','num_false_positives','num_misses','precision','recall','num_objects']
    result=mm.metrics.create().compute(accumulator,metrics=metrics,name=subset['sequence']).iloc[0].to_dict()
    return {k:float(v) for k,v in result.items()}

def quality(subset,pred):
    rows=[]
    for tid,truth in subset['tracks'].items():
        gt=np.array([x['box'] for x in truth]);ious=[];centers=[];widths=[];heights=[];validpred=[]
        for f in range(1,len(gt)-1):
            b=pred[tid][f]
            if b is None:ious.append(0.);validpred.append(False);continue
            b=np.array(b);g=gt[f];wh=g[2:]-g[:2]
            ious.append(float(overlap([b],[g])[0,0]));validpred.append(True)
            centers.append(float(np.linalg.norm((b[:2]+b[2:]-g[:2]-g[2:])/2)))
            widths.append(float(abs((b[2]-b[0])-wh[0])));heights.append(float(abs((b[3]-b[1])-wh[1])))
        pp=[np.array(x) if x is not None else np.full(4,np.nan) for x in pred[tid]]
        pc=(np.array(pp)[:,:2]+np.array(pp)[:,2:])/2;gc=(gt[:,:2]+gt[:,2:])/2
        acceleration=np.linalg.norm(np.diff(pc,n=2,axis=0)-np.diff(gc,n=2,axis=0),axis=1)
        acceleration=acceleration[np.isfinite(acceleration)]
        rows.append({'id':tid,'mean_box_iou':float(np.mean(ious)),
            'iou_below_05_fraction':float(np.mean(np.array(ious)<.5)),
            'missing_fraction':float(1-np.mean(validpred)),
            'center_error_pixels_visible_predictions':float(np.mean(centers)) if centers else None,
            'width_mae_pixels_visible_predictions':float(np.mean(widths)) if widths else None,
            'height_mae_pixels_visible_predictions':float(np.mean(heights)) if heights else None,
            'center_acceleration_error_pixels':float(acceleration.mean()) if len(acceleration) else None,
            'occluded_frames':sum(x['occlusion']>0 for x in truth)})
    return rows

def render(subset,all_methods):
    out=ROOT/'runs'/'video'/f'comparison-{subset["sequence"]}.mp4'
    methods=list(all_methods);panelw=640;panelh=400
    writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*'mp4v'),15,(panelw*2,panelh*2))
    assert writer.isOpened()
    for f,ip in enumerate(subset['frames']):
        im=cv2.imread(ip);h,w=im.shape[:2];panels=[]
        for method in methods[:4]:
            panel=im.copy()
            for tid,truth in subset['tracks'].items():
                b=list(map(int,truth[f]['box']));cv2.rectangle(panel,tuple(b[:2]),tuple(b[2:]),(0,255,0),2)
                pp=all_methods[method][tid][f]
                if pp is not None:
                    b=list(map(int,pp));cv2.rectangle(panel,tuple(b[:2]),tuple(b[2:]),(0,0,255),2)
                    cv2.putText(panel,tid,tuple(b[:2]),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),1)
            panel=cv2.resize(panel,(panelw,panelh));cv2.rectangle(panel,(0,0),(panelw,28),(0,0,0),-1)
            cv2.putText(panel,f'{method} | frame {f} | GT green / prediction red',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
            panels.append(panel)
        while len(panels)<4:panels.append(np.zeros_like(panels[0]))
        writer.write(np.concatenate([np.concatenate(panels[:2],axis=1),np.concatenate(panels[2:],axis=1)],axis=0))
    writer.release()

def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',default=str(ROOT/'runs'/'det-full-50'/'weights'/'best.pt'))
    p.add_argument('--frames',type=int,default=120);a=p.parse_args()
    torch.manual_seed(42);torch.cuda.manual_seed_all(42);np.random.seed(42);cv2.setRNGSeed(42)
    provenance={'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'seed':42,'torch':torch.__version__,
        'detector_weights_sha256':hashlib.file_digest(Path(a.weights).open('rb'),'sha256').hexdigest(),
        'sam2_source':json.loads((ROOT/'reports'/'sam2.1-tiny-source.json').read_text())}
    protocol=prepare_subset(a.frames);summary=[]
    for subset in protocol['subsets']:
        methods={'linear':interpolate(subset),'log-size':interpolate(subset,True)};times={};mot={};failures={}
        expected=set(subset['tracks'])
        for method in ['sam2','bytetrack','botsort']:
            start=time.perf_counter()
            try:
                isolated=copy.deepcopy(subset)
                if method=='sam2':candidate=sam2(isolated)
                else:
                    candidate,predictions=tracker(isolated,a.weights,method)
                    mot[method]=mot_metrics(isolated,predictions)
                    (ROOT/'runs'/'video'/f'{subset["sequence"]}-{method}-all.json').write_text(json.dumps(predictions))
                candidate={str(k):v for k,v in candidate.items()}
                print(f'{subset["sequence"]} {method} expected IDs={sorted(expected)} output IDs={sorted(candidate)}',flush=True)
                assert set(candidate)==expected,f'{method} target IDs changed: {sorted(candidate)}'
                methods[method]=candidate
                (ROOT/'runs'/'video'/f'{subset["sequence"]}-predictions.json').write_text(json.dumps(methods))
            except Exception as e:
                import traceback
                failures[method]=traceback.format_exc();print(failures[method],flush=True)
                gc.collect();torch.cuda.empty_cache()
            times[method]=time.perf_counter()-start
        result={'sequence':subset['sequence'],'quality':{k:quality(subset,v) for k,v in methods.items()},
            'mot':mot,'seconds':times,'failures':failures,'provenance':provenance}
        summary.append(result)
        (ROOT/'runs'/'video'/f'{subset["sequence"]}-predictions.json').write_text(json.dumps(methods))
        render(subset,{k:v for k,v in methods.items() if k!='log-size'})
        (ROOT/'reports'/'video-results.json').write_text(json.dumps(summary,indent=2))
        print(result,flush=True)

if __name__=='__main__':main()
