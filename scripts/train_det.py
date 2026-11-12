"""One reproducible GPU training job; resumable, with evidence of weight updates."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
os.environ['YOLO_CONFIG_DIR']=str(Path(__file__).resolve().parents[1]/'work-config')
import ultralytics
from ultralytics import YOLO
import yaml

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='smoke')
    p.add_argument('--name',default='smoke')
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--final-val-only',action='store_true',help='Skip intermediate validation; final epoch is still validated')
    args=p.parse_args()
    os.chdir(ROOT)
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    initial=ROOT/'models'/'yolo26n.pt'
    if not initial.exists():
        YOLO('yolo26n.pt')
        (ROOT/'yolo26n.pt').replace(initial)
    run=ROOT/'runs'/args.name
    if run.exists() and not args.resume:
        raise FileExistsError(f'{run} already exists; use --resume or a new name')
    model=YOLO(str(run/'weights'/'last.pt') if args.resume else str(initial))
    # Backbone key is independent of the changed class head.
    key='model.0.conv.weight'
    before=model.model.state_dict()[key].detach().float().cpu().clone()
    config=ROOT/'configs'/f'{args.data}.yaml'
    start=time.perf_counter()
    record={'command':sys.argv,'git':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'initial_weight_sha256':sha(initial),'data_config_sha256':sha(config),
        'torch':torch.__version__,'ultralytics':ultralytics.__version__,
        'gpu':torch.cuda.get_device_name(0),'seed':args.seed,'status':'running'}
    data_config=yaml.safe_load(config.read_text())
    train_images=sorted(Path(data_config['train']).glob('*.jpg'))
    labels_hash=hashlib.sha256()
    for ip in train_images:
        lp=Path(str(ip).replace(os.sep+'images'+os.sep,os.sep+'labels'+os.sep)).with_suffix('.txt')
        labels_hash.update(ip.name.encode());labels_hash.update(lp.read_bytes())
    record['training_image_count']=len(train_images)
    record['training_labels_sha256']=labels_hash.hexdigest()
    validation_images=sorted(Path(data_config['val']).glob('*.jpg'))
    validation_hash=hashlib.sha256()
    for ip in validation_images:
        validation_hash.update(ip.name.encode())
        with ip.open('rb') as stream:validation_hash.update(hashlib.file_digest(stream,'sha256').digest())
    record['actual_validation_image_count']=len(validation_images)
    record['actual_validation_images_sha256']=validation_hash.hexdigest()
    record['clean_validation_manifest_sha256']=sha(ROOT/'reports'/'manifest-val.json')
    evidence=ROOT/'reports'/f'{args.name}-training.json'
    evidence.write_text(json.dumps(record,indent=2))
    torch.cuda.reset_peak_memory_stats()
    common=dict(data=str(config),epochs=args.epochs,imgsz=args.imgsz,batch=args.batch,
        device=0,workers=2,seed=args.seed,deterministic=True,amp=True,
        optimizer='AdamW',lr0=.001,lrf=.01,weight_decay=.0005,
        project=str(ROOT/'runs'),name=args.name,exist_ok=args.resume,
        patience=0,save=True,save_period=5,plots=True,cache=False,
        close_mosaic=0,mosaic=0.0,mixup=0.0,copy_paste=0.0,
        max_det=500,verbose=False,val=not args.final_val_only)
    try:
        result=model.train(resume=True,**common) if args.resume else model.train(**common)
    except Exception:
        import traceback
        record.update(status='failed',error=traceback.format_exc(),seconds=time.perf_counter()-start)
        evidence.write_text(json.dumps(record,indent=2))
        raise
    torch.cuda.synchronize()
    record.update(seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),status='completed')
    last=YOLO(str(run/'weights'/'last.pt'))
    assert all(bool(p.isfinite().all()) for p in last.model.parameters()),'Non-finite trained weights'
    after=last.model.state_dict()[key].detach().float().cpu()
    record['backbone_parameter_max_abs_delta']=float((after-before).abs().max())
    assert record['backbone_parameter_max_abs_delta']>0,'No backbone update detected'
    record['checkpoints']={x.name:sha(x) for x in (run/'weights').glob('*.pt')}
    record['validation_metrics']=result.results_dict
    record['per_class']=[{'class':model.names[int(c)],'precision':float(pr),'recall':float(re),
        'ap50':float(ap[0]),'ap50_95':float(ap.mean())}
        for c,pr,re,ap in zip(result.box.ap_class_index,result.box.p,result.box.r,result.box.all_ap)]
    evidence.write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2),flush=True)
    del model,last;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
