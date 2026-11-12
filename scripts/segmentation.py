"""UAVid palette audit, fixed crops, real SegFormer-B0 GPU training and native-mask evaluation."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import time
ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HOME']=str(ROOT/'work'/'hf-cache')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from transformers import SegformerForSemanticSegmentation

NAMES=['clutter','building','road','tree','low-vegetation','moving-car','static-car','human']
PALETTE=np.array([[0,0,0],[128,0,0],[128,64,128],[0,128,0],[128,128,0],[64,0,128],[192,0,192],[64,64,0]],dtype=np.uint8)
MEAN=np.array([.485,.456,.406],dtype=np.float32)[:,None,None]
STD=np.array([.229,.224,.225],dtype=np.float32)[:,None,None]

def read_mask(path):
    rgb=np.array(Image.open(path).convert('RGB'));packed=(rgb[:,:,0].astype('uint32')<<16)+(rgb[:,:,1].astype('uint32')<<8)+rgb[:,:,2]
    out=np.full(packed.shape,255,dtype=np.uint8)
    for i,c in enumerate(PALETTE):out[packed==((int(c[0])<<16)+(int(c[1])<<8)+int(c[2]))]=i
    if (out==255).any():raise ValueError(f'Unknown palette color in {path}')
    return out

def prepare():
    raw=ROOT/'data'/'uavid'/'UAVid-v1';out=ROOT/'data'/'seg';out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists() and (out/'patches.json').exists() and (ROOT/'reports'/'uavid-data-quality.json').exists():
        previous=json.loads((out/'patches.json').read_text())
        if len(previous)==800 and all(Path(x['image']).exists() and Path(x['mask']).exists() for x in previous):
            print('Verified existing 800 training crops; no regeneration needed.',flush=True)
            return
    records={};audit={};crop_manifest=[]
    for split,expected in [('train',200),('val',70)]:
        ips=sorted((raw/split).glob('seq*/Images/*.png'))
        assert len(ips)==expected,(split,len(ips),expected)
        pairs=[];counts=np.zeros(8,dtype=np.int64);sizes=[];unknown=0
        for ip in ips:
            mp=ip.parent.parent/'Labels'/ip.name
            mask=read_mask(mp)
            with Image.open(ip) as im:
                im.load();assert im.size==(mask.shape[1],mask.shape[0]);image=np.array(im.convert('RGB'))
            counts+=np.bincount(mask.ravel(),minlength=8);sizes.append(list(mask.shape))
            seq=ip.parent.parent.name
            converted=out/'masks'/split/f'{seq}-{ip.name}';converted.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(mask).save(converted)
            pair={'image':str(ip),'mask':str(converted),'sequence':seq,'frame':ip.stem,
                'image_sha256':hashlib.file_digest(ip.open('rb'),'sha256').hexdigest(),
                'original_mask_sha256':hashlib.file_digest(mp.open('rb'),'sha256').hexdigest()}
            pairs.append(pair)
            if split=='train':
                rng=random.Random(int(hashlib.sha256(f'{seq}-{ip.stem}-42'.encode()).hexdigest()[:16],16))
                h,w=mask.shape
                for k in range(4):
                    x=rng.randrange(w-1024+1);y=rng.randrange(h-1024+1)
                    crop=cv2.resize(image[y:y+1024,x:x+1024],(512,512),interpolation=cv2.INTER_AREA)
                    label=cv2.resize(mask[y:y+1024,x:x+1024],(512,512),interpolation=cv2.INTER_NEAREST)
                    cp=out/'patches'/'images'/f'{seq}-{ip.stem}-{k}.jpg';cp.parent.mkdir(parents=True,exist_ok=True)
                    lp=out/'patches'/'masks'/f'{seq}-{ip.stem}-{k}.png';lp.parent.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(crop).save(cp,quality=95);Image.fromarray(label).save(lp)
                    crop_manifest.append({'image':str(cp),'mask':str(lp),'source_sequence':seq,'source_frame':ip.stem,'crop_xywh':[x,y,1024,1024]})
            if len(pairs)<=3:
                color=PALETTE[mask];overlay=(image*.5+color*.5).astype('uint8')
                Image.fromarray(cv2.resize(overlay,(1280,720))).save(ROOT/'reports'/f'uavid-{split}-{seq}-{ip.name}')
        records[split]=pairs;audit[split]={'images':len(pairs),'pixel_counts':dict(zip(NAMES,counts.tolist())),
            'pixel_ratio':dict(zip(NAMES,(counts/counts.sum()).tolist())),'image_shapes':sorted(set(tuple(x) for x in sizes)),'unknown_pixels':unknown}
    trainseq={x['sequence'] for x in records['train']};valseq={x['sequence'] for x in records['val']}
    assert not trainseq&valseq,'Sequence leakage'
    report={'source':'Official DANS DOI:10.17026/dans-x9f-w9sa; UAVid2020 20/7 sequence split',
        'splits':audit,'class_zero_is_valid':True,'palette':PALETTE.tolist(),
        'train_sequences':sorted(trainseq),'val_sequences':sorted(valseq),
        'training_crop_policy':'4 fixed seed42 random native1024 crops per training image, resized to512; no val/test crops in train',
        'train_patches':len(crop_manifest)}
    (out/'manifest.json').write_text(json.dumps(records,indent=2))
    (out/'patches.json').write_text(json.dumps(crop_manifest,indent=2))
    (ROOT/'reports'/'uavid-data-quality.json').write_text(json.dumps(report,indent=2))
    print(report,flush=True)

def tensor_image(im):
    a=np.asarray(im,dtype=np.float32).transpose(2,0,1)/255
    return torch.from_numpy((a-MEAN)/STD)

class Crops(Dataset):
    def __init__(self,rows,variant):self.rows=rows;self.variant=variant
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i];im=Image.open(row['image']).convert('RGB');mask=np.array(Image.open(row['mask'])).copy()
        rng=np.random.default_rng(int(hashlib.sha256((Path(row['image']).name+'42').encode()).hexdigest()[:16],16))
        if self.variant=='boundary':
            boundary=np.zeros(mask.shape,np.uint8)
            boundary[1:]|=(mask[1:]!=mask[:-1]);boundary[:,1:]|=(mask[:,1:]!=mask[:,:-1])
            boundary=cv2.dilate(boundary,np.ones((7,7),np.uint8))>0
            dx,dy=rng.integers(-3,4,size=2);shifted=cv2.warpAffine(mask,np.array([[1,0,dx],[0,1,dy]],dtype=float),mask.shape[::-1],flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_REPLICATE)
            mask[boundary]=shifted[boundary]
        elif self.variant=='wrong':
            x,y=rng.integers(0,384,size=2);mask[y:y+128,x:x+128]=0
        return tensor_image(im),torch.from_numpy(mask.astype('int64'))

def metrics(cm):
    tp=np.diag(cm);union=cm.sum(0)+cm.sum(1)-tp
    iou=np.divide(tp,union,out=np.full(8,np.nan),where=union>0)
    return {'mIoU':float(np.nanmean(iou)),'pixel_accuracy':float(tp.sum()/max(cm.sum(),1)),
        'per_class_IoU':{n:float(v) if np.isfinite(v) else None for n,v in zip(NAMES,iou)},'confusion_matrix':cm.tolist()}

def training_evidence(rows,weighting):
    """Train-only pixel frequencies; validation labels never determine loss weights."""
    counts=np.zeros(8,dtype=np.int64);digest=hashlib.sha256()
    for row in rows:
        mask=np.array(Image.open(row['mask']))
        counts+=np.bincount(mask.ravel(),minlength=8)
        for key in ['image','mask']:
            path=Path(row[key]);digest.update(str(path.relative_to(ROOT)).encode())
            with path.open('rb') as stream:digest.update(hashlib.file_digest(stream,'sha256').digest())
    weights=None
    if weighting=='inverse-sqrt':
        frequency=counts/counts.sum()
        weights=np.sqrt((1/8)/np.maximum(frequency,1e-8)).clip(.25,8)
        weights=(weights/weights.mean()).astype('float32')
    return {'training_data_sha256':digest.hexdigest(),'train_pixel_counts':dict(zip(NAMES,counts.tolist())),
            'class_weighting':weighting,'class_weights':None if weights is None else weights.tolist(),
            'weight_rule':'sqrt((1/8)/train_frequency), clip [.25,8], normalize mean=1; no val statistics'}

@torch.inference_mode()
def evaluate(model,rows,output=None):
    model.eval();cm=np.zeros((8,8),dtype=np.int64)
    for i,row in enumerate(rows):
        with Image.open(row['image']) as im:
            im=im.convert('RGB');w,h=im.size;im=im.resize((round(w*1024/max(w,h)/32)*32,round(h*1024/max(w,h)/32)*32))
            image=tensor_image(im).unsqueeze(0).to('cuda')
        target=np.array(Image.open(row['mask']))
        with torch.autocast('cuda',dtype=torch.float16):logits=model(pixel_values=image).logits
        # Compute native mask metrics. Input remains max1024, explicitly distinct from native/tiled inference.
        pred=F.interpolate(logits.float(),size=target.shape,mode='bilinear',align_corners=False).argmax(1)[0].cpu().numpy().astype('uint8')
        cm+=np.bincount((target.astype('int64')*8+pred).ravel(),minlength=64).reshape(8,8)
        if output and i<6:
            src=np.array(Image.open(row['image']).convert('RGB'))
            gt=PALETTE[target];pr=PALETTE[pred]
            panels=[cv2.resize(x,(640,360)) for x in [src,gt,pr,(src*.5+pr*.5).astype('uint8')]]
            canvas=np.concatenate([np.concatenate(panels[:2],axis=1),np.concatenate(panels[2:],axis=1)],axis=0)
            Image.fromarray(canvas).save(output/f'prediction-{row["sequence"]}-{row["frame"]}.jpg')
            Image.fromarray(pred).save(output/f'mask-{row["sequence"]}-{row["frame"]}.png')
    model.train();return metrics(cm)

def train(args):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,100}',args.name):raise ValueError('Unsafe run name')
    assert args.epochs>0 and args.batch>0 and args.subset>=0
    assert torch.cuda.is_available(),'This training job requires the local CUDA GPU'
    seed=args.seed;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False;torch.set_num_threads(4)
    out=ROOT/'runs'/args.name
    if out.exists() and not args.resume:raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'data'/'seg'/'manifest.json').read_text());rows=json.loads((ROOT/'data'/'seg'/'patches.json').read_text())
    if args.subset:
        # Fixed whole source images, all their four crops. No sequence mixing with validation.
        keys=sorted(set((r['source_sequence'],r['source_frame']) for r in rows));selected=set(random.Random(42).sample(keys,args.subset))
        rows=[r for r in rows if (r['source_sequence'],r['source_frame']) in selected]
    evidence=training_evidence(rows,args.class_weighting)
    model=SegformerForSemanticSegmentation.from_pretrained(ROOT/'models'/'mit-b0',num_labels=8,
        id2label=dict(enumerate(NAMES)),label2id={n:i for i,n in enumerate(NAMES)},ignore_mismatched_sizes=True,local_files_only=True)
    model.config.semantic_loss_ignore_index=255
    before=next(model.segformer.parameters()).detach().clone()
    init=out/'initial.safetensors'
    from safetensors.torch import save_file
    save_file(model.state_dict(),str(init))
    model=model.to('cuda');optimizer=torch.optim.AdamW(model.parameters(),lr=6e-5,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda');history=[];start_epoch=0;best=-1
    checkpoint=out/'last.pt'
    if args.resume:
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        start_epoch=state['epoch']+1;history=state['history'];best=state['best_mIoU']
        if 'rng' in state:
            rng=state['rng'];random.setstate(rng['python']);np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch_cpu']);torch.cuda.set_rng_state_all(rng['torch_cuda'])
        if 'protocol' in state:
            assert state['protocol']=={'variant':args.variant,'data':evidence['training_data_sha256'],'weighting':args.class_weighting},'Resume protocol mismatch'
    record={'command':vars(args),'git':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'initial_state_sha256':hashlib.file_digest(init.open('rb'),'sha256').hexdigest(),
        'source':json.loads((ROOT/'reports'/'mit-b0-source.json').read_text()),
        'seed':seed,'training_patches':len(rows),'validation_images':len(manifest['val']),
        'optimizer':'AdamW lr=6e-5 wd=.01, constant LR','crop_policy':'native1024 random fixed crops resized512',
        'validation_schedule':'each epoch' if not args.subset else 'final epoch only; identical across all three noise conditions',
        'evaluation':'all70 val images; max1024 RGB input maintaining aspect ratio; logits bilinear upsample to native mask; 8 valid classes incl clutter',
        'implementation_sha256':hashlib.file_digest(Path(__file__).open('rb'),'sha256').hexdigest(),
        **evidence,'status':'running'}
    report=ROOT/'reports'/f'{args.name}-training.json';report.write_text(json.dumps(record,indent=2))
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    dataset=Crops(rows,args.variant)
    loss_weights=None if evidence['class_weights'] is None else torch.tensor(evidence['class_weights'],device='cuda')
    for epoch in range(start_epoch,args.epochs):
        generator=torch.Generator().manual_seed(seed+epoch)
        loader=DataLoader(dataset,batch_size=args.batch,shuffle=True,num_workers=2,pin_memory=True,generator=generator)
        total=0;model.train()
        for x,y in loader:
            x=x.to('cuda',non_blocking=True);y=y.to('cuda',non_blocking=True);optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                if loss_weights is None:loss=model(pixel_values=x,labels=y).loss
                else:
                    logits=model(pixel_values=x).logits
                    logits=F.interpolate(logits,size=y.shape[-2:],mode='bilinear',align_corners=False)
                    loss=F.cross_entropy(logits,y,weight=loss_weights,ignore_index=255)
            assert torch.isfinite(loss)
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(optimizer);scaler.update();total+=loss.item()*len(x)
        result=evaluate(model,manifest['val'],out if args.subset else None) if not args.subset or epoch+1==args.epochs else None
        history.append({'epoch':epoch+1,'loss':total/len(dataset),**(result or {})})
        if result is not None and result['mIoU']>best:
            best=result['mIoU'];model.save_pretrained(out/'best')
        state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
            'epoch':epoch,'history':history,'best_mIoU':best,'seed':seed,
            'protocol':{'variant':args.variant,'data':evidence['training_data_sha256'],'weighting':args.class_weighting},
            'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all()}}
        torch.save(state,checkpoint)
        (out/'history.json').write_text(json.dumps(history,indent=2));print(history[-1],flush=True)
    record.update(status='completed',seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        best_mIoU=best,encoder_max_abs_delta=float((next(model.segformer.parameters()).detach().cpu()-before).abs().max()))
    assert record['encoder_max_abs_delta']>0
    bestmodel=SegformerForSemanticSegmentation.from_pretrained(out/'best',local_files_only=True).to('cuda')
    if args.subset:
        # Fresh controlled runs validate only their final epoch. A resumed run may
        # retain an earlier, better validation checkpoint; do not silently call last best.
        selected=max((h for h in history if 'mIoU' in h),key=lambda h:h['mIoU'])
        if selected['epoch']==history[-1]['epoch']:
            assert torch.equal(next(bestmodel.segformer.parameters()),next(model.segformer.parameters()))
        record['best_epoch']=selected['epoch']
        record['best_validation']={k:selected[k] for k in ['mIoU','pixel_accuracy','per_class_IoU','confusion_matrix']}
    else:record['best_validation']=evaluate(bestmodel,manifest['val'],out)
    record['best_weights_sha256']=hashlib.file_digest((out/'best'/'model.safetensors').open('rb'),'sha256').hexdigest()
    record['checkpoint_sha256']=hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest()
    report.write_text(json.dumps(record,indent=2));print(record,flush=True)
    del model,bestmodel;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train']);p.add_argument('--name',default='seg-full-20')
    p.add_argument('--variant',choices=['clean','boundary','wrong'],default='clean');p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch',type=int,default=8);p.add_argument('--seed',type=int,default=42);p.add_argument('--subset',type=int,default=0)
    p.add_argument('--class-weighting',choices=['none','inverse-sqrt'],default='none')
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.action=='prepare':prepare()
    else:
        try:train(a)
        except Exception:
            import traceback
            rp=ROOT/'reports'/f'{a.name}-training.json'
            if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,100}',a.name) and rp.exists():
                state=json.loads(rp.read_text())
                # Refusing an existing completed run must not relabel its evidence as failed.
                if state.get('status')!='completed':
                    state.update(status='failed',error=traceback.format_exc());rp.write_text(json.dumps(state,indent=2))
            raise
