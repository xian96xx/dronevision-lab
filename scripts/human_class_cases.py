"""Actual baseline/weighted inference on a declared human-class diagnostic subset."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

from segmentation import tensor_image

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def predict(model,source):
    w,h=source.size
    resized=source.resize((max(32,round(w*1024/max(w,h)/32)*32),max(32,round(h*1024/max(w,h)/32)*32)))
    with torch.inference_mode():
        logits=model(pixel_values=tensor_image(resized).unsqueeze(0)).logits
        return F.interpolate(logits,size=(h,w),mode='bilinear',align_corners=False).argmax(1)[0].numpy().astype('uint8')

def main():
    torch.set_num_threads(4)
    weighted_record=json.loads((ROOT/'reports/seg-full-weighted-20-training.json').read_text())
    assert weighted_record['status']=='completed','Use finalized weights only'
    rows=json.loads((ROOT/'data/seg/manifest.json').read_text())['val']
    candidates=[]
    for row in rows:
        mask=np.array(Image.open(row['mask']));candidates.append((int((mask==7).sum()),row))
    selected=sorted(candidates,key=lambda x:(-x[0],x[1]['sequence'],x[1]['frame']))[:3]
    models={name:SegformerForSemanticSegmentation.from_pretrained(ROOT/'runs'/run/'best',local_files_only=True).eval()
            for name,run in [('baseline','seg-full-20'),('weighted','seg-full-weighted-20')]}
    out=ROOT/'runs/refinement-human-cases';out.mkdir(exist_ok=True)
    font_path=Path('C:/Windows/Fonts/msyh.ttc')
    font=ImageFont.truetype(str(font_path),19) if font_path.exists() else ImageFont.load_default()
    records=[]
    for pixels,row in selected:
        assert pixels>0
        source=Image.open(row['image']).convert('RGB');rgb=np.array(source);truth=np.array(Image.open(row['mask']))==7
        count,labels,stats,centers=cv2.connectedComponentsWithStats(truth.astype('uint8'))
        component=1+np.argmax(stats[1:,cv2.CC_STAT_AREA]);x,y,w,h,_=stats[component]
        side=max(192,int(max(w,h)*3));cx=x+w/2;cy=y+h/2
        left=max(0,min(source.width-side,round(cx-side/2)));top=max(0,min(source.height-side,round(cy-side/2)))
        bounds=(left,top,left+side,top+side)
        masks={name:predict(model,source)==7 for name,model in models.items()}
        panels=[];stats_by_model={}
        for title,mask in [('原图',None),('GT human（绿色）',truth),('原始模型 human（蓝色）',masks['baseline']),('加权模型 human（蓝色）',masks['weighted'])]:
            shown=rgb.copy()
            if mask is not None:
                color=np.array([45,220,120] if title.startswith('GT') else [70,150,255])
                shown[mask]=(shown[mask]*.4+color*.6).astype('uint8')
            panel=Image.fromarray(shown).crop(bounds).resize((384,384),Image.Resampling.NEAREST)
            canvas=Image.new('RGB',(384,425),'#111920');canvas.paste(panel,(0,41));ImageDraw.Draw(canvas).text((9,8),title,font=font,fill='white');panels.append(canvas)
        result=Image.new('RGB',(1536,475),'#111920')
        for i,panel in enumerate(panels):result.paste(panel,(384*i,50))
        ImageDraw.Draw(result).text((12,12),f'{row["sequence"]}/{row["frame"]} · 按val human像素量选择，局部最大GT连通域；非随机样本',font=font,fill='white')
        target=out/f'human-{row["sequence"]}-{row["frame"]}.jpg';result.save(target,quality=94)
        for name,mask in masks.items():
            intersection=int((truth&mask).sum());union=int((truth|mask).sum())
            stats_by_model[name]={'human_IoU_on_selected_full_image':intersection/union,'human_predicted_pixels':int(mask.sum())}
            Image.fromarray(mask.astype('uint8')*255).save(out/f'{name}-human-{row["sequence"]}-{row["frame"]}.png')
        records.append({'image':row['image'],'source_sha256':sha(Path(row['image'])),'human_GT_pixels':pixels,
                        'crop_xyxy':bounds,'diagnostic_image':str(target.relative_to(ROOT)),**stats_by_model})
    record={'selection':'Top 3 official val images by human GT pixel count, ties by sequence/frame; zoom at largest GT human component.',
            'inference':'CPU float32; max1024 input; logits upsampled to native mask. Diagnostic only, not a new 70-image GPU score.',
            'weights':{name:sha(ROOT/'runs'/run/'best/model.safetensors') for name,run in [('baseline','seg-full-20'),('weighted','seg-full-weighted-20')]},
            'cases':records,'limitations':'Deliberately selected class-specific cases; not representative random images or an independent test set.'}
    (ROOT/'reports/human-class-diagnostic.json').write_text(json.dumps(record,indent=2))
    print('Saved 3 real baseline/weighted human-class comparisons.')

if __name__=='__main__':main()
