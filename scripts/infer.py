"""Run a trained local detector or segmenter on a user-specified image."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HOME']=str(ROOT/'work'/'hf-cache')
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'work-config')
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['detect','segment'],default='detect')
    p.add_argument('--image',type=Path,required=True);p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'runs'/'user-inference');p.add_argument('--device',default='cuda:0');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    if a.task=='detect':
        from ultralytics import YOLO
        model=YOLO(a.weights);result=model.predict(str(a.image),device=a.device,imgsz=640,conf=.25,verbose=False)[0]
        result.save(filename=str(a.output/'detection.jpg'))
        lines=[]
        for c,b,conf in zip(result.boxes.cls.cpu().tolist(),result.boxes.xywhn.cpu().tolist(),result.boxes.conf.cpu().tolist()):
            lines.append(' '.join([str(int(c)),*[f'{v:.8f}' for v in b],f'{conf:.8f}']))
        (a.output/'detection.txt').write_text('\n'.join(lines))
        (a.output/'detection.json').write_text(result.to_json())
        summary={'objects':len(result.boxes),'class_names':model.names}
    else:
        from transformers import SegformerForSemanticSegmentation
        from segmentation import PALETTE,tensor_image
        model=SegformerForSemanticSegmentation.from_pretrained(a.weights,local_files_only=True).to(a.device).eval()
        with Image.open(a.image) as im:
            im=im.convert('RGB');original=np.array(im);w,h=im.size
            im=im.resize((max(32,round(w*1024/max(w,h)/32)*32),max(32,round(h*1024/max(w,h)/32)*32)))
            with torch.inference_mode():
                logits=model(pixel_values=tensor_image(im).unsqueeze(0).to(a.device)).logits
                pred=F.interpolate(logits,size=(h,w),mode='bilinear',align_corners=False).argmax(1)[0].cpu().numpy().astype('uint8')
        Image.fromarray(pred).save(a.output/'mask-index.png')
        Image.fromarray(PALETTE[pred]).save(a.output/'mask-color.png')
        Image.fromarray((original*.5+PALETTE[pred]*.5).astype('uint8')).save(a.output/'overlay.jpg')
        summary={'class_names':model.config.id2label,'pixel_counts':np.bincount(pred.ravel(),minlength=8).tolist()}
    summary.update(image=str(a.image.resolve()),input_sha256=hashlib.file_digest(a.image.open('rb'),'sha256').hexdigest(),
        weights=str(a.weights.resolve()),device=a.device,seconds=time.perf_counter()-start)
    (a.output/'inference.json').write_text(json.dumps(summary,indent=2));print(summary)

if __name__=='__main__':main()
