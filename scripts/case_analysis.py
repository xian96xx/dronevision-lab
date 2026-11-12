"""Actual prediction-vs-GT error cases, using the saved evaluator outputs."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

def build_cases(out,raw,truth,ignored,paths,names):
    from evaluate_det import overlap
    predictions=json.loads((out/'predictions.json').read_text())
    byimage={}
    for p in predictions:
        if p['score']>=.25:
            x,y,w,h=p['bbox']
            byimage.setdefault(p['image_id'],[]).append({'class':p['category_id']-1,'score':p['score'],'box':[x,y,x+w,y+h]})
    analyses=[];classified={}
    for image_id,ip in enumerate(paths,1):
        matched=set();ps=[]
        for d in sorted(byimage.get(image_id,[]),key=lambda x:-x['score']):
            candidates=[(i,t) for i,t in enumerate(truth[image_id]) if i not in matched and t[0]==d['class']]
            ious=overlap([d['box']],[t[1:] for _,t in candidates])[0]
            if len(ious) and ious.max()>=.5:
                matched.add(candidates[int(ious.argmax())][0]);ps.append({**d,'kind':'TP','iou':float(ious.max())})
            elif ignored[image_id] and overlap([d['box']],ignored[image_id],True).max()>=.5:
                continue
            else:ps.append({**d,'kind':'FP','nearest_same_class_iou':float(ious.max()) if len(ious) else 0.})
        missed=[t for i,t in enumerate(truth[image_id]) if i not in matched]
        small_missed=sum((t[3]-t[1])*(t[4]-t[2])<1024 for t in missed)
        analyses.append({'image_id':image_id,'image':ip.name,'FP':sum(x['kind']=='FP' for x in ps),
            'FN':len(missed),'TP':len(matched),'small_FN_native_area_lt1024':small_missed,
            'localization_near_misses':sum(x['kind']=='FP' and .1<=x.get('nearest_same_class_iou',0)<.5 for x in ps)})
        classified[image_id]=(ps,missed)
    selected=[]
    for field in ['FN','FP','localization_near_misses','small_FN_native_area_lt1024']:
        for row in sorted(analyses,key=lambda x:x[field],reverse=True):
            if row['image_id'] not in [x['image_id'] for x in selected]:selected.append({**row,'selection_reason':field});break
    for record in selected:
        image_id=record['image_id'];ip=paths[image_id-1]
        left=Image.open(ip).convert('RGB');right=left.copy();dl=ImageDraw.Draw(left);dr=ImageDraw.Draw(right)
        for t in truth[image_id]:dl.rectangle(t[1:],outline='lime',width=2)
        ps,missed=classified[image_id]
        for p in ps:
            dr.rectangle(p['box'],outline='cyan' if p['kind']=='TP' else 'red',width=2)
        for t in missed:dr.rectangle(t[1:],outline='yellow',width=2)
        dl.rectangle([0,0,left.width,24],fill='black');dl.text((5,5),'Reference GT: green',fill='white')
        dr.rectangle([0,0,right.width,24],fill='black');dr.text((5,5),'TP cyan | FP red | FN yellow | conf>=0.25 IoU>=0.5',fill='white')
        canvas=Image.new('RGB',(left.width*2,left.height));canvas.paste(left);canvas.paste(right,(left.width,0));canvas.thumbnail((1920,1080))
        filename=f'error-{record["selection_reason"]}-{ip.name}'
        canvas.save(out/filename);record['visualization']=filename
    report={'selected_cases':selected,'all_image_counts':analyses,
        'limits':'GT is public annotation, not perfect manual re-audit. Case selection uses val only for descriptive analysis. Small defined by native bbox area<1024.'}
    (out/'error-analysis.json').write_text(json.dumps(report,indent=2))
    return selected
