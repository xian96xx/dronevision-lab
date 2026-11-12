"""Summarize completed paired experiments, never turn running jobs into results."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1]

def read(name):
    path=ROOT/'reports'/name
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else None

def main():
    rows=[];checks=[]
    for seed in [42,43,44]:
        group=[]
        for variant in ['clean','boundary','wrong']:
            name=f'seg-noise-{variant}'+('' if seed==42 else f'-seed{seed}')
            record=read(name+'-training.json')
            if not record or record.get('status')!='completed':continue
            val=record['best_validation']
            cm=np.array(val['confusion_matrix'],dtype=float)
            union=cm.sum(0)+cm.sum(1)-cm.diagonal()
            miou=float(np.nanmean(np.divide(cm.diagonal(),union,out=np.full(8,np.nan),where=union>0)))
            assert abs(miou-val['mIoU'])<1e-10
            group.append(record)
            rows.append({'name':name,'seed':seed,'variant':variant,'mIoU':miou,
                         'pixel_accuracy':val['pixel_accuracy'],'human_IoU':val['per_class_IoU']['human'],
                         'per_class_IoU':val['per_class_IoU'],'seconds':record['seconds']})
        if len(group)==3:
            check={'seed':seed,'same_initial_weights':len({r['initial_state_sha256'] for r in group})==1,
                   'same_budget':all(r['command']['epochs']==10 and r['command']['subset']==40 and
                                     r['training_patches']==160 and r['validation_images']==70 for r in group),
                   'same_training_inputs':len({r['training_data_sha256'] for r in group})==1 if all('training_data_sha256' in r for r in group) else None,
                   'per_file_input_fingerprint_available':all('training_data_sha256' in r for r in group)}
            assert check['same_initial_weights'] and check['same_budget'] and check['same_training_inputs'] is not False
            checks.append(check)
    paired=[]
    for variant in ['boundary','wrong']:
        deltas=[];seeds=[]
        for seed in [42,43,44]:
            v=next((r for r in rows if r['seed']==seed and r['variant']==variant),None)
            c=next((r for r in rows if r['seed']==seed and r['variant']=='clean'),None)
            if v and c:deltas.append(v['mIoU']-c['mIoU']);seeds.append(seed)
        if deltas:
            mean=statistics.mean(deltas)
            half=float(t.ppf(.975,len(deltas)-1)*statistics.stdev(deltas)/(len(deltas)**.5)) if len(deltas)>1 else None
            paired.append({'variant':variant,'seeds':seeds,'n':len(deltas),'delta_mIoU_mean':mean,
                           'paired_deltas':deltas,'exploratory_t95':None if half is None else [mean-half,mean+half]})
    weighted=read('seg-full-weighted-20-training.json');baseline=read('seg-full-20-training.json')
    comparison={'status':'pending' if weighted is None else weighted.get('status')}
    if weighted and weighted.get('status')=='completed':
        assert weighted['initial_state_sha256']==baseline['initial_state_sha256']
        assert weighted['command']['epochs']==baseline['command']['epochs']==20
        assert weighted['training_patches']==baseline['training_patches']==800
        comparison.update(same_initial_weights=True,same_crops_and_budget=True,
            baseline_mIoU=baseline['best_validation']['mIoU'],weighted_mIoU=weighted['best_validation']['mIoU'],
            baseline_human_IoU=baseline['best_validation']['per_class_IoU']['human'],
            weighted_human_IoU=weighted['best_validation']['per_class_IoU']['human'],
            delta_mIoU=weighted['best_validation']['mIoU']-baseline['best_validation']['mIoU'],
            baseline_pixel_accuracy=baseline['best_validation']['pixel_accuracy'],
            weighted_pixel_accuracy=weighted['best_validation']['pixel_accuracy'],
            per_class_IoU=weighted['best_validation']['per_class_IoU'],class_weights=weighted['class_weights'])
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'completed_noise_runs':len(rows),'target_noise_runs':9,
            'noise_rows':rows,'protocol_checks':checks,'paired_differences':paired,'weighted_comparison':comparison,
            'limitations':['Three training seeds share one fixed corruption realization and one validation set.',
                          't intervals with only 2-3 seeds are exploratory; they do not establish generalization.',
                          'Seed42 legacy runs lack per-file training-input fingerprints; selection policy and manifests are retained.',
                          'Class weighting is a single-seed development experiment, not a blind test result.',
                          'No company or user-supplied sequence entered training.']}
    path=ROOT/'reports/refinement-results.json';path.write_text(json.dumps(result,indent=2))
    lines=['# 第二轮训练与质量实验', '',f'已完成分割噪声实验 {len(rows)}/9 组。只汇总 status=completed 的真实训练结果。','',
           '| seed | 标签版本 | mIoU | human IoU |', '| --- | --- | --- | --- |']
    lines.extend(f'| {r["seed"]} | {r["variant"]} | {r["mIoU"]:.5f} | {r["human_IoU"]:.5f} |' for r in rows)
    lines+=['','## 同种子配对差值','']
    for row in paired:
        lines.append(f'- {row["variant"]} 相对干净组：平均 mIoU 差值 {row["delta_mIoU_mean"]:+.5f}，配对种子 {row["seeds"]}；探索性 t 区间 {row["exploratory_t95"]}。')
    lines+=['','## 类别加权实验','']
    if comparison['status']=='completed':
        lines+=['| 相同初始化与20轮预算 | mIoU | human IoU |','| --- | --- | --- |',
                f'| 原始交叉熵 | {comparison["baseline_mIoU"]:.5f} | {comparison["baseline_human_IoU"]:.5f} |',
                f'| 训练频率类别加权 | {comparison["weighted_mIoU"]:.5f} | {comparison["weighted_human_IoU"]:.5f} |','']
        lines.append(f'像素准确率：{comparison["baseline_pixel_accuracy"]:.5f} → {comparison["weighted_pixel_accuracy"]:.5f}；应同时检查总体与类别表现，不能宣称所有指标同时改善。')
    else:lines+=['状态：'+comparison['status']+'；没有最终成绩。','']
    lines+=[
            '3个种子仍不足以证明跨数据集结论；同一验证集被重复用于开发，没有盲测 test 成绩。',
            '加权组与原全量组相同初始化、800裁块、20轮与验证协议。只改变由训练掩码频率计算的损失类别权重；不保证 human 或总体 mIoU 改善。',
            '旧单种子实验和权重保留，不用新的结果覆盖旧记录。']
    (ROOT/'reports/第二轮训练结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans']
    fig,ax=plt.subplots(figsize=(7,4))
    for i,variant in enumerate(['clean','boundary','wrong']):
        values=[r['mIoU'] for r in rows if r['variant']==variant]
        if values:
            ax.bar(i,statistics.mean(values),yerr=statistics.stdev(values) if len(values)>1 else 0,capsize=5,color=['#67b7af','#e6ac64','#ac94cc'][i])
            ax.scatter([i]*len(values),values,c='black',s=16,zorder=3)
    ax.set_xticks([0,1,2],['clean','boundary','wrong']);ax.set_ylabel('验证集 mIoU');ax.set_title('分割标签噪声：点为真实种子，误差线为样本标准差')
    fig.tight_layout();fig.savefig(ROOT/'reports/seg-noise-multiseed.png',dpi=160);plt.close(fig)
    if comparison['status']=='completed':
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for name,label in [('seg-full-20','原始交叉熵'),('seg-full-weighted-20','训练频率类别加权')]:
            history=json.loads((ROOT/'runs'/name/'history.json').read_text())
            axes[0].plot([h['epoch'] for h in history],[h['mIoU'] for h in history],label=label)
            axes[1].plot([h['epoch'] for h in history],[h['per_class_IoU']['human'] for h in history],label=label)
        axes[0].set_ylabel('全70图验证 mIoU');axes[1].set_ylabel('全70图 human IoU')
        for ax in axes:ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend()
        fig.suptitle('同初始化、同800裁块、同20轮：仅改变损失类别权重，seed42')
        fig.tight_layout();fig.savefig(ROOT/'reports/seg-weighted-training.png',dpi=160);plt.close(fig)
    print(json.dumps({'completed_noise_runs':len(rows),'weighted_status':comparison['status']}))

if __name__=='__main__':main()
