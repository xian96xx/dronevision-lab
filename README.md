# DroneVision Lab：无人机视觉训练与标注质量影响研究

用公开无人机数据集，在单张 RTX 5060 Ti 16GB 上真实训练检测和分割模型，再用受控对照实验观察标注错误（框偏移、漏标、掩码边界扰动、局部错标）和自动补齐（区间插值、SAM2）对模型和标注质量的影响。

*An independent study on public UAV datasets (VisDrone2019-DET/MOT, UAVid): real fine-tuning of YOLO26n and SegFormer-B0 on one RTX 5060 Ti, controlled label-noise experiments, keyframe interpolation vs. SAM2 vs. trackers, with every number traceable to a JSON record in `reports/`.*

## 说明

- 独立的个人研究，只用公开数据集和公开模型，不含任何公司或机构的内部数据、代码和模型。
- 代码和实验的执行主要由 AI 编程助手完成；作者负责提出研究问题和实验要求、试用、检查结果并反馈迭代。
- 所有数字都来自真实运行，原始记录在 `reports/*.json`；完整分析见 [技术报告](reports/技术报告.md)。
- 仓库不包含数据集、训练权重、预测图和视频（许可原因见 [THIRD_PARTY.md](THIRD_PARTY.md)）。

## 主要结果

评测用的是本项目自定义的 COCO 式口径（忽略区按 crowd 处理），**不是官方 VisDrone 分数**，不能和排行榜直接比较。

| 实验 | 设置 | 结果 | 口径与限制 |
| --- | --- | --- | --- |
| 检测：COCO 预训练 → VisDrone 微调（共享六类） | YOLO26n，640，50 轮 | mAP50-95：val 0.092 → 0.205；test-dev 0.041 → 0.178 | 只比两种类别体系共有的六类；固定阈值下 Precision 下降、Recall 上升 |
| 检测：微调后十类 | 同上 | val mAP50 0.300、mAP50-95 0.164；test-dev 0.251、0.136 | test-dev 每个模型只评一次；其标注公开，不是官方盲测 |
| 检测标签噪声 | 固定 256 张图、20 轮、3 个训练种子 | mAP50-95：干净 0.0425，框偏移 0.0250，漏标 30% 0.0399 | 小预算下的相对效应；每种噪声只试了一个强度 |
| 区间插值的关键帧间距 | 3 段 × 120 帧、12 条轨迹 | 每 10 帧给一个人工框：平均 IoU 0.80；只给首尾两帧：0.16 | 人工框数量不同；视频与 DET 有同源航次 |
| SAM2.1 tiny（首帧框提示） | 同上 | 三段视频的平均框 IoU 在 0.42–0.72 之间 | 可见掩码的紧致框与遮挡 GT 框语义不同 |
| ByteTrack / BoT-SORT | 微调后的 YOLO | MOTA 0.16–0.44，IDF1 0.31–0.65 | HOTA 未实现；不是严格未见场景的成绩 |
| SegFormer-B0（UAVid） | 800 个裁块、20 轮 | mIoU 0.547；类别加权后 0.567，human IoU 从 0 升到 0.178 | 资源受限协议，不同于论文的高分辨率滑窗评测 |
| 分割标签噪声 | 40 张图、10 轮、3 个种子 | 边界扰动 mIoU −0.0071（95% 区间含 0），局部错标 −0.0003 | 小预算下没看到显著影响；human 类在这些实验里 IoU 都是 0 |

## 复现

### 环境

已验证环境：Windows，RTX 5060 Ti 16GB（sm_120），Python 3.12.10，PyTorch 2.9.1+cu128，torchvision 0.24.1+cu128，Ultralytics 8.4.14。RTX 50 系显卡需要 CUDA 12.8 及以上的 PyTorch 构建。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe scripts\verify_gpu.py
.\.venv\Scripts\python.exe -m pip check
```

`verify_gpu.py` 会做几步真实的 CUDA 反向传播和参数更新，确认 GPU 训练可用。

### 数据与初始模型

先读 [THIRD_PARTY.md](THIRD_PARTY.md)：VisDrone 只用于非商业研究且不再分发；UAVid 按 CC BY-NC-SA 4.0 使用。

```powershell
.\.venv\Scripts\python.exe scripts\download_det.py
.\.venv\Scripts\python.exe scripts\prepare_det.py --subset 256 --smoke 32
.\.venv\Scripts\python.exe scripts\download_mot.py --split train
.\.venv\Scripts\python.exe scripts\download_mot.py --split val
.\.venv\Scripts\python.exe scripts\download_uavid.py
.\.venv\Scripts\python.exe scripts\fetch_hf_models.py
.\.venv\Scripts\python.exe scripts\segmentation.py prepare
```

- 下载脚本会校验 ZIP CRC 和 SHA256；`fetch_file.py` 支持分块断点下载。
- `prepare_det.py` 把 VisDrone 转成 YOLO 格式（排除忽略区和非 1–10 类，原始标注不改），生成带 SHA256 的数据清单，并用固定种子 42 生成噪声标签版本。它会按你本机的绝对路径重写 `configs/*.yaml`；仓库里的配置用 `<REPO_ROOT>` 占位。
- `fetch_hf_models.py` 下载 SegFormer 编码器和 SAM2.1 tiny，并锁定仓库 revision。YOLO26n 初始权重在第一次运行 `train_det.py` 时由 Ultralytics 下载到 `models/yolo26n.pt`。

### 检测：训练与评测

```powershell
.\.venv\Scripts\python.exe scripts\train_det.py --data smoke --name smoke --epochs 2 --batch 8
.\.venv\Scripts\python.exe scripts\train_det.py --data det-full --name det-full-50 --epochs 50 --batch 16
# 中断后只对同一实验续训
.\.venv\Scripts\python.exe scripts\train_det.py --data det-full --name det-full-50 --epochs 50 --batch 16 --resume

.\.venv\Scripts\python.exe scripts\evaluate_det.py --weights models/yolo26n.pt --name baseline-common --ontology common --pretrained-coco
.\.venv\Scripts\python.exe scripts\evaluate_det.py --weights runs/det-full-50/weights/best.pt --name trained-common --ontology common
.\.venv\Scripts\python.exe scripts\evaluate_det.py --weights runs/det-full-50/weights/best.pt --name trained-full
```

test-dev 只在最终模型上评一次，不拿它调参：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_det.py --weights runs/det-full-50/weights/best.pt --name testdev-trained-full --split test-dev
.\.venv\Scripts\python.exe scripts\evaluate_det.py --weights runs/det-full-50/weights/best.pt --name testdev-trained-common --ontology common --split test-dev
.\.venv\Scripts\python.exe scripts\evaluate_det.py --weights models/yolo26n.pt --name testdev-baseline-common --ontology common --pretrained-coco --split test-dev
```

- 共享六类把 pedestrian 和 people 合并成 human，另有 bicycle、car、truck、bus、motorcycle。COCO 80 类和 VisDrone 10 类的 mAP 不能直接相减。
- P/R 是 conf=0.25、IoU=0.5 下的微平均；AP 用 COCO 101 点。
- `runs/<name>/weights/best.pt` 用于推理，`last.pt` 用于恢复没跑完的训练。

### 检测标签噪声对照

固定 256 张训练图，三组只改标签：干净、框偏移（每个框的 x、y 各自在 ±20% 宽/高内均匀平移）、漏标（每个框以 30% 概率删除）。噪声实现固定用种子 42；训练种子分别取 42、43、44，每组 20 轮，只在最后一轮验证。

```powershell
.\.venv\Scripts\python.exe scripts\train_det.py --data noise-shift --name noise-shift-seed42 --epochs 20 --batch 16 --seed 42 --final-val-only
.\.venv\Scripts\python.exe scripts\evaluate_det.py --weights runs/noise-shift-seed42/weights/best.pt --name noise-shift-seed42-eval
```

把 `noise-shift` 换成 `noise-clean`、`noise-drop`，种子换成 43、44，即得全部 9 组。`audit_noise.py` 统计标签实际被改动了多少。

### 视频：插值、SAM2 与跟踪

```powershell
.\.venv\Scripts\python.exe scripts\audit_mot.py
.\.venv\Scripts\python.exe scripts\video_experiment.py --weights runs/det-full-50/weights/best.pt --frames 120
.\.venv\Scripts\python.exe scripts\interpolation_sweep.py
```

固定 MOT-val 前三段、每段 120 帧、最多 4 条轨迹，只评估中间帧。插值用首尾真值框；SAM2.1 tiny 用首帧框提示，走 Transformers 的原生 PyTorch 视频实现，在 Windows 上直接运行；检测加跟踪不用后续真值。三种方法用到的人工框数量不同，不做同等成本下的优劣排名。

### 语义分割

```powershell
.\.venv\Scripts\python.exe scripts\segmentation.py train --name seg-full-20 --epochs 20
.\.venv\Scripts\python.exe scripts\segmentation.py train --name seg-noise-clean --subset 40 --epochs 10
.\.venv\Scripts\python.exe scripts\segmentation.py train --name seg-noise-boundary --variant boundary --subset 40 --epochs 10
.\.venv\Scripts\python.exe scripts\segmentation.py train --name seg-noise-wrong --variant wrong --subset 40 --epochs 10
# 第二轮：分割噪声补跑种子 43、44，以及类别加权全量 20 轮（共 7 组）
.\.venv\Scripts\python.exe scripts\run_refinement.py
```

SegFormer-B0 用 ImageNet 预训练编码器、8 类头、像素交叉熵；训练取 800 个固定 512 裁块，验证用全部 70 张验证图，输入最长边 1024、在原生掩码上计分。类别加权用 `--class-weighting inverse-sqrt`，权重只由训练掩码的像素频率计算。

### 推理

```powershell
.\.venv\Scripts\python.exe scripts\infer.py --task detect --image path\to\image.jpg --weights runs\det-full-50\weights\best.pt
.\.venv\Scripts\python.exe scripts\infer.py --task segment --image path\to\image.png --weights runs\seg-full-20\best
```

加 `--device cpu` 可以在 CPU 上运行。不要加载来源不可信的 PyTorch pickle 权重。

## 目录

```text
configs/          数据配置快照（prepare_det.py 会按本机路径重新生成）
scripts/          下载、转换、训练、评测、视频与分割实验
reports/          指标 JSON、数据清单、图表、技术报告、方法对照、失败记录
references/       第三方许可文本快照与参考仓库版本
reproducibility/  第二轮分割训练实际执行时的源码快照
```

`data/`、`models/`、`runs/`、`logs/` 和 `.venv/` 在本地运行时生成，都已加入 `.gitignore`。

## 局限

- val 在开发中反复使用，`best.pt` 也是按 val 选出的；test-dev 只评了一次，但它的标注是公开的，不能当作官方盲测。
- 噪声实验的预算很小：检测 256 张图训 20 轮，分割 40 张图训 10 轮，每种噪声只试一个强度。人为噪声也不能完全代表真实的人工标注错误。
- 分割的 human 类仍然很弱；视频的 3 段序列与 DET 数据有同源航次；HOTA 没有实现。
- 没有做伪标签再训练，也没有测人工改标的耗时。

## 许可

本仓库的原创代码按 [AGPL-3.0](LICENSE) 发布。数据集、初始模型和由它们训练出的权重仍受各自上游条款约束，见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 致谢与引用

- VisDrone：Zhu et al., *Detection and Tracking Meet Drones Challenge*, IEEE TPAMI 44(11), 7380–7399, DOI:10.1109/TPAMI.2021.3119563。
- UAVid：https://uavid.nl/ ，DANS 存档 DOI:10.17026/dans-x9f-w9sa。
- Ultralytics YOLO：https://github.com/ultralytics/ultralytics
- SAM 2：https://github.com/facebookresearch/sam2
- SegFormer：https://github.com/NVlabs/SegFormer
- 方法对照中参考的四个公开项目见 [研究方法对照](reports/研究方法对照.md)。
