# 数据与第三方许可

本项目是独立的非商业研究，只使用公开数据集与公开模型。仓库**不包含**任何数据集图像、标注、训练得到的模型权重、预测图或视频。

本项目原创代码按根目录 AGPL-3.0 许可提供。第三方数据、初始模型及其衍生权重仍分别受上游条款约束，源码许可不会把非商业模型或数据变成可商用资产。上游许可文本快照和 SHA256 见 `references/license-snapshots.json`。

## VisDrone

- 原始发布：https://github.com/VisDrone/VisDrone-Dataset
- 下载说明：https://aiskyeye.com/download/
- 竞赛划分规则：https://aiskyeye.com/test-guidelines-2022/
- 标注与忽略区域：https://github.com/VisDrone/VisDrone2018-DET-toolkit
- 镜像来源：https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/VisDrone.yaml

官方公开鼓励下载使用，并要求引用。已读页面没有给出明确的数据再分发许可证，所以本仓库不再分发原始数据，也不推断它可以商用。

使用约定：只用 DET train 训练，用 DET val 验证；test-dev 只在最终模型上做一次留出评测，不用于调参；MOT 数据不混入检测训练；原始标注不修改，噪声标签另存。

引用：Zhu et al., Detection and Tracking Meet Drones Challenge, IEEE TPAMI 44(11), 7380–7399, DOI:10.1109/TPAMI.2021.3119563。

## Ultralytics

代码和模型按上游 AGPL-3.0 / 商业许可条款使用：https://github.com/ultralytics/ultralytics/blob/main/LICENSE 。用 Ultralytics 训练得到的权重同样受这些条款约束。本仓库不发布权重；发布或商用衍生项目前需复核适用条款。

## UAVid

来源：https://uavid.nl/ 。官方声明的许可是 CC BY-NC-SA 4.0：保留作者署名、仅限非商业使用、相同方式共享。

官方 DANS 存档：https://doi.org/10.17026/dans-x9f-w9sa 。存档中机器可读的顶层 license 字段写的是 CC-BY-SA，但存档内的 readme 和官方网页都写 CC-BY-NC-SA。本项目按更严格的 NC-SA 约束使用，不把元数据上的差异理解为商业授权。数据不随仓库分发。

## SAM2 与 SegFormer

SAM2.1 tiny 的模型卡标记为 Apache-2.0：https://huggingface.co/facebook/sam2.1-hiera-tiny 。代码许可：https://github.com/facebookresearch/sam2/blob/main/LICENSE 。本项目使用 Transformers 的原生 PyTorch 视频实现。

SegFormer-B0 编码器来源：https://huggingface.co/nvidia/mit-b0 。模型卡指向 NVIDIA SegFormer Source Code License（https://github.com/NVlabs/SegFormer/blob/master/LICENSE），只允许研究、评估等非商业用途。

两个模型都锁定了仓库 revision 并保存了 SHA256，见 `reports/mit-b0-source.json` 和 `reports/sam2.1-tiny-source.json`。

## imageio-ffmpeg

依赖锁定文件里包含 `imageio-ffmpeg==0.8.1`（https://pypi.org/project/imageio-ffmpeg/）。它的 Python 包装代码是 BSD-2-Clause；Windows 轮子附带的 FFmpeg 二进制和 libx264 另受各自的 GPL 等许可约束。本仓库的脚本不直接调用它。

## 四个参考项目

Pi-Seg、RSKT-Seg、MTRefSeg、MARIS 只作为公开学术研究参考，见 `reports/研究方法对照.md` 和 `references/repositories.json`。每个仓库的许可都单独检查过；没有许可文件的，不复制实现，也不擅自重新授权。本仓库不包含它们的代码。
