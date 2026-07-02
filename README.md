# CLA-LIDNet

CLA-LIDNet 是一个面向语言识别实验和毕业设计展示的深度学习项目。当前仓库主体位于 `language_recognition_system/`，提供从公开语音数据准备、模型训练、单文件推理到 Flask 网页演示的一套最小可运行流程。

## 项目定位

本项目用于验证多语言语音片段的语言识别能力，适合做课程设计、毕业设计原型、语音识别前端实验或后续模型结构扩展。核心实现采用 `CNN + BiLSTM + Attention`，通过音频特征提取、监督训练和网页上传识别完成端到端演示。

## 核心功能

- 支持按语言目录组织 `.wav` 数据集
- 提供 FLEURS、CommonLanguage、VoxLingua 等公开数据准备脚本
- 支持本地 CPU 调试与云端 GPU 训练
- 提供训练前环境检查、训练、推理脚本
- 提供 Flask 上传页面，用于演示单段音频语言识别
- 输出 `best_model.pt` 与训练摘要，便于复现实验结果

## 技术栈

- Python
- PyTorch
- NumPy
- Flask
- Hugging Face Datasets / Hub

## 目录结构

```text
CLA-LIDNet/
├── README.md
├── language_recognition_system/
│   ├── README.md
│   ├── app.py
│   ├── requirements.txt
│   ├── scripts/
│   │   ├── check_training_ready.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── download_*.py
│   ├── src/language_recognition/
│   │   ├── audio.py
│   │   ├── dataset.py
│   │   ├── features.py
│   │   ├── model.py
│   │   ├── training.py
│   │   └── inference.py
│   ├── templates/
│   └── static/
└── skills/
```

## 快速开始

进入主项目目录：

```bash
cd language_recognition_system
```

安装依赖：

```bash
pip install -r requirements.txt
```

准备数据目录：

```text
dataset/
├── zh/
│   ├── zh_001.wav
│   └── zh_002.wav
├── en/
│   └── en_001.wav
└── ja/
    └── ja_001.wav
```

运行训练前检查：

```bash
python scripts/check_training_ready.py --dataset-root /path/to/dataset --device cpu
```

启动训练：

```bash
python scripts/train.py \
  --dataset-root /path/to/dataset \
  --output-dir checkpoints/run_demo \
  --epochs 5 \
  --batch-size 8 \
  --device cpu
```

单文件推理：

```bash
python scripts/predict.py \
  --checkpoint checkpoints/run_demo/best_model.pt \
  --audio /path/to/demo.wav \
  --device cpu
```

启动网页演示：

```bash
export LID_CHECKPOINT=checkpoints/run_demo/best_model.pt
export LID_DEVICE=cpu
python app.py
```

浏览器访问：

```text
http://127.0.0.1:5000
```

## 进一步说明

更完整的数据下载、训练参数和云服务器训练流程见：

- `language_recognition_system/README.md`
- `language_recognition_system/CLOUD_TRAINING.md`

## 当前状态

该仓库已具备语言识别实验的基础闭环，后续可以继续扩展更多语言类别、更大规模数据集、模型对比实验、评估报告和在线部署能力。