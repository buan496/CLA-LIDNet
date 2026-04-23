# Language Recognition System

一个适合毕业设计演示和后续扩展的深度学习语言识别系统，包含：

1. `CNN + BiLSTM + Attention` 识别模型
2. 训练脚本和单文件推理脚本
3. 简单的 Flask 上传识别页面
4. 面向本地调试和云服务器训练的运行说明
5. `FLEURS` / `CommonLanguage` 数据准备脚本

## 1. 推荐环境

你当前机器上已经确认可用的环境是：

```powershell
conda run -n nlp100 python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

后续所有命令都建议在 `nlp100` 环境里运行。

## 2. 数据目录格式

只要按下面这种目录组织 `.wav` 文件就可以直接训练：

```text
dataset/
  zh/
    zh_001.wav
    zh_002.wav
  en/
    en_001.wav
  ja/
    ja_001.wav
```

说明：

1. 当前版本只读取 `.wav` 文件。
2. 一级子目录名就是语言标签。
3. 本地先用 3 到 6 个语言类别调通最稳妥。

## 2.1 下载公开训练数据

当前项目优先推荐使用 `FLEURS` 小规模子集，因为它更适合论文实验，也不依赖远程 dataset script：

```powershell
conda run -n nlp100 python H:\deepmode\language_recognition_system\scripts\download_fleurs_subset.py `
  --output-root H:\deepmode\dataset `
  --configs cmn_hans_cn en_us ja_jp `
  --config-map "cmn_hans_cn:zh en_us:en ja_jp:ja" `
  --max-per-split 120 `
  --report-path H:\deepmode\language_recognition_system\checkpoints\fleurs_report.json
```

这会把公开数据导出成我们训练脚本能直接读取的结构：

```text
dataset/
  zh/
    train_zh_00000.wav
    validation_zh_00000.wav
  en/
  ja/
```

如果后面想扩成更多语言，只需要修改 `--configs` 和 `--config-map`。

项目里也保留了 `CommonLanguage` 下载脚本：

```powershell
conda run -n nlp100 python -m pip install datasets
conda run -n nlp100 python H:\deepmode\language_recognition_system\scripts\download_commonlanguage.py --help
```

但当前更建议先用 `FLEURS` 把训练跑通。

## 3. 开始训练

训练前先做一次体检：

```powershell
conda run -n nlp100 python H:\deepmode\language_recognition_system\scripts\check_training_ready.py `
  --dataset-root H:\deepmode\dataset `
  --device cuda
```

这个脚本会检查：

1. 当前 `torch` 和 `CUDA` 是否可用
2. 音频读取、特征提取、模型前向是否正常
3. 数据集目录是否存在
4. 每个语言类别的样本量是否足够开始训练

本地 CPU 调试：

```powershell
conda run -n nlp100 python H:\deepmode\language_recognition_system\scripts\train.py `
  --dataset-root H:\deepmode\dataset `
  --output-dir H:\deepmode\language_recognition_system\checkpoints\trial_cpu_debug `
  --epochs 5 `
  --batch-size 8 `
  --device cpu
```

云服务器 GPU 训练：

```powershell
conda run -n nlp100 python /path/to/language_recognition_system/scripts/train.py \
  --dataset-root /path/to/dataset_fleurs240 \
  --output-dir /path/to/language_recognition_system/checkpoints/run_aug_fleurs240 \
  --epochs 30 \
  --batch-size 32 \
  --train-repeat-factor 3 \
  --device cuda
```

训练完成后会生成：

1. `best_model.pt`
2. `train_summary.json`

## 4. 命令行推理

```powershell
conda run -n nlp100 python H:\deepmode\language_recognition_system\scripts\predict.py `
  --checkpoint H:\deepmode\language_recognition_system\checkpoints\run_aug_fleurs240\best_model.pt `
  --audio H:\deepmode\demo.wav `
  --device cpu
```

如果云服务器上有 GPU，也可以改成 `--device cuda`。

## 5. 启动网页

```powershell
$env:LID_CHECKPOINT="H:\deepmode\language_recognition_system\checkpoints\run_aug_fleurs240\best_model.pt"
$env:LID_DEVICE="cpu"
conda run -n nlp100 python H:\deepmode\language_recognition_system\app.py
```

浏览器打开：

```text
http://127.0.0.1:5000
```

## 6. 云服务器部署建议

如果数据量较大，建议把训练放到云服务器：

1. 本地负责调试目录结构、确认脚本能启动。
2. 将项目和数据集上传到云服务器。
3. 在云端用 `conda run -n nlp100 ... --device cuda` 训练。
4. 训练完成后把 `best_model.pt` 拉回本机，网页推理直接复用。

这个流程对毕业设计很实用，因为你本地只需要负责展示，长时间训练交给云端即可。
