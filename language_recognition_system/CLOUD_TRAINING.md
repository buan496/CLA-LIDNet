# Cloud Training Prep

This project is ready for a Tencent Cloud single-GPU training run that expands the VoxLingua107 subset to a thesis-friendly scale.

## Recommended Server

- Instance family: single-GPU compute instance
- Preferred GPU: NVIDIA A10 24 GB
- Acceptable fallback: NVIDIA T4 16 GB
- vCPU: 16 or more
- Memory: 64 GB or more
- OS: Ubuntu 22.04 LTS
- Disk: 100 GB system disk plus 100 GB or more data disk

The current training code is single-GPU only, so one strong GPU is better value than a multi-GPU machine.

Your selected Tencent Cloud `GN10Xp.2XLARGE40` instance is also suitable:

- GPU: 1 x NVIDIA V100 32 GB
- vCPU: 10
- Memory: 40 GiB
- OS: Ubuntu Server 22.04 LTS 64-bit
- System disk: 100 GiB SSD cloud disk
- Data disk: 200 GiB SSD cloud disk
- Driver/CUDA/cuDNN: 570.158.01 / 12.8.1 / 9.10.2

This is enough for the planned `16500`-sample VoxLingua run. Because the V100 is a Volta GPU, do not install CUDA 13 PyTorch wheels for this project. Use a CUDA 12.x PyTorch build, preferably `cu128` for this server image.

## Dataset Target

For thesis presentation, use real audio samples instead of only augmentation-based expansion.

- Languages: `en ja zh`
- Target raw samples: `16500`
- Per language target: `5500`
- Expected split after the current code path:
  - train: about `10560`
  - val: about `2640`
  - test: about `3300`

This is driven by `11` VoxLingua shards per language at about `500` WAV files per shard.

## Before You Start

Clone the repo to the server, then install the Python dependencies in your environment:

```bash
cd ~/CLA-LIDNet/language_recognition_system
python -m pip install -r requirements.txt
```

If your environment does not already include a CUDA-enabled PyTorch build, install that first using the matching PyTorch command for your CUDA version.

For the Tencent Cloud V100 + CUDA 12.8 server, use the pinned helper instead:

```bash
cd ~/CLA-LIDNet/language_recognition_system
bash scripts/cloud_setup_v100_cuda128.sh
```

It installs Flask, Hugging Face Hub, NumPy, and CUDA-enabled PyTorch wheels from the `cu128` index, then verifies that PyTorch sees the V100.

## Readiness Check

Use the diagnostic script before downloading data or training:

```bash
cd ~/CLA-LIDNet/language_recognition_system
python scripts/cloud_check_environment.py --workspace-root .
```

It reports Python, Torch, CUDA, free disk space, and optional dataset counts.

## One-Command Server Run

The script below downloads the `16500`-sample VoxLingua subset, checks readiness, and launches training:

```bash
cd ~/CLA-LIDNet/language_recognition_system
bash scripts/cloud_train_vox_16500.sh
```

Default settings:

- dataset root: `~/cla_lidnet_runs/dataset_vox_16500`
- batch size: `32`
- epochs: `30`
- repeat factor: `3`
- min epochs: `15`
- early stop patience: `8`
- device: `cuda`

Outputs land under `language_recognition_system/checkpoints/run_vox_16500_b32_<timestamp>/`.

## Useful Overrides

You can tune the run without editing code:

```bash
cd ~/CLA-LIDNet/language_recognition_system
RUN_NAME=run_vox_16500_b48 \
BATCH_SIZE=48 \
EPOCHS=40 \
TRAIN_REPEAT_FACTOR=3 \
bash scripts/cloud_train_vox_16500.sh
```

If you already downloaded the dataset:

```bash
cd ~/CLA-LIDNet/language_recognition_system
SKIP_DOWNLOAD=1 \
DATASET_ROOT=~/cla_lidnet_runs/dataset_vox_16500 \
bash scripts/cloud_train_vox_16500.sh
```

## What To Watch

After training starts, the key files are:

- `checkpoints/<run_name>/train.log`
- `checkpoints/<run_name>/train_summary.json`
- `checkpoints/<run_name>/best_model.pt`

When `train_summary.json` appears, the run has completed and you can compare its metrics with the local Vox and FLEURS baselines.
