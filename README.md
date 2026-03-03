<div align="center">

# 🎬 Generative Regression Based Watch Time Prediction for Short-Video Recommendation

[![Paper](https://img.shields.io/badge/arXiv-2412.20211-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2412.20211)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

<div align="center">

[Hongxu Ma](mailto:hxma24@m.fudan.edu.cn)\*†&emsp;
[Kai Tian](mailto:tiank311@gmail.com)\*&emsp;
[Tao Zhang](mailto:zhangtao08@kuaishou.com)&emsp;
[Xuefeng Zhang](mailto:zhangxuefeng06@kuaishou.com)&emsp;
[Han Zhou](mailto:zhouhan@stu.sufe.edu.cn)&emsp;
[Chenghou Jin](mailto:jinch24@m.fudan.edu.cn)&emsp;
[Chunjie Chen](mailto:chencj517@gmail.com)&emsp;
[Han Li](mailto:lihan08@kuaishou.com)&emsp;
[Jihong Guan](mailto:jhguan@tongji.edu.cn)&emsp;
[Shuigeng Zhou](mailto:sgzhou@fudan.edu.cn)‡

🏫 Fudan University &emsp; 🏢 Kuaishou Technology &emsp; 🏫 Shanghai University of Finance and Economics &emsp; 🏫 Tongji University

<sub>\* Both authors contributed equally to this research &emsp; † Work done during the internship at Kuaishou Technology &emsp; ‡ Corresponding author</sub>

</div>

---

## 📖 Introduction

This repository contains the official implementation of **GR (Generative Regression)**, a novel framework for watch time prediction in short-video recommendation systems. GR reformulates continuous watch time regression as a sequence generation problem via a Seq2Seq Transformer, enabling richer modeling of the output distribution through a dynamic vocabulary construction and windowed soft-argmax decoding.

---

## 🔧 Requirements

```bash
pip install torch numpy scikit-learn tqdm
```

| Package | Version |
| :--- | :--- |
| Python | ≥ 3.8 |
| PyTorch | ≥ 1.12 |
| NumPy | ≥ 1.21 |
| scikit-learn | ≥ 1.0 |
| tqdm | ≥ 4.0 |

---

## 📂 Data Preparation

### Option A — Use our pre-processed files (recommended)

We provide ready-to-use `.npy` feature arrays. Download them directly from Google Drive:

🔗 **[Download pre-processed data (Google Drive)](https://drive.google.com/file/d/1nKxE31n_tAFE-ceAHAhsC9dxQ9P_w4O4/view?usp=drive_link)**

After downloading, place the files anywhere convenient and pass their paths to `train.py` via `--train_data` and `--test_data`.

---

### Option B — Process from raw KuaiRec data

Our preprocessing follows the pipeline provided by the original KuaiRec dataset authors ([EasyRL4Rec / KuaiData.py](https://github.com/chongminggao/EasyRL4Rec/blob/main/src/core/envs/KuaiRec/KuaiData.py)).

**Step 1 — Download the raw dataset**

```bash
# Via wget (recommended)
wget https://nas.chongminggao.top:4430/easyrl4rec/data.tar.gz

# Or download manually from:
# https://rec.ustc.edu.cn/share/a3bdc320-d48e-11ee-8c50-4b1c32c31e9c
```

**Step 2 — Set up the directory structure**

Create a `data_raw/` folder under `data/` and place the following files inside:

```
data/
└── data_raw/
    ├── big_matrix_processed.csv    # pre-normalised big-matrix interactions
    ├── small_matrix_processed.csv  # pre-normalised small-matrix interactions
    ├── item_categories.csv         # item category tags
    └── user_features.csv           # raw user features
```

**Step 3 — Run the preprocessing script**

```bash
python data_process.py
```

This will produce:

```
data/
└── data_processed/
    ├── train_kauiRec.npy
    └── test_kauiRec.npy
```

---

## 🚀 Usage

### Basic Training

```bash
python train.py \
    --train_data data/data_processed/train_kauiRec.npy \
    --test_data  data/data_processed/test_kauiRec.npy  \
    --log_dir    checkpoints/
```

### ✨ With Embedding Mixup

```bash
python train.py \
    --train_data data/data_processed/train_kauiRec.npy \
    --test_data  data/data_processed/test_kauiRec.npy  \
    --log_dir    checkpoints/ \
    --use_embedding_mixup
```

### 📈 With Curriculum Learning

Supports three decay schedules: `linear`, `exp`, `sigmoid`.

```bash
python train.py \
    --train_data              data/data_processed/train_kauiRec.npy \
    --test_data               data/data_processed/test_kauiRec.npy  \
    --log_dir                 checkpoints/ \
    --use_curriculum_learning \
    --curriculum_learning_type    exp \
    --curriculum_learning_decay   0.9999
```

## 📋 Arguments

### Data

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--train_data` | str | **Required** | Path to training data (.npy) |
| `--test_data` | str | **Required** | Path to test data (.npy) |

### Model Architecture

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--feat_dim` | int | 128 | Feature dimension |
| `--hidden_dim` | int | 128 | Model hidden dimension |
| `--n_head` | int | 8 | Number of attention heads |
| `--dec_layers` | int | 3 | Number of decoder layers |
| `--dropout` | float | 0.1 | Dropout rate |
| `--ffn_dim` | int | 256 | Feedforward dimension |
| `--window_size` | int | 20 | Window size for soft-argmax decoding |

### Training

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--batch_size` | int | 512 | Batch size |
| `--num_epochs` | int | 20 | Number of training epochs |
| `--lr` | float | 5e-4 | Learning rate |
| `--cls_weight` | float | 10.0 | Cross-entropy loss weight |
| `--huber_weight` | float | 1.0 | Huber loss weight |
| `--log_dir` | str | `checkpoints/` | Directory to save best model |
| `--seed` | int | 2024 | Random seed |

### Vocabulary

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--q_start` | float | 0.9999 | Starting quantile |
| `--q_end` | float | 0.9 | Ending quantile |
| `--q_decay_rate` | float | 0.99 | Quantile decay rate |
| `--epsilon` | float | 1e-6 | Convergence threshold |

### Model Variants

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--use_embedding_mixup` | flag | False | Enable embedding mixup |
| `--use_curriculum_learning` | flag | False | Enable curriculum learning |
| `--curriculum_learning_type` | str | `linear` | Decay schedule: `linear`, `exp`, `sigmoid` |
| `--curriculum_learning_decay` | float | 0.9999 | Decay rate for teacher forcing |
| `--teacher_force_ratio` | float | 0.5 | Initial teacher forcing ratio |

---

## 💖 Citation

If you find GR useful for your project or research, welcome to ⭐ this repo and cite our work:

```bibtex
@article{ma2025generativeregressionbasedwatch,
  title   = {Generative Regression Based Watch Time Prediction for Short-Video Recommendation},
  author  = {Hongxu Ma and Kai Tian and Tao Zhang and Xuefeng Zhang and Han Zhou and
             Chenghou Jin and Chunjie Chen and Han Li and Jihong Guan and Shuigeng Zhou},
  journal = {arXiv preprint arXiv:2412.20211},
  year    = {2025}
}
```
