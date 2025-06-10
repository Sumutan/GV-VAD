# GV-VAD

This repository contains the official implementation of our paper:
[**GV-VAD: Exploring Video Generation for Weakly-Supervised Video Anomaly Detection**](https://arxiv.org).

---

## 🔥 What's New

* 🚀 We introduce **GV-VAD**, a novel framework utilizing **text-conditioned video generation** to significantly augment weakly-supervised video anomaly detection datasets.
* 🎞️ Our approach generates semantically controllable and physically plausible synthetic videos, effectively mitigating the scarcity and high annotation costs associated with real-world anomalies.
* ⚙️ We employ a **synthetic sample loss scaling strategy** to manage the influence of synthetic samples during training, enhancing model efficiency and performance.
* 📈 GV-VAD achieves competable results on the widely-used **UCF-Crime** dataset.
* 📦 The code, pre-trained models, and processed data will be available for full reproducibility.

---

## 💎 Framework

![Framework](assets/framework_1.png)

GV-VAD comprises two integral components:

1.	Video Generator
A conditional diffusion model that generates realistic anomalous and normal videos from structured textual descriptions. These descriptions encompass key anomaly-related elements: camera viewpoint, location, subject, and anomalous event, enabling diverse synthetic video generation to address real-world data scarcity.

2.	Synthetic Sample Loss Scaling (SSLS)
A loss scaling strategy designed to balance the influence of synthetic and real video samples during training. SSLS prevents model overfitting to synthetic data, ensuring robust anomaly detection performance on real-world videos.

A fusion mechanism integrates generated synthetic videos with real video samples to enhance the model’s ability to detect and generalize across various anomaly scenarios effectively.



---

## 🏁 Quick Start

### 📁 Dataset Preparation

GV-VAD is evaluated on the **UCF-Crime** dataset.

Download the dataset from [UCF-Crime](https://URL_ADDRESS.crcv.ucf.edu/data/UCF_Crimes/).

Please extract video features using CLIP as guided in the provided script:

```bash
python tools/extract_features.py --dataset ucfcrime --feature_extractor clip
```

---

### 🛠️ Environment Setup

Set up the Python environment using `conda`:

```bash
conda create -n gvvad python=3.8
conda activate gvvad
pip install -r requirements.txt
```

---

## 🚀 Training

To train GV-VAD with VL Verifier:

```bash
python train.py --dataset ucfcrime --backbone clip --use_vl_verifier
```

To train without VL Verifier (baseline):

```bash
python train.py --dataset ucfcrime --backbone clip
```

---

## 🔍 Inference

Run inference with the trained model:

```bash
python infer.py --ckpt_path checkpoints/gvvad_ucf_best.pth --dataset ucfcrime
```

---

## 📊 Evaluation

Evaluate anomaly detection performance (AUC):

```bash
python eval.py --result_path results/ucfcrime_preds.npy --gt_path data/ucfcrime_labels.npy
```

---

<!-- ## 🎯 Performance

| Dataset   | AUC (%) |
| --------- | ------- |
| UCF-Crime | 79.8    |
| UCF-Crime | 79.8    | -->

<!-- Qualitative results:

![Qualitative Results](assets/qualitative_examples.png)

--- -->

## 💖 Acknowledgements

We sincerely thank the contributors of [LAP](https://arxiv.org/html/2403.01169) and prior works in the field of video anomaly detection for their invaluable resources and inspiration.

---

## 📚 Citation

If you find GV-VAD useful for your research, please cite our paper:

```bibtex
@article{cai2025gvvad,
  title={GV-VAD: Exploring Video Generation for Weakly-Supervised Video Anomaly Detection},
  author={Suhang Cai, Chong Wang, Xiaojie Cai, and Xiaohao Peng},
  journal={arXiv preprint arXiv:2403.01169},
  year={2025}
}
```
