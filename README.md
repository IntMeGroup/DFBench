<div align="center">

<h1>💡 DFBench: Benchmarking Deepfake Image Detection Capability of Large Multimodal Models</h1>

</div>

<div align="center">


<a href="https://huggingface.co/datasets/IntMeGroup/DFBench/tree/main">
   <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-green" alt="Hugging Face Dataset Badge"/>
</a>

</div>
<p align="center">
  <img width="1000" alt="teaser" src="https://github.com/user-attachments/assets/f0b63cd9-dda3-4437-84ad-a64c2109a806" />
</p>

---
## 🤗 DFBench Database Download

[![🤗 Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-green)](https://huggingface.co/datasets/IntMeGroup/DFBench/tree/main)

*Linux*
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

*Windows Powershell*
```bash
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

Download with huggingface-cli:
```bash
huggingface-cli download IntMeGroup/DFBench --repo-type dataset --local-dir ./DFBench
```

<p align="center">
  <img width="1000" alt="example" src="https://github.com/user-attachments/assets/5f02efc7-ab54-458d-8c7a-45c772fe5c26" />
</p>

# Zeo-Shot Model Comparison

<p align="center">
  <img width="1000" alt="example" src="https://github.com/user-attachments/assets/e9e42d42-ca0a-420c-a01d-3bacdb699a3b" />
</p>

---


# 💡 The MoA-DF Method



<p align="center">
  <img width="1000" alt="example" src="https://github.com/user-attachments/assets/7136b1b0-db17-44a0-bf21-9cf409825b16" />
</p>

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/IntMeGroup/DFBench.git
```

Create and activate a conda environment:

```bash
conda create -n DFBench python=3.9 -y
conda activate DFBench
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install `flash-attn==2.3.6` (pre-built):

```bash
pip install flash-attn==2.3.6 --no-build-isolation
```

Or compile from source:

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.3.6
python setup.py install
```





## 🔧 Preparation for Qwen2.5-VL
<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-VL/qwen2.5vl_logo.png" width="400"/>
<p>

### 📁 Prepare dataset

```bash
huggingface-cli download IntMeGroup/DFBench img_train.json --repo-type dataset --local-dir ./qwen2.5/datasets
huggingface-cli download IntMeGroup/DFBench img_test.json --repo-type dataset --local-dir ./qwen2.5/datasets
```

### 📦 Prepare model weights

```bash
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local_dir ./Qwen/Qwen2.5-VL-7B-Instruct
```

---



## 🚀 Training for Qwen2.5-VL

```bash
cd qwen2.5
sh train.sh
```

---

## 🚀 Evaluation & Real/Fake Prediction for Qwen2.5-VL

### 📦 Merge LoRA weights

change merge_lora.sh line3 --adapters ./output_ckpt/your_weights

```bash
sh merge_lora.sh 
```

### 📈 Evaluate & Real/Fake Prediction (with logit probabilities)

```bash
python evaluate_logit.py --model_path ./output_ckpt/your_weights_merged
```

---
## 🔧 Preparation for InternVL2.5

### 📁 Prepare dataset

```bash
huggingface-cli download IntMeGroup/DFBench img_train.json --repo-type dataset --local-dir ./qwen2.5/datasets
huggingface-cli download IntMeGroup/DFBench img_test.json --repo-type dataset --local-dir ./qwen2.5/datasets
```

### 📦 Prepare model weights

```bash
huggingface-cli download OpenGVLab/InternVL2_5-8B --local_dir ./OpenGVLab/InternVL2_5-8B
```

## 🚀 Training for InternVL2.5

```bash
cd internvl2.5
sh shell/train_deepfake.sh
```

## 🚀 Evaluation & Real/Fake Prediction for InternVL2.5

```bash
sh shell/eval_deepfake.sh
```

---
## 🔧 Preparation for InternVL3

### 📁 Prepare dataset

```bash
huggingface-cli download IntMeGroup/DFBench img_train.json --repo-type dataset --local-dir ./qwen2.5/datasets
huggingface-cli download IntMeGroup/DFBench img_test.json --repo-type dataset --local-dir ./qwen2.5/datasets
```

### 📦 Prepare model weights

```bash
huggingface-cli download OpenGVLab/InternVL3-9B --local_dir ./OpenGVLab/InternVL3-9B
```

## 🚀 Training for InternVL3

```bash
cd internvl3
sh shell/train_deepfake.sh
```

## 🚀 Evaluation & Real/Fake Prediction for InternVL3

```bash
sh shell/eval_deepfake.sh
```

