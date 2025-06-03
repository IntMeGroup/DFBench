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


## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/IntMeGroup/LOVE.git
```

Create and activate a conda environment:

```bash
conda create -n LOVE python=3.9 -y
conda activate LOVE
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

---

## 🔧 Preparation

### 📁 Prepare dataset

```bash
huggingface-cli download anonymousdb/AIGVE-60K data.zip --repo-type dataset --local-dir ./
unzip data.zip -d ./data
```
### 📦 Prepare model weights

```bash
huggingface-cli download OpenGVLab/InternVL3-9B --local_dir OpenGVLab/InternVL3-9B
huggingface-cli download anonymousdb/LOVE-pretrain temporal.pth ./
```

---



## 🚀 Training


### 📈 Stage 1: Text-based quality training

```bash
sh shell/st1_train.sh
```

### 🎨 Stage 2: Fine-tune vision encoder and LLM with LoRA

```bash
sh shell/st2_train.sh
```

### ❓ Question-Answering (QA) Training

```bash
sh shell/train_qa.sh
```

---

## 🚀 Evaluation

### 📦 Download pretrained weights

```bash
huggingface-cli download anonymousdb/LOVE-Perception --local-dir ./weights/stage2/stage2_mos1
huggingface-cli download anonymousdb/LOVE-Correspondence --local-dir ./weights/stage2/stage2_mos2
huggingface-cli download anonymousdb/LOVE-QA --local-dir ./weights/qa
```

### 📈 Evaluate perception & correspondence scores

[![HF](https://img.shields.io/badge/%F0%9F%A4%97%20LOVE--Perception-orange)](https://huggingface.co/anonymousdb/LOVE-Perception)  
[![HF](https://img.shields.io/badge/%F0%9F%A4%97%20LOVE--Correspondence-green)](https://huggingface.co/anonymousdb/LOVE-Correspondence)

```bash
sh shell/eval_score.sh
```
