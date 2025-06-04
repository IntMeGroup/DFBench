<div align="center">

<h1>💡 DFBench: Benchmarking Deepfake Image Detection Capability of Large Multimodal Models</h1>

</div>

<div align="center">

   <div>
      <!-- <a href="https://arxiv.org/abs/2506.03007"><img src="https://arxiv.org/abs/2506.03007"/></a> -->
      <a href="https://arxiv.org/abs/2506.03007"><img src="https://img.shields.io/badge/Arxiv-2506.03007-red"/></a>
<a href="https://huggingface.co/datasets/IntMeGroup/DFBench/tree/main">
   <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-green" alt="Hugging Face Dataset Badge"/>
</a>
</div>

</div>
<p align="center">
  <img width="1000" alt="teaser" src="https://github.com/user-attachments/assets/f0b63cd9-dda3-4437-84ad-a64c2109a806" />
</p>
<h3>If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil:</h3>

---
# 🤗 DFBench Database Download

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



---


# 💡 The MoA-DF Method



<p align="center">
  <img width="1000" alt="example" src="https://github.com/user-attachments/assets/7136b1b0-db17-44a0-bf21-9cf409825b16" />
</p>
Overview of the MoA-DF architecture. Three LMMs are chosen as core detectors. Each model independently produces log-probabilities corresponding to the likelihood of the input image belonging to A (real) or B (fake). These log-probabilities are converted into normalized probabilities via the softmax function. The final decision is made based on the aggregation of these probabilities across all models.
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

Install `ms-swift` (pre-built):
```bash
pip install ms-swift -U
```

Or compile from source:
```bash
# pip install git+https://github.com/modelscope/ms-swift.git
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e .
```


Alternatively if you are cuda12 you can use the packed env from
```
huggingface-cli download IntMeGroup/env swift.tar.gz --repo-type dataset --local-dir /home/user/anaconda3/envs
mkdir -p /home/user/anaconda3/envs/swift
tar -xzf swift.tar.gz -C /home/user/anaconda3/envs/swift
```




## 🔧 Preparation for Qwen2.5-VL
<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-VL/qwen2.5vl_logo.png" width="500"/>
<p>

### 📁 Prepare dataset

```bash
huggingface-cli download IntMeGroup/DFBench img_train_shuffled.json --repo-type dataset --local-dir ./qwen2.5/datasets
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
## 🌈 Evaluation & Real/Fake Prediction for Qwen2.5-VL

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
## 🔧 Preparation for InternVL2.5 & InternVL3
<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/930e6814-8a9f-43e1-a284-118a5732daa4">
  <br>
</div>

### 📁 Prepare dataset

```bash
huggingface-cli download IntMeGroup/DFBench img_train_shuffled.jsonl --repo-type dataset --local-dir ./internvl2.5/data
huggingface-cli download IntMeGroup/DFBench img_test.jsonl --repo-type dataset --local-dir ./internvl2.5/data
huggingface-cli download IntMeGroup/DFBench img_train_shuffled.jsonl --repo-type dataset --local-dir ./internvl3/data
huggingface-cli download IntMeGroup/DFBench img_test.jsonl --repo-type dataset --local-dir ./internvl3/data
```

### 📦 Prepare model weights

```bash
huggingface-cli download OpenGVLab/InternVL2_5-8B --local_dir ./internvl25/OpenGVLab/InternVL2_5-8B
huggingface-cli download OpenGVLab/InternVL3-9B --local_dir ./internvl3/OpenGVLab/InternVL3-9B
```
---
## 🚀 Training for InternVL2.5

```bash
cd internvl2.5
sh shell/train_deepfake.sh
```

## 🌈 Evaluation & Real/Fake Prediction for InternVL2.5

```bash
sh shell/eval_deepfake.sh
```

---
## 🔧 Preparation for InternVL3

## 🚀 Training for InternVL3

```bash
cd internvl3
sh shell/train_deepfake.sh
```

## 🌈 Evaluation & Real/Fake Prediction for InternVL3

```bash
sh shell/eval_deepfake.sh
```
---
# 📈 Calculate the final logit results and accuracy

```bash
python logit_calculation.py
python process_results.py
```
---
# Feature Distribution and Plot
<p align="center">
  <img width="1000" alt="feature" src="https://github.com/user-attachments/assets/20fee5f6-995d-48d7-a8bb-c7ac2563b88f" />
</p>
Feature distribution of the DFBench. (a) Feature distribution of real images with no distortion. (b) Feature distribution of real images with distortions. (c) Feature distribution of AI-edited images. (d) Feature distribution of AI-generated images.

```bash
python feature_distribution.py
python plot_features.py
```
# Zeo-Shot Model Comparison

<p align="center">
  <img width="1000" alt="example" src="https://github.com/user-attachments/assets/e9e42d42-ca0a-420c-a01d-3bacdb699a3b" />
</p>
(a) Performance comparison of image generation models (b) Performance comparison of image detection models
---

## 📌 TODO
- ✅ Release the training code 
- ✅ Release the evaluation code 
- ✅ Release the DFBench Database

## 📧 Contact
If you have any inquiries, please don't hesitate to reach out via email at `wangjiarui@sjtu.edu.cn`


## 🎓Citations

If you find our work useful, please cite our paper as:
```
@misc{wang2025dfbenchbenchmarkingdeepfakeimage,
      title={DFBench: Benchmarking Deepfake Image Detection Capability of Large Multimodal Models}, 
      author={Jiarui Wang and Huiyu Duan and Juntong Wang and Ziheng Jia and Woo Yi Yang and Xiaorong Zhu and Yu Zhao and Jiaying Qian and Yuke Xing and Guangtao Zhai and Xiongkuo Min},
      year={2025},
      eprint={2506.03007},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.03007}, 
}
```
