# KIDL

This repository contains the official code and datasets for  
**A Knowledge-Informed Deep Learning Paradigm for Generalizable and Stability-Optimized Car-Following Models**

**Authors:** Chengming Wang, Dongyao Jia, Wei Wang, Dong Ngoduy, Bei Peng, Jianping Wang

---

## 🚀 How to Run

### 1. Download the Dataset

- **LLM-generated sample data**:  
  Available at [Google Drive](https://drive.google.com/drive/folders/1xUn302jjgZXQ9XTrf04lJeSPtW8inwJq?usp=sharing)

- **Reconstructed NGSIM-I80 dataset**:  
  Publicly available upon request from the [MULTITUDE Project](http://www.multitude-project.eu/enhanced-ngsim.html)

### 2. Preprocess and Run

```bash
# Move the downloaded LLM sample data to the expected directory
mv <your_download_path> llm/models/dnn_sg/data/

# Move the downloaded NGSIM-I80 dataset to the expected directory
mv <your_download_path> data/ngsim/

# Extract car-following trajectories from the dataset
python preprocess/preprocess_ngsim_i80.py

# Run distillation training and evaluation
python main.py
```

### 3. Distill other LLMs

```bash
# Generate distillation samples using a specified LLM.
# Provide the appropriate API key and base URL to interface with your target LLM.
python llm/gen_samples.py
```