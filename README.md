# ⚡ LLM Quantization: Qwen 2.5 on CPU (INT8) & GPU (4-bit NF4)

A practical, hands-on exploration of model quantization techniques to optimize and run Large Language Models locally on consumer hardware. 

This repository demonstrates how to take **`Qwen/Qwen2.5-0.5B-Instruct`** and run it efficiently across two distinct quantization pipelines:
1. **CPU Quantization (INT8)** using `optimum-quanto` (low RAM footprint).
2. **GPU Quantization (4-bit NF4)** using `bitsandbytes` (accelerated inference on NVIDIA RTX 4050 GPU).

---

## 📌 Project Architecture & Features

- **Memory Estimator (`test.py`)**: A quick calculator script to estimate theoretical VRAM/RAM requirements across different precision levels (FP32, FP16, INT8, INT4) for various parameter scales.
- **CPU INT8 Quantization (`quantize_qwen.py` & `int8_qwen.py`)**:
  - Compresses FP32/FP16 weights to 8-bit integers using `optimum-quanto`.
  - Saves the quantized model shards locally (`./qwen-int8`) for offline deployment.
- **GPU 4-bit NormalFloat Quantization (`gpu_bnb_qwen.py`)**:
  - Uses `bitsandbytes` with `NF4` (NormalFloat4), double quantization, and `fp16` compute dtype.
  - Runs with minimal VRAM (~350 MB) on an NVIDIA RTX 4050 Laptop GPU.
  - Interactive multi-turn CLI chatbot interface with context memory.

---

## 📊 Precision & Memory Comparison (0.5B Parameter Model)

| Precision | Bits per Weight | Approx. Weight Size | Primary Target |
| :--- | :--- | :--- | :--- |
| **FP32** | 32-bit | ~1.86 GB | High-end servers / Training |
| **FP16 / BF16** | 16-bit | ~0.93 GB | Standard GPU Inference |
| **INT8 (Quanto)** | 8-bit | ~0.47 GB | Resource-constrained CPU |
| **4-bit NF4 (BNB)** | 4-bit | **~0.35 GB** | Consumer GPU (Fast token generation) |

---

## 🛠️ Tech Stack & Requirements

- **Python**: 3.12+ (managed with `uv`)
- **PyTorch**: CUDA 12.4 enabled (`torch >= 2.6.0+cu124`)
- **Hugging Face**: `transformers`, `tokenizers`, `accelerate`
- **Quantization Libraries**:
  - `optimum-quanto` (CPU INT8)
  - `bitsandbytes` (GPU 4-bit NF4)

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/Quantize.git
cd Quantize
```

### 2. Set up the Environment (using `uv`)
```bash
# Create and activate virtual environment
uv venv
.venv\Scripts\activate   # On Windows

# Install CUDA-enabled PyTorch
uv pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124

# Install project dependencies
uv pip install -r requirements.txt
uv pip install bitsandbytes optimum-quanto
```

---

## 💻 How to Run

### 1. Check Memory Calculation
```bash
python test.py
```

### 2. Quantize & Run on CPU (INT8)
```bash
# Step A: Download & Quantize to local folder (./qwen-int8)
python quantize_qwen.py

# Step B: Run inference using quantized weights
python int8_qwen.py
```

### 3. Interactive Chatbot on GPU (4-bit NF4)
Loads model into GPU VRAM using on-the-fly 4-bit NormalFloat quantization:
```bash
python gpu_bnb_qwen.py
```

---

## 💡 Key Learnings

- **Activation-aware & Distribution-aware quantization**: Why NF4 outperforms standard INT4 for LLMs by adapting to normal weight distributions.
- **Compute vs. Storage Precision**: Keeping weights in 4-bit while dequantizing to FP16 dynamically for fast tensor core computation.
- **Hardware Bottlenecks**: Overcoming CPU latency limitations by offloading quantized weights to dedicated NVIDIA CUDA cores.

