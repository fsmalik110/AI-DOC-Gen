# 🤖 AI Documentation Generator

<div align="center">

![Demo](UI%20Example%201.png)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-000000.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Automatically generate documentation for Python functions using Deep Learning**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Training](#-training)

</div>

---

## ✨ Features

- 🧠 **Deep Learning Model**: 3-layer Bidirectional LSTM with **15,890,283 parameters**
- 🎨 **Beautiful Web UI**: Modern, responsive Flask-based interface
- ⚡ **Real-time Generation**: Instant documentation generation (< 1 second)
- 📊 **High Accuracy**: 0.0008 validation loss after 15 epochs
- 🎯 **Easy to Use**: Simple one-click examples and intuitive interface
- 💻 **CPU Compatible**: Works on any PC, no GPU required

---

## 🎬 Demo

### 🌐 Web Interface

<div align="center">

![UI Example 1](UI%20Example%201.png)
*Main interface with code input and documentation output*

![UI Example 2](UI%20Example%202.png)
*Real-time generation with multiple examples*

![UI Example 3](UI%20Example%203.png)
*Clean, professional UI design*

</div>

### 📊 Training Process & Results

<div align="center">

![Training Process](Training%20Process.png)
*Model training with real-time metrics showing 15 epochs*

![Training Metrics](Training%20Metrics.png)
*Training progress with loss curves and validation metrics*

</div>

### 🎯 Example Outputs

<div align="center">

![Examples](Examples.png)
*Testing interface with pre-defined examples*

![Results](Results.png)
*Generated documentation results showing model accuracy*

</div>

### 📋 Sample Generations

| Input Function | Generated Documentation |
|---------------|------------------------|
| `def add(a, b): return a + b` | ✅ Add two numbers and return their sum |
| `def is_even(n): return n % 2 == 0` | ✅ Check if a number is even |
| `def reverse(s): return s[::-1]` | ✅ Reverse a string |
| `def find_max(a, b): return a if a > b else b` | ✅ Find maximum of two numbers |
| `def multiply(x, y): return x * y` | ✅ Multiply two numbers and return product |

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/fsmalik110/AI-DOC-Gen.git
cd AI-DOC-Gen

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the web interface
python web_ui.py
Open your browser: http://localhost:5000

---

## 💻 Usage
Option 1: Web Interface (Recommended)

python web_ui.py

Then navigate to http://localhost:5000 and:

-Enter your Python function in the input box
-Click "Generate Documentation"
-Get instant AI-generated documentation!

Option 2: Command Line Testing

python test_model_enhanced.py

Interactive mode with pre-loaded examples.

Option 3: Python API

import torch
import pickle

# Load model
checkpoint = torch.load('models/best_model.pth', map_location='cpu')

# Your code here
code = "def add(a, b): return a + b"

# Generate documentation
# (Full implementation in test_model.py)

---

## 📂 Project Structure
AI-DOC-Gen/
│
├── 📁 data/                       # Dataset files
│   ├── train_data.pkl            # Training dataset (10,000 samples)
│   └── val_data.pkl              # Validation dataset (1,000 samples)
│
├── 📁 models/                     # Trained model files
│   ├── best_model.pth            # Best model checkpoint (186 KB)
│   └── training_history.pkl      # Training metrics history
│
├── 📁 outputs/                    # Generated outputs
│
├── 📸 Screenshots/
│   ├── UI Example 1.png
│   ├── UI Example 2.png
│   ├── UI Example 3.png
│   ├── Training Process.png
│   ├── Training Metrics.png
│   ├── Results.png
│   ├── Examples.png
│   └── Project Structure.png
│
├── 📄 main.py                     # Simple training script
├── 📄 train_improved_CORRECT.py   # Advanced training pipeline
├── 📄 test_model.py               # Basic model testing
├── 📄 test_model_enhanced.py      # Interactive testing interface
├── 📄 web_ui.py                   # Flask web application
├── 📄 visualize_training.py       # Training visualization
├── 📄 create_sample_dataset.py    # Dataset creation
├── 📄 download_dataset.py         # CodeSearchNet downloader
├── 📄 requirements.txt            # Python dependencies
├── 📄 README.md                   # This file
└── 📄 .gitignore                  # Git ignore rules

---

## 🏗️ Model Architecture
## Network Design

Input (Python Code)
    ↓
Tokenization & Embedding (256 dim)
    ↓
3× Bidirectional LSTM (512 hidden units each)
    ↓
Dropout Layer (0.3)
    ↓
Fully Connected Layer
    ↓
Softmax Output
    ↓
Documentation Text

---

### Technical Specifications

| Component | Configuration |
|-----------|--------------|
| **Architecture** | 3-layer Bidirectional LSTM |
| **Embedding Dimension** | 256 |
| **Hidden Units** | 512 per layer |
| **Total Parameters** | 15,890,283 |
| **Dropout Rate** | 0.3 |
| **Vocabulary Size** | 10,000 tokens |
| **Max Sequence Length** | 100 tokens |
| **Optimizer** | Adam (lr=0.001) |
| **Loss Function** | Cross Entropy Loss |
| **Batch Size** | 32 |

---
## 🎓 Training Process

### Dataset

**Source:** CodeSearchNet (Python subset)
- **Training Samples:** 10,000 Python functions with documentation
- **Validation Samples:** 1,000 Python functions
- **Vocabulary:** 10,000 most common tokens

### Training Configuration

```python
Epochs: 15
Batch Size: 32
Learning Rate: 0.001 (with ReduceLROnPlateau)
Early Stopping: Patience = 5 epochs
Optimizer: Adam
Loss Function: Cross Entropy
Device: CPU/CUDA (auto-detect)

---
### Training Results

**Final Metrics:**
- ✅ **Training Loss:** 0.0001
- ✅ **Validation Loss:** 0.0008
- ✅ **Learning Rate:** 1e-06 (after scheduling)
- ✅ **Epochs Completed:** 15/15
- ✅ **Model Size:** 186 KB

![Training Curves](Training%20Metrics.png)
*Training and validation loss curves over 15 epochs*

---

## 🔬 Train Your Own Model

### Step 1: Prepare Dataset

 Option A: Download full CodeSearchNet dataset (large, ~2GB)
python download_dataset.py

 Option B: Create sample dataset (quick, for testing)
python create_sample_dataset.py

### Step 2: Train Model

### Train with full pipeline
python train_improved_CORRECT.py

## Training will:

- ✅ Load and process dataset
- ✅ Build vocabulary from code/docs
- ✅ Create data loaders
- ✅ Train BiLSTM model
- ✅ Save best model checkpoint
- ✅ Generate training history

### Expected Time:

CPU: ~30-60 minutes
GPU: ~10-20 minutes

## Step 3: Visualize Results

python visualize_training.py

