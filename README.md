# 🔬 MedLens AI - Explainable Medical Image Classifier

> Detect pneumonia from chest X-rays using deep learning with visual explanations via Grad-CAM

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Overview

MedLens AI is an explainable deep learning system that detects pneumonia from chest X-rays with **94% accuracy**. Unlike traditional "black-box" AI models, it uses Grad-CAM (Gradient-weighted Class Activation Mapping) to show exactly which regions of the lung influenced its decision, building trust and enabling clinical validation.

### Key Features

- ✅ **94.3% Validation Accuracy** on chest X-ray classification
- ✅ **Grad-CAM Visualization** showing model reasoning
- ✅ **Real-time Inference** (147ms on GPU)
- ✅ **Interactive Web Interface** via Gradio
- ✅ **Transfer Learning** from ImageNet for efficiency

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/John-Tenkorang-Anim/medlens-ai.git
cd medlens-ai

# Create virtual environment
python -m venv medlens_env
source medlens_env/bin/activate  # On Windows: medlens_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt