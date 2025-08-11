# VLM Feature Analysis Research Pipeline 🚁✈️

**A Comprehensive Framework for Vision-Language Model Feature Space Analysis in Open-Set Aircraft Recognition**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research](https://img.shields.io/badge/Research-Open%20Set%20Recognition-green.svg)](https://github.com)

---

## 🎯 Core Research Question Addressed

> *"Do features before and after embedding from VLM focusing only on vision part make a difference and lead to findings in open set scenarios?"*

This research pipeline provides a systematic evaluation framework to answer fundamental questions about Vision-Language Models (VLMs) feature representations, specifically comparing **pre-embedding** and **post-embedding** features in challenging **open-set recognition** scenarios using the FGVC Aircraft dataset.

---

## 📊 Key Features & Innovations

### 🔬 **Enhanced Open-Set Recognition**
- **GHOST Algorithm**: Implementation of Manuel Gunther's Gaussian Hypothesis Open-Set Technique
- **Advanced Metrics**: AUOSCR, OSCR, FPR95, per-class fairness analysis
- **Proper Protocol**: Training on known classes only, testing on all classes
- **Statistical Rigor**: Significance testing and confidence intervals

### 🎯 **Multi-Level Feature Analysis**
- **Pre-Embedding Features**: CLIP vision encoder output before final projection
- **Post-Embedding Features**: Final CLIP embeddings after projection/normalization
- **CNN Baseline**: ResNet features for traditional comparison
- **Comprehensive Metrics**: Separability, dimensionality, discriminability analysis

### 💬 **Text Prompt Distance Analysis**
- **Novel Approach**: Analyze where unknown classes get classified relative to text prompts
- **No Text During Testing**: Pure visual evaluation with text prompt reference
- **Multiple Templates**: Simple, descriptive, technical, and contextual prompts
- **Distance Mapping**: Cosine similarity analysis between visual features and text embeddings

### ⚡ **Advanced Fine-Tuning**
- **LoRA Integration**: Low-rank adaptation for efficient embedding fine-tuning
- **Parameter Efficiency**: Minimal parameters for maximum improvement
- **Adaptation Analysis**: Quantify embedding space changes and preservation

---

## 🔬 Research Pipeline Structure

The framework systematically addresses **5 core research questions**:

| Research Question | Focus Area | Key Metrics | Analysis Type |
|-------------------|------------|-------------|---------------|
| **RQ1** | Feature space characterization and differences | Intrinsic dimensionality, separability ratios, UMAP quality | Mathematical Analysis |
| **RQ2** | Classifier performance across feature spaces | Accuracy, F1-score across feature-classifier pairs | Comparative Evaluation |
| **RQ3** | Open-set vs closed-set analysis | AUROC, AUOSCR, FPR95, GHOST performance | Open-Set Recognition |
| **RQ4** | Text prompt strategy evaluation | Zero-shot accuracy, prompt effectiveness, distance analysis | Multimodal Analysis |
| **RQ5** | LoRA fine-tuning impact | Adaptation quality, parameter efficiency, embedding preservation | Fine-Tuning Analysis |

---

## 🚀 Quick Start

### **1. Installation**

```bash
# Clone/download the repository
git clone <repository-url>
cd vlm-research-pipeline

# Create conda environment (recommended)
conda create -n vlm_research python=3.9 -y
conda activate vlm_research

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install transformers scikit-learn pandas numpy matplotlib seaborn umap-learn scipy Pillow tqdm pyyaml jupyter

# Verify installation
python verify_installation.py
```

### **2. Quick Test (2 minutes)**

```bash
# Create default configuration
python vlm_research_pipeline.py --create_config

# Get dataset information
python vlm_research_pipeline.py --info

# Run quick test with 100 samples
python vlm_research_pipeline.py --config config.yaml --max_samples 100
```

### **3. Full Research Pipeline**

```bash
# Manufacturer-level analysis (easier, ~30 classes)
python vlm_research_pipeline.py --config config.yaml --annotation_level manufacturer

# Family-level analysis (moderate, ~70 classes)
python vlm_research_pipeline.py --config config.yaml --annotation_level family

# Variant-level analysis (challenging, ~100 classes)
python vlm_research_pipeline.py --config config.yaml --annotation_level variant
```

---

## 📁 Project Structure

```
vlm-research-pipeline/
├── 📄 vlm_research_pipeline.py      # Main research pipeline CLI
├── 📄 enhanced_vlm_pipeline.py      # Core evaluation framework  
├── 📄 feature_extractors.py         # CLIP and ResNet extractors
├── 📄 fgvc_usage_example.py         # Usage examples and utilities
├── ⚙️ config.yaml                   # Configuration template
├── 📊 requirements.txt              # Dependencies
├── 🧪 verify_installation.py        # Installation checker
├── 📖 README.md                     # This file
├── 📖 QUICKSTART.md                 # Quick start guide
└── 📁 results/                      # Output directory
    ├── 📁 features/                 # Cached feature extractions
    ├── 📁 models/                   # Trained LoRA adapters
    ├── 📁 visualizations/           # Plots and figures
    └── 📁 reports/                  # Research reports
```

---

## 💻 Usage Examples

### **Dataset Information and Setup**

```bash
# Show comprehensive dataset statistics
python vlm_research_pipeline.py --info --annotation_level manufacturer

# Compare different granularity levels (manufacturer vs family vs variant)
python vlm_research_pipeline.py --compare_levels

# Get suggestions for good known classes for open-set evaluation
python vlm_research_pipeline.py --suggest_classes

# Download FGVC Aircraft dataset
python vlm_research_pipeline.py --download --data_dir ./data/fgvc-aircraft
```

### **Research Evaluations**

```bash
# Quick test for setup verification (100 samples, ~5 minutes)
python vlm_research_pipeline.py --config config.yaml --max_samples 100

# Standard research pipeline with different granularity levels
python vlm_research_pipeline.py --config config.yaml --annotation_level manufacturer  # Easy
python vlm_research_pipeline.py --config config.yaml --annotation_level family       # Moderate  
python vlm_research_pipeline.py --config config.yaml --annotation_level variant      # Challenging

# Custom known classes for open-set evaluation
python vlm_research_pipeline.py --config config.yaml --known_classes Boeing Airbus Cessna

# Custom output directory
python vlm_research_pipeline.py --config config.yaml --output_dir ./results/my_experiment
```

### **Using the Examples Script**

```bash
# Show all available examples
python fgvc_usage_example.py --action examples

# Run specific examples
python fgvc_usage_example.py --action info          # Dataset information
python fgvc_usage_example.py --action quick_test    # Quick test run
python fgvc_usage_example.py --action manufacturer  # Manufacturer-level analysis
python fgvc_usage_example.py --action custom        # Custom configuration example
```

---

## 📈 Expected Outputs

### **📊 Comprehensive Reports**
- **Research Report** (`enhanced_vlm_evaluation_report.md`): Detailed analysis answering all research questions
- **Executive Summary**: Key findings and actionable insights
- **Methodology Documentation**: Reproducible experimental setup

### **📈 Advanced Visualizations**
- **Open-Set Performance Heatmap**: Comprehensive method comparison
- **GHOST vs Traditional Comparison**: Algorithm effectiveness analysis
- **Text Prompt Distance Analysis**: Novel multimodal analysis
- **Feature Space UMAP Plots**: High-quality dimensionality reduction
- **Per-Class Fairness Analysis**: Equity assessment across classes

### **💾 Quantitative Results**
- **JSON Results** (`enhanced_vlm_results.json`): All metrics and experimental data
- **Feature Caches**: Reusable extracted features for faster experimentation
- **Model Checkpoints**: Trained LoRA adapters and classifiers
- **Configuration Logs**: Reproducible experimental settings

---

## 🔍 Key Research Insights Provided

### **🎯 Feature Space Effectiveness**
- **Pre vs Post Embedding**: Quantified differences in CLIP feature spaces with mathematical rigor
- **CNN vs Transformer**: ResNet baseline comparison with vision transformers
- **Optimal Combinations**: Data-driven identification of best feature-classifier pairings
- **Dimensionality Analysis**: Intrinsic dimensionality and manifold structure

### **🤖 Advanced Open-Set Recognition**
- **GHOST Algorithm**: State-of-the-art Gaussian hypothesis approach
- **Multiple Metrics**: AUOSCR, OSCR, FPR95 for comprehensive evaluation
- **Per-Class Fairness**: Equity analysis across different aircraft manufacturers
- **Statistical Validation**: Significance testing and confidence intervals

