# Enhanced VLM Open-Set Recognition Framework

A comprehensive framework for evaluating Vision-Language Models (VLMs) in open-set recognition scenarios, with advanced metrics, multi-classifier evaluation, and LoRA fine-tuning support.

## 🚀 Overview

This framework addresses the critical challenge of open-set recognition in computer vision, where models must not only classify known categories accurately but also detect and handle unknown categories effectively. We provide a complete evaluation pipeline comparing VLMs (like CLIP) with traditional CNNs (like ResNet) across multiple dimensions.

### Key Features

- **OSCR (Open Set Classification Rate)** metric implementation
- **Comprehensive feature space analysis** at multiple network levels
- **Multi-classifier evaluation** (KNN, SVM, MLP, Random Forest)
- **Advanced open-set metrics** (Wilderness Risk, Open Space Risk)
- **Text prompt optimization** with ensemble strategies
- **LoRA fine-tuning** for domain adaptation
- **Rich visualizations** and detailed reports

## 📊 Research Questions Addressed

1. **Enhanced Feature Space Analysis**: How do VLM embeddings differ from ResNet features in terms of known/unknown discriminability?
2. **Comprehensive Classifier Performance**: Which classifiers work best with VLM embeddings for open-set recognition?
3. **Advanced Open-Set Discriminability**: How well can VLMs detect out-of-distribution samples using OSCR metrics?
4. **Text Prompt Strategy Optimization**: What prompt engineering strategies maximize open-set performance?
5. **LoRA Fine-tuning**: Can domain-specific fine-tuning improve open-set recognition while preserving zero-shot capabilities?

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/vlm-openset-recognition.git
cd vlm-openset-recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Packages

```txt
torch>=1.9.0
torchvision>=0.10.0
transformers>=4.20.0
numpy>=1.19.0
pandas>=1.3.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
scipy>=1.7.0
umap-learn>=0.5.0
tqdm>=4.60.0
Pillow>=8.2.0
peft>=0.5.0  # For LoRA support
```

## 📁 Project Structure

```
vlm-openset-recognition/
├── enhanced_metrics.py          # OSCR and open-set metrics
├── feature_analysis.py          # Feature space analysis tools
├── prompt_optimization.py       # Text prompt strategies
├── lora_finetuning.py          # LoRA fine-tuning module
├── aircraft_protocol.py         # Base VLM protocol
├── enhanced_aircraft_analysis.py # Main analysis script
├── run_analysis.py             # Quick start script
├── requirements.txt            # Dependencies
├── README.md                   # This file
└── data/                       # Dataset directory
    └── fgvc-aircraft/
        ├── train.csv
        ├── val.csv
        ├── test.csv
        └── images/
```

## 🏃 Quick Start

### Basic Analysis
```bash
python enhanced_aircraft_analysis.py --data_dir ./data/fgvc-aircraft
```

### Full Analysis with All Features
```bash
python enhanced_aircraft_analysis.py \
    --data_dir ./data/fgvc-aircraft \
    --compare_resnet \
    --resnet_model resnet50 \
    --enable_lora \
    --output_dir results \
    --timestamp
```

### Using the Quick Start Script
```bash
python run_analysis.py
```

## 🎯 Usage Examples

### 1. VLM-Only Analysis
```bash
python enhanced_aircraft_analysis.py \
    --data_dir ./data/fgvc-aircraft \
    --clip_model openai/clip-vit-base-patch32 \
    --output_dir vlm_results
```

### 2. VLM vs ResNet Comparison
```bash
python enhanced_aircraft_analysis.py \
    --data_dir ./data/fgvc-aircraft \
    --compare_resnet \
    --resnet_model resnet50 \
    --output_dir comparison_results
```

### 3. With LoRA Fine-tuning
```bash
python enhanced_aircraft_analysis.py \
    --data_dir ./data/fgvc-aircraft \
    --enable_lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_epochs 10 \
    --output_dir lora_results
```

### 4. Custom Prompt Strategies
```bash
python enhanced_aircraft_analysis.py \
    --data_dir ./data/fgvc-aircraft \
    --prompt_strategies basic detailed contextual \
    --optimize_prompts \
    --output_dir prompt_results
```

## 📊 Metrics and Evaluation

### Open-Set Metrics

1. **OSCR (Open Set Classification Rate)**
   - Measures the trade-off between correct classification on known classes and false positive rate on unknown samples
   - Higher OSCR AUC indicates better open-set performance

2. **AUROC**
   - Standard ROC curve for binary known/unknown detection
   - Complementary to OSCR for overall detection performance

3. **Wilderness Risk**
   - Risk of misclassifying unknown samples as known
   - Lower values indicate better safety in deployment

4. **Open Space Risk**
   - Measures how much of the feature space is incorrectly covered by known class boundaries
   - Critical for understanding model overconfidence

### Feature Space Metrics

- **Discriminability Ratio**: Inter-class distance / Intra-class distance
- **Silhouette Score**: Clustering quality measure
- **Text-Image Alignment**: Cosine similarity between paired embeddings

## 📈 Outputs

After running the analysis, you'll find:

```
results_[timestamp]/
├── visualizations/
│   ├── data_distribution.png         # Known/unknown sample distribution
│   ├── embedding_umap.png           # UMAP visualization of embeddings
│   └── feature_evolution.png        # Features across network layers
├── feature_analysis/
│   ├── embedding_comparison.png     # VLM vs ResNet comparison
│   ├── text_image_alignment.png     # Alignment visualization
│   └── discriminability_stats.csv   # Numerical metrics
├── classifier_comparison/
│   ├── classifier_evaluation.png    # Performance comparison
│   ├── confidence_distributions.png # Score distributions
│   └── classifier_results.csv       # Detailed results
├── open_set_analysis/
│   ├── VLM_oscr_curve.png          # OSCR curves
│   ├── threshold_analysis.png       # Optimal threshold selection
│   ├── model_comparison.png         # Side-by-side comparison
│   └── oscr_metrics.csv            # Numerical OSCR values
├── prompt_optimization/
│   ├── prompt_analysis.png          # Strategy comparison
│   ├── optimal_weights.png          # Ensemble weights
│   └── prompt_results.csv           # Performance by strategy
├── lora_results/                    # If LoRA enabled
│   ├── training_curves.png          # Fine-tuning progress
│   ├── before_after_comparison.png  # Performance improvement
│   └── lora_checkpoint/             # Saved LoRA weights
└── reports/
    └── comprehensive_analysis.md    # Full markdown report
```

## 🔧 Configuration Options

### Model Options
- `--clip_model`: CLIP model variant (default: `openai/clip-vit-base-patch32`)
- `--resnet_model`: ResNet variant for comparison (`resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`)

### Analysis Options
- `--compare_resnet`: Enable ResNet comparison
- `--enable_lora`: Enable LoRA fine-tuning analysis
- `--optimize_prompts`: Run prompt optimization
- `--prompt_strategies`: List of strategies to evaluate

### LoRA Parameters
- `--lora_rank`: LoRA rank (default: 16)
- `--lora_alpha`: LoRA alpha scaling (default: 32)
- `--lora_dropout`: LoRA dropout (default: 0.1)
- `--lora_epochs`: Fine-tuning epochs (default: 10)
- `--lora_lr`: Learning rate (default: 1e-4)

### Execution Options
- `--batch_size`: Batch size for processing (default: 32)
- `--seed`: Random seed (default: 42)
- `--max_samples`: Limit samples for testing
- `--num_workers`: Data loader workers (default: 4)

## 🎓 Understanding the Results

### Key Findings to Look For

1. **OSCR Performance**
   - VLMs typically achieve higher OSCR AUC (>0.85) compared to ResNet (<0.75)
   - This indicates better known/unknown separation

2. **Feature Discriminability**
   - VLM discriminability ratio should be >2.0 for good performance
   - Higher ratios indicate better class separation

3. **Prompt Optimization**
   - Contextual prompts often provide 10-15% improvement
   - Ensemble strategies reduce variance

4. **LoRA Fine-tuning**
   - Typically improves known class accuracy by 5-10%
   - Should maintain or improve unknown detection

### Interpreting Visualizations

- **UMAP Plots**: Look for clear separation between known (clustered) and unknown (scattered) samples
- **OSCR Curves**: Higher curves indicate better performance; compare area under curves
- **Threshold Analysis**: Optimal threshold balances known accuracy and unknown rejection

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone with submodules
git clone --recursive https://github.com/yourusername/vlm-openset-recognition.git

# Install in development mode
pip install -e .

# Run tests
pytest tests/
```

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@software{vlm_openset_2024,
  title={Enhanced VLM Open-Set Recognition Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/vlm-openset-recognition}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- FGVC-Aircraft dataset creators
- OpenAI for CLIP models
- Hugging Face for transformers library
- PyTorch team for the deep learning framework

## 📞 Contact

- **Issues**: Please use [GitHub Issues](https://github.com/yourusername/vlm-openset-recognition/issues)
- **Discussions**: Join our [Discord Server](https://discord.gg/yourinvite)
- **Email**: your.email@example.com

## 🗺️ Roadmap

- [ ] Multi-modal fusion support
- [ ] Additional VLM architectures (ALIGN, FLAVA)
- [ ] Real-time inference optimization
- [ ] Web-based visualization dashboard
- [ ] Extended dataset support

---

**Note**: This framework is actively maintained. Check back for updates and new features!