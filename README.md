# QUETT: Quadratic Edge Transformer for Brain Networks

A novel transformer-based architecture for brain network analysis and classification tasks. QUETT (Quadratic Edge Transformer) introduces innovative approaches to modeling brain connectivity patterns using quadratic edge interactions and advanced attention mechanisms.

##  Overview

QUETT is a state-of-the-art deep learning model specifically designed for brain network analysis. It combines the power of transformer architectures with specialized components for handling brain connectivity data, including quadratic edge modeling, ASPP (Atrous Spatial Pyramid Pooling) integration, and rank-based parameter optimization.

##  Project Structure

```
brain_Networks/
├── QUETT/                    # Main QUETT model implementation
│   ├── data/                 # Brain connectivity datasets
│   ├── model/                # QUETT model architecture
│   ├── result/               # Experimental results
│   ├── main.py              # Main training script
│   ├── train_test.py        # Training and testing functions
│   ├── data_utils.py        # Data loading utilities
│   ├── utils.py             # Helper functions
│   └── parse.py             # Argument parsing
├── analysis_scripts/         # QUETT analysis and visualization scripts
│   ├── create_edgequad_rank_accuracy_plot.py
│   ├── run_rank_accuracy_analysis.py
│   ├── detailed_quett_parameter_analysis.py
│   └── edgequad_aspp_ablation.py
└── results/                  # QUETT experimental results and plots
```

## 🚀 QUETT Model Architecture

### **QUETT (Quadratic Edge Transformer)**
- **Location**: `QUETT/`
- **Description**: A novel transformer-based architecture for brain network analysis
- **Key Features**: 
  - **Quadratic Edge Modeling**: Captures complex edge interactions in brain networks
  - **ASPP Integration**: Atrous Spatial Pyramid Pooling for multi-scale feature extraction
  - **Rank-based Optimization**: Efficient parameter reduction through low-rank approximations
  - **Attention Mechanisms**: Advanced self-attention for brain connectivity patterns
  - **Adaptive Pooling**: Flexible pooling strategies for different network topologies
- **Main Script**: `QUETT/main.py`

### **Core Components**:
1. **Edge Quadratic Module**: Models quadratic interactions between brain regions
2. **CNN Correlation Module**: Processes correlation matrices with convolutional layers
3. **Transformer Layers**: Multi-head attention for global connectivity patterns
4. **Rank Decomposition**: Efficient parameterization using low-rank matrices
5. **ASPP Module**: Multi-scale feature extraction with different dilation rates

## 📊 Datasets

The project supports multiple brain imaging datasets:

### Available Datasets:
- **ABIDE** (Autism Brain Imaging Data Exchange)
  - AAL116 parcellation
  - Harvard48 parcellation  
  - Schaefer100 parcellation
  - K-means100 parcellation
  - Ward100 parcellation

- **PPMI** (Parkinson's Progression Markers Initiative)
  - AAL116 parcellation
  - Schaefer100 parcellation

### Data Format:
- Brain connectivity matrices stored as `.npy` files
- Preprocessed correlation matrices
- Multiple parcellation schemes for different granularities

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.7+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd brain_Networks

# Install dependencies (create requirements.txt based on your environment)
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn scikit-learn
```

## 🏃‍♂️ Quick Start

### Running QUETT Model
```bash
cd QUETT
python main.py --dataset abide_AAL116 --rank 32 --epochs 100
```

### Basic Training Example
```bash
# Train QUETT on ABIDE dataset with AAL116 parcellation
python main.py \
    --dataset abide_AAL116 \
    --rank 32 \
    --layers 4 \
    --dropout 0.1 \
    --epochs 100 \
    --batch_size 16 \
    --base_lr 0.001
```

### Running Analysis Scripts
```bash
# Generate rank vs accuracy plots for QUETT
python create_edgequad_rank_accuracy_plot.py

# Run comprehensive rank analysis
python run_rank_accuracy_analysis.py

# Generate parameter analysis
python run_detailed_quett_analysis.py

# ASPP ablation study
python edgequad_aspp_ablation.py
```

## 📈 Analysis & Visualization

### Key QUETT Analysis Scripts:

1. **`create_edgequad_rank_accuracy_plot.py`**
   - Creates comprehensive rank vs accuracy plots for QUETT
   - Generates multi-panel visualizations (Accuracy, ROC-AUC, Sensitivity, Specificity)
   - Exports results to CSV for further analysis

2. **`run_rank_accuracy_analysis.py`**
   - Performs detailed rank-based performance analysis
   - Compares different QUETT configurations
   - Generates performance summaries and best model identification

3. **`detailed_quett_parameter_analysis.py`**
   - Analyzes hyperparameter sensitivity for QUETT
   - Performs comprehensive ablation studies
   - Generates parameter importance rankings

4. **`edgequad_aspp_ablation.py`**
   - ASPP (Atrous Spatial Pyramid Pooling) ablation studies
   - Evaluates different dilation rates [1, 2, 4]
   - Performance impact analysis of multi-scale features

### Generated Outputs:
- Performance plots (`.png` files)
- Results CSV files with detailed metrics
- Model comparison tables
- Parameter sensitivity analysis

## 📊 Results & Performance

### Key Metrics Evaluated:
- **Accuracy**: Classification accuracy percentage
- **ROC-AUC**: Area under the ROC curve
- **Sensitivity**: True positive rate
- **Specificity**: True negative rate


## 🔧 Configuration

### QUETT Model Parameters:
- **Rank**: Model complexity parameter (16, 32, 64, 128, 256) - controls low-rank decomposition
- **Layers**: Number of transformer layers (typically 2-6)
- **Dropout**: Regularization strength (0.1-0.3)
- **DropPath**: Stochastic depth regularization
- **Base Learning Rate**: Main learning rate for backbone (0.0001-0.001)
- **Adapter Learning Rate**: Learning rate for adapter modules (0.001-0.01)
- **Weight Decay**: L2 regularization strength
- **Batch Size**: Training batch size (8-32)
- **Epochs**: Number of training epochs (50-200)
- **CNN Base Channels**: Number of base channels in CNN correlation module
- **ASPP Rates**: Dilation rates for ASPP module [1, 2, 4]
- **Quadratic ASPP Rates**: Dilation rates for quadratic edge module
- **Cluster Number**: Number of clusters for pooling operations

### Dataset Configuration:
- **Train/Val/Test Split**: 70%/10%/20% by default
- **Stratified Sampling**: Maintains class balance
- **Data Augmentation**: Optional preprocessing steps

## 📝 Usage Examples

### Basic QUETT Training:
```bash
# QUETT model with custom parameters
python main.py \
    --dataset abide_AAL116 \
    --rank 32 \
    --layers 4 \
    --dropout 0.1 \
    --epochs 100 \
    --batch_size 16 \
    --base_lr 0.001 \
    --adapter_lr 0.01
```

### Advanced QUETT Configuration:
```bash
# QUETT with ASPP and quadratic edge modeling
python main.py \
    --dataset abide_AAL116 \
    --rank 64 \
    --layers 6 \
    --dropout 0.2 \
    --droppath 0.1 \
    --cnncorr_base_ch 32 \
    --cnncorr_aspp_rates [1,2,4] \
    --quadratic_aspp_rates [1,2,4] \
    --cluster_num 8 \
    --epochs 150
```

### Rank-based Analysis:
```bash
# Analyze performance across different ranks
python run_rank_accuracy_analysis.py
```

## 🗂️ File Organization

### Results Storage:
- **CSV Files**: Detailed performance metrics
- **PNG Files**: Visualization plots and charts
- **Log Files**: Training logs and experiment tracking
- **Model Checkpoints**: Saved model weights

### Archive Management:
- **`zip_all_7_models.sh`**: Script to create comprehensive archives
- **Timestamped Archives**: Automatic versioning of model collections

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for research purposes. Please cite appropriately if used in academic work.

## 📞 Contact

For questions or collaboration opportunities, please contact the research team.

## 🔬 Research Context

QUETT contributes to the field of computational neuroscience by:
- **Novel Architecture**: Introducing quadratic edge modeling for brain networks
- **Efficient Design**: Rank-based parameter reduction for scalable brain network analysis
- **Multi-scale Features**: ASPP integration for capturing different scales of connectivity
- **Comprehensive Evaluation**: Extensive analysis across multiple brain imaging datasets
- **Reproducible Framework**: Complete experimental setup with detailed analysis tools
- **Clinical Applications**: Potential for autism, ADHD, and Parkinson's disease classification

---

