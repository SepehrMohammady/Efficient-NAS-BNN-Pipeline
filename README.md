# NAS-BNN Multi-Dataset Pipeline

🔬 **Neural Architecture Search for Binary Neural Networks with Multi-Dataset Support**

This repository provides a complete implementation of the NAS-BNN framework with support for multiple datasets and comprehensive pipeline automation. The project demonstrates the evolution from ImageNet → CIFAR-10 → WakeVision, showcasing the adaptability of neural architecture search for different domains.

## 🎯 **Current Focus: WakeVision Person Detection**
- **Dataset:** WakeVision (Harvard-Edge/Wake-Vision)
- **Task:** Binary classification (person detection)
- **Architecture:** `superbnn_wakevision_large` (128×128 input)
- **Pipeline:** Complete NAS-BNN workflow with visualization and analysis

## 📈 **Dataset Evolution Timeline**
1. **ImageNet** (Original) → 2. **CIFAR-10** (Adaptation) → 3. **WakeVision** (Current Focus)

## ✨ **Key Features**

### 🔄 **Multi-Dataset Support**
- **WakeVision**: Person detection (current focus)
- **CIFAR-10**: 10-class image classification (legacy support)
- **ImageNet**: Original framework support (maintained compatibility)

### 🚀 **Enhanced Pipeline**
- **Resume Capability**: Automatic checkpoint detection and resumption
- **Interactive Jupyter Notebook**: Step-by-step execution with analysis
- **PowerShell Automation**: Complete pipeline automation for Windows
- **Comprehensive Logging**: Detailed logs with accuracy parsing
- **ONNX Export**: Model export for visualization with Netron

### 📊 **Analysis & Visualization**
- Pareto front visualization
- Architecture performance comparison
- Fine-tuning improvement analysis
- Search result inspection tools

## 🔬 **Supported Architectures**
- `superbnn_wakevision_large` - Optimized for WakeVision (128×128)
- `superbnn_cifar10_large` - Enhanced CIFAR-10 model
- `superbnn_cifar10` - Standard CIFAR-10 model
- Legacy ImageNet architectures (`superbnn`, `superbnn_100`)

## Modifications and Additions
- Adapted model configuration (`superbnn_wakevision_large`) for WakeVision dataset (128x128 images, 2 classes - person detection).
- Created `prepare_local_wake_vision_from_csv.py` and `prepare_wakevision.py` for WakeVision dataset preparation.
- Support for multiple dataset preparation methods (local CSV files and HuggingFace datasets).
- Implemented resume logic and enhanced logging in `search.py`.
- Developed orchestrator scripts to run the full pipeline:
    - `run_all.ps1` (PowerShell for Windows) - **Note: Currently configured for CIFAR-10, needs update**
    - `run_all.ipynb` (Jupyter Notebook for interactive execution and analysis) - **Updated for WakeVision**
- Diagnostic scripts: `check_ops.py`, `check_cuda.py`.
- Addressed Windows-specific execution issues (e.g., DataLoader workers).
- **Legacy Support:** Previous CIFAR-10 model configurations remain available (`superbnn_cifar10`, `superbnn_cifar10_large`).

## 🚀 **Quick Start**

### **Option 1: Interactive Notebook (Recommended)**
```bash
jupyter lab run_all.ipynb
```

### **Option 2: PowerShell Automation**
```powershell
.\run_all.ps1
```

### **Option 3: Manual Steps**
```bash
# 1. Prepare data
python prepare_local_wake_vision_from_csv.py

# 2. Check architecture OPs range
python check_ops.py -a superbnn_wakevision_large --img-size 128

# 3. Train supernet
python train.py --dataset WakeVision -a superbnn_wakevision_large

# 4. Search architectures
python search.py --dataset WakeVision -a superbnn_wakevision_large

# 5. Test and fine-tune selected architectures
python test.py --dataset WakeVision -a superbnn_wakevision_large --ops 5
python train_single.py --dataset WakeVision -a superbnn_wakevision_large --ops 5
```

## 🛠 **Installation**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SepehrMohammady/Efficient-NAS-BNN-Pipeline.git
   cd Efficient-NAS-BNN-Pipeline
   ```

2. **Create Python environment:**
   ```bash
   python -m venv nasbnn
   # Windows
   .\nasbnn\Scripts\activate
   # Linux/macOS
   source nasbnn/bin/activate
   ```

3. **Install PyTorch with CUDA:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📁 **Project Structure**

```
├── 📊 run_all.ipynb              # Interactive pipeline notebook
├── 🔧 run_all.ps1                # PowerShell automation script
├── 📝 prepare_local_wake_vision_from_csv.py  # WakeVision data preparation
├── 🔍 check_ops.py               # Architecture complexity analysis
├── 🏋️ train.py                   # Supernet training
├── 🔬 search.py                  # Architecture search
├── 🧪 test.py                    # Architecture testing
├── 🎯 train_single.py            # Fine-tuning
├── 📊 eval_finetuned.py          # Model evaluation
├── 🔧 models/                    # Model architectures
│   ├── superbnn.py              # Main SuperBNN implementations
│   ├── dynamic_operations.py    # Dynamic operations
│   └── operations.py            # Basic operations
├── 🛠 utils/                     # Utility functions
└── 📁 WakeVision/               # Dataset directory
```

## 📈 **Results & Performance**

### **WakeVision Person Detection**
- ✅ Successfully adapted NAS-BNN for binary classification
- 🎯 Achieved competitive accuracy on person detection task
- ⚡ Optimized architectures with 3.8M-6.2M operations
- 📊 Complete Pareto front analysis available

### **Key Improvements**
- 🔄 **Resume capability** for long training sessions
- 📝 **Enhanced logging** with accuracy parsing
- 🖥️ **Windows compatibility** with proper DataLoader handling
- 📊 **Comprehensive analysis** with visualization tools

## 🎯 **Use Cases**

- **Research**: Neural architecture search experimentation
- **Education**: Understanding NAS-BNN methodology
- **Application**: Person detection in edge devices
- **Benchmarking**: Comparing architectures across datasets

## 🔄 **Migration Guide**

### **From CIFAR-10 to Custom Dataset**
1. Update `models/superbnn.py` with new architecture
2. Modify data preparation script
3. Adjust image size and number of classes
4. Update default parameters in scripts

## 📚 **Documentation**

- **Interactive Tutorial**: Open `run_all.ipynb`
- **Original Paper**: [NAS-BNN: Neural Architecture Search for Binary Neural Networks](https://arxiv.org/abs/2408.15484)
- **Original Repository**: [VDIGPKU/NAS-BNN](https://github.com/VDIGPKU/NAS-BNN)
- **Authors' README**: See `README-Authors.md`

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 **License**

This project follows the same license as the original NAS-BNN implementation. See `LICENSE` file for details.

## 🙏 **Credits & Acknowledgments**

### **Original Work**
- **Paper**: [NAS-BNN: Neural Architecture Search for Binary Neural Networks](https://arxiv.org/abs/2408.15484) (Pattern Recognition 2025)
- **Authors**: VDIG-PKU Team
- **Repository**: [https://github.com/VDIGPKU/NAS-BNN](https://github.com/VDIGPKU/NAS-BNN)

### **Dataset**
- **WakeVision**: Harvard-Edge/Wake-Vision dataset for person detection

### **Modifications & Enhancements**
- Multi-dataset adaptation and pipeline automation
- Enhanced logging and resume capabilities
- Windows compatibility improvements
- Comprehensive analysis and visualization tools

## 📞 **Contact**

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas

---

⭐ **If this project helps your research, please consider starring it!**
