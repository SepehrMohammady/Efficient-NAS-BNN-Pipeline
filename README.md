# NAS-BNN Multi-Dataset Pipeline 🚀

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

## 🔍 **Projects at a Glance**

| Repository | Purpose | Key Features |
|------------|---------|--------------|
| [VDIGPKU/NAS-BNN](https://github.com/VDIGPKU/NAS-BNN) | Original Implementation | NAS-BNN framework for ImageNet, Linux/multi-GPU focus |
| [NAS-BNN-CIFAR10-Exploration](https://github.com/SepehrMohammady/NAS-BNN-CIFAR10-Exploration) | CIFAR-10 Adaptation | Windows compatibility, CIFAR-10 support, resume logic |
| [Efficient-NAS-BNN-Pipeline](https://github.com/SepehrMohammady/Efficient-NAS-BNN-Pipeline) (this repo) | Multi-Dataset Pipeline | WakeVision support, unified workflow, enhanced analysis tools |

**This repository consolidates the previous work and is the recommended version for all use cases.**

## 🎯 **Major Updates - WakeVision Integration with 500k Sample Dataset**

### ✅ **Successfully Adapted NAS-BNN for Person Detection**
- **🏆 Achieved 88.81% accuracy** on WakeVision person detection after fine-tuning
- **📈 1.13% improvement** from initial architecture search results
- **⚡ Optimized architectures** with 3.8M-6.2M operations for edge deployment
- **📊 Complete Pareto front analysis** with 4 optimal architectures discovered

### 🔧 **Enhanced Pipeline Features**
- **🖥️ Windows compatibility** with proper DataLoader handling (`workers=0`)
- **🔄 Resume capability** for long-running training sessions
- **📝 Enhanced logging** with improved accuracy parsing from multiple log formats
- **📊 Comprehensive analysis** with automated visualization tools
- **🎯 Multi-dataset support** - ImageNet, CIFAR-10, and WakeVision

---

## 📈 **WakeVision Results Summary**

### **Architecture Search Results:**
| OPs Key | Operations (M) | Search Accuracy | Test Accuracy | Fine-tuned Accuracy | Improvement |
|---------|----------------|-----------------|---------------|-------------------|-------------|
| **5** ⭐ | 5.236M | 87.77% | 87.68% | **88.81%** | **+1.04%** |
| **6** | 6.026M | 87.81% | 87.7-87.8% | **88.81%** | **+1.00%** |

### **Key Findings:**
- **Both Key 5 and Key 6 achieve excellent results**: 88.81% accuracy after fine-tuning
- **Key 5 offers better efficiency**: Slightly fewer operations with same accuracy
- **Successful fine-tuning**: Significant accuracy improvements achieved
- **Edge-ready deployment**: Models optimized for resource-constrained devices
- **Larger dataset provides better results**: Training with 500,000 samples yielded higher accuracy

---

## 🔍 **Project Origins & Acknowledgment**

### **Original Work**
This project is based on the official implementation of **["NAS-BNN: Neural Architecture Search for Binary Neural Networks"](https://arxiv.org/abs/2408.15484)** by [VDIGPKU/NAS-BNN](https://github.com/VDIGPKU/NAS-BNN).

### **Development History**
- **Initial Adaptation**: First adapted for CIFAR-10 in the [NAS-BNN-CIFAR10-Exploration](https://github.com/SepehrMohammady/NAS-BNN-CIFAR10-Exploration) repository, which focused on extending the original work to smaller datasets with enhanced Windows compatibility.
- **Current Repository**: This repository consolidates and extends both works with multi-dataset support, focusing on WakeVision for person detection, while maintaining support for ImageNet and CIFAR-10.

The original README from the authors is preserved in `README-Authors.md`.

---

## 🚀 **Quick Start for WakeVision**

### **1. Setup Environment**
```bash
# Clone and install dependencies
git clone https://github.com/SepehrMohammady/Efficient-NAS-BNN-Pipeline.git
cd Efficient-NAS-BNN-Pipeline
pip install -r requirements.txt
```

### **2. Configure for WakeVision**
```python
# In run_all.ipynb Cell 2 - Configuration
dataset_name = "WakeVision"
architecture_name = "superbnn_wakevision_large"
wakevision_img_size = 128  # Image size (64 or 128) - must match across all components

# ⚠️ IMPORTANT: Image size consistency required!
# - superbnn.py: superbnn_wakevision_large(img_size=128)  
# - prepare_local_wake_vision_from_csv.py: TARGET_IMAGE_SIZE = (128, 128)
# - run_all.ipynb: wakevision_img_size = 128
```

### **3. Prepare Data**
Choose your data preparation method:
- **Local CSV**: Use existing local WakeVision data and CSV files  
- **Online**: Automatic download from HuggingFace datasets

**⚠️ Important:** Ensure image size consistency across all components before starting!

### **4. Image Size Consistency Checklist** ✅
Before running the pipeline, verify these three files have matching image sizes:

```bash
# 1. Check model architecture (should be 128 for best results)
grep "def superbnn_wakevision_large" models/superbnn.py
# Expected: def superbnn_wakevision_large(sub_path=None, img_size=128):

# 2. Check data preparation script
grep "TARGET_IMAGE_SIZE" prepare_local_wake_vision_from_csv.py  
# Expected: TARGET_IMAGE_SIZE = (128, 128)

# 3. Check notebook configuration (Cell 2)
grep "wakevision_img_size" run_all.ipynb
# Expected: wakevision_img_size = 128
```

**If values don't match:** Update all three locations to use the same size before training!

### **5. Run Complete Pipeline**
Execute cells sequentially in `run_all.ipynb`:
1. **Data Preparation** → 2. **Supernet Training** → 3. **Architecture Search** → 4. **Testing & Fine-tuning** → 5. **Analysis & Export**

---

## 📊 **Pipeline Architecture**

### **Complete NAS-BNN Workflow**

```mermaid
flowchart TD
    %% Data Preparation Stage
    subgraph DP[🗃️ Data Preparation]
        A1[WakeVision Dataset<br/>📊 500k Images]
        A2[Image Processing<br/>🖼️ 128×128 Resize]
        A3[Data Validation<br/>✅ Size Consistency]
        A1 --> A2 --> A3
    end

    %% Supernet Training Stage  
    subgraph ST[🏗️ Supernet Training]
        B1[Architecture Definition<br/>🧠 superbnn_wakevision_large]
        B2[Binary Weight Training<br/>⚡ 120 Epochs]
        B3[Weight Sharing<br/>🔄 Subnetwork Sampling]
        B1 --> B2 --> B3
    end

    %% Architecture Search Stage
    subgraph AS[🔍 Neural Architecture Search] 
        C1[Population Init<br/>👥 50 Architectures]
        C2[Evolutionary Search<br/>🧬 10 Generations]
        C3[Pareto Optimization<br/>⚖️ Accuracy vs Efficiency]
        C4[Optimal Solutions<br/>🎯 Key 3,4,5,6]
        C1 --> C2 --> C3 --> C4
    end

    %% Testing & Fine-tuning Stage
    subgraph TF[🧪 Testing & Fine-tuning]
        D1[Architecture Testing<br/>📊 Key 5 & 6]
        D2[Performance Validation<br/>✅ 87.7-87.8%]
        D3[Fine-tuning Training<br/>🎯 From Scratch]
        D4[Final Results<br/>🏆 88.81% Accuracy]
        D1 --> D2 --> D3 --> D4
    end

    %% Export & Deployment Stage
    subgraph ED[📦 Export & Deployment]
        E1[Model Selection<br/>🎯 Key 5 or Key 6]
        E2[ONNX Export<br/>📤 Optimization]
        E3[Deployment Package<br/>🚀 17-18 MB]
        E1 --> E2 --> E3
    end

    %% Main Flow
    DP --> ST
    ST --> AS  
    AS --> TF
    TF --> ED

    %% Key Results Annotations
    AR1[🎯 Key Results<br/>• Key 5: 5.236M ops, 88.81%<br/>• Key 6: 6.026M ops, 88.81%<br/>• ONNX: 17-18 MB<br/>• Edge-ready deployment]
    
    %% Connect annotation
    ED -.-> AR1

    %% Enhanced Styling for Better Visibility
    classDef dataStyle fill:#E3F2FD,stroke:#1976D2,stroke-width:2px,color:#0D47A1
    classDef trainStyle fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px,color:#4A148C
    classDef searchStyle fill:#E8F5E8,stroke:#388E3C,stroke-width:2px,color:#1B5E20
    classDef testStyle fill:#FFF3E0,stroke:#F57C00,stroke-width:2px,color:#E65100
    classDef exportStyle fill:#FCE4EC,stroke:#C2185B,stroke-width:2px,color:#880E4F
    classDef resultStyle fill:#FFFDE7,stroke:#F9A825,stroke-width:2px,color:#F57F17

    %% Apply Styles
    class A1,A2,A3 dataStyle
    class B1,B2,B3 trainStyle  
    class C1,C2,C3,C4 searchStyle
    class D1,D2,D3,D4 testStyle
    class E1,E2,E3 exportStyle
    class AR1 resultStyle
```

### **🔄 Pipeline Flow Highlights**

| Stage | Duration | Key Output | Next Action |
|-------|----------|------------|-------------|
| **🗃️ Data Prep** | ~1 hour | Formatted dataset | Start supernet training |
| **🏗️ Supernet** | ~24-48 hours | Trained weights | Launch architecture search |
| **🔍 Search** | ~6-8 hours | Pareto front | Test best candidates |
| **🧪 Testing** | ~2-4 hours | Validated archs | Fine-tune winners |
| **📦 Export** | ~30 minutes | ONNX models | Deploy to edge |

### **💡 Key Features**
🎯 **Multi-objective Optimization**: Balances 88.81% accuracy with 5-6M operations  
⚡ **Efficient Search**: 10 generations discover optimal architectures automatically  
🚀 **Deployment Ready**: Outputs 17-18MB ONNX models for immediate edge deployment  
🔧 **Robust Pipeline**: Complete error handling and resume capabilities

---

## 🔧 **Technical Improvements**

### **Enhanced Log Parsing**
- ✅ Fixed accuracy parsing for multiple log formats
- ✅ Support for `test.py`, `train.py`, and `train_single.py` outputs
- ✅ Robust pattern matching for different output styles

### **Windows Compatibility**
- ✅ DataLoader workers set to 0 for Windows single-GPU setups
- ✅ Proper path handling for Windows file systems
- ✅ CUDA device management optimized for single-GPU workflows

### **Modular Dataset Support**
- ✅ Easy switching between ImageNet, CIFAR-10, and WakeVision
- ✅ Conditional dataset preparation cells
- ✅ Automatic configuration validation

---

## 📁 **Project Structure**

```
Efficient-NAS-BNN-Pipeline/
├── run_all.ipynb                 # 🎯 Main pipeline notebook (UPDATED)
├── prepare_local_wake_vision_from_csv.py  # 📁 WakeVision local data prep
├── prepare_wakevision.py         # 🌐 WakeVision online data prep  
├── prepare_cifar10.py            # 🎯 CIFAR-10 preparation
├── models/                       # 🧠 Architecture definitions
├── utils/                        # 🔧 Utilities and helpers
├── work_dirs/                    # 📊 Training outputs and results
└── requirements.txt              # 📦 Dependencies
```

---

## 🎯 **Use Cases**

- **🔬 Research**: Neural architecture search experimentation
- **📚 Education**: Understanding NAS-BNN methodology  
- **📱 Applications**: Person detection for edge devices
- **⚖️ Benchmarking**: Comparing architectures across datasets

---

## 🏆 **Key Achievements**

### **Successful WakeVision Integration**
- ✅ Binary classification adaptation (person/no-person)
- ✅ Custom data loading and preprocessing
- ✅ Architecture search parameter optimization
- ✅ Complete pipeline validation

### **Robust Implementation**
- ✅ Error handling and recovery mechanisms
- ✅ Comprehensive logging and analysis
- ✅ Cross-platform compatibility
- ✅ Production-ready ONNX export

### **Performance Optimization**
- ✅ Memory-efficient training configurations
- ✅ GPU utilization optimization
- ✅ Batch size tuning for target hardware

---

## 📋 **Future Work**

- [ ] **Multi-GPU distributed training** support
- [ ] **Additional datasets** integration (COCO, OpenImages)
- [ ] **Quantization-aware training** for further optimization
- [ ] **Mobile deployment** with TensorFlow Lite conversion
- [ ] **Real-time inference** benchmarking

---

## 🤝 **Contributing**

Contributions are welcome! Please feel free to submit:
- 🐛 Bug reports and fixes
- ✨ Feature enhancements  
- 📖 Documentation improvements
- 🧪 Additional dataset integrations

---

## 📄 **Citation and License**

### **Citation**
If you use this work, please cite both the original NAS-BNN paper and this adaptation:

```bibtex
@article{wang2024nasbnn,
  title={NAS-BNN: Neural Architecture Search for Binary Neural Networks},
  author={Wang, Yingting and Zhang, Huixia and Chen, Sheng and Li, Jiashuai and Xu, Chang and Lin, Mingbao and Yan, Junchi},
  journal={Pattern Recognition},
  volume={147},
  pages={110001},
  year={2024},
  publisher={Elsevier}
}

@article{mohammady2025efficient,
  title={Efficient NAS-BNN Pipeline: Multi-Dataset Neural Architecture Search for Binary Neural Networks},
  author={Sepehr Mohammady},
  journal={GitHub Repository},
  url={https://github.com/SepehrMohammady/Efficient-NAS-BNN-Pipeline},
  year={2025}
}
```

### **License**
- The **original NAS-BNN code** is available for academic research purposes only, and requires authorization for commercial use (see `README-Authors.md`). For commercial permission, please contact wyt@pku.edu.cn.
- **Modifications and additions** in this repository are provided under the MIT License (see `LICENSE`).

---

## 📞 **Support**

- 📖 **Documentation**: See `run_all.ipynb` for detailed pipeline walkthrough
- 🐛 **Issues**: Report bugs via GitHub Issues
- 💬 **Discussions**: Join GitHub Discussions for questions

---

**🎉 Ready for edge deployment with optimized binary neural networks!** 🚀
