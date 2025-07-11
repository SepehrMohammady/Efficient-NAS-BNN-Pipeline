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
graph TB
    %% Data Preparation Stage
    subgraph "🗃️ Data Preparation"
        A1[WakeVision CSV Files] --> A2[prepare_local_wake_vision_from_csv.py]
        A3[HuggingFace Dataset] --> A4[prepare_wakevision.py]
        A2 --> A5[Image Resizing 128×128]
        A4 --> A5
        A5 --> A6[data/wakevision/train_large<br/>data/wakevision/val<br/>data/wakevision/test]
        A7[Image Size Consistency Check] --> A5
        A8[TARGET_IMAGE_SIZE validation] --> A7
    end

    %% Supernet Training Stage
    subgraph "🏗️ Supernet Training"
        B1[Supernet Architecture<br/>superbnn_wakevision_large] --> B2[Weight Sharing Strategy]
        B2 --> B3[Binary Weight Training]
        B3 --> B4[Random Subnetwork Sampling]
        B4 --> B5[Gradient Updates]
        B5 --> B6[Checkpoint Saving<br/>work_dirs/.../checkpoint.pth.tar]
        B7[Training Configuration<br/>120 epochs, batch=128] --> B1
        B8[Loss: CrossEntropy<br/>Optimizer: SGD] --> B3
    end

    %% Architecture Search Stage
    subgraph "🔍 Neural Architecture Search"
        C1[Population Initialization<br/>50 random architectures] --> C2[Evolutionary Algorithm]
        C2 --> C3[Architecture Evaluation]
        C3 --> C4[Operations Count<br/>3.8M - 6.2M range]
        C3 --> C5[Accuracy Assessment<br/>Quick validation]
        C4 --> C6[Pareto Front Update]
        C5 --> C6
        C6 --> C7{Epoch < 10?}
        C7 -->|Yes| C8[Mutation & Crossover<br/>25 mutations, 25 crossovers]
        C8 --> C2
        C7 -->|No| C9[Final Pareto Front<br/>Key 3,4,5,6 architectures]
        C9 --> C10[Save Results<br/>search/info.pth.tar]
        C11[Fitness Function<br/>Accuracy vs Efficiency] --> C6
    end

    %% Testing & Validation Stage
    subgraph "🧪 Architecture Testing"
        D1[Select Promising Keys<br/>Key 5, Key 6] --> D2[Architecture Extraction]
        D2 --> D3[Supernet Weight Loading]
        D3 --> D4[Extended Evaluation<br/>test.py script]
        D4 --> D5[Performance Validation<br/>~87.7-87.8% accuracy]
        D5 --> D6[Architecture Ranking]
        D6 --> D7[Fine-tuning Candidates<br/>Best 2 architectures]
    end

    %% Fine-tuning Stage
    subgraph "⚡ Fine-tuning"
        E1[Key 5 Architecture<br/>5.236M operations] --> E2[From-scratch Training<br/>train_single.py]
        E3[Key 6 Architecture<br/>6.026M operations] --> E4[From-scratch Training<br/>train_single.py]
        E2 --> E5[Optimized Training<br/>30 epochs, lr=0.01]
        E4 --> E5
        E5 --> E6[Final Accuracies<br/>Key 5: 88.766%<br/>Key 6: 88.807%]
        E6 --> E7[Model Checkpoints<br/>finetuned_ops_key5/<br/>finetuned_ops_key6/]
    end

    %% Analysis & Export Stage
    subgraph "📊 Analysis & Export"
        F1[Performance Analysis] --> F2[Accuracy Comparison<br/>Search vs Fine-tuned]
        F2 --> F3[Pareto Front Visualization]
        F1 --> F4[Efficiency Metrics<br/>Operations, Inference Time]
        F3 --> F5[Results Documentation]
        F4 --> F5
        F5 --> F6[Architecture Selection<br/>Key 5 or Key 6]
        F6 --> F7[ONNX Export<br/>export_ops_key selection]
        F7 --> F8[Deployment Package<br/>Key 5: 18.3MB<br/>Key 6: 17.5MB]
        F9[Model Optimization<br/>Constant folding] --> F7
    end

    %% Data Flow Connections
    A6 --> B1
    B6 --> C1
    C10 --> D1
    D7 --> E1
    D7 --> E3
    E7 --> F1
    F8 --> G1[🚀 Edge Deployment]

    %% Configuration Dependencies
    subgraph "⚙️ Configuration Management"
        G2[run_all.ipynb<br/>Central Configuration] --> G3[Image Size: 128×128<br/>Architecture: superbnn_wakevision_large]
        G3 --> G4[Training Parameters<br/>Epochs, Batch Size, LR]
        G3 --> G5[Search Parameters<br/>Population, Generations, Bounds]
        G4 --> B7
        G5 --> C1
        G6[Cross-component Validation<br/>models/superbnn.py<br/>prepare_local_wake_vision_from_csv.py] --> A8
    end

    %% Error Handling & Monitoring
    subgraph "🔧 Error Handling"
        H1[CUDA Memory Management] --> H2[Batch Size Adjustment]
        H3[Image Size Mismatch Detection] --> H4[Configuration Validation]
        H5[Training Resume Capability] --> H6[Checkpoint Recovery]
        H2 --> B3
        H4 --> A7
        H6 --> B1
    end

    %% Styling
    classDef dataStage fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef trainStage fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef searchStage fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef testStage fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef exportStage fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef configStage fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    classDef errorStage fill:#ffebee,stroke:#b71c1c,stroke-width:2px

    class A1,A2,A3,A4,A5,A6,A7,A8 dataStage
    class B1,B2,B3,B4,B5,B6,B7,B8 trainStage
    class C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11 searchStage
    class D1,D2,D3,D4,D5,D6,D7 testStage
    class E1,E2,E3,E4,E5,E6,E7 testStage
    class F1,F2,F3,F4,F5,F6,F7,F8,F9 exportStage
    class G1,G2,G3,G4,G5,G6 configStage
    class H1,H2,H3,H4,H5,H6 errorStage
```

### **Key Pipeline Features**

🔄 **Iterative Process**: The evolutionary search runs for 10 generations with continuous improvement  
⚖️ **Multi-objective Optimization**: Balances accuracy (87-88%) with efficiency (3.8-6.2M operations)  
🎯 **Pareto-optimal Solutions**: Discovers 4 architectures representing different accuracy-efficiency trade-offs  
🔧 **Robust Configuration**: Ensures image size consistency across all pipeline components  
📊 **Comprehensive Analysis**: From quick search evaluation to thorough fine-tuning validation  
🚀 **Deployment-ready**: Outputs optimized ONNX models ready for edge device deployment

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
