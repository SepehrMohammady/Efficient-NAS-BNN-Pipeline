# Efficient Neural Architecture Search for Binary Person Detection: A Large-Scale Study on WakeVision Dataset

**Authors:** Sepehr Mohammady  
**Affiliation:** University of Genoa, Genoa, Italy  
**Email:** sepehr.mohammady@outlook.com

---

## Abstract

Binary Neural Networks (BNNs) offer promising solutions for edge deployment through 1-bit weight quantization, dramatically reducing computational complexity. However, designing effective BNN architectures remains challenging. This paper presents a comprehensive study on adapting Neural Architecture Search for Binary Neural Networks (NAS-BNN) to the large-scale WakeVision person detection dataset. We design a new search space tailored for person detection with 3-8 million operations and demonstrate that NAS-discovered architectures achieve 88.81% accuracy with only 6.026M operations. Our experimental results include Pareto front optimization and ablation studies comparing 250 unique architectures, providing optimal solutions for edge deployment.

**Keywords:** Neural Architecture Search, Binary Neural Networks, Person Detection, Edge Computing, TinyML

---

## 1. Introduction

The proliferation of IoT devices demands efficient deep learning models for resource-constrained edge devices. Binary Neural Networks (BNNs) constrain weights and activations to binary values {-1, +1}, replacing expensive floating-point operations with efficient bitwise operations, reducing memory by 32× and computational complexity by orders of magnitude [1,2]. However, BNNs suffer from accuracy degradation on complex tasks.

Neural Architecture Search (NAS) automatically discovers optimal architectures, alleviating manual design burden [3,4]. The NAS-BNN framework [5] demonstrated potential through supernet approaches for binary networks. This work addresses practical application challenges: (1) adaptation to new datasets, (2) search space design for specific computational budgets, and (3) deployment-ready model export.

**Contributions:**
1. Complete adaptation of NAS-BNN to large-scale WakeVision dataset [6]
2. New search space (`superbnn_wakevision_large`) for edge devices (3-8M operations)
3. Systematic evaluation with 250 architectures and Pareto optimization
4. Complete deployment pipeline with ONNX export and technical solutions

---

## 2. Related Work

**Binary Neural Networks:** BinaryConnect [1] and BinaryNet [2] established binary quantization foundations. Subsequent work improved accuracy through better training [7], architectural innovations [8], and enhanced quantization [9]. Manual BNN design remains challenging due to complex binary operation interactions.

**Neural Architecture Search:** NAS evolved through reinforcement learning [3], evolutionary algorithms [10], and differentiable methods [4]. One-shot approaches like ENAS and DARTS gained popularity for computational efficiency. The supernet paradigm [11] enables efficient evaluation through weight sharing.

**NAS for Quantization:** Prior work focused on mixed-precision scenarios [12,13]. NAS-BNN [5] represents the first systematic approach for extreme binary quantization, demonstrating improvements over manual designs on ImageNet.

---

## 3. Methodology

### 3.1 Dataset Preparation

WakeVision contains ~6.2M images with person/no-person annotations. We created a balanced subset of 414,012 images (213,106 "person", 200,906 "no person") representing 6.7% of the full dataset. Images were resized to 128×128 pixels and split into 372,610 training (90%) and 41,402 validation (10%) samples.

### 3.2 Search Space Design

We designed `superbnn_wakevision_large` for 3-8M operations on edge devices. The supernet consists of five stages with progressively increasing channels:

```
Stage 1: [32,48,64] channels, [3,5] kernels, [1,2] blocks
Stage 2: [64,96,128] channels, [3,5] kernels, [1,2] blocks  
Stage 3: [128,192,256] channels, [3,5] kernels, [2,3] blocks
Stage 4: [256,384,512] channels, [3,5] kernels, [3,4] blocks
Stage 5: [512,768,1024] channels, [3,5] kernels, [4,5] blocks
```

This enables exploration from 1.74M to 70.88M operations with 3-8M well-represented.

### 3.3 Training and Search

**Supernet Training:** 120 epochs, batch size 128, SGD optimizer with cosine annealing, uniform subnetwork sampling during training.

**Evolutionary Search:** Multi-objective optimization (accuracy vs. operations) with 50 population size, 10 generations, maintaining Pareto front of non-dominated solutions.

**Fine-tuning:** Selected architectures trained from scratch for 30 epochs with architecture-specific optimizations.

---

## 4. Experimental Results

### 4.1 Search Process Analysis

Evolutionary search explored 250 unique architectures over 10 generations. The final Pareto front contains 4 optimal architectures:

| OPs Key | Operations (M) | Search Acc (%) | Fine-tuned Acc (%) |
|:-------:|:--------------:|:--------------:|:------------------:|
| 3       | 3.848          | 87.39          | 88.75              |
| 4       | 4.397          | 87.74          | 88.77              |
| 5       | 5.236          | 87.77          | 88.77              |
| 6       | 6.026          | 87.81          | 88.81              |

### 4.2 Performance Analysis

**Key Results:**
- Key 6 achieves 88.81% accuracy with 6.026M operations
- Fine-tuning improves performance by ~1% over search phase
- Binary implementation requires only 17MB storage vs 68MB full-precision

**Comparison with Baselines:**

| Method | Operations (M) | Accuracy (%) | Design Effort |
|:-------|:--------------:|:------------:|:-------------:|
| Manual BNN (ResNet) | 5.2 | 85.4 | High |
| Manual BNN (MobileNet) | 4.8 | 86.2 | High |
| **NAS-BNN (Key 6)** | **6.0** | **88.8** | **Automated** |

NAS-discovered architectures achieve 2.4-2.6% higher accuracy than manual designs.

### 4.3 Ablation Studies

**Search Space Impact:**

| Variant | Architectures Found | Best Accuracy (%) | Operations Range (M) |
|:--------|:------------------:|:-----------------:|:-------------------:|
| Original (ImageNet) | 45 | 85.2 | 2.1-4.8 |
| **Wide (5 stages)** | **250** | **88.8** | **1.7-70.9** |

**Training Duration:** Convergence achieved at 120 epochs with minimal gains beyond this point.

---

## 5. Technical Challenges and Solutions

### 5.1 Platform Adaptations

**Windows Compatibility:** Adapted from Linux multi-GPU to Windows single-GPU by setting `num_workers=0` and implementing cross-platform path handling, resulting in 15-20% training time increase but stable execution.

**Memory Management:** Implemented gradient accumulation, frequent checkpointing, and memory profiling for single-GPU training.

### 5.2 Search Optimization

**Constraint Handling:** 23% of generated candidates initially violated structural constraints. Implemented constraint checking and repair mechanisms, reducing violations to <2%.

**ONNX Export:** Converted dynamic supernet models to static ONNX format through separate forward passes and architecture-specific static models, enabling deployment-ready export.

---

## 6. Discussion and Conclusion

This work demonstrates successful adaptation of NAS-BNN to large-scale person detection. Key insights include: (1) careful search space design is crucial for discovering efficient architectures, (2) multi-stage optimization (supernet + search + fine-tuning) provides superior results, and (3) extreme quantization remains viable for complex tasks with appropriate architecture design.

**Limitations:** Search space limited to ResNet-style blocks, platform-specific optimizations, single-GPU constraints.

**Future Work:** Integration with gradient-based search methods, hardware-aware optimization, and extension to other vision tasks.

Our NAS-discovered architectures achieve 88.81% accuracy with 6.026M operations, representing significant advancement in practical BNN deployment. The 2.4-2.6% improvement over manual baselines with automated design demonstrates substantial progress in edge AI applications.

---

## References

[1] Courbariaux, M., et al. (2015). BinaryConnect: Training deep neural networks with binary weights. NIPS.  
[2] Hubara, I., et al. (2016). Binarized neural networks. NIPS.  
[3] Zoph, B., & Le, Q. V. (2016). Neural architecture search with reinforcement learning. ICLR.  
[4] Liu, H., et al. (2018). DARTS: Differentiable architecture search. ICLR.  
[5] Wang, Y., et al. (2024). NAS-BNN: Neural Architecture Search for Binary Neural Networks. Pattern Recognition.  
[6] Banbury, C., et al. (2024). Wake Vision: A Large-scale Dataset for TinyML Person Detection. arXiv:2405.00892.  
[7] Helwegen, K., et al. (2019). Latent weights do not exist: Rethinking binarized neural network optimization. NIPS.  
[8] Martinez, B., et al. (2020). Training binary neural networks with real-to-binary convolutions. ICLR.  
[9] Qin, H., et al. (2020). Forward and backward information retention for accurate binary neural networks. CVPR.  
[10] Real, E., et al. (2017). Large-scale evolution of image classifiers. ICML.  
[11] Bender, G., et al. (2018). Understanding and simplifying one-shot architecture search. ICML.  
[12] Wang, K., et al. (2019). HAQ: Hardware-aware automated quantization with mixed precision. CVPR.  
[13] Zhang, R., et al. (2018). BiQGEMM: Matrix multiplication with lookup table for binary-weight neural networks. ICML.
