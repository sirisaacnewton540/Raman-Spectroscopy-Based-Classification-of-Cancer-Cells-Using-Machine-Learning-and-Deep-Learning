# Raman Spectroscopy-Based Classification of Cancer Cells Using Machine Learning and Deep Learning

This project presents a complete pipeline for classifying cancerous vs normal cell types using Surface-Enhanced Raman Spectroscopy (SERS) data. We analyze spectroscopic responses of cell media functionalized with three different chemical groups — NH₂, COOH, and (COOH)₂ — and apply machine learning, dimensionality reduction, and deep learning models for accurate identification of cell phenotypes.

![image](https://github.com/user-attachments/assets/868e4d6c-9348-401e-b297-c54b0a065bcf)

---

## 🧪 [Dataset (with link)](https://www.kaggle.com/datasets/andriitrelin/cells-raman-spectra/data)

**Source**: University of Chemistry and Technology, Prague
**Description**:
Raman spectra were collected from cell culture media associated with 12 types of normal and cancer cell lines. Each sample was treated with gold nanourchins functionalized with specific chemical groups (NH₂, COOH, and (COOH)₂). The goal is to classify the sample type using the spectral signature of each group.

* 📊 **Spectral Range**: 100–4278 cm⁻¹ (2090 points)
* 📁 **Folders**: 12 cell types × 3 CSVs (one per functional group)
* 🎯 **Classes**: 12 (e.g., A2058, G361, HPM, HF, ZAM, DMEM, and serum variations)

---

## 🎯 Objectives

* Normalize and preprocess raw Raman spectra
* Visualize separability of classes using PCA, t-SNE, and UMAP
* Apply classical ML (SVM + PCA/LDA) and compare with 1D CNNs
* Evaluate classification performance for each chemical group
* Demonstrate model potential for non-invasive cancer diagnostics

---

## 🧠 Research Questions

1. Can Raman spectra from functionalized culture media reliably differentiate between cancerous and non-cancerous cells?
2. Which dimensionality reduction method (PCA, t-SNE, UMAP) best visualizes class separability?
3. How do classical classifiers (e.g., SVM) compare to CNNs for 1D signal classification?
4. Are some chemical groups (NH₂, COOH, (COOH)₂) more discriminative than others?

---

## ⚙️ Methodology

### 1. Preprocessing

* Normalization of Raman spectra using linear min-max scaling
* Data structured into three independent feature matrices: `[NH₂]`, `[COOH]`, `[COOH₂]`
* Concatenated datasets (`NH₂+COOH`, `COOH+(COOH)₂`, etc.) for fusion analysis

### 2. Visualization

* **PCA** for linear variance-driven structure
* **t-SNE & UMAP** to explore nonlinear clustering and manifold structure
* Color-coded by cell line to reveal separability patterns

### 3. Unsupervised Clustering

* Applied **K-Means** on t-SNE results
* Evaluated clustering effectiveness using label permutation and confusion matrices

### 4. Classical Classification

* **PCA + SVM** and **LDA + SVM** pipelines with RandomizedSearchCV for hyperparameter tuning
* Metrics: Accuracy, Confusion Matrix, Classification Report

### 5. Deep Learning

* **1D Convolutional Neural Networks (1D CNNs)**

  * Input: (2090 features, 1 channel)
  * Architecture: Conv → Pool → Conv → Pool → Dense → Softmax
  * Loss: Categorical Crossentropy, Optimizer: Adam
  * Evaluation: Accuracy and loss plotted over 200 epochs

---

## 📊 Results

### Visual Exploration

* **t-SNE** showed high inter-class separability, outperforming PCA.
* NH₂+COOH combinations improved clustering, suggesting informative spectral fusion.

![image](https://github.com/user-attachments/assets/c3be62d6-d80d-4bcd-ba46-a5e99ec3664d)

![image](https://github.com/user-attachments/assets/44ae2b3f-f106-49e8-86cb-88e337163f7c)



### Clustering (K-Means)

* Achieved near-alignment with ground truth labels.
* Confusion matrix showed dominant diagonal structure, indicating high true positive rates.

### Classical ML

| Feature Set      | Accuracy | Notes                                     |
| ---------------- | -------- | ----------------------------------------- |
| NH₂ + SVM (PCA)  | \~85%    | Sensitive to class balance                |
| COOH + SVM (LDA) | \~88%    | Better discrimination on fibroblast lines |
| (COOH)₂ + SVM    | \~80%    | Slightly less informative individually    |

### Deep Learning (1D CNN)

| Input   | Accuracy |
| ------- | -------- |
| NH₂     | \~94%    |
| COOH    | \~91%    |
| (COOH)₂ | \~89%    |

![image](https://github.com/user-attachments/assets/16a96e34-5e31-4c94-a772-0af0318c391e)

![image](https://github.com/user-attachments/assets/7f903ebc-5d21-4285-b8fb-c7b9b6a2d1ea)

![image](https://github.com/user-attachments/assets/8af9469f-2231-425d-a5d9-e9640276a121)

![image](https://github.com/user-attachments/assets/4aeb229b-6fb0-40e6-b898-a9da65431739)


CNNs significantly outperformed classical models, capturing subtle patterns in high-dimensional Raman signals.

---

## 📌 Conclusion

* Raman spectroscopy of cell media, enhanced with nanourchins and chemical functionalization, holds great promise for **label-free cancer detection**.
* Dimensionality reduction techniques validated inherent spectral separability between classes.
* CNNs demonstrated robust performance across all functional groups, showing the viability of deep learning in Raman biosensing applications.
* Future directions include ensemble models, attention-based networks, and real-time biosensor integration.



## 💡 Future Enhancements

* Integrate spectral denoising and baseline correction
* Use advanced architectures (Transformers, ResNet1D)
* Apply domain adaptation for real-world, multi-lab deployment
