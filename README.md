# GFTEL-CPDP: Graph-Based Feature Transformation and Ensemble Learning for Cross-Project Defect Prediction (CPDP)

## 📌 Overview
GFTEL-CPDP is an advanced **Cross-Project Defect Prediction (CPDP)** framework that integrates:
- **Graph Neural Networks (GCN, GAT)** for feature transformation.
- **SHAP-based Feature Selection** to enhance model interpretability.
- **SMOTE and Oversampling** to address class imbalance.
- **Ensemble Learning (Random Forest, Gradient Boosting, AdaBoost)** to improve classification.
- **Deep Neural Network (DNN) with Focal Loss** to handle imbalanced defect data.

The model is evaluated on **multiple open-source software defect datasets**.

---

## 📂 Dataset
The framework is tested on **10 publicly available defect datasets**:
- `ant-1.7.csv`
- `camel-1.6.csv`
- `forrest-0.8.csv`
- `ivy-2.0.csv`
- `log4j-1.2.csv`
- `poi-3.0.csv`
- `synapse-1.2.csv`
- `velocity-1.6.csv`
- `xalan-2.7.csv`
- `xerces-1.4.csv`

Each dataset contains software metrics with **defect labels (0: Non-defective, 1: Defective)**.

---

## 🛠 Installation
Ensure you have Python 3.8+ and install the required dependencies:

```bash
pip install pandas numpy torch torch-geometric scikit-learn imbalanced-learn shap

Execute the script with:
python GFTEL-CPDP.py

The results are saved in:
cpdp_results.csv

This README file clearly describes the **purpose, dataset, installation, execution, evaluation, and key features** of GFTEL-CPDP. Let me know if you need modifications! 🚀
