# 🧬 Breast Cancer Prediction Using Machine Learning

This project focuses on predicting whether a breast cancer diagnosis is **benign** or **malignant** using supervised machine learning models. It leverages the widely used **Breast Cancer Wisconsin Diagnostic Dataset**, consisting of 30 numeric features derived from digitized images of fine needle aspirate (FNA) of breast masses.

---

## 📌 Project Objective

To build a machine learning model that classifies breast tumors as **malignant** (cancerous) or **benign** (non-cancerous), using various tumor features extracted from medical images.

---

## 📊 Dataset Overview

- **Source**: Breast Cancer Wisconsin Diagnostic Dataset
- **Number of Instances**: 569
- **Number of Features**: 30 numerical features
- **Target Classes**:
  - `Malignant` (212 samples)
  - `Benign` (357 samples)

### 🔍 Key Features:
- `radius` – mean of distances from center to points on the perimeter  
- `texture` – standard deviation of gray-scale values  
- `perimeter`  
- `area`  
- `smoothness` – local variation in radius lengths  
- `compactness` – perimeter² / area − 1.0  
- `concavity` – severity of concave portions of the contour  
- `concave points` – number of concave portions of the contour  
- `symmetry`  
- `fractal dimension` – coastline approximation  

All 30 features are used in model training, and the dataset is **linearly separable**.

---

## 🧠 Models Used

- Logistic Regression  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  
- Decision Tree  
- Random Forest  
- Naive Bayes  

Performance evaluation is done using accuracy, confusion matrix, and classification report (precision, recall, F1-score).

---

## 🧪 Steps Followed

1. **Import Libraries & Load Dataset**
2. **Data Preprocessing**
   - Handling missing values
   - Label encoding target variable
   - Feature scaling
3. **Model Training**
4. **Model Evaluation**
5. **Model Comparison**

---

## 📈 Results

The models performed well given the dataset's linearly separable nature. Best-performing models achieved high accuracy and precision on both classes.

---

## 📁 File Structure

```
📦 Breast-Cancer-Prediction
 ┣ 📜 Breast Cancer.ipynb     # Main Jupyter Notebook with full analysis and models
 ┗ 📜 README.md               # Project overview and documentation
```

---

## 📌 Requirements

- Python 3.x
- Jupyter Notebook / Google Colab
- Libraries:
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `scikit-learn`

Install dependencies:

```bash
pip install -r requirements.txt
```

> (You can auto-generate `requirements.txt` from the notebook using tools like `pipreqs` or manually add package names.)

---

## 💡 Future Work

- Hyperparameter tuning using GridSearchCV
- Cross-validation
- Deploy model with Flask or Streamlit for web access
- Add SHAP or LIME for interpretability

---

## 🙌 Acknowledgements

- Dataset provided by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- Inspiration from various ML model comparison studies

---

## 📬 Contact

**Author**: Pavithra Yasiru  
📧 [YourEmail@example.com]  
🔗 [LinkedIn](https://www.linkedin.com/in/pavithra-yasiru) | [GitHub](https://github.com/pavithra-yasiru)

---
