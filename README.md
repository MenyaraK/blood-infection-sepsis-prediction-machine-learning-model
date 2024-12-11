# Development of a Machine Learning Model for Predicting Blood Infections (Sepsis)

This repository contains the code, data, and results from my master's thesis, "Early Prediction of Sepsis using Machine Learning Algorithms", submitted to the National Institute of Applied Science and Technology, University of Passau, and Klinikum Passau. The project uses machine learning to analyze routine blood test metrics and Procalcitonin (PCT) levels, achieving an accuracy of 86.52% in early sepsis detection.

---

## Motivation
Sepsis is a global healthcare challenge with high mortality rates (20% as of 2020) and rapid progression. Current diagnostic methods are often slow and costly, causing delays that impact patient outcomes. This project aims to address these issues with a cost-effective and reliable machine-learning-based diagnostic tool.

---

## Abstract
This thesis explores the use of machine learning to improve early sepsis detection by analyzing routine blood test metrics and PCT levels. The study identifies subtle patterns in blood data that signal sepsis onset. The resulting model achieved 86.52% accuracy, demonstrating its potential as a cost-effective, reliable diagnostic tool that reduces diagnosis time and enables earlier, more precise interventions.

---

## Key Features
- **High Accuracy**: Achieved 86.52% accuracy in predicting sepsis.
- **Cost-Effective**: Utilizes routine blood tests , making it practical for real-world applications.
- **Advanced Machine Learning Techniques**: Hyperparameter tuning and feature selection for optimized performance.
- **Clinical Relevance**: Directly applicable to improving patient outcomes by enabling timely diagnosis.
  
---


### Technologies Used:
- **Languages:** Python
- **Machine Learning Libraries:** TensorFlow, Scikit-learn
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

## Data
- **Source**: Patient data was collected from Klinikum Passau.
- **Features**: Routine blood test metrics and Procalcitonin (PCT) levels.
- **Target Variable**: Sepsis diagnosis.

> Note: Due to confidentiality, the original dataset is not included.

---

## Repository Structure
```plaintext
.
├── data/                # This file is empty, data is confidential
├── docs/                # Documentation of the project: report submitted for my thesis.
├── models/              # Stores trained grid search machine learning models and best models.
├── notebooks/           # Jupyter notebooks for data exploration, modeling, and evaluation
├── results/             # Performance metrics, confusion matrices, and feature importance charts
├── src/                 # Scripts for preprocessing, training, and testing models
└── README.md            # Project overview and usage instructions
```

## Methodology
1. **Data Preprocessing**:
   - Addressed missing values and outliers.
   - Transformed and standardized features.
2. **Feature Selection**:
   - Techniques such as clustering, correlation analysis, and random forest feature importance.
3. **Model Development**:
   - Tested models: Random Forest, AdaBoost, and RUSBoost.
   - Implemented hyperparameter tuning using GridSearchCV.
4. **Evaluation**:
   - Metrics: Accuracy, Precision, Recall, F1-Score, and AUROC.
   - Confusion matrix for clinical relevance.

---
## Results
- **Performance Metrics**:
  - Accuracy: 86.52%
  - AUROC: 93% High discrimination capability (details in results/ folder).
- **Feature Importance**:
  - White Blood features were significant predictors.

Visualization examples:
- Spearman's correlation visualization

- RUSBoost model learning curve:
![learningcurve2](https://github.com/user-attachments/assets/bbb1b85b-33a7-4e90-97f8-72bb443b40e3)

- Feature Importance of RUSBoost Chart:
![feature importance model](https://github.com/user-attachments/assets/c0f89e4e-c7e9-42c9-9223-b8671508d8c1)

- Feature Importance of XGBoost Chart:
![randomforestresult](https://github.com/user-attachments/assets/2f3abab3-3c38-4715-b7d9-3995c47a0d76)

- Output example
<img width="179" alt="output" src="https://github.com/user-attachments/assets/01999ca9-4319-4e9a-b591-e283f866415f" />

---

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sepsis-prediction-ml-model.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run Jupyter notebooks for data exploration and model training.

---

## Usage

To preprocess the dataset, run:
```bash
python src/preprocessing.py
```
--------------------------------------------
To set different daasets, run:
```bash
python src/dataset_setting.py
```
--------------------------------------------
To perform grid search of the best model parameters, run:
```bash
python src/grid_search.py
```
--------------------------------------------
To train the model, run:
```bash
python src/train_model.py
```
--------------------------------------------
To evaluate the model performance, run:
```
python src/evaluate_model.py
```
--------------------------------------------
To plot learning curves of the model, run:
```
python src/plot_results.py
```
--------------------------------------------

## Key Contributions
- Conducted detailed feature analysis and model evaluation.
- Developed a comprehensive ML pipeline for sepsis prediction.
- Proposed a cost-effective diagnostic tool for clinical use.

---

## Future Work
- Extend the model to include additional biomarkers and datasets.
- Explore deep learning models for improved performance.
- Collaborate with more healthcare institutions for real-world implementation.

---

## Contact
Feel free to reach out for collaboration or inquiries:
- **Name**: Khaireddine Menyara
- **Email**: [khaire01@ads.uni-passau.de](mailto:khaire01@ads.uni-passau.de)
- **LinkedIn**: [Menyara Khaireddine](https://www.linkedin.com/in/menyara-k/) 

---

## Acknowledgments
This project was hosted by Klinikum Passau and the University of Passau, under the supervision of Prof. Imen Harbaoui, Dr. Wiem Fekih Hassen, and Mr. Johannes Böhm.
