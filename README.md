# Development of a Machine Learning Model for Predicting Blood Infections (Sepsis)

This repository contains the code, data, and results from my master's thesis, "Early Prediction of Sepsis using Machine Learning Algorithms", submitted to the National Institute of Applied Science and Technology, University of Passau, and Klinikum Passau. The project uses machine learning to analyze routine blood test metrics and Procalcitonin (PCT) levels, achieving an accuracy of 93% in early sepsis detection.

---

## Motivation
Sepsis is a global healthcare challenge with high mortality rates (20% as of 2020) and rapid progression. Current diagnostic methods are often slow and costly, causing delays that impact patient outcomes. This project aims to address these issues with a cost-effective and reliable machine-learning-based diagnostic tool.

---

## Abstract
This thesis explores the use of machine learning to improve early sepsis detection by analyzing routine blood test metrics and PCT levels. The study identifies subtle patterns in blood data that signal sepsis onset. The resulting model achieved 93% accuracy, demonstrating its potential as a cost-effective, reliable diagnostic tool that reduces diagnosis time and enables earlier, more precise interventions.

---


### Key Features:
- Utilization of advanced data analytics to examine interrelations between blood count behaviors and PCT levels.
- Rigorous testing of machine learning algorithms and hyperparameter tuning to ensure diagnostic accuracy and reliability.
- Development of a predictive model that significantly cuts down the time to sepsis diagnosis.

### Technologies Used:
- **Languages:** Python
- **Machine Learning Libraries:** TensorFlow, Scikit-learn
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

## Repository Structure
```
/
├── data/                   # Contains datasets used in the project.
├── models/                 # Stores trained grid search machine learning models and best models.
├── notebooks/              # Jupyter notebooks for exploratory data analysis and visualization.
├── src/                    # Source code for training machine learning models and making predictions.
└── README.md               # Provides an overview of the project and setup instructions.
```
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
