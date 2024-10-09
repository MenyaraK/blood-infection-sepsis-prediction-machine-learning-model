# Early Prediction of Sepsis Using Machine Learning Algorithms

This repository contains the source code for the master thesis titled "Early Prediction of Sepsis using Machine Learning Algorithms," submitted to the National Institute of Applied Science and Technology, University of Passau and Klinikum Passau.

## Thesis Overview
The thesis presents a machine learning framework aimed at enhancing the early detection of sepsis by leveraging the predictive potential of routine blood test metrics and Procalcitonin (PCT) levels. Through advanced data analytics, this study identifies subtle patterns in blood count behaviors and PCT that signal the onset of sepsis. The primary goal is to develop a cost-effective and reliable diagnostic tool that reduces the diagnostic timeframe and improves patient outcomes by enabling earlier and more precise interventions.

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
├── models/                 # Stores trained machine learning models and serialization files.
├── notebooks/              # Jupyter notebooks for exploratory data analysis and visualization.
├── src/                    # Source code for training machine learning models and making predictions.
├── tests/                  # Test scripts for validating the codebase.
└── README.md               # Provides an overview of the project and setup instructions.
```
## Usage

To train the model, run:
```bash
python src/train_model.py
```
To evaluate the model performance, run:
```
python src/evaluate_model.py
```
