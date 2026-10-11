
# HR Analytics: Job Change of Data Scientists

Analyze and predict the factors influencing data scientists' decisions to look for new job opportunities. This project implements an end-to-end machine learning pipeline, starting from Exploratory Data Analysis (EDA) and data preprocessing, to training and evaluating multiple classification models (Random Forest, Logistic Regression, and LightGBM). Finally, it leverages SHAP (SHapley Additive exPlanations) to interpret the model's predictions and identify the most significant features driving job change intentions.

The dataset used in this project is sourced from the Kaggle: [HR Analytics: Job Change of Data Scientists](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists).

## Overview

The `jobchange.ipynb` notebook is structured into the following main phases:

1. **Data Loading & Initial Inspection**: Loading the training and testing datasets and inspecting data types, missing values, and duplicates.
2. **Exploratory Data Analysis (EDA)**: Visualizing the distribution of the target variable and analyzing relationships between features (e.g., city development index, gender, education level) and the target.
3. **Data Cleaning & Preprocessing**: 
   - Handling missing values using domain-aware imputation (e.g., 'Unknown' for categorical variables, median for numerical).
   - Feature engineering (e.g., creating missingness flags).
4. **Data Transformation**: Encoding categorical variables using Binary, Ordinal, and One-Hot Encoding techniques.
5. **Modeling & Evaluation**: 
   - Splitting data with stratification to handle class imbalance.
   - Training baseline and class-balanced models.
   - Evaluating models using ROC-AUC and Cross-Validation.
6. **Model Interpretability**: Using SHAP values to explain the predictions of the best-performing model and understand feature importance.

## Dataset Setup

Because the dataset is hosted on Kaggle, you must download it manually before running the notebook:
1. Go to the [Kaggle Dataset Page](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists).
2. Download the dataset (requires a free Kaggle account).
3. Extract the `.zip` file and place `aug_train.csv` and `aug_test.csv` in the **root directory** of this project (the exact same folder where `jobchange.ipynb` is located).

You want your project structure to look like below:

```txt
HRAnalytics-DataScientists/
├── aug_test.csv
├── aug_train.csv
├── jobchange.ipynb
└── README.md
```

## Running Locally

### Prerequisites

- [Python](https://www.python.org/downloads) >= 3.10
- [Jupyter Notebook](https://jupyter.org/install) or [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)

### Setup

1. **Clone the project**

```bash
git clone https://github.com/JulHernn/HRAnalytics-DataScientists.git
cd HRAnalytics-DataScientists
```

2. **Create and activate a Virtual Environment (Optional but recommended)**

```python
python -m venv .venv
```

Activate Environment

| Shell Environment        | Activation Command           |
| ------------------------ | ---------------------------- |
| Windows (Command Prompt) | `.venv\Scripts\activate.bat` |
| Windows (PowerShell)     | `.venv\Scripts\Activate.ps1` |
| macOS/Linux (bash/zsh) | `source .venv/bin/activate`  |

3. **Install dependencies**

```python
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm shap jupyter
```

### Run

> **⚠️ A Note on `jupyter execute`:** While `jupyter execute jobchange.ipynb` will run the code cells headlessly in your terminal, it will **not** display the interactive visualizations (like Seaborn plots or SHAP graphs), nor will it save the generated outputs into the notebook file.

Because this project relies heavily on visual EDA and SHAP summaries, it is **highly recommended** to open the notebook in the Jupyter web interface or an IDE.

#### Option 1: Jupyter Notebook (Classic Web Interface)

```bash
jupyter notebook jobchange.ipynb
```

#### Option 2: JupyterLab (Modern Web Interface)

```bash
jupyter lab jobchange.ipynb
```

#### Option 3: VS Code (IDE)

If you prefer using an IDE, simply open the `jobchange.ipynb` file in [Visual Studio Code](https://code.visualstudio.com) with the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) installed. Select your Python kernel accordingly, and run the cells interactively.

### Export Model

If you would like to use the model for another dataset, you can export existing model inside this Jupyter Notebook using code snippet below:

```python
import joblib

joblib.dump(<replace with model accordingly>, '<replaceMeWithModelName>.joblib')
```

Existing options include: `rf`, `rf_balanced`, `log_reg`, `log_reg_balanced`, `lgbm`, `lgbm_balanced`

## Optimizations

I haven't done the pipeline for `aug_test.csv`, feel free to do it on your own, to test the model more.
