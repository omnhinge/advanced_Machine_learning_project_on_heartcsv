
# 🫀 Heart Disease Prediction — Advanced Machine Learning Project

This project focuses on predicting the presence of **heart disease** using machine learning techniques.  
It walks through everything from **EDA → preprocessing → feature scaling → model training → evaluation**, all built in Python using Jupyter Notebook / Google Colab.

---

## 🚩 About the Project
Cardiovascular disease is one of the leading causes of death worldwide.  
The goal here is to use clinical parameters (like age, cholesterol, resting BP, etc.) to build a model that can **classify whether a patient has heart disease**.

The notebook includes:
- End-to-end data cleaning and preprocessing  
- Handling missing values and categorical features  
- Applying **Z-score scaling (StandardScaler)**  
- Model training & hyperparameter tuning using `RandomizedSearchCV`  
- Evaluation using metrics like R², RMSE, and accuracy  

---

## 📂 Dataset
Dataset used — **`heart.csv`**

| Column | Meaning |
|--------|----------|
| age | Patient age |
| sex | Gender (1 = male, 0 = female) |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol |
| fbs | Fasting blood sugar |
| restecg | Resting electrocardiographic results |
| thalach | Max heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression |
| slope | Slope of ST segment |
| ca | Number of major vessels |
| thal | Thalassemia |
| target | 1 = heart disease present, 0 = not present |

---

## 🧠 What’s Inside
**Notebook name:** `file.ipynb`  
Here’s the rough flow:

1. **Load and Inspect Data**
   - Read dataset, check shape, dtypes, and missing values.

2. **Data Cleaning**
   - Impute missing values  
   - Encode categorical columns  
   - Basic outlier and skewness treatment  

3. **Feature Scaling**
   - Apply **Z-score (StandardScaler)** for numeric features  

4. **Modeling**
   - Build and tune **RandomForestRegressor**  
   - Use `RandomizedSearchCV` for parameter optimization  

5. **Evaluation**
   - Check R², MSE, RMSE  
   - Compare multiple models (Logistic Regression, RF, etc.)

6. **Result**
   - Final model trained with best params on the full dataset  

---

## 🧰 Tech Stack

| Category | Tools |
|-----------|--------|
| Language | Python |
| Notebook | Jupyter / Colab |
| Libraries | pandas, numpy, seaborn, matplotlib, scikit-learn |
| ML Models | RandomForest, LogisticRegression, GradientBoosting |
| Optimization | RandomizedSearchCV |
| Scaling | StandardScaler |

---

## ⚙️ How to Run

### Step 1: Clone the Repo
```bash
git clone https://github.com/omnhinge/advanced_Machine_learning_project_on_heartcsv.git
cd advanced_Machine_learning_project_on_heartcsv
```

### Step 2: Install Requirements
```bash
pip install -r requirements.txt
```

### Step 3: Run the Notebook
- Open `file.ipynb` in **Jupyter Notebook** or **Google Colab**  
- Make sure your `heart.csv` is in the same directory  
- Run all cells in order  

Or open directly in Colab:  
👉 [**Open in Colab**](https://colab.research.google.com/github/omnhinge/advanced_Machine_learning_project_on_heartcsv/blob/main/file.ipynb)

---

## 📊 Model Results
| Model | R² | RMSE | Comment |
|--------|----|------|----------|
| Random Forest | ~0.85 | ~3.2 | Best overall performance |
| Logistic Regression | ~0.79 | ~4.1 | Simpler, interpretable |
| Gradient Boosting | ~0.84 | ~3.3 | Competitive alternative |

*(Scores may vary slightly by random seed or split.)*

---

## 🔍 Directory Layout
```
advanced_Machine_learning_project_on_heartcsv/
│
├── file.ipynb              # Main notebook
├── heart.csv               # Dataset
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

---

## 🚀 Future Work
- Add feature importance visualization (SHAP / LIME)  
- Deploy via Streamlit for live predictions  
- Test additional ensemble models (XGBoost, CatBoost)  
- Explore ANN or hybrid stacking approach  

---

## ✨ Author
**Om N. Hinge**  
📍 Machine Learning Enthusiast  
🔗 [GitHub Profile](https://github.com/omnhinge)

If you find this useful, drop a ⭐ on the repo!  

