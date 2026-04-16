# 🌧️ Rain Prediction in Australia

**Machine Learning Project — Predicting Rainfall in Australian Cities**

---

## What This Project Is About

I've always been curious about what meteorologists actually look at
when they make a weather forecast. So when I came across this
Australian weather dataset on Kaggle, I wanted to see if machine
learning could predict whether it's going to rain tomorrow based
on today's weather conditions.

The dataset covers four years of daily weather measurements from
various locations across Australia — things like temperature,
humidity, wind speed, and pressure. The goal is simple: will it
rain tomorrow? Yes or No.

---

## The Business Problem

Weather forecasting isn't just about knowing whether to grab an
umbrella. Inaccurate forecasts cost money — farmers lose crops,
airlines reroute flights, and emergency services get caught
off guard.

If we can predict rain accurately using historical weather data,
we can:

- 🌾 Help farmers plan irrigation and harvesting
- ✈️ Support airlines with route and fuel planning
- 🏗️ Help construction companies schedule outdoor work
- 🚨 Give emergency services earlier warning of severe weather

**The question:** Can we use daily weather measurements to reliably
predict whether it will rain the next day?

---

## 🛠️ Skills & Tools

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

**What I applied:**
- 🔵 Classification (Decision Tree, ANN)
- 🟡 Clustering (K-Means)
- 🧹 Data Cleaning & Missing Value Imputation
- ⚙️ Feature Engineering (cyclical date encoding)
- 📊 Data Visualisation (heatmaps, confusion matrix, pair plots)
- 📉 Model Evaluation (accuracy, precision, recall, F1-score)

---

## The Dataset

- **Source:** Kaggle — Australian Weather Dataset
- **Size:** 62,327 rows × 23 columns
- **Coverage:** 4 years of daily weather data across multiple
  Australian cities
- **Target variable:** `RainTomorrow` (Yes / No)

**Key features used:**

| Feature | Description |
|---|---|
| MinTemp / MaxTemp | Min and max temperature (°C) |
| Rainfall | Amount of rainfall (mm) |
| Humidity9am / Humidity3pm | Humidity at 9am and 3pm (%) |
| Pressure9am / Pressure3pm | Atmospheric pressure (hPa) |
| WindGustSpeed | Speed of strongest wind gust (km/h) |
| WindGustDir | Direction of strongest wind gust |
| Cloud9am / Cloud3pm | Cloud cover (oktas) |
| RainToday | Whether it rained today |

---

## What I Did

### 🧹 Data Cleaning
The dataset had a fair amount of missing values — Sunshine was
missing for ~37,000 rows and Evaporation for ~32,000. I filled
categorical columns with the mode and numerical columns with the
median. No rows were dropped at this stage.

I also engineered cyclical features for month and day using sine
and cosine encoding this helps models understand that December
and January are close together, not far apart.

### 📊 Exploratory Data Analysis

The data is imbalanced  most days it doesn't rain.

Non rainy days significantly outnumber rainy ones, which is
something to keep in mind when evaluating model performance.

Looking at location vs rainy days, some cities are noticeably
wetter than others Brisbane and coastal areas see more rain
days than inland locations like Mildura.

The scatter plot of Temp9am vs MaxTemp shows a strong positive
relationship if the morning is warm, the afternoon tends to
be warm too. And when it does rain, the gap between MinTemp
and MaxTemp tends to be smaller.

The correlation heatmap confirmed that humidity (especially
Humidity3pm) and pressure variables are the strongest
predictors of rain.

### 🔵 Decision Tree

Trained a Decision Tree classifier with max depth of 20 and
sqrt feature selection.

| Metric | Score |
|---|---|
| Accuracy | 79.8% |
| Precision (No Rain) | 0.86 |
| Precision (Rain) | 0.52 |
| F1 (No Rain) | 0.87 |
| F1 (Rain) | 0.50 |

Humidity3pm came out as the most important feature, followed
by Humidity9am and Pressure3pm. The model struggled more with
predicting rain (class 1) than no rain (class 0) which is
expected given the class imbalance.

### 🟡 K-Means Clustering

Applied K-Means with 3 clusters on the scaled weather features.
The pair plot showed distinct groupings each cluster represents
a different type of weather day. Some clusters aligned closely
with rainy vs dry conditions, showing that weather patterns
naturally form distinct groups even without supervision.

**ANN Results:**

| Metric | Score |
|---|---|
| Accuracy | **85%** |
| Precision (No Rain) | 0.87 |
| Precision (Rain) | 0.70 |
| Recall (No Rain) | 0.95 |
| Recall (Rain) | 0.47 |
| F1 (No Rain) | 0.91 |
| F1 (Rain) | 0.56 |

The ANN outperformed the Decision Tree by 5 percentage points.
The confusion matrix shows the model is very good at predicting
no rain (75% of test cases correct) but still misses some rainy
days — a known challenge with imbalanced weather data.

---

## 📊 Model Comparison

| Model | Accuracy | Notes |
|---|---|---|
| Decision Tree | 79.8% | Fast, interpretable, good feature importance |
| K-Means | N/A | Unsupervised used for pattern discovery |
| **ANN ⭐ Best** | **85%** | Best overall accuracy, handles complexity well |

---

## 💡 What This Means

The ANN is the strongest model here. At 85% accuracy it's
genuinely useful for practical weather prediction not perfect,
but better than random and better than a simple rule based system.

The fact that humidity and pressure are the top predictors makes
meteorological sense these are exactly the variables weather
forecasters pay attention to.

The model is better at predicting dry days than rainy ones. That's
partly because dry days outnumber rainy days in the dataset. In a
real deployment you'd want to address this imbalance maybe with
SMOTE oversampling or class weighting.

---

## 🔭 What I'd Do Differently / Next Steps

Honest limitations of this project:

- Class imbalance wasn't addressed the model sees far more
  dry days than rainy ones which biases predictions
- Some features like Sunshine and Evaporation had huge amounts
  of missing data better imputation or removal might help
- Only tested on Australian data would need retraining for
  other regions

Things I'd build next:
- [ ] Address class imbalance with SMOTE or class weights
- [ ] Try Random Forest and XGBoost for comparison
- [ ] Add cross validation instead of single train/test split
- [ ] Build a simple web app where you input today's weather
      and get a rain prediction
- [ ] Test on real time weather API data

---

## 📁 Files in This Repo

```
├── DOCUMENT.pdf
├── Dataset1.csv
├── R CODE.ipynb
└── README.md
```

---
