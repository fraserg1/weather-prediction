# weather-prediction
An ML Model developed to predict the Weather based on meteorological data

## Business Problem

The objective of this project is to build a machine learning model that predicts weather types (rain, drizzle, fog, sun, and snow) based on meteorological measurements including precipitation, wind speed, and temperature data. This is a multi-class classification problem with imbalanced classes.

**Key Business Question:** Can we accurately predict weather conditions using weather measurements to help in weather forecasting applications?

## Repository Structure

```
project_1__Weather_Prediction.ipynb
├── 1. Imports
├── 2. EDA (Exploratory Data Analysis)
│   ├── Data Loading
│   ├── Missing Value Analysis
│   ├── Weather Type Distribution
│   └── Statistical Analysis by Weather Type
├── 3. Train/Test Set Preparation
│   ├── Feature Encoding
│   └── Data Normalization
├── 4. Model Training & Evaluation
│   └── Classification Models
└── 5. Results & Insights

seattle-weather.csv - Input dataset containing historical weather data
```

## Tools and Programming Languages

- **Language:** Python 3
- **Data Processing:** 
  - Pandas (data manipulation and analysis)
  - NumPy (numerical operations)
- **Machine Learning:**
  - scikit-learn (model training, preprocessing, metrics)
  - LabelEncoder (categorical encoding)
- **Data Normalization:** Custom normalization functions for feature scaling
- **Visualization:** Matplotlib (distribution plots and analysis)

## Key Insights & Results

### Dataset Characteristics
- **Classes:** 5 weather types (rain, drizzle, fog, sun, snow)
- **Class Balance:** Highly imbalanced dataset with uneven distribution of weather types
- **Missing Values:** No missing values detected in the dataset

### Weather Statistics
- **Precipitation:** Zero precipitation for drizzle, fog, and sun events
- **Temperature:** Snow events have significantly lower average temperatures compared to other weather types
- **Similarity:** Sun, drizzle, and fog weather events exhibit very similar weather statistics on average, which may challenge model discrimination between these classes

### Data Preprocessing
- Target variable (weather) encoded using Label Encoder
- Features normalized using min-max scaling (dividing by max value)
- Dataset prepared for supervised learning tasks

### Best Model & Performance

**Winning Model:** Random Forest Classifier with SMOTE (Synthetic Minority Over-sampling Technique)

**Key Performance Metrics:**
- **F1-Macro Score:** 58% (primary metric for multi-class imbalanced classification)
- **Method:** SMOTE was applied to balance minority weather classes in training data
- **Optimization:** GridSearchCV hyperparameter tuning with parameters including:
  - n_estimators: [100, 200]
  - max_depth: [None, 10, 20]
  - min_samples_split: [2, 5]
  - class_weight: ['balanced', 'balanced_subsample', None]
- **Comparison:** Also evaluated XGBoost model, but Random Forest with SMOTE demonstrated superior generalization

### Challenges Identified
1. **Class Imbalance:** Unequal distribution of weather types requiring careful handling
2. **Feature Similarity:** High overlap in feature values between certain weather classes (sun, drizzle, fog)
3. **Limited Discriminative Features:** Precipitation being zero for multiple classes reduces information content

### Model Considerations
- Multi-class classification requiring appropriate evaluation metrics (precision, recall, F1-score per class)
- Potential need for class weight balancing or resampling techniques
- Evaluation should account for class imbalance
