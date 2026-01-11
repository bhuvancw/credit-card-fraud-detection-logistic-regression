# Credit Card Fraud Detection - Logistic Regression

A machine learning project that detects fraudulent credit card transactions using Logistic Regression. This project demonstrates binary classification techniques with class imbalance handling and comprehensive model evaluation.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Key Features](#key-features)
- [Project Workflow](#project-workflow)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technologies Used](#technologies-used)

## 🎯 Overview

Credit card fraud is a significant problem in financial systems. This project builds a machine learning classifier to identify fraudulent transactions from legitimate ones using Logistic Regression, a foundational yet powerful classification algorithm.

The model achieves:
- **Test Accuracy:** 96.95%
- **Training Accuracy:** 94.28%
- **AUC Score:** Strong discriminative ability

## 📊 Dataset

### Dataset Description
- **Total Records:** 284,807 transactions
- **Features:** 31 (28 PCA-transformed features + Time + Amount)
- **Target Variable:** Class (0 = Legitimate, 1 = Fraud)

### Class Distribution (Original Dataset)
- Legitimate transactions: 284,315 (99.83%)
- Fraudulent transactions: 492 (0.17%)

**Class Imbalance Challenge:** The dataset is heavily imbalanced, which is addressed through balanced sampling:
- 492 fraudulent transactions
- 492 legitimate transactions (randomly sampled)
- **Balanced Dataset:** 984 total transactions for model training

### Features
- **Time:** Time elapsed (in seconds) between transactions
- **V1-V28:** Principal Component Analysis (PCA) transformed features (confidential for privacy)
- **Amount:** Transaction amount in euros
- **Class:** Target variable (0 or 1)

## 🔑 Key Features

1. **Data Preprocessing:**
   - Null value verification (no missing values found)
   - Column name normalization to lowercase
   - Class distribution analysis

2. **Class Balancing:**
   - Random sampling of legitimate transactions to match fraud count
   - Creates balanced dataset (1:1 ratio) to avoid bias

3. **Train-Test Split:**
   - 80% Training (787 samples)
   - 20% Testing (197 samples)
   - Stratified split to maintain class distribution

4. **Model Evaluation:**
   - Accuracy Score
   - ROC Curve and AUC
   - Precision-Recall Curve
   - Confusion Matrix with threshold analysis

## 🔄 Project Workflow

### 1. Data Loading & Exploration
```python
df = pd.read_csv('creditcard.csv')
df.head()  # View first records
df.info()  # Check data types and missing values
df['Class'].value_counts()  # Analyze class distribution
```

### 2. Handling Class Imbalance
- Separate legitimate and fraud transactions
- Sample 492 legitimate transactions (matching fraud count)
- Concatenate to create balanced dataset

### 3. Feature and Target Separation
```python
X = balanced_data.drop('class', axis=1)  # Features
y = balanced_data['class']  # Target
```

### 4. Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    random_state=42, 
    stratify=y, 
    test_size=0.2
)
```

### 5. Model Training
```python
model = LogisticRegression()
model.fit(X_train, y_train)
```

### 6. Model Evaluation
- Training accuracy prediction
- Test accuracy prediction
- ROC curve analysis
- Precision-Recall curve analysis
- Confusion matrix visualization

## 📈 Model Performance

### Accuracy Metrics
| Metric | Value |
|--------|-------|
| Training Accuracy | 94.28% |
| Test Accuracy | 96.95% |

### ROC Curve
- Measures trade-off between True Positive Rate and False Positive Rate
- AUC Score provides aggregate measure of classifier performance
- Higher AUC indicates better discrimination between classes

### Precision-Recall Curve
- Evaluates trade-off between precision and recall
- Important for imbalanced datasets
- Average Precision (AP) Score provided

### Confusion Matrix
- Threshold set at 0.5 for classification
- Shows True Positives, False Positives, True Negatives, False Negatives
- Helps understand specific error types

## 💻 Installation

### Prerequisites
- Python 3.7+
- pip or conda

### Required Libraries
```bash
pip install pandas numpy scikit-learn matplotlib
```

### Setup
```bash
# Clone the repository
git clone https://github.com/bhuvancw/credit-card-fraud-detection-logistic-regression.git

# Navigate to project directory
cd credit-card-fraud-detection-logistic-regression
```

## 🚀 Usage

### Running the Notebook
1. Ensure the `creditcard.csv` file is in the project directory
2. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open `Credit Card Fraud Detection - ML - Logistic Regression.ipynb`
4. Run cells sequentially (Shift + Enter)

### Interpreting Results
- **Training vs Test Accuracy:** Both scores are above 94%, indicating good generalization
- **ROC-AUC:** Visualizes model's ability to distinguish between fraud and legitimate transactions
- **Precision-Recall:** Shows model's effectiveness at different operating points
- **Confusion Matrix:** Reveals false positive and false negative rates

## 📊 Results

### Model Insights
1. **High Accuracy:** The model achieves ~97% test accuracy, demonstrating strong performance
2. **Balanced Performance:** Training and test accuracies are close, indicating no overfitting
3. **Effective Classification:** ROC and Precision-Recall curves show the model can effectively distinguish between fraud and legitimate transactions
4. **Business Impact:** Even small improvements in fraud detection can save significant financial losses

### Key Findings
- Fraudulent transactions have different feature patterns compared to legitimate ones
- The model learned meaningful patterns from balanced training data
- Threshold-based decision boundaries (0.5) provide reasonable classification

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|----------|
| **Python** | Programming language |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computing |
| **Scikit-learn** | Machine learning algorithms and metrics |
| **Matplotlib** | Data visualization |
| **Jupyter** | Interactive notebook environment |

## 📚 Learning Concepts

This project demonstrates:
- Binary classification with imbalanced data
- Train-test split and stratified sampling
- Logistic Regression fundamentals
- Model evaluation metrics (accuracy, ROC, precision-recall)
- Confusion matrix interpretation
- Data preprocessing and exploration

## 🔍 Future Enhancements

- Try ensemble methods (Random Forest, Gradient Boosting)
- Implement cross-validation for robust performance estimation
- Explore different class weights or oversampling techniques (SMOTE)
- Tune hyperparameters using grid search
- Deploy model as a REST API
- Add feature importance analysis

## 📝 Notes

- The dataset features V1-V28 are PCA-transformed, maintaining transaction privacy
- Class imbalance is handled by balanced sampling during preprocessing
- The model uses default Logistic Regression parameters without hyperparameter tuning
- Threshold of 0.5 is used for binary classification; this can be adjusted based on business requirements

## 📄 License

This project is open source and available for educational purposes.

## 👤 Author

**Bhuvan CW**
- GitHub: [@bhuvancw](https://github.com/bhuvancw)

---

**Last Updated:** January 2026

*A project demonstrating machine learning fundamentals in fraud detection.*
