# [Python: Working with Predictive Analytics](https://www.linkedin.com/learning/python-working-with-predictive-analytics-25321500/predict-data-in-python?u=69919578)

## 1. Data Preprocessing

### 1.1 Differentiate data types
- Data can be either numerical or categorical.
- Numerical data can be expressed as interval or ratio.
- Categorical data can be nominal or ordinal.
- **Nominal scale**: only compare if data is equal or not equal, cannot order, add, subtract, multiply, or divide.
- **Ordinal scale**: can order values, but cannot add, subtract, multiply, or divide.
- **Interval scale**: can compare, order, add, or subtract values, but lacks a true zero point.
- **Ratio scale**: has a true zero point, supports all mathematical operations.
- Prediction models cannot process categorical data; they need numbers.
- Need to convert categorical data into numerical data for predictive analytics.

### 1.2 Python libraries and data import
- Python libraries are collections of functions and methods that save time.
- Libraries used: pandas, NumPy, scikit-learn, Matplotlib, Seaborn.
- Install libraries with `pip install library_name`.
- Import libraries using the import statement, e.g., `import pandas as pd`.
- Load dataset with `pd.read_csv('insurance.csv')`.
- Display first X rows with `print(data.head(X))`.

### 1.3 Handling missing values
- In Python, missing values are represented as NaN.
- Prediction methods cannot work with missing data.
- Three main strategies: drop the column, drop the rows, or fill them in.
- Check missing values with `data.isnull().sum()`.
- Drop column: `data.drop('Bmi', axis=1, inplace=True)`.
- Drop rows: `data.dropna(inplace=True)`.
- Fill missing values with mean using `SimpleImputer(strategy='mean')`.

### 1.4 Convert categorical data into numbers
- Prediction models only accept numerical data.
- Two ways: label encoding (for two distinct values), one hot encoding (for three or more).
- **Label encoding**
  - example: replace yes/no with 1/0
  - Use `LabelEncoder` from sklearn
  - Use `factorize` from Pandas
- **One hot encoding** 
  - example: add new columns for each category, use 1/0 for presence
  - Use `OneHotEncoder` from sklearn.
  - Use `get_dummies` from Pandas
- Use label encoding for binary features, one hot encoding for multi-class features.

### 1.5 Divide the data into test and train
- Divide data into train and test datasets.
- Train dataset: used to train the model.
- Test dataset: used to evaluate model performance on unseen data.
- Common split: two thirds for training, one third for testing.
- Combine numerical features with encoded categorical features.
- Use `train_test_split` from sklearn.
- Display shapes of resulting datasets with `.shape`.

### 1.6 Feature scaling
- Apply feature scaling to prevent features with larger magnitudes from dominating the model.
- Two common methods: normalization (MinMax scaling) and standardization.
- **Normalization**
  - using `MinMaxScaler` from sklearn, scale to target range via `feature_range`.
  - default range: [0,1]  
  - you can specify different range, e.g., [0, 10]
    - example: scaler = MinMaxScaler(feature_range=(0, 10))
- **Standardization**
  - using `StandardScaler` from sklearn
  - transform the data to have a mean of 0 and a standard deviation of 1
- Scaling is commonly applied to X, not Y.
- Feature scaling ensures all variables are treated equally in the model.

---

## 2. Predictive Models

### 2.1 Introduction to predictive models
- Prediction uses patterns in past data to estimate unknown outcomes.
- Machine learning models: supervised, unsupervised, reinforcement learning.
- Supervised learning: regression (numerical outputs), classification (categorical outputs).
- Unsupervised learning: clustering, association.
- Focus on regression models: linear regression, polynomial regression, support vector regression, decision tree regression, random forest regression.

### 2.2 Linear regression
- Simple linear regression: explain dependent variable Y with independent variable X.
- Error is the distance between data points and the mean or regression line; square and sum the error.
- R squared = 1 - SSR/SST, shows proportion of total variation explained by the model.
- Multiple linear regression: include more than one independent variable.
- Use `LinearRegression` from sklearn.
- Fit model: `lr.fit(X_train, y_train)`.
- Predict: `lr.predict(X_train)`, `lr.predict(X_test)`.
- Print coefficients, intercept, R-squared scores.

### 2.3 Polynomial regression
- Data may not have a linear relationship; use polynomial regression.
- Add powers to each variable as new features.
- Use `PolynomialFeatures` from sklearn.
- Fit model on polynomial features.
- Degree affects underfitting/overfitting.
- Predict on both training and test datasets.

### 2.4 Support Vector Regression (SVR)
- SVR creates an optimal margin to separate data points.
- Hyperplane separates classes; support vectors are closest points.
- Use kernel tricks for non-linear data: linear, RBF, poly, exponential.
- SVR is sensitive to outliers; standardization is important.
- Use `SVR` from sklearn.
- Fit: `svr.fit(X_train, y_train)`.
- Predict: `svr.predict(X_train)`, `svr.predict(X_test)`.

### 2.5 Decision tree regression
- Decision trees: nodes represent features, branches are decisions, leaves are outcomes.
- Decision tree algorithms are similar to human decision making.
- Prone to overfitting, especially with small datasets.
- Use `DecisionTreeRegressor` from sklearn.
- Fit: `dt.fit(X_train, y_train)`.
- Predict: `dt.predict(X_train)`, `dt.predict(X_test)`.

### 2.6 Random forest regression
- Random forest: multiple decision trees, ensemble learning.
- Bagging: subdivide data into smaller samples, create multiple trees.
- Aggregation: take mean for regression.
- Use `RandomForestRegressor` from sklearn.
- Parameters: n_estimators, criterion, random_state, n_jobs.
- Fit: `forest.fit(X_train, y_train)`.
- Predict: `forest.predict(X_train)`, `forest.predict(X_test)`.
- Reduces overfitting compared to single decision tree.

### 2.7 Evaluation of predictive models
- R-squared measures regression model success.
- Linear regression: advantage with linear relationship, disadvantage otherwise.
- Polynomial regression: strong for non-linear, degree is key.
- SVR: good with different kernels, sensitive to outliers if not scaled.
- Decision tree: no scaling needed, intuitive, prone to overfitting.
- Random forest: reduces overfitting, less interpretable.
- Combine R-squared with visualization, domain knowledge, and further tests.

### 2.8 Hyperparameter optimization
- Hyperparameter optimization: find ideal set of parameters for a prediction algorithm.
- GridSearch: make a grid of the search space, evaluate each parameter setting.
- Other methods: random search, bayesian optimization, gradient-based, evolutionary, population-based.
- Use `GridSearchCV` from sklearn.
- Fit: `svr.fit(X_train, y_train)` with GridSearchCV.
- Evaluate improved performance.
