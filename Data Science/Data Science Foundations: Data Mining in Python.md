# [Data Science Foundations: Data Mining in Python](https://www.linkedin.com/learning/data-science-foundations-data-mining-in-python/python-for-data-mining?resume=false&u=69919578)

## 1. Preliminaries

### 1.1 Tools for data mining
- The two most common tools for data mining are Python and R.
- Python is currently the most popular language for Data Science and Machine Learning.
- Python is a general purpose language, easy to learn, and well adapted for data.
- R is a programming language specifically developed for data analysis.
- Other options: SPSS, SAS, jamovi, JASP, Tableau.
- SQL (Structured Query Language) is essential for accessing databases.
- Spreadsheets like Microsoft Excel and Google Sheets are the single most common data tools in the world.
- Recommended order: start with spreadsheets, then applications like jamovi or SPSS, then programming languages.
- KDnuggets survey: Python is number one, followed by RapidMiner, R, Excel, Anaconda, SQL.
- Choose tools based on client requirements and project needs.

### 1.2 The CRISP-DM data mining model
- CRISP-DM: cross-industry standard process for data mining.
- Six phases: business understanding, data understanding, data preparation, modeling, evaluation, deployment.
- Business understanding: determine business objectives, situation assessment, data mining goals, project plan.
- Data understanding: collect initial data, describe, explore, verify data quality.
- Data preparation: describe dataset, select data, clean, construct, integrate, format data.
- Modeling: select modeling technique, check assumptions, generate test design, build and assess model.
- Evaluation: evaluate results, review process, determine next steps.
- Deployment: apply conclusions, plan deployment, monitoring, maintenance, final report, review project.

### 1.3 Privacy, copyright, and bias
- Data mining takes place in a human social context.
- Privacy: GDPR (EU), CCPA (California), HIPAA (US), FERPA (US education).
- Copyright: data may be restricted or proprietary, check licenses.
- Bias: GIGO (garbage in, garbage out), check data quality, representativeness, diversity, weighting, construct validity.
- Interpretation: ensure you are measuring what you think you are measuring.

### 1.4 Validating results
- Over-fitting: model fits dataset perfectly but may not work well in practice.
- Validation: statistical model validation, external validity, generalizability.
- Assess validity: check data, fit, sample, replicate study.
- Validation methods: holdout validation (split into training and testing datasets), cross validation (k-folds).
- Holdout: train on one part, test on another, possibly use a validation dataset.
- Cross validation: split into k-folds, rotate training/testing, find best model.

---

## 2. Dimensionality Reduction

### 2.1 Dimensionality reduction overview
- High dimensionality: exponentially increasing complexity, idiosyncratic error, risk of over-fitting, interpretability issues.
- Solution: combine variables to reduce dimensionality.
- Dimensionality reduction: project data into lower dimensional space while preserving useful variability.
- Metaphor: hand's shadow (3D to 2D), scatterplot rotation.
- Main algorithms: PCA (Principal Component Analysis), LDA (Linear Discriminant Analysis), t-SNE (t-distributed Stochastic Neighbor Embedding).

### 2.2 Handwritten digits dataset
- Load libraries and dataset (CSV file or UCI repository).
- Rename columns: pixel variables as P, class variable as y.
- Restrict dataset to numbers 1, 3, and 6.
- Split into training and testing datasets using train test split.
- Visualize first 20 images in training dataset.
- Explore attribute variables (e.g., p25, p30, p45, p60) and their associations.
- Save datasets as CSV using `.to_csv()`.

### 2.3 PCA
- Principal component analysis (PCA) is the most common way to reduce dimensionality.
- Use scikit-learn (sklearn) for PCA.
- Scree plot: shows variance attributed to each factor.
- Project data onto first two principal components (from 64 to 2 dimensions).
- Color points by class (digit).
- Compare average log-likelihood between training and testing data.

### 2.4 LDA
- Linear discriminant analysis (LDA) is a classification algorithm that uses dimensionality reduction.
- Use sklearn for LDA.
- Fit model, transform data, plot transformed data.
- LDA creates maximum separation between groups.
- Accuracy: 99.63% on training data, 99.71% on testing data.

### 2.5 t-SNE
- t-SNE (t-distributed stochastic neighbor embedding) is for visualizing high dimensional data.
- Use sklearn for t-SNE.
- Specify number of components (e.g., 2), set random_state.
- Main parameter: perplexity.
- Increase perplexity to see better separation between classes.

---

## 3. Clustering

### 3.1 Clustering overview
- Clustering is about bringing order by grouping similar things together; there is no single correct answer.
- Clustering is based on the data and variables you provide, and is designed to serve practical goals.
- Data is represented in multi-dimensional space; standardize variables so each has mean 0 and standard deviation 1.
- Measure distance between points to find dense areas (clusters).
- Three common clustering methods:
  - Hierarchical clustering: arranges cases in a hierarchy (agglomerative or divisive), visualized with a dendrogram.
  - k-Means: specify number of groups (K), algorithm finds k-centroids and assigns cases to closest centroid; works best for spherical, non-overlapping clusters.
  - DBSCAN (Density Based Spatial Clustering of Applications with Noise): works on local density, can find non-convex and non-linearly separable clusters, identifies noise points.

### 3.2 Penguin dataset
- Demonstrations use the Palmer Penguins dataset.
- Install the Palmer Penguins library, load the dataset, remove unhelpful variables, rename class variable to y, drop missing data.
- Explore the data: bar plot for species counts, pair grid for variable relationships (bill length, bill depth, flipper length, body mass).
- Visualize separation between species using histograms, scatterplots, and density plots.
- Save cleaned data as CSV for future use.

### 3.3 Hierarchical clustering
- Hierarchical clustering is exploratory and shows every possible level of clustering.
- Use `scipy` and `sklearn` for clustering and visualization.
- Reduce dataset size for visualization, separate class variable.
- Use `linkage` (agglomerative) and `dendrogram` functions; common linkage methods: ward, average, single, complete.
- Dendrogram visualizes how cases are combined; interpret clusters and similarities between species.

### 3.4 K-means
- k-Means is used when you know the number of clusters (K).
- Standardize variables before clustering.
- Use `KMeans` from sklearn, specify number of clusters, random state, initialization method.
- Visualize clusters and centroids in scatter plots.
- Use silhouette score and `GridSearchCV` to find the ideal number of clusters.
- Empirical results may suggest a different number of clusters than expected (e.g., two clusters for three species).
- Clustering results depend on the data and variables provided.

### 3.5 DBSCAN
- DBSCAN stands for density-based spatial clustering of applications with noise.
- Good for clusters that are not spherical or convex, such as circles, rings, or irregular shapes.
- Builds clusters by accretion from individual neighbors; can leave certain points out as noise.
- Two main parameters: min_samples (minimum number of neighboring points for clustering) and epsilon (neighborhood radius).
- Use a graph to find the best value for epsilon (look for a knee in the curve).
- DBSCAN can find clusters of different shapes and identify noise points that do not belong to any cluster.
- In the penguins dataset example, DBSCAN found four clusters, showing its flexibility compared to k-means and hierarchical clustering.
- Useful for adapting to peculiarities of real data and finding actionable clusters.

### 3.6 Text mining and network graphs
- Text mining can analyze word pairs (n-grams, bigrams) to find connections between words.
- Use `networkx` library for network graphs.
- Tokenize text, create n-grams, remove stop words, split pairs into columns.
- Visualize frequent word pairs with graphs to understand sentence construction and ideas in the text.

---

## 4. Classification

### 4.1 Classification overview
- Clustering is unsupervised learning with unlabeled data; classification is supervised learning with labeled data.
- Classification assigns new objects to existing categories based on labeled data.
- Effectiveness is judged by accuracy (e.g., positive cases labeled as positive).
- Common methods: k-NN (k-nearest neighbors), Naive Bayes, decision trees.
- k-NN: assign new case to the most common category among its k nearest neighbors in feature space.
- Naive Bayes: uses Bayes' theorem, assumes predictors are independent, calculates posterior probabilities.
- Decision trees: series of binary splits, chooses most informative split at each point, easy to interpret, works with many data types.
- There are many more classification methods, but k-NN, Naive Bayes, and decision trees are most useful for quick application.

### 4.2 Spambase dataset
- Use Spambase dataset (spam and not spam emails, with various attributes).
- Prepare data: rename variables X0-X56, Y as class variable (0 for not spam, 1 for spam).
- Split data: 70% training, 30% testing.
- Explore class distribution and attribute variables with bar plots and colored graphs.
- Outliers and separation between classes are visible, indicating useful features for classification.
- Save prepared data for modeling.

### 4.3 KNN
- k-Nearest Neighbors (kNN) classifies a new case by looking at its k closest neighbors in multidimensional space and assigning the most common category among them.
- In Python, use scikit-learn's `KNeighborsClassifier`.
- Load and split the data into training and testing sets; variables are X0–X56, class variable is y (1 for spam, 0 for not spam).
- Train the kNN model and check mean accuracy on training data (about 87%).
- Optimize k by testing a range (e.g., 3 to 15, step 2) and plot mean cross-validation score; best accuracy at k=3.
- Use a confusion matrix to evaluate performance on test data:
  - True label vs. predicted label (spam/not spam).
  - 15% false positives (non-spam identified as spam), 26% false negatives (spam missed).
- Mean accuracy on test data: 81.17%.
- kNN is a simple, intuitive algorithm and a good starting point for classification tasks.

### 4.4 Naive Bayes
- Naive Bayes uses Bayes' theorem to calculate posterior probabilities, assuming predictors are independent (the "naive" assumption).
- In Python, use scikit-learn's `GaussianNB` for Naive Bayes classification.
- Load and split the data into training and testing sets.
- Fit the model to the training data (`fit(X_train, y_train)`).
- Training accuracy: about 82% (similar to kNN).
- Evaluate on test data using a confusion matrix:
  - 26% false positives (regular messages identified as spam), 6% false negatives (spam identified as not spam).
- Mean accuracy on test data: about 82%.
- Naive Bayes is fast, easy to implement, and effective in many situations; can adjust parameters for more balanced performance.

### 4.5 Decision Trees
- Decision trees provide a visual, interpretable method for classification.
- In Python, use scikit-learn's `DecisionTreeClassifier`.
- Load and split the data into training and testing sets.
- Fit the model, specifying criteria (e.g., 'gini' or 'entropy') and number of leaf nodes.
- Training accuracy: about 89% (higher than kNN and Naive Bayes).
- Optimize by tuning max leaf nodes and criteria, using 5-fold cross-validation; best result with Gini index and about 38 leaves.
- Visualize the tree to see decision splits (e.g., if X52 < 0.054, go left/right).
- Test on the testing data and evaluate with a confusion matrix:
  - Very few false positives and a small number of false negatives.
- Mean accuracy on test data: 91% (higher than other models).
- Decision trees are highly interpretable and often provide strong performance for classification tasks.

---

## 5. Association Analysis

### 5.1 Association analysis overview
- Association analysis (association rules mining/market basket analysis) finds "if this, then that" rules in transactional data (e.g., if a person buys X, what's the probability they buy Y?).
- Useful for commercial settings (e.g., product placement, incentives) and other domains.
- Data can be in tabular (rows/columns) or transactional (list of items per transaction) format; most algorithms prefer transactional format.
- Three main algorithms:
  - Apriori: counts itemsets, builds rules based on support, confidence, and lift; easy to interpret.
  - Eclat: similar to Apriori but more efficient for large datasets; uses equivalent class transformation.
  - FP-Growth: uses a tree structure for efficient pattern mining, especially with many transactions and few items.
- Key metrics:
  - Support: proportion of transactions containing an itemset.
  - Confidence: conditional probability of B given A.
  - Lift: confidence compared to expected frequency of B.

### 5.2 Groceries dataset
- Demonstration uses the groceries dataset (CSV, transactional format) from the R package "A Rules".
- Load with Pandas; each row is a transaction, columns are items purchased (with NaN for unused columns).
- Data can be used as-is, cleaned to list format, or converted to standard spreadsheet format depending on the algorithm.

### 5.3 Apriori
- Apriori is the most common association analysis method; implemented in Python with the `Apyori` package.
- Prepare data as a list of transactions (remove NaN, split items).
- Specify minimum support, confidence, and rule length.
- Output rules with support, confidence, lift, and sample sizes.
- Convert rules to a readable table and sort by support.
- Visualize rules with a graph: left-hand (A) and right-hand (B) items, color-coded by support.
- Example: "other vegetables" → "whole milk" has highest support.

### 5.4 Eclat
- Eclat is efficient for large datasets; implemented in Python with `pyEclat`.
- Use original transactional format (with NaN for missing items).
- Specify minimum support and combination size.
- Output support levels for item pairs, convert to readable table, and sort by support.
- Visualize rules with a graph, similar to Apriori.
- Example: "other vegetables" → "whole milk" is a frequent association.

### 5.5 FP-growth
- FP-Growth (Frequent Pattern Growth) uses frequent pattern trees for efficient mining of repeated associations.
- Implemented in Python with `mlxtend`.
- Prepare data in standard spreadsheet format using a transaction encoder (items as columns, True/False for each transaction).
- Specify minimum support and rule length.
- Output rules with support, convert to readable table, and visualize with a graph.
- FP-Growth is especially useful for large datasets with repeated patterns.
- All three algorithms help uncover actionable associations in transactional data.

---

## 6. Time-Series Mining

### 6.1 Time-series mining
- Time-series data is always moving forward, presenting unique analytical challenges.
- Key methods for time-series analysis:
  - Decomposition: separates data into trend, seasonal variation, and noise; can use additive or multiplicative models.
  - ARIMA (AutoRegressive Integrated Moving Average): combines autoregression (AR), differencing (I), and moving average (MA); can be extended to ARMA, SARIMA, SARIMAX.
  - MLP (Multilayer Perceptron): a type of feedforward neural network with input, hidden, and output layers; can model non-linearly separable data.

### 6.2 Air Passengers dataset
- Uses the Air Passengers dataset (monthly international airline passengers, 1949–1960).
- Prepare data: parse dates, set index, plot time series to observe trend and seasonality.
- Data shows overall increase and strong seasonal patterns (e.g., more travel in summer).

### 6.3 Time-Series decomposition
- Decomposition is a descriptive, visual method to break time-series into components.
- Use `seasonal_decompose` from statsmodels; specify period (e.g., 12 for monthly data).
- Additive model: trend + seasonal + residual; multiplicative model: trend * seasonal * residual.
- Multiplicative model is better for data with changing variance (heteroscedasticity).

### 6.4 ARIMA
- ARIMA is used for forecasting and modeling time-series data.
- Requires non-stationary data (changing mean/variance/covariance over time).
- Split data into training (earlier years) and testing (later years); do not use random splits.
- Test for stationarity and autocorrelation (lags).
- Use `auto_arima` to find best parameters; can result in SARIMAX model for seasonality.
- Evaluate residuals (should be close to zero, not autocorrelated); use Ljung-Box test for autocorrelation.
- Forecast future values and plot predictions with confidence intervals; ARIMA can capture both trend and seasonality.

### 6.5 MLP
- MLP (Multilayer Perceptron) is a neural network useful for time-series forecasting.
- Prepare data: reshape into matrix, split into training/testing (e.g., 80/20), standardize features.
- Fit MLP model, tune number of hidden nodes to minimize mean squared error (MSE).
- Forecast and plot results; convert predictions back to original scale.
- Can use cross-validation (e.g., 5-fold) to select best model.
- MLP can model complex, non-linear patterns and provides strong performance for time-series prediction.

---

## 7. Text Mining

### 7.1 Text mining overview
- Text data is unstructured, highly variable, and voluminous (books, news, social media).
- Key challenges: unstructured format, variability, misspellings, colloquialisms.
- Main applications:
  - Topic modeling (identifying subjects of texts)
  - Summarization (condensing content)
  - Classification (e.g., spam filtering)
- Sentiment analysis: finding emotional content (positive/negative) in text, important for online marketing.
- Sentiment analysis can be binary (positive/negative), scored (degree of sentiment), or explore word associations (word pairs).

### 7.2 Iliad dataset
- Demonstration uses "The Iliad" from Project Gutenberg (plain text format).
- Load text, drop empty lines and metadata, and prepare for analysis.

### 7.3 Sentiment analysis: Binary classification
- Binary sentiment analysis classifies words as positive or negative using a published lexicon (e.g., opinion_lexicon).
- Use NLTK for tokenization, stop word removal, and frequency analysis.
- Tokenize text, convert to lowercase, remove non-letters, split into words.
- Remove stop words, count and compare positive/negative words.
- Visualize results (e.g., bar graph of positive vs. negative word counts).
- Example: "The Iliad" has more negative than positive words, fitting for a war story.

### 7.4 Sentiment analysis: Sentiment scoring
- Sentiment scoring assigns numeric values to words (e.g., -5 to +5) using a lexicon (e.g., afinn).
- Tokenize and score words, drop neutral (zero) scores.
- Visualize sentiment distribution (bar chart of sentiment scores).
- Analyze sentiment over narrative arc by breaking text into sections (e.g., 100 lines each) and plotting average sentiment per section.
- Rolling average shows overall emotional trajectory of the text.

### 7.5 Word pairs
- Analyze word pairs (bigrams/n-grams) to find associations between words.
- Use `networkx` for network graph visualization.
- Tokenize text, create n-grams (e.g., bigrams), split pairs into columns.
- Remove stop words from either word in the pair.
- Sort and count frequent word pairs; visualize most common pairs with a network graph.
- Example: "old man", "peleus's son", "brass clad" are frequent pairs in "The Iliad".
- Word pair analysis reveals sentence structure and thematic connections in text.
