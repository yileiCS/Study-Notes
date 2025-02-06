# MACHINE LEARNING Study Notes

## Some terminologies

#### **<u>Supervised Learning**</u>：
- The model learns from labeled training data
- Common superviesd learning tasks include: **Classification** and **Regression**
- For supervised learning the training data always contains labels
  - Each training data always contains labels
  - This label is a **known output** or **target value** used to guide the model's learning process
  - Labels can be **numeric 数值型**
    - When the label is numeric, the supervised learning task is usually referred to as a **Regression** problem
  - Labels can be **categorical 类别型**
    - When the label is categorical, the supervised learning task is usually referred to as a **Classification** problem, where the model's goal is to predict a **discrete category** (离散的类别)
- Model objective：to learn the relationship between **input features and output labels**，so as to make accurate predictions on new, unseen data
- Data can be anything 数据可以是任何类型 ➡️ 监督学习可以处理各种类型的输入数据
  - Numbers (数值)：e.g. age, price, measurements
  - Categories (类别)： e.g. gender, color, country
  - Images (图像)： e.g. medical imaging, satellite imagery, photographs
  - Text (文本)： e.g. reviews, articles, social media posts
- Common supervised learning algorithms:
  - **Linear & Logistic Regression**
    - Linear Regression: used for predicting continuous numeric labels
    - Logistic Regression: used for classification problems, predicting categorical labels
  - **Decision Trees & Random Forests**
    - Decision Trees: tree-like structures used for classification and regression tasks
    - Random Forests: ensembles of multiple decision trees, used to **improve the accuracy and robustness of the model**
  - **Boosting & Bagging**
    -  Boosting: an ensemble technique that **combines multiple weak classifiers to create a strong classifier** (通过组合多个弱分类器来创建一个强分类器)
    -  Bagging: an ensemble technique that **reduces variance by creating multiple copies of the training set and training multiple models** (通过创建多个训练集的副本并训练多个模型来减少方差)
  - **Support Vector Machines**
    - abbre. SVM
    - a powerful classification algorithm
    - can be used for regression (known as Support Vector Regression, SVR)
  - **Neural Networks**
    - models inspired by the structure of the human brain and can be used for various complex classification and regression tasks

#### **<u>Unsupervised Learning**</u>
- For handling unlabeled data
- Model objective：to **discover patterns and structures** in the data
- Important in **data exploration**, **pattern recognition**, and **data preprocessing**, and is widely used in customer segmentation, image segmentation, gene data analysis, etc. (在数据探索、模式识别和数据预处理等方面非常重要，广泛应用于客户细分、图像分割、基因数据分析等领域。)
- **Clusering** (divides the samples in a dataset into several groups or clusters, with **high similarity within the same cluster** and **low similarity between different clusters**)
  - **K-means**
    - Divides the data into k clusters through **iteration** (迭代)
    - Randomly selects k points as the **initial cluster centers**, assigns each sample to the **nearest cluster center**, and then **recalculates the center** of each cluster **until the cluster centers no longer change** or the preset number of iterations is reached
  - **DBScan**
    - A **density-based** clustering algorithm (基于密度的聚类算法)
    - Can discover any shape and identify noise points by **examining the neighborhood of samples to determine the cluster boundaries** (能发现任意形状的簇，识别噪声点，通过检查样本的邻域来确定簇的边界)
- **Anomaly Detection** (异常检测): Identifies samples in the dataset that significantly deviate from other data points, which are called **anomalies** (异常点) or **outliers** (离群点)
  - One class SVM
    - A variant of SVM that learns the boundary of normal data to **identify anomalies outside the boundary**. (支持向量机的变体，通过学习正常数据的边界，识别边界之外的异常点) 
    - It **maximizes the distance between normal data points and the boundary**, reducing the impact of anomalies (最大化正常数据点与边界之间的距离，减少异常点的影响)
- **Dimensionality Reduction** (Reduce the dimensionality of the data while **preserving the useful information in the data**, which is particularly important when dealing with **high-dimensional** data)
  - **PCA: Principal component analysis** (主成分分析)
    - Projects the original data onto a **new coordinate system** through **linear transformation** (通过线性变换将原始数据投影到新的坐标系中)
    - **Maximizes variance** in the data (方差最大化)
    - Calculates the **covarianze matrix** of the data (计算数据的协方差矩阵)
    - Selects the **eigenvectors with the largest variance** as the principal components (选择方差最大的几个特征向量作为主成分)


#### **<u>Semi or Self Supervised Learning</u>**
- Partially labelled training dataset
  - ∵ acquiring a large amount of labeled data is costly and time-consuming, while unlabeled data is relatively easy to obtain
- Usually combines supervised and unsupervised algorithm
- Uses labeled data to train the model and unlabeled data to discover the structure and patterns in the data
- Useful in fields such as: image recognition, natural language processing, and speech recognition
- Significantly improve the performance of the model, especially when labeled data is scarce (在标记数据稀缺的情况下，提升模型的性能)

#### **<u>Batch Learning</u>**
- A large mount of data is available at once
- Train offline and then use in production
- Takes a lot of resources and memory
- Suitable for scenarios where large amounts of data are available and real-time model updates are not required

#### **<u>Online Learning</u>**
- System trained incrementally
  - Online learning models process a small portion of data at a time and update the model gradually, instead of training on all data at once
- System continues learning during production
  - important in rapidly changing environments
  - Models can still receive and learning from new data in real-world applications
  - crucial for environments that require rapid adaptation to changes, e.g. financial markets, social media trends
- Can be used for huge datasets that cannot fit in memory at once (适用于无法一次性加载到内存的大型数据集)
  - Online learning processes data in batches, **effectively managing memory usage** and **avoiding overload**
- Need to control how fast they learn
  - Balance between learning changes quickly and forgetting old data
- Need careful monitoring of performance
  - Since models are constantly updating, their performance needs to be continuously monitored to ensure accuracy and reliability
- Suitable for handling large-scale, dynamically changing data


#### **<u>Instance Based Learning</u>**
- Also known as lazy learning or memory-based learning
- Makes predictions or classifications based on the **similarity between new instances and the training examples**
- Instead of learning a general model from the training data, instance-based learning stores the training instances and uses them directly for **inference** when new instances are encountered
  - <div style="color: grey">the training data serves as the model itself</div>
  - <div style="color: grey">the core idea: similar instances should have similar outputs or labels
  - <div style="color: grey">when a new instance is presented, the algorithm searches for the most similar instances in the training data and used their labels or values to make predictions for the new instance
- 【Key characteristics and steps involved】
  - **Instance storage**
    - The training instances, comprising feature vectors and associated labels or values, are stored in memory
    - This storage enables efficient retrieval and comparison during the prediction phase
  - **Similarity measure**
    - A similarity measure or distance metric is defined to quantify the similarity between instances
    - Common distance metrics include Euclidean distance, Manhattan distance, or cosine similarity, depending on the type of data and the problem at hand
  - **Nearest neighbor search**
    - When a new instance is presented, the algorithm searches for the nearest neighbors in the training data based on the **defined similarity measure**
    - The number of neighbors to consider, known as ***k***, is a parameter that can be set based on the problem requirements
  - **Prediction or classification**
    - Once the nearest neighbors are identified, the algorithm **assigns a prediction or label to the new instance** based on the labels or values of the nearest neighbors
    - This can involve various techniques, such as majority voting for classification tasks or weighted averaging for regression tasks**
  - **Adaptation to local data**
    - Instance-based learning allows for **adaptation to local patterns** in the data. 
    - As the training instances are stored, the algorithm can **adjust predictions** based on the distribution and characteristics of the nearest neighbors
- 【Advantages】
  - the ability to handle complex and non-linear relationships
  - the flexibility to adapt to changing data distributions
  - the potential for incremental learning (增量学习)
  - particularly useful in domains where the underlying function or decision boundaries are unknown or difficult to model explicitly (底层函数或决策边界未知或难以显式建模)
- 【Limitations】
  - can be computationally expensive <div style="color: grey"> (especially when dealing with large datasets, as it requires searching through the entire training data for each prediction)
  - is sensitive to noisy or irrelevant features, and it may struggle with high-dimensional data

#### **<u>Model Based Learning</u>**
- Also known as eager learning
- build a model from the known data
- Use the model to make **predictions** for unknown data
- When training the model usually define a **performance measure** of how good or bad it is
- If the performance is similar, we usually prefer the simple model
- This approach focuses more on **extracting general patterns or rules** from the data rather than directly using known data for prediction
- 【primarily implemented through the following steps】
  - **Building the model**: Learning from known data to construct a model, such as linear regression or decision trees
  - **Applying the model**: Using the constructed model to make predictions on unknown data
  - **Evaluating the model**: Assessing the model's performance by defining metrics like accuracy, recall, F1 score, mean squared error, etc.


---
## Reasons to use ML
- Existing solutions require extensive fine tuning: 现有解决方案需要广泛的微调
  - <div style="color: grey;">传统解决方案可能需要大量手动调整和优化才能达到满意的性能。机器学习算法能自动调整参数，减少人工干预的需求。</div>
- Problem too complex for traditional approach: 问题过于复杂，传统方法难以应对
  - <div style="color: grey;">当问题涉及大量变量、非线性关系或复杂模型时，机器学习算法，特别是深度学习，能处理这些复杂性，自动学习数据中的模式</div>
- Fluctuating Environments: 环境波动
  - <div style="color: grey;">在环境不断变化的情况下，传统静态模型可能无法适应新的条件。机器学习模型能从新的数据中学习，适应环境变化，从而保持预测的准确性</div>
    <div style="color: grey;">
    - 适应环境变化：通常指模型能随时间的推移和数据更新而调整其预测或决策，以应对外部条件或数据分布的变化。这种能力被称为模型的泛化能力或在线学习能力
    - 环境：指多种不同的外部条件或数据特征，具体包括但不限于：数据分布变化；概念漂移；非平稳性；外部事件的影响
    - 为了适应这些环境变化，机器学习模型可能需要：Online Learning在线学习; Incremental Learning增量学习; Transfer Learning迁移学习; Multi-task Learning多任务学习
- Gain insight from large amounts of data: 从大量数据中获得洞察
  - <div style="color: grey;">机器学习算法能处理/分析大量数据，从中提取有用的信息/洞察。这对于理解复杂现象、发现隐藏的关联和趋势非常有用。</div>

---

## Challenges for ML
### What affects the performance of a machine learning algorithm?
| **Reason**              | **Specific**                                                                                                                                                    |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Bad data                | - not enough data </br> - not representative data </br> - poor quality </br> - poor features                                                                    |
| Not enough data         | - unreasonable effectiveness of data </br> - you never have enough data! (mroe data, mroe fine grained problems) </br> - good ML more important with small data |
| Data not representative |                                                                                                                                                                 |
| Data quality            | - bad quality data </br> - missing data                                                                                                                         |
| Bad algorithms          | - over fitting </br> - under fitting                                                                                                                            |

### **<u>Correct Features</u>**
- **Feature Selection 特征选择**
  - Features must be relevant
    - 相关性高的特征通常能更好地帮助模型进行预测
  - Must have enough features
    - 确保选择的特征数量足够，以覆盖数据的多样性和复杂性
    - 特征过少可能导致模型**欠拟合** (Under-fitting)，无法捕捉数据中的重要模式
  - Not too many irrelevant features
    - 不相关的特征无助于模型预测，还可能增加模型复杂度，导致**过拟合** (Over-fitting)
- **Feature Engineering 特征工程** 
  ➡️ 通过创建新的特征或对现有特征进行转换来提高模型性能的过程
  - Create new features
    - 根据现在数据生成新特征，这些新特征可能更好地捕捉数据中的模式。
  - Combine, split or transform
    - Combine (组合): 将多个特征组合成一个新特征。e.g.将两个数值特征相加或相乘
    - Split (分割): 将一个特征分割成多个特征。e.g.将一个包含多个信息的字符串特征分割成多个数值特征
    - Transform: 对特征进行数学转换。e.g.对数值特征取对数、平方等，以减少特征的偏态或非线性关系
  
- 通过特征选择和特征工程，可以提高模型的预测性能，减少过拟合和欠拟合的风险，并使模型更加简洁和易于解释。

---
### What can machine learning use for?
- Forecasting future revenue based on performance metrics: 基于绩效指标预测未来收入
- Calculating online insurance quotes: 计算在线保险报价
- Detecting fraudulent bank transactions: 检测欺诈性银行交易
- Segmenting clients for marketing purposes: 为营销目的进行客户细分
- Detecting tumours in medical scans: 在医学扫描中检测肿瘤
- Spam filters: 垃圾邮件过滤器
- Recommendation algorithms: 推荐算法

---
### **<u>Test Data</u>**
- Must test on unseen data before deploying algorithm
- Split data into **training** (often 70-80% ) and **test** (often 20-30% ) sets
- Don't touch test set until final algorithm created
  - <div style="color: grey; ">This helps prevent overfitting and ensures that the test set provides an unbiased evaluation of the model's performance</div>
- Use just once to measure performance of final algorithm
  - Repeated use of the test set for tuning or validation can lead to overfitting and an inaccurate assessment of the model's **generalization ability**

How do we know how the model will perform on new data if we can’t test on the test data?
Reply on **validation techniques** during development, here's how it works:
- **Cross-Validation** (e.g., k-fold):
  - Split the training data into ***k*** subsets. Train the model ***k*** times, each time using ***k*** subsets for training and ***1*** subset for validation.
  -  This provides an average performance metric (e.g., accuracy, F1-score) that estimates how the model generalizes to unseen data.
- **Hold-Out Validation Set**
  - Divide the data into **training**, **validation**, and **test** sets
  - Use the validation set to **tune hyperparameters** and **monitor performance** during training
  - This mimics testing on unseen data without touching the final test set. (模拟了在不接触最终测试集的情况下对看不见的数据进行测试)
- Why Avoid the Test Set Early?
  - The test set acts as a "final exam" to evaluate the model’s **true generalization** (真实泛化能力)
  - Using it prematurely (e.g., for hyperparameter tuning) risks overfitting to the test data, inflating performance estimates (可能会过度拟合测试数据，从而夸大性能估计)

---
#### **<u>Validation data</u>** 验证数据
- Need a measure of how model will perform on unseen data before we use the test data
  - <div style="color: grey">验证集用于调整模型的参数和评估模型的性能
  - <div style="color: grey">测试集用于最终评估模型的泛化能力
- Split data again
  - Corss-validation or validation set 交叉验证或验证集
    - <div style="color: grey">将训练集分成多个小的子集，每次用一个子集作为验证集，其余子集作为训练集
    - <div style="color: grey">通过多次重复该过程以更全面地评估模型的性能
    - <div style="color: grey">从训练集中划分出一部分数据作为验证集，用于调整模型的超参数和评估模型的性能
- Use multiple times to tune algorithm 调整算法
  - The algorithm starts to learn this data too

Key Takeaway:
Validation techniques (cross-validation, hold-out validation) **provide reliable estimates** of model performance on new data, ensuring the test set remains pristine for unbiased evaluation. <div style="color: grey; ">验证技术（交叉验证、保持验证）提供了对新数据模型性能的可靠估计，确保测试集保持原始状态，以便进行无偏评估</div>

**There are 3 properties necessary for a good training and validation strategy:**
- Train the model on a large proportion of the dataset. Otherwise we’ll fail to read and recognise underlying trends in the data, resulting in underfitting. 在很大一部分数据集上训练模型。否则，我们将无法读取和识别数据中的潜在趋势，从而导致拟合不足。
- Need a good number of validation data points or we might fail to detect overfitting. 需要大量的验证数据点，否则我们可能无法检测到过拟合。
- Iterate on the training and validation process multiple times, using various training and validation dataset distributions, to be confident in validating model effectiveness properly. 使用各种训练和验证数据集分布多次迭代训练和验证过程，以确保正确验证模型的有效性。

**K-fold cross validation is a method that addresses all three**
  - Choosing K is a tradeoff; 5 and 10 are commonly used. 选择K是一种权衡；5和10是常用的
  - Disadvantage: the increased computational cost (计算成本较高，因为需进行K次训练和验证，特别是在数据集大或模型复杂的情况下)

#### **<u>K-fold cross-validation</u>  K折交叉验证** 
- It helps ensure that the model generalizes well to unseen data by using different portions of the dataset for training and testing in multiple iterations. 
- Randomly split your entire dataset into ***K*** ”folds” 将整个数据集随机拆分为K个“折叠”
- For each fold in your dataset, build your model on ***K–1*** folds of the dataset 对于数据集中的每个折叠，在数据集的K-1个折叠上构建模型
- Then, test the model to check the effectiveness for ***Kth*** fold 测试模型以检查第K次折叠的有效性
- Record the error you see on each of the predictions 记录您在每个预测中看到的错误
- Repeat this until each of the folds has served as the test set 重复此操作，直到每个fold都作为测试集
- The ***average of K recorded errors*** is called the ***cross-validation error*** and will serve as your **performance metric** for the model K个记录错误的平均值称为交叉验证错误，将作为模型的性能指标
  
【K-Fold Cross-Validation vs. Train-Test Split】
- Train-Test Split (Image by Vinod Chugani)
<img src="pic_ML/test_split_illustration.png" style="width: 100%; max-width: 90%" />
- K-Fold Cross-Validation (Image by Vinod Chugani)
<img src="pic_ML/5-Fold_Cross-Validation.png" style="width: 100%; max-width: 90%" />

[A Comprehensive Guide to K-Fold Cross Validation](https://www.datacamp.com/tutorial/k-fold-cross-validation)


None of the models produced in cross validation are the final model.
- <div style="color: grey">交叉验证中的模型不是最终模型，因为它们只基于部分数据训练，无法充分利用所有信息
Use the best hyper-parameters from CV
- <div style="color: grey">通过交叉验证，我们可以选择最佳的超参数，比如正则化强度或树的深度。这些超参数是在多个数据子集上表现一致的组合
- retrain on the whole training set to get your final model <div style="color: grey">确定最佳超参数后，使用这些参数在完整训练集上重新训练模型，以获得最终模型


#### **<u>Fine tuning models</u> 模型微调** 
Fine-tuning involves the process of adjusting the hyperparameters of a model to improve its performance.
- **Parameters** are learnt by the model <div style="color: grey">参数是模型在训练过程中学习到的变量。这些是模型的内部变量，通过调整来最小化损失函数
- Hyper Parameters are set by user <div style="color: grey">超参数是用户在训练过程开始之前设置的变量。这些是控制学习过程和模型结构的外部变量</div>
  - hyperparameters are not learned from the data
  - hyperparameters are the setting that are manually specified before the training process begins
  - hyperparameters control the learning process and the structure of the model
  - examples of hyperparameters: gradient descent, the number of layers in a neural network, or the max depth of a decision tree.
- Fine-tune algorithms by finding a good set of hyper-parameters
- 模型微调方法
  - 网格搜索: 系统地尝试指定范围内的所有可能的超参数组合。虽然计算成本高，但可以确保找到最佳组合。
  - 随机搜索: 从指定分布中随机采样超参数组合。计算成本较低，通常比网格搜索更高效地找到好的超参数
  - 贝叶斯优化: 使用概率模型来指导搜索最优超参数。在高维超参数空间中比网格和随机搜索更高效
  - 交叉验证: 通常与超参数调优结合使用，以确保模型性能在不同数据子集上公平且一致地评估。
- 模型微调步骤
  - 定义超参数空间: 确定要调优的超参数，并为每个超参数指定值的范围
  - 选择调优方法: 选择搜索超参数空间的方法（例如，网格搜索、随机搜索、贝叶斯优化）
  - 评估模型性能: 使用交叉验证评估每组超参数的模型性能。
  - 选择最佳超参数: 选择在验证集上表现最佳的超参数。
  - 重新训练模型: 使用最佳超参数在完整训练集上重新训练模型，以获得最终模型。

---
### **Notation**
- <img src="pic_ML/Notation.png" style="width: 100%; max-width: 50%" />
- Columns contain features / variables
  - <img src="pic_ML/features.png" style="width: 40%; max-width: 50%" />
- Rows contain observations
  - <img src="pic_ML/Observations.png" style="width: 100%; max-width: 50%" />

**矩阵**
- 定义: 矩阵用大写加粗字母表示，例如 A,B,C,X,…。
- 用途: 这些符号用于在数学表达式和方程中表示矩阵。矩阵是线性代数中的基本数据结构，在数据挖掘和机器学习中广泛用于表示数据、特征和变换。

**矩阵元素**
- 定义: 矩阵的元素用相应的小写字母加上两个下标表示，例如 a<sub>mq</sub>, b<sub>rs</sub>, c<sub>tu</sub>, x<sub>ij</sub> ,…
- 用途: 这些下标表示元素在矩阵中的位置。例如，a<sub>mq</sub> 表示矩阵 A 中第 m 行第 q 列的元素。这种表示法在进行矩阵运算或访问特定数据点时非常重要。

**数据矩阵或特征矩阵**
- 定义: 矩阵 X 通常用于表示数据矩阵或特征矩阵。
- 用途: 在数据挖掘和机器学习中，数据矩阵 X 通常包含数据点的特征值。矩阵 X 的每一行表示一个数据点，每一列表示一个特征。例如，x<sub>ij</sub> 是矩阵 X 中第 i 行第 j 列的元素，表示第 i 个数据点的第 j 个特征值。

- We use lower case bold letters for vectors, ***y***, ***x***
- A vector is a matrix with only
  - one row (a row vector) or,
  - one column (a column vector)
- If not specified, a vector will be a column vector
- Use italic letters for scalar values(标量值), ***i***, ***y***, ***p***
- Transpose operator(转置算子): superscript ***T***（上标T）
  - <img src="pic_ML/transpose_operator.png" style="width: 40%; max-width: 50%" />

### **Supervised Learning Notation**
***y*** = ***f*** (***X***,***θ***) + ***ϵ***
***y***: outcome; response; label; dimensions ***n × 1***
***f***: a function
***X***: data or feature matrix, dimensions ***n × p***
***θ***: a set of “parameters”
***ϵ***: a vector of “errors” or “noise” dimensions ***n × 1***

### Inputs to training are:
- ***X*** (a feature matrix) and,
- ***y*** the values we want to predict

### During learning:
- try to find a function ***f*** and parameters ***θ***
- that give results close to ***y*** when applied to ***X*** 
- <img src="pic_ML/observation.png" style="width: 60%; max-width: 60%" />

---
### **ML Project Structure** 

#### **Steps in an ML Project** 
- Look at the big picture
  - Objective: Understand the overall goals and context of the project
  - Activities: Define the problem statement, identify the business objectives, and set clear success metrics.
- Organise the data
- Data exploration
  - Objective: Gain insights into the data and understand its characteristics
  - Activities: Perform exploratory data analysis (EDA), visualize data distributions, and identify patterns and anomalies.
- Data wrangling (数据整理)
  - Objective: Clean and preprocess the data to make it suitable for modelling
  - Activities: Handle missing values, remove duplicates, normalize or standardize data (对数据进行归一化或标准化), and perform feature engineering.
- Select and train a model
- Fine-tune model
  - Objective: Optimize the model's performance by adjusting hyperparameters
  - Activities: Use techniques like grid search or random search to find the best hyperparameters, and validate the model's performance using the validation set
- Present your solution
  - Objective: Communicate the model's performance and insights to stakeholders
  - Activities: Prepare a presentation or report that explains the model's performance, key insights, and business implication
- Launch, monitor and maintain model
  - Objective: Deploy the model into production and ensure it continues to perform well over time
  - Activities: Deploy the model, set up monitoring systems to track its performance, and perform regular maintenance and updates as needed

#### **Frame the solution** 
- Supervised / Unsupervised
  - 监督学习：使用带标签的数据训练模型，每个样本都有对应的输出标签，目标是学习输入特征与输出标签之间的映射关系，进行准确预测。常见任务包括分类和回归。
  - 无监督学习：使用没标签的数据，目标是发现数据中的结构和模式，常见任务包括聚类和降维。
- online / offline
  - 在线学习：模型在接收到新数据时立即更新，适用于数据流不断更新的场景，需要处理数据的顺序性和实时性。
  - 离线学习：模型在训练阶段使用整个数据集进行学习，训练完成后不再更新，适用于数据集相对静态的场景，便于管理和复现。
- classification / regression
  - 分类：预测离散的类别标签，输出是有限的类别集合。
  - 回归：预测连续的数值，输出是一个连续的数值范围。
- How will solution be used
  - 确定模型的类型和学习方式后，考虑解决方案的具体应用场景，包括模型的部署方式、用户交互方式以及模型的维护和更新策略。
