# [Predictive Analytics Essential Training: Data Mining](https://www.linkedin.com/learning/predictive-analytics-essential-training-data-mining/data-mining-and-predictive-analytics-24925114?u=69919578)

## 1. What is Data Mining and Predictive Analytics

### 1.1 Introducing the essential elements
- Essential elements are a collection of requirements, sometimes hurdles, encountered in predictive analytics projects.
- Clarifying the characteristics of a true data mining project versus related areas like statistics and business intelligence.

### 1.2 Defining data mining
- Data mining is finding patterns in historical data and leveraging those patterns on current data to make future predictions.
- "Data mining is the selection and analysis of data accumulated during the normal course of doing business, to find and confirm previously unknown relationships that can produce positive and verifiable outcomes through the deployment of predictive models when applied to new data."
- The data is not new; it was collected in the running of a business.
- Not testing hypothesis, but exploring for patterns.
- Need verifiable outcomes and measurable benefit.
- Must work on new data; deployment is essential.

### 1.3 Introducing Crisp-DM
- CRISP-DM: Cross-Industry Standard Process for Data Mining.
- Six phases, 24 tasks, detailed advice.
- The essential elements are the what, CRISP-DM is the how-to, and the nine laws are the why of data mining.

---

## 2. Problem Definition

### 2.1 Beginning with a solid first step: Problem definition
- Poor problem definition is the biggest reason projects fail.
- Four essential elements of problem definition to avoid mistakes.

### 2.2 Framing the problem in terms of a micro-decision
- Data mining is about making a model to help make a specific decision that comes up frequently and has a measurable impact on the business.
- Micro decisions: specific decision about a single instance.
- "If only we knew X, we would be able to make a better decision about Y."
- Begin by working with management to frame the problem using this approach.

### 2.3 Why every model needs an effective intervention strategy
- Need an intervention strategy: the action you take for high scores that you don't take for low scores.
- Predictions occur at an individual level and are time sensitive.
- The intervention is the specific step you take when the model score indicates you need to react.

### 2.4 Evaluate a project's potential with business metrics and ROI
- Identify and define a measurable benefit.
- "To call it a data mining project, you must eventually deploy."
- Estimate how much you'll gain monetarily if predictions are accurate and interventions are successful.
- Gather appropriate business metrics to measure progress.

### 2.5 Translating business problems into data mining problems
- Analytics project management: define the boundaries of the project.
- One micro decision equals one project.
- The project is defined by a micro decision tied to an intervention strategy; the program is tied to a business objective.

---

## 3. Data Requirements

### 3.1 Understanding data requirements
- Data in a predictive analytics project has to be crafted and custom fit.
- Getting and prepping the data is 70 to 90% of the work.

### 3.2 Gathering historical data
- First data requirement element is history: need good historical data.
- Historical data in the form of a customer footprint is the cornerstone of the dataset.

### 3.3 Meeting the flat file requirement
- Next data requirement element is a flat file: all records and characteristics in one big, rectangular table.
- The resulting flat file will be unique, built for the current project.

### 3.4 Determining your target variable
- Need a target variable: labeled data.
- The final outcome is known and recorded in the flat file.
- The algorithm defines the relationship between input variables and the target variable.

### 3.5 Selecting relevant data
- Be thoughtful about the data you select; focus on selecting the cases or instances (rows).
- The historical dataset should mimic the data you'll be deploying the model on.

### 3.6 Hints on effective data integration
- Extensive data integration is needed; combine as many different sources of data as possible.
- The harder it is to integrate the data, the better the project is going to be.

### 3.7 Understanding feature engineering
- Feature engineering: creating new variables is one of the most powerful things you can do.
- More than 90% of variables may come from feature engineering.
- "Applied machine learning is feature engineering."

### 3.8 Developing your craft
- Data mining, especially data prep, is a craft.
- Professional development is best accomplished through apprenticeship.
- No other task influences model accuracy more than effective data prep.

---

## 4. Resources You Will Need

### 4.1 Skill sets and resources that you'll need
- Need the whole organization supporting you.
- Be truly collaborative from the start.
- Need specialized data mining algorithms, a cross-functional team, realistic timetable, and access to subject matter experts.

### 4.2 Compare machine learning and statistics
- Data mining algorithms are inherently a product of the computer age.
- Statistics is about yes-no questions and hypothesis testing; data mining is highly iterative and requires computing power.
- Machine learning algorithms are qualitatively different from traditional statistical techniques.

### 4.3 Assessing team requirements
- Data mining is a team sport; necessary mix of skills comes from the team.
- Assemble a diverse team; cross train and work closely together.
- Mix of ages, college majors, and backgrounds is best for the project and professional development.

### 4.4 Budgeting sufficient time
- Data mining projects take time: many weeks or months.
- Problem definition and data preparation take up much of the time.
- Not done until you've deployed something.

### 4.5 Working with subject matter experts
- Need subject matter experts for context.
- Let the computer narrow the search, let the SME help you widen the search.
- SMEs help with interpretation and data quality problems.

---

## 5. Problems You Will Face

### 5.1 Predictive analytics essential training: Data mining
- Every project will face unique problems, but always missing data and organizational resistance.
- All models will degrade over time.

### 5.2 Addressing missing data
- Assess missing data during data understanding before modeling.
- Missing data is not just a statistics issue; assess the journey the data took from collection to modeling.

### 5.3 Addressing organizational resistance
- All patterns are subject to change; models will degrade.
- Using predictive models changes the future.
- The laws of data mining provide a firm foundation in the what, how to, and why of predictive analytics.

---

## 6. Finding the Solution

### 6.1 Preparing for the modeling phase tasks
- Discussion of modeling focuses on common sources of misunderstanding around data mining modeling.
- Compare and contrast data mining with statistics; avoid confusing the two.
- Review strategic goals and implications for modeling practice.
- Review what does and doesn't constitute proof in this approach.

### 6.2 Searching for optimal solutions
- No need for hypotheses in machine learning; avoid guessing or speculating about relationships.
- Be systematic about trying all possible relationships between input variables and the target variable.
- Data mining is exploratory and data-driven, not based on experimental data or the scientific method.
- Present the algorithm with the most comprehensive search space possible.

### 6.3 Seeking surprise results
- Surprises can be good; don't be too frugal with predictors.
- Unanticipated interactions between variables can provide valuable insights.
- Data mining algorithms are designed to handle lots of variables; leaving out variables can sacrifice accuracy.
- Data reduction is important but should be done cautiously.

### 6.4 Establishing proof that the model works
- Need evidence that a model is good before deployment.
- The same data used to uncover the pattern must never be used to prove it applies to future data.
- Use train/test (hold-out) validation: build model on train data, verify on test data.
- Hold-out validation is essential for generalizing to new data.

### 6.5 Embracing a trial and error approach
- Modeling phase involves a lot of trial and error.
- List all possible methods and parameters, then systematically try them all.

---

## 7. Putting the Solution to Work

### 7.1 Preparing for the deployment phase
- Deployment is essential; the project isn't complete unless you deploy the model.
- Deployment often gets bogged down in technology, but strategic issues are critical.

### 7.2 Using probabilities and propensities
- Models produce probabilities, not direct predictions; these are called propensity scores.
- Propensity scores drive intervention strategies.
- Binary classification is powerful and recommended.
- Model scores are used in business rules to drive actions.

### 7.3 Understanding meta modeling
- One model almost never does the job alone; models can be combined (metamodeling).
- Ensembles (e.g., random forest) are one approach; metamodeling is more inclusive.
- More than one micro-decision implies more than one model.

### 7.4 Understanding reproducibility
- Everything about the project must ultimately be reproducible.
- Data mining must result in a reproducible series of steps that can be performed on new data.
- Reproducibility is key for deployment and production environments.

### 7.5 Preparing for model deployment
- Deployment requires collaboration with enterprise data teams (data engineering).
- Follow the data from input to model to output scores.
- Predictive scores may be sent to various systems or devices.

### 7.6 How to approach project documentation
- Documentation is critical for knowledge transfer and project continuity.
- Write milestone reports at the end of each CRISP-DM phase.
- Document every phase, including variables used, data sources, code, handling of missing data, and score destinations.
- Prepare for end-user training and ensure documentation is understandable by others.

---

## 8. The Nine Laws of Data Mining

### 8.1 CRISP-DM and the laws of data mining
- CRISP-DM is the de facto standard for data mining process.
- Tom Khabaza authored the "Nine Laws of Data Mining" to explain what data mining is and why the process has its properties.

### 8.2 Understanding CRISP-DM
- CRISP-DM has six phases: business understanding, data understanding, data preparation, modeling, evaluation, and deployment.
- 24 tasks under the six phases; explicit documentation suggestions for each phase and task.
- Emphasizes consensus, data exploration, feature engineering, and business evaluation.

### 8.3 Advice for using CRISP-DM
- Establish consensus among team members during business understanding.
- Data understanding phase is often neglected but critical.
- Data prep and integration are key components.
- Don't skip the first three phases.
- Evaluation should focus on business evaluation (KPIs, ROI).
- Review project for technical and non-technical improvements.

### 8.4 Understanding the nine laws of data mining
- The nine laws are philosophical and practical, capturing important truths about data mining.
- Reflecting on the laws helps navigate projects more successfully.

### 8.5 Understanding the first and second laws
- First law: Business objectives are the origin of every data mining solution.
- Second law: Business knowledge is central to every step of the data mining process.
- Technology can't help with business knowledge; the project relies on you.

### 8.6 Understanding the data preparation law
- Data preparation is more than half of every data mining process.
- Data prep in support of data mining is about problem space shaping, not just cleaning.
- Data prep embeds business knowledge in the data.

### 8.7 Understanding the laws about patterns
- There are always patterns in data mining problems; patterns are a byproduct of running a business.
- No free lunch: the right model can only be discovered by experiment; requires trial and error.
- All problems require more than one model; business knowledge guides the process.

### 8.8 Understanding the insight and prediction laws
- Data mining amplifies perception in the business domain; helps us see challenges and opportunities.
- Prediction increases information locally by generalization; models add information to each instance.

### 8.9 Understanding the value law
- Value does not derive from accuracy alone; business metrics and ROI are critical.
- "Data miners should not focus on predictive accuracy, model stability, or any other technical metric for predictive models at the expense of business insight and business fit."
- Value comes from how well you've solved the business problem.

### 8.10 Understanding why models change
- All patterns are subject to change; models will degrade over time.
- Data, customers, and our understanding change; business knowledge evolves.
- Using predictive models changes the future (Rohaczynski's Law).
