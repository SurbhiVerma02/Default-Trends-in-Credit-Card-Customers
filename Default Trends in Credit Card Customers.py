# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC #Default Trends in Credit Card Customers
# MAGIC
# MAGIC #####  Surbhi Verma 
# MAGIC
# MAGIC ## ***Business Problem***
# MAGIC * Credit cards are typically issued by banks based on a person's credit history and score. There is no set mechanism for identifying defaulters because any client can default at any time. Although, banks make money by selling the assets of defaulters in extreme circumstances, this usually comes at a high litigation costs and takes a long time. As a result, it is critical that banks lend responsibly in order to reduce defaults. 
# MAGIC
# MAGIC ## ***Solution***
# MAGIC * In this project, we attempted to predict the defaults of credit card clients using various Machine Learning Models in R. This can be used by the banks to identify the chances of a particular client defaulting on their repayment, thereby lending responsibly.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## ***Install Libraries and Download dataset***

# COMMAND ----------

pip install squarify


# COMMAND ----------

# DBTITLE 1,Install libraries
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()


from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.stat import ChiSquareTest
from pyspark.ml.linalg import Vectors
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import IndexToString
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import clustering
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import evaluation
from pyspark.ml import tuning
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import requests
import pandas as pd

from pyspark.sql.functions import col, lit, round
import matplotlib.pyplot as plt

# For Data Preprocessing
from pyspark.ml.feature import Imputer
from pyspark.ml.feature import StandardScaler, MinMaxScaler, RobustScaler
from pyspark.ml.feature import SQLTransformer
from pyspark.ml.feature import PCA
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.ml.feature import Bucketizer

# For Neural Network
from pyspark.ml.classification import MultilayerPerceptronClassifier

# For Model Selection
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import evaluation

# Other PySpark-related Libraries
from pyspark.sql.functions import when, col

# COMMAND ----------

# DBTITLE 1,Downloading and displaying data from Google Drive 
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import requests
import pandas as pd
from io import StringIO

# Downloaded the data from Kaggle and uploaded in a drive.
# Kaggle data link : https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset/data

drive_link = "https://drive.google.com/uc?export=download&id=1wnAHCZQxOi7ADV2IRoJcBkgAL7sBZcgt"

# Request to get the file content
response = requests.get(drive_link)

# Check if the request was successful
if response.status_code == 200:
    # Use pandas to read the CSV from the response content
    df = pd.read_csv(StringIO(response.text))

    # Now 'df' is a pandas DataFrame with the data
    # If needed, you can convert it to a PySpark DataFrame
    spark_df = spark.createDataFrame(df)

    # Show the DataFrame
    display(spark_df)
else:
    print(f"Failed to download the file. Status code: {response.status_code}")

# COMMAND ----------

# DBTITLE 1,Checking the number of rows and columns in the dataset
# Get the number of rows and columns
len_df = spark_df.count()
total_columns = len(df.columns)

# Print the results
print('Total rows: {} and total columns: {}'.format(len_df, total_columns))
display(spark_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ***Data Pre-processing***

# COMMAND ----------

# DBTITLE 1,Remove unnecessary columns
#Remove index column 
spark_df = spark_df.drop(spark_df.columns[0])

display(spark_df)

# COMMAND ----------

# DBTITLE 1,Rename pay columns for clarity and readability
#Rename coloumns
new_column_names = ["limit_bal", "sex", "education", "marriage", "age", "pay_sep",
                    "pay_aug", "pay_jul", "pay_jun", "pay_may", "pay_arp", "bill_amt_sep",
                    "bill_amt_aug", "bill_amt_jul", "bill_amt_jun", "bill_amt_may", 
                    "bill_amt_arp", "pay_amt_sep", "pay_amt_aug", "pay_amt_jul", 
                    "pay_amt_jun", "pay_amt_may", "pay_amt_arp", "def_pay"]
    
for old_name, new_name in zip(spark_df.columns, new_column_names):
    spark_df = spark_df.withColumnRenamed(old_name, new_name)
    
display(spark_df)

# COMMAND ----------

# DBTITLE 1,Handling unknown, missing and null values
#Remove rows with unknown values in education and marriage columns
spark_df = spark_df.filter((col("education") != 0) & (col("marriage") != 0))

#Remove rows with NaN values
spark_df = spark_df.dropna()

display(spark_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ***Exploratory Data Analysis***

# COMMAND ----------

# MAGIC %md
# MAGIC ####***Calculating and visualizing the sex distribution within the dataset***
# MAGIC

# COMMAND ----------

import matplotlib.pyplot as plt
import squarify

target_variable = "def_pay"

#create sex table which has counts in ratio of male and female
sex_ratio = spark_df.groupBy("sex").count()
total_count = sex_ratio.groupBy().sum("count").collect()[0][0]
sex_ratio = sex_ratio.withColumn("ratio", round((col("count") / total_count) * 100, 2))
sex_ratio = sex_ratio.withColumn("sex", 
                                 when(col("sex") == 1, "male")
                                 .otherwise("female"))
sex_ratio_pd = sex_ratio.toPandas()

# Prepare data
sizes = sex_ratio_pd['ratio'].values
labels = sex_ratio_pd.apply(lambda x: str(x['sex']) + '\n' + str(x['ratio']) + '%', axis=1)

# Create a color list
colors = ['lightblue', 'pink']

# Create Tree map
plt.figure(figsize=(8, 8))
squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.7)
plt.title('Sex Ratio')
plt.axis('off') # Removes the axes for visual appeal
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ####***Calculating and visualizing the defaulter ratio - Default payment (1=yes, 0=no)***
# MAGIC

# COMMAND ----------

#Create the default pay next month(defaulter ratio) and plot it

# Create defaulter ratio
def_ratio = spark_df.groupBy(target_variable).count()
total_count = def_ratio.groupBy().sum("count").collect()[0][0]
def_ratio = def_ratio.withColumn("ratio", round((col("count") / total_count) * 100, 2))

# Convert to Pandas DataFrame for plotting
def_ratio_pd = def_ratio.toPandas()

# Define two colors for the pie chart segments
colors = ['lightblue', 'lightgreen']

# Plotting a pie chart with different colors
plt.figure(figsize=(8, 8))
plt.pie(def_ratio_pd['ratio'], labels=def_ratio_pd[target_variable], autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Defaulter Ratio')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Inference**:
# MAGIC We can see that the dataset consists of 77.8% of clients are not expected to default payment(0) whereas 22.1% of clients are expected to default payment(1)

# COMMAND ----------

# MAGIC %md
# MAGIC ####***Calculating and visualizing the ratio of marital status - 1=married, 2=single, 3-others***
# MAGIC

# COMMAND ----------


# Create marriage ratio
marriage_ratio = spark_df.groupBy("marriage").count()
total_count = marriage_ratio.groupBy().sum("count").collect()[0][0]
marriage_ratio = marriage_ratio.withColumn("ratio", round((col("count") / total_count) * 100, 2))

# Convert to Pandas DataFrame for plotting
marriage_ratio_pd = marriage_ratio.toPandas()

# Define three colors for the segments
colors = ['lightskyblue', 'lightcoral', 'lightgreen']

# Plotting the donut chart with specified colors
plt.figure(figsize=(8, 8))
plt.pie(marriage_ratio_pd['ratio'], labels=marriage_ratio_pd['marriage'], autopct='%1.1f%%', startangle=90, colors=colors)

# Adding a circle at the center to create a donut shape
centre_circle = plt.Circle((0,0), 0.30, color='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title('Marriage Status Ratio')
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC **Inference**:
# MAGIC We can see that the dataset consists of 45.6% of married clients, 53.3% of singles, and 1.1% who did not  mention their marital status.

# COMMAND ----------

# MAGIC %md
# MAGIC ####***Create age groups and Plot credit limit by age group***
# MAGIC

# COMMAND ----------

# Age Groups and Credit Limit by Age Group

from pyspark.sql.functions import when, col
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

# create age groups
bins = [0, 20, 30, 40, 50, 60, 70, float('inf')]
labels = ["0-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71+"]

age_expr = when(col("age") < bins[0], None)  # Default case, if needed
for i in range(1, len(bins)):
    lower_bound = bins[i-1]
    upper_bound = bins[i]
    label = labels[i-1]
    age_expr = age_expr.when((col("age") > lower_bound) & (col("age") <= upper_bound), label)

spark_df = spark_df.withColumn("age_group", age_expr)

# Convert the DataFrame to Pandas for plotting
pd_df = spark_df.toPandas()

custom_palette = ["lightskyblue",  
                  "lightcoral",  
                  "lightgreen", 
                  "lightpink", 
                  "#E6E6FA",  
                  "#AFEEEE",  
                  "#B19CD9"]  

# Convert the DataFrame to Pandas for plotting
pd_df = spark_df.toPandas()

# Plot credit limit by age group with custom palette
plt.figure(figsize=(12, 6))
ax = sns.boxplot(x="age_group", y="limit_bal", data=pd_df, palette=custom_palette)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))
plt.title("Credit Limits in Different Age Groups")
plt.xlabel("Age Group")
plt.ylabel("Credit Limit")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC **Inference**:
# MAGIC The box plot suggests that the credit limit generally increases with age, peaking in the 61-70 age group, with a wide range of limits indicated by outliers in each category.

# COMMAND ----------

# MAGIC %md
# MAGIC ####***Create total bill and total pay columns***
# MAGIC
# MAGIC We are creating two new columns by summing up all the bill amounts and all the payments made by the clients.
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import col

# Column names for bills and payments
bill_cols = ["bill_amt_sep", "bill_amt_aug", "bill_amt_jul", "bill_amt_jun", "bill_amt_may", "bill_amt_arp"]
pay_cols = ["pay_amt_sep", "pay_amt_aug", "pay_amt_jul", "pay_amt_jun", "pay_amt_may", "pay_amt_arp"]

# Adding total bill and total payment columns
spark_df = spark_df.withColumn("total_bill", sum(col(x) for x in bill_cols))
spark_df = spark_df.withColumn("total_pay", sum(col(x) for x in pay_cols))

display(spark_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **We are plotting total bill and total payed amount by age group**
# MAGIC
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import sum as _sum
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Aggregate total bill and pay by age groups
bill_by_groups = spark_df.groupBy("age_group").agg(_sum("total_bill").alias("total_bill"))
pay_by_groups = spark_df.groupBy("age_group").agg(_sum("total_pay").alias("total_pay"))

# Join the two dataframes
bill_pay_df = bill_by_groups.join(pay_by_groups, "age_group")

# Convert to Pandas for plotting
bill_pay_pd = bill_pay_df.toPandas()

# Convert 'age_group' to a categorical data type with the desired order
age_group_order = ["0-20", "21-30", "31-40", "41-50", "51-60", "61-70", "Above 70"]
bill_pay_pd['age_group'] = pd.Categorical(bill_pay_pd['age_group'], categories=age_group_order, ordered=True)

# Melt the DataFrame for Seaborn plotting
bill_pay_melted = pd.melt(bill_pay_pd, id_vars=["age_group"], var_name="type", value_name="amount")

# Plotting
plt.figure(figsize=(12, 6))
ax = sns.lineplot(data=bill_pay_melted, x="age_group", y="amount", hue="type")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))
plt.title("Bill Compared to Payment by Age Group")
plt.xlabel("Age Group")
plt.ylabel("New Taiwan Dollar (NTD)")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Inference**:
# MAGIC
# MAGIC Despite this decrease in billing amounts, the payment amounts remain consistently low across all age groups, suggesting that older age groups may have a higher rate of bill settlement relative to their total bill amounts compared to younger age groups.

# COMMAND ----------

# DBTITLE 1,Schema Conversion and Inspection
spark_df.printSchema()

# Assuming 'credit_card' is your PySpark DataFrame
columns_to_convert = ['limit_bal', 'education', 'pay_sep', 'pay_aug', 'pay_jul', 'pay_jun', 'pay_may', 'pay_arp',"def_pay"]

# Print the updated schema
spark_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ####***MODELS***
# MAGIC

# COMMAND ----------

# DBTITLE 1,Logistic regression:
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType, FloatType
from pyspark.mllib.evaluation import MulticlassMetrics
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

target_variable = "def_pay"

# Define the feature columns
feature_columns = [col for col in spark_df.columns if col != target_variable]

# Convert 'age_group' to numerical format
age_group_indexer = StringIndexer(inputCol='age_group', outputCol='age_group_index')
spark_df = age_group_indexer.fit(spark_df).transform(spark_df)

# Drop the original 'age_group' column
spark_df = spark_df.drop('age_group')

# Convert target_variable to DoubleType
spark_df = spark_df.withColumn(target_variable, col(target_variable).cast(DoubleType()))

# Define the feature columns
feature_columns = [col for col in spark_df.columns if col != target_variable]

# Split data into training and testing sets
(train_data, test_data) = spark_df.randomSplit([0.8, 0.2], seed=123)

# Define the logistic regression model pipeline
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
log_reg = LogisticRegression(featuresCol="features", labelCol="def_pay")
pipeline = Pipeline(stages=[assembler, log_reg])

# Define parameter grid for model tuning
param_grid = ParamGridBuilder().build()

# Define evaluator and cross-validator
evaluator = BinaryClassificationEvaluator(labelCol=target_variable, metricName="areaUnderROC")
cross_val = CrossValidator(estimator=pipeline, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3)

# Fit the model
cvModel = cross_val.fit(train_data)

# Make predictions on test data
test_results = cvModel.transform(test_data)

# Extracting actual and predicted probabilities
y_true = test_results.select(target_variable).rdd.map(lambda row: float(row[target_variable])).collect()
y_prob = test_results.select("probability").rdd.map(lambda row: float(row["probability"][1])).collect()

precision, recall, _ = precision_recall_curve(y_true, y_prob)

# Plotting Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Convert predictions and labels for use with MulticlassMetrics
predictionAndLabels = test_results.select("prediction", target_variable).rdd.map(lambda x: (float(x[0]), float(x[1])))
metrics = MulticlassMetrics(predictionAndLabels)

# Compute and print confusion matrix
confusion_matrix = metrics.confusionMatrix().toArray()
print("Confusion Matrix:")
print(confusion_matrix)

# Compute accuracy
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol=target_variable, predictionCol="prediction", metricName="accuracy")
accuracy = accuracy_evaluator.evaluate(test_results)
print(f"Accuracy: {accuracy}")

# Extract probability for ROC
get_prob = udf(lambda v: float(v[1]), FloatType())
test_results = test_results.withColumn("probability", get_prob("probability"))

# ROC Curve
roc_curve_df = test_results.select(target_variable, "probability").toPandas()
fpr, tpr, _ = roc_curve(roc_curve_df[target_variable], roc_curve_df['probability'])

# Plot ROC Curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Area under ROC
roc_auc = evaluator.evaluate(test_results)
print(f"Area under ROC: {roc_auc}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Inference**:
# MAGIC
# MAGIC Logistic Regression model's accuracy of 81% suggests that it's reasonably effective in predicting credit card defaults. Here's how it might contribute to cost reduction and efficiency:
# MAGIC
# MAGIC ***Credit Allocation***: The model's ability to identify potential defaulters aids in smarter credit allocation. It helps banks in lending responsibly by offering credit to customers with lower risk profiles and avoiding excessive lending to individuals who might struggle with repayments.
# MAGIC
# MAGIC ***Operational Costs***: Identifying potential defaulters using predictive models can reduce the operational costs associated with managing defaults. It enables proactive measures to be taken to prevent defaults, thus minimizing the need for extensive recovery processes and legal actions.
# MAGIC
# MAGIC ***Litigation Costs***: By accurately pinpointing customers at higher risk of default, the model aids in preventing cases that might lead to litigation. This minimizes legal fees and expenses incurred during the recovery of defaulted amounts.
# MAGIC
# MAGIC We plan to execute more models and then compare which is the best for us.
