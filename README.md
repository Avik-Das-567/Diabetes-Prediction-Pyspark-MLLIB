# Diabetes Prediction With PySpark MLlib

## Project Overview

This project builds a binary classification pipeline in PySpark to predict whether a patient is diabetic or non-diabetic using a logistic regression model from `pyspark.ml`. The workflow is implemented in a notebook and follows a typical Spark ML pattern: ingest structured data into a Spark DataFrame, inspect schema and distributions, clean problematic values, assemble features into a vector column, train a classifier, evaluate predictive performance, persist the trained model, and reuse the saved model for inference on new records.

The implementation uses Spark's DataFrame-based machine learning API rather than a single-machine workflow. That design is useful for tabular classification problems because it keeps the preprocessing, feature assembly, model training, and prediction steps inside the Spark ecosystem, which is well-suited for scalable distributed data processing.

## Problem Statement

The goal of the project is to classify patient records into one of two categories:

- `1`: diabetic
- `0`: non-diabetic

The notebook trains a logistic regression classifier on medical and demographic attributes, then applies the trained model to unseen patient records. This makes the project a compact end-to-end example of supervised classification with Spark MLlib on structured healthcare-style data.

## Dataset Description

The notebook loads a diabetes dataset into a Spark DataFrame from CSV and infers the schema automatically. The resulting dataset contains `2,000` rows and `9` columns.

### Target Variable

- `Outcome`: binary label indicating diabetic (`1`) or non-diabetic (`0`)

### Input Features

- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`

### Schema

The notebook infers the following column types:

| Column | Type |
| --- | --- |
| `Pregnancies` | `integer` |
| `Glucose` | `integer` |
| `BloodPressure` | `integer` |
| `SkinThickness` | `integer` |
| `Insulin` | `integer` |
| `BMI` | `double` |
| `DiabetesPedigreeFunction` | `double` |
| `Age` | `integer` |
| `Outcome` | `integer` |

### Class Distribution

The label distribution is moderately imbalanced:

| Outcome | Count |
| --- | ---: |
| `0` | `1316` |
| `1` | `684` |

This means approximately `65.8%` of the samples are non-diabetic and `34.2%` are diabetic.

## Data Inspection and Cleaning

The notebook begins by profiling the dataset with:

- `df.show()` to inspect records
- `df.printSchema()` to verify datatypes
- `df.groupBy('Outcome').count()` to inspect label balance
- `df.describe()` to review summary statistics

### Summary Statistics

The descriptive summary shows the central tendency and spread of the main variables. A few notable values from the notebook output are:

- Mean `Glucose`: `121.1825`
- Mean `BloodPressure`: `69.1455`
- Mean `SkinThickness`: `20.935`
- Mean `Insulin`: `80.254`
- Mean `BMI`: `32.193`
- Mean `Age`: `33.0905`
- Mean `Outcome`: `0.342`

The summary also reveals that several medical measurement columns contain `0` values, which are not realistic physiological readings in this context and are treated as placeholders for missing information.

### Null-Value Check

The notebook checks each column for nulls and finds none:

- `Pregnancies`: `0`
- `Glucose`: `0`
- `BloodPressure`: `0`
- `SkinThickness`: `0`
- `Insulin`: `0`
- `BMI`: `0`
- `DiabetesPedigreeFunction`: `0`
- `Age`: `0`
- `Outcome`: `0`

### Zero-Value Diagnostics

Although the dataset contains no null values, several clinically meaningful features contain zeros that are treated as invalid placeholders:

| Column | Zero Count |
| --- | ---: |
| `Glucose` | `13` |
| `BloodPressure` | `90` |
| `SkinThickness` | `573` |
| `Insulin` | `956` |
| `BMI` | `28` |

### Imputation Strategy

The notebook replaces zero values in the following columns with their column means using `when(...).otherwise(...)` from `pyspark.sql.functions`:

| Column | Replacement Value Used |
| --- | ---: |
| `Glucose` | `121` |
| `BloodPressure` | `69` |
| `SkinThickness` | `20` |
| `Insulin` | `80` |
| `BMI` | `32` |

This is a simple mean-imputation strategy applied directly within the Spark DataFrame. It preserves the row count while removing clearly invalid zero placeholders from important physiological measurements.

## Feature Engineering

After cleaning, the notebook evaluates feature-to-target relationships with `df.stat.corr('Outcome', column)` and reports the following correlations with `Outcome`:

| Feature | Correlation with `Outcome` |
| --- | ---: |
| `Pregnancies` | `0.22443699263363961` |
| `Glucose` | `0.48796646527321064` |
| `BloodPressure` | `0.17171333286446713` |
| `SkinThickness` | `0.1659010662889893` |
| `Insulin` | `0.1711763270226193` |
| `BMI` | `0.2827927569760082` |
| `DiabetesPedigreeFunction` | `0.1554590791569403` |
| `Age` | `0.23650924717620253` |

Among the input variables, `Glucose` shows the strongest linear relationship with the target in the notebook's correlation analysis.

### Vector Assembly

Spark ML models expect features in a single vector column, so the notebook uses `VectorAssembler` to combine the eight predictors into a column named `Features`:

```python
assembler = VectorAssembler(
    inputCols=[
        'Pregnancies',
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI',
        'DiabetesPedigreeFunction',
        'Age'
    ],
    outputCol='Features'
)
```

The transformed DataFrame retains the original columns and appends:

- `Features`: `vector`

For modeling, the notebook then creates a final two-column training frame:

- `Features`
- `Outcome`

## Model Training

The classifier used in this project is `pyspark.ml.classification.LogisticRegression`.

### Train/Test Split

The final DataFrame is partitioned with:

```python
train, test = final_data.randomSplit([0.7, 0.3])
```

This creates a roughly `70/30` split between training and test data. In the captured notebook execution, the model summary reports `1397` training predictions, which reflects the training subset used in that run.

### Algorithm Choice

Logistic regression is a natural baseline for binary classification because it models the probability of the positive class and returns both class predictions and class probabilities. In Spark ML, it integrates cleanly with vectorized features and DataFrame-based workflows.

The notebook instantiates and fits the model as follows:

```python
models = LogisticRegression(labelCol='Outcome')
model = models.fit(train)
```

## Evaluation Results

Once training is complete, the notebook inspects the fitted model summary:

```python
summary = model.summary
summary.predictions.describe().show()
```

The training-summary output reports:

| Metric | `Outcome` | `prediction` |
| --- | ---: | ---: |
| `count` | `1397` | `1397` |
| `mean` | `0.3457408732999284` | `0.2634216177523264` |
| `stddev` | `0.47577952789741335` | `0.44064686485175947` |
| `min` | `0.0` | `0.0` |
| `max` | `1.0` | `1.0` |

For held-out evaluation, the notebook uses:

```python
from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(
    rawPredictionCol='rawPrediction',
    labelCol='Outcome'
)
evaluator.evaluate(model.transform(test))
```

The resulting evaluation score is:

```text
0.8537536199599021
```

Because `BinaryClassificationEvaluator` is used without overriding `metricName`, this score corresponds to the evaluator's default binary classification metric in Spark, which is area under ROC.

The notebook also displays test-set prediction outputs containing:

- `Features`
- `Outcome`
- `rawPrediction`
- `probability`
- `prediction`

This provides both the predicted class and the model's estimated probability distribution for each record.

## Saved Model and Prediction Workflow

The project includes model persistence and reuse inside the notebook workflow.

### Saving the Trained Model

The fitted logistic regression model is saved with:

```python
model.save("Model")
```

### Reloading the Model

The notebook reloads the saved artifact using:

```python
from pyspark.ml.classification import LogisticRegressionModel
model = LogisticRegressionModel.load("Model")
```

This demonstrates how a trained Spark ML model can be serialized and restored without retraining.

### Predicting on New Data

For inference, the notebook:

1. Loads `new_test.csv` into a Spark DataFrame.
2. Reuses the same `VectorAssembler` to build the `Features` column.
3. Applies `model.transform(test_data)` to generate predictions.
4. Selects `features` and `prediction` for inspection.

The displayed predictions for the four new records are:

| Record | Prediction |
| --- | ---: |
| 1 | `1.0` |
| 2 | `0.0` |
| 3 | `1.0` |
| 4 | `1.0` |

This final stage shows the full deployment-style path from raw incoming tabular records to feature vector generation and binary classification using the persisted model.

## Technical Stack

The implementation relies on the following components:

- `PySpark` for distributed data processing and machine learning workflows
- `SparkSession` for creating the Spark application context
- `Spark SQL DataFrames` for schema-aware tabular processing
- `pyspark.sql.functions` for conditional column updates during cleaning
- `VectorAssembler` for feature vector construction
- `pyspark.ml.classification.LogisticRegression` for binary classification
- `BinaryClassificationEvaluator` for model evaluation
- `LogisticRegressionModel.load()` for restoring the saved model artifact

## Key Takeaways

- The project demonstrates a complete Spark ML classification workflow on structured medical data.
- Data cleaning is a central part of the pipeline, especially because the dataset contains many zero-valued placeholders in clinically meaningful fields.
- Feature preparation is handled with Spark-native transformations rather than manual NumPy-style preprocessing.
- Logistic regression provides both class labels and probabilities, making it suitable for binary medical risk-style classification tasks.
- The notebook does not stop at training; it also evaluates the classifier, saves the model, reloads it, and performs inference on new samples.
