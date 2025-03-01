# Week 3 Assignment:  Data Pipeline Components for Production ML

In this last graded programming exercise of the course, you will put together all the lessons we've covered so far to handle the first three steps of a production machine learning project - Data ingestion, Data Validation, and Data Transformation.

Specifically, you will build the production data pipeline by:

*   Performing feature selection
*   Ingesting the dataset
*   Generating the statistics of the dataset
*   Creating a schema as per the domain knowledge
*   Creating schema environments
*   Visualizing the dataset anomalies
*   Preprocessing, transforming and engineering your features
*   Tracking the provenance of your data pipeline using ML Metadata

Most of these will look familiar already so try your best to do the exercises by recall or browsing the documentation. If you get stuck however, you can review the lessons in class and the ungraded labs. 

Let's begin!

## Table of Contents

- [1 - Imports](#1)
- [2 - Load the Dataset](#2)
- [3 - Feature Selection](#4)
  - [Exercise 1 - Feature Selection](#ex-1)
- [4 - Data Pipeline](#4)
  - [4.1 - Setup the Interactive Context](#4-1)
  - [4.2 - Generating Examples](#4-2)
    - [Exercise 2 - ExampleGen](#ex-2)
  - [4.3 - Computing Statistics](#4-3)
    - [Exercise 3 - StatisticsGen](#ex-3)
  - [4.4 - Inferring the Schema](#4-4)
    - [Exercise 4 - SchemaGen](#ex-4)
  - [4.5 - Curating the Schema](#4-5)
    - [Exercise 5 - Curating the Schema](#ex-5)
  - [4.6 - Schema Environments](#4-6)
    - [Exercise 6 - Define the serving environment](#ex-6)
  - [4.7 - Generate new statistics using the updated schema](#4-7)
      - [Exercise 7 - ImporterNode](#ex-7)
      - [Exercise 8 - StatisticsGen with the new schema](#ex-8)
  - [4.8 - Check anomalies](#4-8)
      - [Exercise 9 - ExampleValidator](#ex-9)
  - [4.9 - Feature Engineering](#4-9)
      - [Exercise 10 - preprocessing function](#ex-10)
      - [Exercise 11 - Transform](#ex-11)
- [5 - ML Metadata](#5)
  - [5.1 - Accessing stored artifacts](#5-1)
  - [5.2 - Tracking artifacts](#5-2)
    - [Exercise 12 - Get parent artifacts](#ex-12)

<a name='1'></a>
## 1 - Imports


```python
import tensorflow as tf
import tfx

# TFX components
from tfx.components import CsvExampleGen
from tfx.components import ExampleValidator
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Transform
from tfx.components import ImporterNode

# TFX libraries
import tensorflow_data_validation as tfdv
import tensorflow_transform as tft
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext

# For performing feature selection
from sklearn.feature_selection import SelectKBest, f_classif

# For feature visualization
import matplotlib.pyplot as plt 
import seaborn as sns

# Utilities
from tensorflow.python.lib.io import file_io
from tensorflow_metadata.proto.v0 import schema_pb2
from google.protobuf.json_format import MessageToDict
from  tfx.proto import example_gen_pb2
from tfx.types import standard_artifacts
import os
import pprint
import tempfile
import pandas as pd

# To ignore warnings from TF
tf.get_logger().setLevel('ERROR')

# For formatting print statements
pp = pprint.PrettyPrinter()

# Display versions of TF and TFX related packages
print('TensorFlow version: {}'.format(tf.__version__))
print('TFX version: {}'.format(tfx.__version__))
print('TensorFlow Data Validation version: {}'.format(tfdv.__version__))
print('TensorFlow Transform version: {}'.format(tft.__version__))
```

    TensorFlow version: 2.3.1
    TFX version: 0.24.0
    TensorFlow Data Validation version: 0.24.1
    TensorFlow Transform version: 0.24.1


<a name='2'></a>
## 2 - Load the dataset

You are going to use a variant of the [Cover Type](https://archive.ics.uci.edu/ml/datasets/covertype) dataset. This can be used to train a model that predicts the forest cover type based on cartographic variables. You can read more about the *original* dataset [here](https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.info) and we've outlined the data columns below:

| Column Name | Variable Type | Units / Range | Description |
| --------- | ------------ | ----- | ------------------- |
| Elevation | quantitative |meters | Elevation in meters |
| Aspect | quantitative | azimuth | Aspect in degrees azimuth |
| Slope | quantitative | degrees | Slope in degrees |
| Horizontal_Distance_To_Hydrology | quantitative | meters | Horz Dist to nearest surface water features |
| Vertical_Distance_To_Hydrology | quantitative | meters | Vert Dist to nearest surface water features |
| Horizontal_Distance_To_Roadways | quantitative | meters | Horz Dist to nearest roadway |
| Hillshade_9am | quantitative | 0 to 255 index | Hillshade index at 9am, summer solstice |
| Hillshade_Noon | quantitative | 0 to 255 index | Hillshade index at noon, summer soltice |
| Hillshade_3pm | quantitative | 0 to 255 index | Hillshade index at 3pm, summer solstice |
| Horizontal_Distance_To_Fire_Points | quantitative | meters | Horz Dist to nearest wildfire ignition points |
| Wilderness_Area (4 binary columns) | qualitative | 0 (absence) or 1 (presence) | Wilderness area designation |
| Soil_Type (40 binary columns) | qualitative | 0 (absence) or 1 (presence) | Soil Type designation |
| Cover_Type (7 types) | integer | 1 to 7 | Forest Cover Type designation |

As you may notice, the qualitative data has already been one-hot encoded (e.g. `Soil_Type` has 40 binary columns where a `1` indicates presence of a feature). For learning, we will use a modified version of this dataset that shows a more raw format. This will let you practice your skills in handling different data types. You can see the code for preparing the dataset [here](https://github.com/GoogleCloudPlatform/mlops-on-gcp/blob/master/datasets/covertype/wrangle/prepare.ipynb) if you want but it is **not required for this assignment**. The main changes include:

* Converting `Wilderness_Area` and `Soil_Type` to strings.
* Converting the `Cover_Type` range to [0, 6]

Run the next cells to load the **modified** dataset to your workspace. 


```python
# # OPTIONAL: Just in case you want to restart the lab workspace *from scratch*, you
# # can uncomment and run this block to delete previously created files and
# # directories. 

# !rm -rf pipeline
# !rm -rf data
```


```python
# Declare paths to the data
DATA_DIR = './data'
TRAINING_DIR = f'{DATA_DIR}/training'
TRAINING_DATA = f'{TRAINING_DIR}/dataset.csv'

# Create the directory
!mkdir -p {TRAINING_DIR}
```


```python
# download the dataset
!wget -nc https://storage.googleapis.com/workshop-datasets/covertype/full/dataset.csv -P {TRAINING_DIR}
```

    File ‘./data/training/dataset.csv’ already there; not retrieving.
    


<a name='3'></a>
## 3 - Feature Selection

For your first task, you will reduce the number of features to feed to the model. As mentioned in Week 2, this will help reduce the complexity of your model and save resources while training. Let's assume that you already have a baseline model that is trained on all features and you want to see if reducing the number of features will generate a better model. You will want to select a subset that has great predictive value to the label (in this case the `Cover_Type`). Let's do that in the following cells.



```python
# Load the dataset to a dataframe
df = pd.read_csv(TRAINING_DATA)

# Preview the dataset
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Elevation</th>
      <th>Aspect</th>
      <th>Slope</th>
      <th>Horizontal_Distance_To_Hydrology</th>
      <th>Vertical_Distance_To_Hydrology</th>
      <th>Horizontal_Distance_To_Roadways</th>
      <th>Hillshade_9am</th>
      <th>Hillshade_Noon</th>
      <th>Hillshade_3pm</th>
      <th>Horizontal_Distance_To_Fire_Points</th>
      <th>Wilderness_Area</th>
      <th>Soil_Type</th>
      <th>Cover_Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2596</td>
      <td>51</td>
      <td>3</td>
      <td>258</td>
      <td>0</td>
      <td>510</td>
      <td>221</td>
      <td>232</td>
      <td>148</td>
      <td>6279</td>
      <td>Rawah</td>
      <td>C7745</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2590</td>
      <td>56</td>
      <td>2</td>
      <td>212</td>
      <td>-6</td>
      <td>390</td>
      <td>220</td>
      <td>235</td>
      <td>151</td>
      <td>6225</td>
      <td>Rawah</td>
      <td>C7745</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2804</td>
      <td>139</td>
      <td>9</td>
      <td>268</td>
      <td>65</td>
      <td>3180</td>
      <td>234</td>
      <td>238</td>
      <td>135</td>
      <td>6121</td>
      <td>Rawah</td>
      <td>C4744</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2785</td>
      <td>155</td>
      <td>18</td>
      <td>242</td>
      <td>118</td>
      <td>3090</td>
      <td>238</td>
      <td>238</td>
      <td>122</td>
      <td>6211</td>
      <td>Rawah</td>
      <td>C7746</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2595</td>
      <td>45</td>
      <td>2</td>
      <td>153</td>
      <td>-1</td>
      <td>391</td>
      <td>220</td>
      <td>234</td>
      <td>150</td>
      <td>6172</td>
      <td>Rawah</td>
      <td>C7745</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Show the data type of each column
df.dtypes
```




    Elevation                              int64
    Aspect                                 int64
    Slope                                  int64
    Horizontal_Distance_To_Hydrology       int64
    Vertical_Distance_To_Hydrology         int64
    Horizontal_Distance_To_Roadways        int64
    Hillshade_9am                          int64
    Hillshade_Noon                         int64
    Hillshade_3pm                          int64
    Horizontal_Distance_To_Fire_Points     int64
    Wilderness_Area                       object
    Soil_Type                             object
    Cover_Type                             int64
    dtype: object



Looking at the data types of each column and the dataset description at the start of this notebook, you can see that most of the features are numeric and only two are not. This needs to be taken into account when selecting the subset of features because numeric and categorical features are scored differently. Let's create a temporary dataframe that only contains the numeric features so we can use it in the next sections.


```python
# Copy original dataset
df_num = df.copy()

# Categorical columns
cat_columns = ['Wilderness_Area', 'Soil_Type']

# Label column
label_column = ['Cover_Type']

# Drop the categorical and label columns
df_num.drop(cat_columns, axis=1, inplace=True)
df_num.drop(label_column, axis=1, inplace=True)

# Preview the resuls
df_num.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Elevation</th>
      <th>Aspect</th>
      <th>Slope</th>
      <th>Horizontal_Distance_To_Hydrology</th>
      <th>Vertical_Distance_To_Hydrology</th>
      <th>Horizontal_Distance_To_Roadways</th>
      <th>Hillshade_9am</th>
      <th>Hillshade_Noon</th>
      <th>Hillshade_3pm</th>
      <th>Horizontal_Distance_To_Fire_Points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2596</td>
      <td>51</td>
      <td>3</td>
      <td>258</td>
      <td>0</td>
      <td>510</td>
      <td>221</td>
      <td>232</td>
      <td>148</td>
      <td>6279</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2590</td>
      <td>56</td>
      <td>2</td>
      <td>212</td>
      <td>-6</td>
      <td>390</td>
      <td>220</td>
      <td>235</td>
      <td>151</td>
      <td>6225</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2804</td>
      <td>139</td>
      <td>9</td>
      <td>268</td>
      <td>65</td>
      <td>3180</td>
      <td>234</td>
      <td>238</td>
      <td>135</td>
      <td>6121</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2785</td>
      <td>155</td>
      <td>18</td>
      <td>242</td>
      <td>118</td>
      <td>3090</td>
      <td>238</td>
      <td>238</td>
      <td>122</td>
      <td>6211</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2595</td>
      <td>45</td>
      <td>2</td>
      <td>153</td>
      <td>-1</td>
      <td>391</td>
      <td>220</td>
      <td>234</td>
      <td>150</td>
      <td>6172</td>
    </tr>
  </tbody>
</table>
</div>



You will use scikit-learn's built-in modules to perform [univariate feature selection](https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection) on our dataset's numeric attributes. First, you need to prepare the input and target features:


```python
# Set the target values
y = df[label_column].values

# Set the input values
X = df_num.values
```

Afterwards, you will use [SelectKBest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest) to score each input feature against the target variable. Be mindful of the scoring function to pass in and make sure it is appropriate for the input (numeric) and target (categorical) values.

<a name='ex-1'></a>
### Exercise 1: Feature Selection

Complete the code below to select the top 8 features of the numeric columns.


```python
### START CODE HERE ###

# Create SelectKBest object using f_classif (ANOVA statistics) for 8 classes
select_k_best = SelectKBest(f_classif, k=8)

# Fit and transform the input data using select_k_best
X_new = select_k_best.fit_transform(X,y)

# Extract the features which are selected using get_support API
features_mask = select_k_best.get_support()

### END CODE HERE ###

# Print the results
reqd_cols = pd.DataFrame({'Columns': df_num.columns, 'Retain': features_mask})
print(reqd_cols)
```

                                  Columns  Retain
    0                           Elevation    True
    1                              Aspect   False
    2                               Slope    True
    3    Horizontal_Distance_To_Hydrology    True
    4      Vertical_Distance_To_Hydrology    True
    5     Horizontal_Distance_To_Roadways    True
    6                       Hillshade_9am    True
    7                      Hillshade_Noon    True
    8                       Hillshade_3pm   False
    9  Horizontal_Distance_To_Fire_Points    True


**Expected Output:**

```
                              Columns  Retain
0                           Elevation    True
1                              Aspect   False
2                               Slope    True
3    Horizontal_Distance_To_Hydrology    True
4      Vertical_Distance_To_Hydrology    True
5     Horizontal_Distance_To_Roadways    True
6                       Hillshade_9am    True
7                      Hillshade_Noon    True
8                       Hillshade_3pm   False
9  Horizontal_Distance_To_Fire_Points    True
```

If you got the expected results, you can now select this subset of features from the original dataframe and save it to a new directory in your workspace.


```python
# Set the paths to the reduced dataset
TRAINING_DIR_FSELECT = f'{TRAINING_DIR}/fselect'
TRAINING_DATA_FSELECT = f'{TRAINING_DIR_FSELECT}/dataset.csv'

# Create the directory
!mkdir -p {TRAINING_DIR_FSELECT}
```


```python
# Get the feature names from SelectKBest
feature_names = list(df_num.columns[features_mask])

# Append the categorical and label columns
feature_names = feature_names + cat_columns + label_column

# Select the selected subset of columns
df_select = df[feature_names]

# Write CSV to the created directory
df_select.to_csv(TRAINING_DATA_FSELECT, index=False)

# Preview the results
df_select.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Elevation</th>
      <th>Slope</th>
      <th>Horizontal_Distance_To_Hydrology</th>
      <th>Vertical_Distance_To_Hydrology</th>
      <th>Horizontal_Distance_To_Roadways</th>
      <th>Hillshade_9am</th>
      <th>Hillshade_Noon</th>
      <th>Horizontal_Distance_To_Fire_Points</th>
      <th>Wilderness_Area</th>
      <th>Soil_Type</th>
      <th>Cover_Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2596</td>
      <td>3</td>
      <td>258</td>
      <td>0</td>
      <td>510</td>
      <td>221</td>
      <td>232</td>
      <td>6279</td>
      <td>Rawah</td>
      <td>C7745</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2590</td>
      <td>2</td>
      <td>212</td>
      <td>-6</td>
      <td>390</td>
      <td>220</td>
      <td>235</td>
      <td>6225</td>
      <td>Rawah</td>
      <td>C7745</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2804</td>
      <td>9</td>
      <td>268</td>
      <td>65</td>
      <td>3180</td>
      <td>234</td>
      <td>238</td>
      <td>6121</td>
      <td>Rawah</td>
      <td>C4744</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2785</td>
      <td>18</td>
      <td>242</td>
      <td>118</td>
      <td>3090</td>
      <td>238</td>
      <td>238</td>
      <td>6211</td>
      <td>Rawah</td>
      <td>C7746</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2595</td>
      <td>2</td>
      <td>153</td>
      <td>-1</td>
      <td>391</td>
      <td>220</td>
      <td>234</td>
      <td>6172</td>
      <td>Rawah</td>
      <td>C7745</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



<a name='4'></a>
## 4 - Data Pipeline

With the selected subset of features prepared, you can now start building the data pipeline. This involves ingesting, validating, and transforming your data. You will be using the TFX components you've already encountered in the ungraded labs and you can look them up here in the [official documentation](https://www.tensorflow.org/tfx/api_docs/python/tfx/components).

<a name='4-1'></a>
### 4.1 - Setup the Interactive Context

As usual, you will first setup the Interactive Context so you can manually execute the pipeline components from the notebook. You will save the sqlite database in a pre-defined directory in your workspace. Please do not modify this path because you will need this in a later exercise involving ML Metadata.


```python
# Location of the pipeline metadata store
PIPELINE_DIR = './pipeline'

# Declare the InteractiveContext and use a local sqlite file as the metadata store.
context = InteractiveContext(pipeline_root=PIPELINE_DIR)
```

    WARNING:absl:InteractiveContext metadata_connection_config not provided: using SQLite ML Metadata database at ./pipeline/metadata.sqlite.


<a name='4-2'></a>
### 4.2 - Generating Examples

The first step in the pipeline is to ingest the data. Using [ExampleGen](https://www.tensorflow.org/tfx/guide/examplegen), you can convert raw data to TFRecords for faster computation in the later stages of the pipeline.

<a name='ex-2'></a>
#### Exercise 2: ExampleGen

Use `ExampleGen` to ingest the dataset we loaded earlier. Some things to note:

* The input is in CSV format so you will need to use the appropriate type of `ExampleGen` to handle it. 
* This function accepts a *directory* path to the training data and not the CSV file path itself. 

This will take a couple of minutes to run.


```python
# # NOTE: Uncomment and run this if you get an error saying there are different 
# # headers in the dataset. This is usually because of the notebook checkpoints saved in 
# # that folder.
# !rm -rf {TRAINING_DIR}/.ipynb_checkpoints
# !rm -rf {TRAINING_DIR_FSELECT}/.ipynb_checkpoints
# !rm -rf {SERVING_DIR}/.ipynb_checkpoints
```


```python
### START CODE HERE

# Instantiate ExampleGen with the input CSV dataset
example_gen = CsvExampleGen(input_base=TRAINING_DIR_FSELECT)

# Run the component using the InteractiveContext instance
context.run(example_gen)

### END CODE HERE
```






<style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object expanded"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExecutionResult</span><span class="deemphasize"> at 0x7fa59cc13fd0</span></div><table class="attr-table"><tr><td class="attr-name">.execution_id</td><td class = "attrvalue">5</td></tr><tr><td class="attr-name">.component</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">CsvExampleGen</span><span class="deemphasize"> at 0x7fa59cbc8820</span></div><table class="attr-table"><tr><td class="attr-name">.inputs</td><td class = "attrvalue">{}</td></tr><tr><td class="attr-name">.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7fa59cbc8580</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/CsvExampleGen/examples/5)<span class="deemphasize"> at 0x7fa59cbd58b0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/CsvExampleGen/examples/5</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.exec_properties</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['input_base']</td><td class = "attrvalue">./data/training/fselect</td></tr><tr><td class="attr-name">['input_config']</td><td class = "attrvalue">{
  &quot;splits&quot;: [
    {
      &quot;name&quot;: &quot;single_split&quot;,
      &quot;pattern&quot;: &quot;*&quot;
    }
  ]
}</td></tr><tr><td class="attr-name">['output_config']</td><td class = "attrvalue">{
  &quot;split_config&quot;: {
    &quot;splits&quot;: [
      {
        &quot;hash_buckets&quot;: 2,
        &quot;name&quot;: &quot;train&quot;
      },
      {
        &quot;hash_buckets&quot;: 1,
        &quot;name&quot;: &quot;eval&quot;
      }
    ]
  }
}</td></tr><tr><td class="attr-name">['output_data_format']</td><td class = "attrvalue">6</td></tr><tr><td class="attr-name">['custom_config']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['span']</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">['version']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['input_fingerprint']</td><td class = "attrvalue">split:single_split,num_files:1,total_bytes:27713036,xor_checksum:1626162109,sum_checksum:1626162109</td></tr><tr><td class="attr-name">['_beam_pipeline_args']</td><td class = "attrvalue">[]</td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">.component.inputs</td><td class = "attrvalue">{}</td></tr><tr><td class="attr-name">.component.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7fa59cbc8580</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/CsvExampleGen/examples/5)<span class="deemphasize"> at 0x7fa59cbd58b0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/CsvExampleGen/examples/5</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr></table></div>



<a name='4-3'></a>
### 4.3 - Computing Statistics

Next, you will compute the statistics of your data. This will allow you to observe and analyze characteristics of your data through visualizations provided by the integrated [FACETS](https://pair-code.github.io/facets/) library.

<a name='ex-3'></a>
#### Exercise 3: StatisticsGen

Use [StatisticsGen](https://www.tensorflow.org/tfx/guide/statsgen) to compute the statistics of the output examples of `ExampleGen`. 


```python
### START CODE HERE

# Instantiate StatisticsGen with the ExampleGen ingested dataset
statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
    

# Run the component
context.run(statistics_gen)
### END CODE HERE
```




<style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object expanded"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExecutionResult</span><span class="deemphasize"> at 0x7fa59cbc8250</span></div><table class="attr-table"><tr><td class="attr-name">.execution_id</td><td class = "attrvalue">6</td></tr><tr><td class="attr-name">.component</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">StatisticsGen</span><span class="deemphasize"> at 0x7fa59cbc81f0</span></div><table class="attr-table"><tr><td class="attr-name">.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7fa59cbc8580</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/CsvExampleGen/examples/5)<span class="deemphasize"> at 0x7fa59cbd58b0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/CsvExampleGen/examples/5</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['statistics']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7fa59cbc8670</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: ./pipeline/StatisticsGen/statistics/6)<span class="deemphasize"> at 0x7fa59c912220</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/StatisticsGen/statistics/6</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.exec_properties</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['stats_options_json']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['exclude_splits']</td><td class = "attrvalue">[]</td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">.component.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7fa59cbc8580</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/CsvExampleGen/examples/5)<span class="deemphasize"> at 0x7fa59cbd58b0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/CsvExampleGen/examples/5</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.component.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['statistics']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7fa59cbc8670</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: ./pipeline/StatisticsGen/statistics/6)<span class="deemphasize"> at 0x7fa59c912220</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/StatisticsGen/statistics/6</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr></table></div>




```python
# Display the results
context.show(statistics_gen.outputs['statistics'])
```


<b>Artifact at ./pipeline/StatisticsGen/statistics/6</b><br/><br/>



<div><b>'train' split:</b></div><br/>



<iframe id='facets-iframe' width="100%" height="500px"></iframe>
        <script>
        facets_iframe = document.getElementById('facets-iframe');
        facets_html = '<script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"><\/script><link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/master/facets-dist/facets-jupyter.html"><facets-overview proto-input="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"></facets-overview>';
        facets_iframe.srcdoc = facets_html;
         facets_iframe.id = "";
         setTimeout(() => {
           facets_iframe.setAttribute('height', facets_iframe.contentWindow.document.body.offsetHeight + 'px')
         }, 1500)
         </script>



<div><b>'eval' split:</b></div><br/>



<iframe id='facets-iframe' width="100%" height="500px"></iframe>
        <script>
        facets_iframe = document.getElementById('facets-iframe');
        facets_html = '<script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"><\/script><link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/master/facets-dist/facets-jupyter.html"><facets-overview proto-input="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"></facets-overview>';
        facets_iframe.srcdoc = facets_html;
         facets_iframe.id = "";
         setTimeout(() => {
           facets_iframe.setAttribute('height', facets_iframe.contentWindow.document.body.offsetHeight + 'px')
         }, 1500)
         </script>


Once you've loaded the display, you may notice that the `zeros` column for `Cover_type` is highlighted in red. The visualization is letting us know that this might be a potential issue. In our case though, we know that the `Cover_Type` has a range of [0, 6] so having zeros in this column is something we expect.

<a name='4-4'></a>
### 4.4 - Inferring the Schema

You will need to create a schema to validate incoming datasets during training and serving. Fortunately, TFX allows you to infer a first draft of this schema with the [SchemaGen](https://www.tensorflow.org/tfx/guide/schemagen) component.

<a name='ex-4'></a>
#### Exercise 4: SchemaGen

Use `SchemaGen` to infer a schema based on the computed statistics of `StatisticsGen`.


```python
### START CODE HERE
# Instantiate SchemaGen with the output statistics from the StatisticsGen
schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])
    
    

# Run the component
context.run(schema_gen)
### END CODE HERE
```




<style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object expanded"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExecutionResult</span><span class="deemphasize"> at 0x7fa59c9bebe0</span></div><table class="attr-table"><tr><td class="attr-name">.execution_id</td><td class = "attrvalue">7</td></tr><tr><td class="attr-name">.component</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">SchemaGen</span><span class="deemphasize"> at 0x7fa58baa6280</span></div><table class="attr-table"><tr><td class="attr-name">.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['statistics']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7fa59cbc8670</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: ./pipeline/StatisticsGen/statistics/6)<span class="deemphasize"> at 0x7fa59c912220</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/StatisticsGen/statistics/6</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['schema']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7fa58baa6460</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline/SchemaGen/schema/7)<span class="deemphasize"> at 0x7fa59cbc8760</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/SchemaGen/schema/7</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.exec_properties</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['infer_feature_shape']</td><td class = "attrvalue">False</td></tr><tr><td class="attr-name">['exclude_splits']</td><td class = "attrvalue">[]</td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">.component.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['statistics']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7fa59cbc8670</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: ./pipeline/StatisticsGen/statistics/6)<span class="deemphasize"> at 0x7fa59c912220</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/StatisticsGen/statistics/6</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.component.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['schema']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7fa58baa6460</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline/SchemaGen/schema/7)<span class="deemphasize"> at 0x7fa59cbc8760</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/SchemaGen/schema/7</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr></table></div>




```python
# Visualize the output
context.show(schema_gen.outputs['schema'])
```


<b>Artifact at ./pipeline/SchemaGen/schema/7</b><br/><br/>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Type</th>
      <th>Presence</th>
      <th>Valency</th>
      <th>Domain</th>
    </tr>
    <tr>
      <th>Feature name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'Soil_Type'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'Soil_Type'</td>
    </tr>
    <tr>
      <th>'Wilderness_Area'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'Wilderness_Area'</td>
    </tr>
    <tr>
      <th>'Cover_Type'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'Elevation'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'Hillshade_9am'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'Hillshade_Noon'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'Horizontal_Distance_To_Fire_Points'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'Horizontal_Distance_To_Hydrology'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'Horizontal_Distance_To_Roadways'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'Slope'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'Vertical_Distance_To_Hydrology'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Values</th>
    </tr>
    <tr>
      <th>Domain</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'Soil_Type'</th>
      <td>'C2702', 'C2703', 'C2704', 'C2705', 'C2706', 'C2717', 'C3501', 'C3502', 'C4201', 'C4703', 'C4704', 'C4744', 'C4758', 'C5101', 'C5151', 'C6101', 'C6102', 'C6731', 'C7101', 'C7102', 'C7103', 'C7201', 'C7202', 'C7700', 'C7701', 'C7702', 'C7709', 'C7710', 'C7745', 'C7746', 'C7755', 'C7756', 'C7757', 'C7790', 'C8703', 'C8707', 'C8708', 'C8771', 'C8772', 'C8776'</td>
    </tr>
    <tr>
      <th>'Wilderness_Area'</th>
      <td>'Cache', 'Commanche', 'Neota', 'Rawah'</td>
    </tr>
  </tbody>
</table>
</div>


<a name='4-5'></a>
### 4.5 - Curating the schema

You can see that the inferred schema is able to capture the data types correctly and also able to show the expected values for the qualitative (i.e. string) data. You can still fine-tune this however. For instance, we have features where we expect a certain range:

* `Hillshade_9am`: 0 to 255
* `Hillshade_Noon`: 0 to 255
* `Slope`: 0 to 90
* `Cover_Type`:  0 to 6

You want to update your schema to take note of these so the pipeline can detect if invalid values are being fed to the model.

<a name='ex-5'></a>
#### Exercise 5: Curating the Schema

Use [TFDV](https://www.tensorflow.org/tfx/data_validation/get_started) to update the inferred schema to restrict a range of values to the features mentioned above.

Things to note:
* You can use [tfdv.set_domain()](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv/set_domain) to define acceptable values for a particular feature.
* These should still be INT types after making your changes.
* Declare `Cover_Type` as a *categorical* variable. Unlike the other four features, the integers 0 to 6 here correspond to a designated label and not a quantitative measure. You can look at the available flags for `set_domain()` in the official doc to know how to set this.


```python
try:
    # Get the schema uri
    schema_uri = schema_gen.outputs['schema']._artifacts[0].uri
    
# for grading since context.run() does not work outside the notebook
except IndexError:
    print("context.run() was no-op")
    schema_path = './pipeline/SchemaGen/schema'
    dir_id = os.listdir(schema_path)[0]
    schema_uri = f'{schema_path}/{dir_id}'
```


```python
# Get the schema pbtxt file from the SchemaGen output
schema = tfdv.load_schema_text(os.path.join(schema_uri, 'schema.pbtxt'))
```


```python
### START CODE HERE ###

# Set the two `Hillshade` features to have a range of 0 to 255
tfdv.set_domain(schema, 'Hillshade_9am', schema_pb2.IntDomain(name='Hillshade_9am', min=0, max=255))
tfdv.set_domain(schema, 'Hillshade_Noon', schema_pb2.IntDomain(name='Hillshade_Noon', min=0, max=255))

# Set the `Slope` feature to have a range of 0 to 90
tfdv.set_domain(schema, 'Slope', schema_pb2.IntDomain(name='Slope', min=0, max=90))

# Set `Cover_Type` to categorical having minimum value of 0 and maximum value of 6
tfdv.set_domain(schema, 'Cover_Type', schema_pb2.IntDomain(name='Cover_Type', min=0, max=6, is_categorical=None))

### END CODE HERE ###

tfdv.display_schema(schema=schema)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Type</th>
      <th>Presence</th>
      <th>Valency</th>
      <th>Domain</th>
    </tr>
    <tr>
      <th>Feature name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'Soil_Type'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'Soil_Type'</td>
    </tr>
    <tr>
      <th>'Wilderness_Area'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'Wilderness_Area'</td>
    </tr>
    <tr>
      <th>'Cover_Type'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>[0,6]</td>
    </tr>
    <tr>
      <th>'Elevation'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'Hillshade_9am'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>[0,255]</td>
    </tr>
    <tr>
      <th>'Hillshade_Noon'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>[0,255]</td>
    </tr>
    <tr>
      <th>'Horizontal_Distance_To_Fire_Points'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'Horizontal_Distance_To_Hydrology'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'Horizontal_Distance_To_Roadways'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'Slope'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>[0,90]</td>
    </tr>
    <tr>
      <th>'Vertical_Distance_To_Hydrology'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Values</th>
    </tr>
    <tr>
      <th>Domain</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'Soil_Type'</th>
      <td>'C2702', 'C2703', 'C2704', 'C2705', 'C2706', 'C2717', 'C3501', 'C3502', 'C4201', 'C4703', 'C4704', 'C4744', 'C4758', 'C5101', 'C5151', 'C6101', 'C6102', 'C6731', 'C7101', 'C7102', 'C7103', 'C7201', 'C7202', 'C7700', 'C7701', 'C7702', 'C7709', 'C7710', 'C7745', 'C7746', 'C7755', 'C7756', 'C7757', 'C7790', 'C8703', 'C8707', 'C8708', 'C8771', 'C8772', 'C8776'</td>
    </tr>
    <tr>
      <th>'Wilderness_Area'</th>
      <td>'Cache', 'Commanche', 'Neota', 'Rawah'</td>
    </tr>
  </tbody>
</table>
</div>


You should now see the ranges you declared in the `Domain` column of the schema.

<a name='4-6'></a>
### 4.6 - Schema Environments

In supervised learning, we train the model to make predictions by feeding a set of features with its corresponding label. Thus, our training dataset will have both the input features and label, and the schema is configured to detect these. 

However, after training and you serve the model for inference, the incoming data will no longer have the label. This will present problems when validating the data using the current version of the schema. Let's demonstrate that in the following cells. You will simulate a serving dataset by getting subset of the training set and dropping the label column (i.e. `Cover_Type`). Afterwards, you will validate this serving dataset using the schema you curated earlier.


```python
# Declare paths to the serving data
SERVING_DIR = f'{DATA_DIR}/serving'
SERVING_DATA = f'{SERVING_DIR}/serving_dataset.csv'

# Create the directory
!mkdir -p {SERVING_DIR}
```


```python
# Read a subset of the training dataset
serving_data = pd.read_csv(TRAINING_DATA, nrows=100)

# Drop the `Cover_Type` column
serving_data.drop(columns='Cover_Type', inplace=True)

# Save the modified dataset
serving_data.to_csv(SERVING_DATA, index=False)

# Delete unneeded variable from memory
del serving_data
```


```python
# Declare StatsOptions to use the curated schema
stats_options = tfdv.StatsOptions(schema=schema, infer_type_from_schema=True)

# Compute the statistics of the serving dataset
serving_stats = tfdv.generate_statistics_from_csv(SERVING_DATA, stats_options=stats_options)

# Detect anomalies in the serving dataset
anomalies = tfdv.validate_statistics(serving_stats, schema=schema)

# Display the anomalies detected
tfdv.display_anomalies(anomalies)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Anomaly short description</th>
      <th>Anomaly long description</th>
    </tr>
    <tr>
      <th>Feature name</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'Cover_Type'</th>
      <td>Column dropped</td>
      <td>Column is completely missing</td>
    </tr>
  </tbody>
</table>
</div>


As expected, the missing column is flagged. To fix this, you need to configure the schema to detect when it's being used for training or for inference / serving. You can do this by setting [schema environments](https://www.tensorflow.org/tfx/tutorials/data_validation/tfdv_basic#schema_environments).

<a name='ex-6'></a>
#### Exercise 6: Define the serving environment

Complete the code below to ignore the `Cover_Type` feature when validating in the *SERVING* environment.


```python
schema.default_environment.append('TRAINING')

### START CODE HERE ###
# Hint: Create another default schema environment with name SERVING (pass in a string)
schema.default_environment.append('SERVING')

# Remove Cover_Type feature from SERVING using TFDV
# Hint: Pass in the strings with the name of the feature and environment 
tfdv.get_feature(schema, 'Cover_Type').not_in_environment.append('SERVING')
### END CODE HERE ###
```

If done correctly, running the cell below should show *No Anomalies*.


```python
# Validate the serving dataset statistics in the `SERVING` environment
anomalies = tfdv.validate_statistics(serving_stats, schema=schema, environment='SERVING')

# Display the anomalies detected
tfdv.display_anomalies(anomalies)
```


<h4 style="color:green;">No anomalies found.</h4>


We can now save this curated schema in a local directory so we can import it to our TFX pipeline.


```python
# Declare the path to the updated schema directory
UPDATED_SCHEMA_DIR = f'{PIPELINE_DIR}/updated_schema'

# Create the said directory
!mkdir -p {UPDATED_SCHEMA_DIR}

# Declare the path to the schema file
schema_file = os.path.join(UPDATED_SCHEMA_DIR, 'schema.pbtxt')

# Save the curated schema to the said file
tfdv.write_schema_text(schema, schema_file)
```

As a sanity check, let's display the schema we just saved and verify that it contains the changes we introduced. It should still show the ranges in the `Domain` column and there should be two environments available.


```python
# Load the schema from the directory we just created
new_schema = tfdv.load_schema_text(schema_file)

# Display the schema. Check that the Domain column still contains the ranges.
tfdv.display_schema(schema=new_schema)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Type</th>
      <th>Presence</th>
      <th>Valency</th>
      <th>Domain</th>
    </tr>
    <tr>
      <th>Feature name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'Soil_Type'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'Soil_Type'</td>
    </tr>
    <tr>
      <th>'Wilderness_Area'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'Wilderness_Area'</td>
    </tr>
    <tr>
      <th>'Cover_Type'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>[0,6]</td>
    </tr>
    <tr>
      <th>'Elevation'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'Hillshade_9am'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>[0,255]</td>
    </tr>
    <tr>
      <th>'Hillshade_Noon'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>[0,255]</td>
    </tr>
    <tr>
      <th>'Horizontal_Distance_To_Fire_Points'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'Horizontal_Distance_To_Hydrology'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'Horizontal_Distance_To_Roadways'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'Slope'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>[0,90]</td>
    </tr>
    <tr>
      <th>'Vertical_Distance_To_Hydrology'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Values</th>
    </tr>
    <tr>
      <th>Domain</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'Soil_Type'</th>
      <td>'C2702', 'C2703', 'C2704', 'C2705', 'C2706', 'C2717', 'C3501', 'C3502', 'C4201', 'C4703', 'C4704', 'C4744', 'C4758', 'C5101', 'C5151', 'C6101', 'C6102', 'C6731', 'C7101', 'C7102', 'C7103', 'C7201', 'C7202', 'C7700', 'C7701', 'C7702', 'C7709', 'C7710', 'C7745', 'C7746', 'C7755', 'C7756', 'C7757', 'C7790', 'C8703', 'C8707', 'C8708', 'C8771', 'C8772', 'C8776'</td>
    </tr>
    <tr>
      <th>'Wilderness_Area'</th>
      <td>'Cache', 'Commanche', 'Neota', 'Rawah'</td>
    </tr>
  </tbody>
</table>
</div>



```python
# The environment list should show `TRAINING` and `SERVING`.
new_schema.default_environment
```




    ['TRAINING', 'SERVING']



<a name='4-7'></a>
### 4.7 - Generate new statistics using the updated schema

You will now compute the statistics using the schema you just curated. Remember though that TFX components interact with each other by getting artifact information from the metadata store. So you first have to import the curated schema file into ML Metadata. You will do that by using an [ImporterNode](https://www.tensorflow.org/tfx/guide/statsgen#using_the_statsgen_component_with_a_schema) to create an artifact representing the curated schema.

<a name='ex-7'></a>
#### Exercise 7: ImporterNode

Complete the code below to create a `Schema` artifact that points to the curated schema directory. Pass in an `instance_name` as well and name it `import_user_schema`.


```python
### START CODE HERE ###

# Use an ImporterNode to put the curated schema to ML Metadata
user_schema_importer = ImporterNode(instance_name='import_user_schema',
                                   source_uri=UPDATED_SCHEMA_DIR,
                                   artifact_type=standard_artifacts.Schema)
    
    
    
# Run the component
context.run(user_schema_importer, enable_cache=False)

### END CODE HERE ###

context.show(user_schema_importer.outputs['result'])
```


<b>Artifact at ./pipeline/updated_schema</b><br/><br/>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Type</th>
      <th>Presence</th>
      <th>Valency</th>
      <th>Domain</th>
    </tr>
    <tr>
      <th>Feature name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'Soil_Type'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'Soil_Type'</td>
    </tr>
    <tr>
      <th>'Wilderness_Area'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'Wilderness_Area'</td>
    </tr>
    <tr>
      <th>'Cover_Type'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>[0,6]</td>
    </tr>
    <tr>
      <th>'Elevation'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'Hillshade_9am'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>[0,255]</td>
    </tr>
    <tr>
      <th>'Hillshade_Noon'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>[0,255]</td>
    </tr>
    <tr>
      <th>'Horizontal_Distance_To_Fire_Points'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'Horizontal_Distance_To_Hydrology'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'Horizontal_Distance_To_Roadways'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'Slope'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>[0,90]</td>
    </tr>
    <tr>
      <th>'Vertical_Distance_To_Hydrology'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Values</th>
    </tr>
    <tr>
      <th>Domain</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'Soil_Type'</th>
      <td>'C2702', 'C2703', 'C2704', 'C2705', 'C2706', 'C2717', 'C3501', 'C3502', 'C4201', 'C4703', 'C4704', 'C4744', 'C4758', 'C5101', 'C5151', 'C6101', 'C6102', 'C6731', 'C7101', 'C7102', 'C7103', 'C7201', 'C7202', 'C7700', 'C7701', 'C7702', 'C7709', 'C7710', 'C7745', 'C7746', 'C7755', 'C7756', 'C7757', 'C7790', 'C8703', 'C8707', 'C8708', 'C8771', 'C8772', 'C8776'</td>
    </tr>
    <tr>
      <th>'Wilderness_Area'</th>
      <td>'Cache', 'Commanche', 'Neota', 'Rawah'</td>
    </tr>
  </tbody>
</table>
</div>


With the artifact successfully created, you can now use `StatisticsGen` and pass in a `schema` parameter to use the curated schema.

<a name='ex-8'></a>
#### Exercise 8: Statistics with the new schema

Use `StatisticsGen` to compute the statistics with the schema you updated in the previous section.


```python
### START CODE HERE ###
# Use StatisticsGen to compute the statistics using the curated schema
statistics_gen_updated = StatisticsGen(examples=example_gen.outputs['examples'],
                                      schema=user_schema_importer.outputs['result'])
    
    

# Run the component
context.run(statistics_gen_updated)
### END CODE HERE ###
```




<style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object expanded"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExecutionResult</span><span class="deemphasize"> at 0x7fa5869d6a60</span></div><table class="attr-table"><tr><td class="attr-name">.execution_id</td><td class = "attrvalue">9</td></tr><tr><td class="attr-name">.component</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">StatisticsGen</span><span class="deemphasize"> at 0x7fa5869719a0</span></div><table class="attr-table"><tr><td class="attr-name">.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7fa59cbc8580</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/CsvExampleGen/examples/5)<span class="deemphasize"> at 0x7fa59cbd58b0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/CsvExampleGen/examples/5</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['schema']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7fa59cc13a30</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline/updated_schema)<span class="deemphasize"> at 0x7fa59cc135e0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/updated_schema</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['statistics']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7fa586971820</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: ./pipeline/StatisticsGen/statistics/9)<span class="deemphasize"> at 0x7fa57a5cb5e0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/StatisticsGen/statistics/9</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.exec_properties</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['stats_options_json']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['exclude_splits']</td><td class = "attrvalue">[]</td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">.component.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7fa59cbc8580</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/CsvExampleGen/examples/5)<span class="deemphasize"> at 0x7fa59cbd58b0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/CsvExampleGen/examples/5</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['schema']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7fa59cc13a30</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline/updated_schema)<span class="deemphasize"> at 0x7fa59cc135e0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/updated_schema</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.component.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['statistics']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7fa586971820</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: ./pipeline/StatisticsGen/statistics/9)<span class="deemphasize"> at 0x7fa57a5cb5e0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/StatisticsGen/statistics/9</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr></table></div>




```python
context.show(statistics_gen_updated.outputs['statistics'])
```


<b>Artifact at ./pipeline/StatisticsGen/statistics/9</b><br/><br/>



<div><b>'train' split:</b></div><br/>



<iframe id='facets-iframe' width="100%" height="500px"></iframe>
        <script>
        facets_iframe = document.getElementById('facets-iframe');
        facets_html = '<script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"><\/script><link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/master/facets-dist/facets-jupyter.html"><facets-overview proto-input="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"></facets-overview>';
        facets_iframe.srcdoc = facets_html;
         facets_iframe.id = "";
         setTimeout(() => {
           facets_iframe.setAttribute('height', facets_iframe.contentWindow.document.body.offsetHeight + 'px')
         }, 1500)
         </script>



<div><b>'eval' split:</b></div><br/>



<iframe id='facets-iframe' width="100%" height="500px"></iframe>
        <script>
        facets_iframe = document.getElementById('facets-iframe');
        facets_html = '<script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"><\/script><link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/master/facets-dist/facets-jupyter.html"><facets-overview proto-input="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"></facets-overview>';
        facets_iframe.srcdoc = facets_html;
         facets_iframe.id = "";
         setTimeout(() => {
           facets_iframe.setAttribute('height', facets_iframe.contentWindow.document.body.offsetHeight + 'px')
         }, 1500)
         </script>


The chart will look mostly the same from the previous runs but you can see that the `Cover Type` is now under the categorical features. That shows that `StatisticsGen` is indeed using the updated schema.

<a name='4-8'></a>
### 4.8 - Check anomalies

You will now check if the dataset has any anomalies with respect to the schema. You can do that easily with the [ExampleValidator](https://www.tensorflow.org/tfx/guide/exampleval) component.

<a name='ex-9'></a>
#### Exercise 9: ExampleValidator

Check if there are any anomalies using `ExampleValidator`. You will need to pass in the updated statistics and schema from the previous sections.


```python
### START CODE HERE ###

example_validator = ExampleValidator(statistics=statistics_gen_updated.outputs['statistics'],
                                    schema=user_schema_importer.outputs['result'])
    
    

# Run the component.
context.run(example_validator)

### END CODE HERE ###
```




<style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object expanded"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExecutionResult</span><span class="deemphasize"> at 0x7fa57a02c760</span></div><table class="attr-table"><tr><td class="attr-name">.execution_id</td><td class = "attrvalue">10</td></tr><tr><td class="attr-name">.component</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExampleValidator</span><span class="deemphasize"> at 0x7fa57a318a30</span></div><table class="attr-table"><tr><td class="attr-name">.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['statistics']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7fa586971820</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: ./pipeline/StatisticsGen/statistics/9)<span class="deemphasize"> at 0x7fa57a5cb5e0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/StatisticsGen/statistics/9</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['schema']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7fa59cc13a30</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline/updated_schema)<span class="deemphasize"> at 0x7fa59cc135e0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/updated_schema</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['anomalies']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleAnomalies'</span> (1 artifact)<span class="deemphasize"> at 0x7fa57a318e80</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleAnomalies</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleAnomalies'</span> (uri: ./pipeline/ExampleValidator/anomalies/10)<span class="deemphasize"> at 0x7fa57a02cbb0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleAnomalies&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/ExampleValidator/anomalies/10</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.exec_properties</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['exclude_splits']</td><td class = "attrvalue">[]</td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">.component.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['statistics']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7fa586971820</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: ./pipeline/StatisticsGen/statistics/9)<span class="deemphasize"> at 0x7fa57a5cb5e0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/StatisticsGen/statistics/9</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['schema']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7fa59cc13a30</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline/updated_schema)<span class="deemphasize"> at 0x7fa59cc135e0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/updated_schema</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.component.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['anomalies']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleAnomalies'</span> (1 artifact)<span class="deemphasize"> at 0x7fa57a318e80</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleAnomalies</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleAnomalies'</span> (uri: ./pipeline/ExampleValidator/anomalies/10)<span class="deemphasize"> at 0x7fa57a02cbb0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleAnomalies&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/ExampleValidator/anomalies/10</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr></table></div>




```python
# Visualize the results
context.show(example_validator.outputs['anomalies'])
```


<b>Artifact at ./pipeline/ExampleValidator/anomalies/10</b><br/><br/>



<div><b>'train' split:</b></div><br/>



<h4 style="color:green;">No anomalies found.</h4>



<div><b>'eval' split:</b></div><br/>



<h4 style="color:green;">No anomalies found.</h4>


<a name='4-10'></a>
### 4.10 - Feature engineering

You will now proceed to transforming your features to a form suitable for training a model. This can include several methods such as scaling and converting strings to vocabulary indices. It is important for these transformations to be consistent across your training data, and also for the serving data when the model is deployed for inference. TFX ensures this by generating a graph that will process incoming data both during training and inference.

Let's first declare the constants and utility function you will use for the exercise.


```python
# Set the constants module filename
_cover_constants_module_file = 'cover_constants.py'
```


```python
%%writefile {_cover_constants_module_file}

SCALE_MINMAX_FEATURE_KEYS = [
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
    ]

SCALE_01_FEATURE_KEYS = [
        "Hillshade_9am",
        "Hillshade_Noon",
        "Horizontal_Distance_To_Fire_Points",
    ]

SCALE_Z_FEATURE_KEYS = [
        "Elevation",
        "Slope",
        "Horizontal_Distance_To_Roadways",
    ]

VOCAB_FEATURE_KEYS = ["Wilderness_Area"]

HASH_STRING_FEATURE_KEYS = ["Soil_Type"]

LABEL_KEY = "Cover_Type"

# Utility function for renaming the feature
def transformed_name(key):
    return key + '_xf'
```

    Overwriting cover_constants.py


Next you will define the `preprocessing_fn` to apply transformations to the features. 

<a name='ex-10'></a>
#### Exercise 10: Preprocessing function

Complete the module to transform your features. Refer to the code comments to get hints on what operations to perform.

Here are some links to the docs of the functions you will need to complete this function:

- [`tft.scale_by_min_max`](https://www.tensorflow.org/tfx/transform/api_docs/python/tft/scale_by_min_max)
- [`tft.scale_to_0_1`](https://www.tensorflow.org/tfx/transform/api_docs/python/tft/scale_to_0_1)
- [`tft.scale_to_z_score`](https://www.tensorflow.org/tfx/transform/api_docs/python/tft/scale_to_z_score)
- [`tft.compute_and_apply_vocabulary`](https://www.tensorflow.org/tfx/transform/api_docs/python/tft/compute_and_apply_vocabulary)
- [`tft.hash_strings`](https://www.tensorflow.org/tfx/transform/api_docs/python/tft/hash_strings)


```python
# Set the transform module filename
_cover_transform_module_file = 'cover_transform.py'
```


```python
%%writefile {_cover_transform_module_file}

import tensorflow as tf
import tensorflow_transform as tft

import cover_constants

_SCALE_MINMAX_FEATURE_KEYS = cover_constants.SCALE_MINMAX_FEATURE_KEYS
_SCALE_01_FEATURE_KEYS = cover_constants.SCALE_01_FEATURE_KEYS
_SCALE_Z_FEATURE_KEYS = cover_constants.SCALE_Z_FEATURE_KEYS
_VOCAB_FEATURE_KEYS = cover_constants.VOCAB_FEATURE_KEYS
_HASH_STRING_FEATURE_KEYS = cover_constants.HASH_STRING_FEATURE_KEYS
_LABEL_KEY = cover_constants.LABEL_KEY
_transformed_name = cover_constants.transformed_name

def preprocessing_fn(inputs):

    features_dict = {}

    ### START CODE HERE ###
    for feature in _SCALE_MINMAX_FEATURE_KEYS:
        data_col = inputs[feature] 
        # Transform using scaling of min_max function
        # Hint: Use tft.scale_by_min_max by passing in the respective column
        features_dict[_transformed_name(feature)] = tft.scale_by_min_max(data_col)

    for feature in _SCALE_01_FEATURE_KEYS:
        data_col = inputs[feature] 
        # Transform using scaling of 0 to 1 function
        # Hint: tft.scale_to_0_1
        features_dict[_transformed_name(feature)] = tft.scale_to_0_1(data_col)

    for feature in _SCALE_Z_FEATURE_KEYS:
        data_col = inputs[feature] 
        # Transform using scaling to z score
        # Hint: tft.scale_to_z_score
        features_dict[_transformed_name(feature)] = tft.scale_to_z_score(data_col)

    for feature in _VOCAB_FEATURE_KEYS:
        data_col = inputs[feature] 
        # Transform using vocabulary available in column
        # Hint: Use tft.compute_and_apply_vocabulary
        features_dict[_transformed_name(feature)] = tft.compute_and_apply_vocabulary(data_col)

    for feature in _HASH_STRING_FEATURE_KEYS:
        data_col = inputs[feature] 
        # Transform by hashing strings into buckets
        # Hint: Use tft.hash_strings with the param hash_buckets set to 10
        features_dict[_transformed_name(feature)] = tft.hash_strings(data_col, hash_buckets=10)
    
    ### END CODE HERE ###  

    # No change in the label
    features_dict[_LABEL_KEY] = inputs[_LABEL_KEY]

    return features_dict

```

    Overwriting cover_transform.py


<a name='ex-11'></a>
#### Exercise 11: Transform

Use the [TFX Transform component](https://www.tensorflow.org/tfx/api_docs/python/tfx/components/Transform) to perform the transformations and generate the transformation graph. You will need to pass in the dataset examples, *curated* schema, and the module that contains the preprocessing function.


```python
### START CODE HERE ###
# Instantiate the Transform component
transform = Transform(examples=example_gen.outputs['examples'],
                      schema=user_schema_importer.outputs['result'],
                      module_file=os.path.abspath(_cover_transform_module_file)
                     )
    
    
    
### END CODE HERE ###

# Run the component
context.run(transform, enable_cache=False)

```

    WARNING:root:This output type hint will be ignored and not used for type-checking purposes. Typically, output type hints for a PTransform are single (or nested) types wrapped by a PCollection, PDone, or None. Got: Tuple[Dict[str, Union[NoneType, _Dataset]], Union[Dict[str, Dict[str, PCollection]], NoneType]] instead.
    WARNING:root:This output type hint will be ignored and not used for type-checking purposes. Typically, output type hints for a PTransform are single (or nested) types wrapped by a PCollection, PDone, or None. Got: Tuple[Dict[str, Union[NoneType, _Dataset]], Union[Dict[str, Dict[str, PCollection]], NoneType]] instead.
    WARNING:apache_beam.typehints.typehints:Ignoring send_type hint: <class 'NoneType'>
    WARNING:apache_beam.typehints.typehints:Ignoring return_type hint: <class 'NoneType'>
    WARNING:apache_beam.typehints.typehints:Ignoring send_type hint: <class 'NoneType'>
    WARNING:apache_beam.typehints.typehints:Ignoring return_type hint: <class 'NoneType'>
    WARNING:apache_beam.typehints.typehints:Ignoring send_type hint: <class 'NoneType'>
    WARNING:apache_beam.typehints.typehints:Ignoring return_type hint: <class 'NoneType'>
    WARNING:apache_beam.typehints.typehints:Ignoring send_type hint: <class 'NoneType'>
    WARNING:apache_beam.typehints.typehints:Ignoring return_type hint: <class 'NoneType'>
    WARNING:apache_beam.typehints.typehints:Ignoring send_type hint: <class 'NoneType'>
    WARNING:apache_beam.typehints.typehints:Ignoring return_type hint: <class 'NoneType'>
    WARNING:apache_beam.typehints.typehints:Ignoring send_type hint: <class 'NoneType'>
    WARNING:apache_beam.typehints.typehints:Ignoring return_type hint: <class 'NoneType'>





<style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object expanded"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExecutionResult</span><span class="deemphasize"> at 0x7fa4f05f9040</span></div><table class="attr-table"><tr><td class="attr-name">.execution_id</td><td class = "attrvalue">14</td></tr><tr><td class="attr-name">.component</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Transform</span><span class="deemphasize"> at 0x7fa4f04709a0</span></div><table class="attr-table"><tr><td class="attr-name">.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7fa59cbc8580</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/CsvExampleGen/examples/5)<span class="deemphasize"> at 0x7fa59cbd58b0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/CsvExampleGen/examples/5</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['schema']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7fa59cc13a30</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline/updated_schema)<span class="deemphasize"> at 0x7fa59cc135e0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/updated_schema</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['transform_graph']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'TransformGraph'</span> (1 artifact)<span class="deemphasize"> at 0x7fa4f0470eb0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">TransformGraph</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'TransformGraph'</span> (uri: ./pipeline/Transform/transform_graph/14)<span class="deemphasize"> at 0x7fa4f0470a90</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.TransformGraph&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/transform_graph/14</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['transformed_examples']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7fa4f0470520</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/Transform/transformed_examples/14)<span class="deemphasize"> at 0x7fa4f0470100</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/transformed_examples/14</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['updated_analyzer_cache']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'TransformCache'</span> (1 artifact)<span class="deemphasize"> at 0x7fa4f04709d0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">TransformCache</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'TransformCache'</span> (uri: ./pipeline/Transform/updated_analyzer_cache/14)<span class="deemphasize"> at 0x7fa4f16f8a30</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.TransformCache&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/updated_analyzer_cache/14</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.exec_properties</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['module_file']</td><td class = "attrvalue">/home/jovyan/work/cover_transform.py</td></tr><tr><td class="attr-name">['preprocessing_fn']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['custom_config']</td><td class = "attrvalue">null</td></tr><tr><td class="attr-name">['splits_config']</td><td class = "attrvalue">None</td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">.component.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7fa59cbc8580</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/CsvExampleGen/examples/5)<span class="deemphasize"> at 0x7fa59cbd58b0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/CsvExampleGen/examples/5</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['schema']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7fa59cc13a30</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline/updated_schema)<span class="deemphasize"> at 0x7fa59cc135e0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/updated_schema</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.component.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['transform_graph']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'TransformGraph'</span> (1 artifact)<span class="deemphasize"> at 0x7fa4f0470eb0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">TransformGraph</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'TransformGraph'</span> (uri: ./pipeline/Transform/transform_graph/14)<span class="deemphasize"> at 0x7fa4f0470a90</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.TransformGraph&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/transform_graph/14</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['transformed_examples']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7fa4f0470520</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/Transform/transformed_examples/14)<span class="deemphasize"> at 0x7fa4f0470100</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/transformed_examples/14</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['updated_analyzer_cache']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'TransformCache'</span> (1 artifact)<span class="deemphasize"> at 0x7fa4f04709d0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">TransformCache</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'TransformCache'</span> (uri: ./pipeline/Transform/updated_analyzer_cache/14)<span class="deemphasize"> at 0x7fa4f16f8a30</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.TransformCache&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/updated_analyzer_cache/14</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr></table></div>



Let's inspect a few examples of the transformed dataset to see if the transformations are done correctly.


```python
try:
    transform_uri = transform.outputs['transformed_examples'].get()[0].uri

# for grading since context.run() does not work outside the notebook
except IndexError:
    print("context.run() was no-op")
    examples_path = './pipeline/Transform/transformed_examples'
    dir_id = os.listdir(examples_path)[0]
    transform_uri = f'{examples_path}/{dir_id}'
```


```python
# Get the URI of the output artifact representing the transformed examples
train_uri = os.path.join(transform_uri, 'train')

# Get the list of files in this directory (all compressed TFRecord files)
tfrecord_filenames = [os.path.join(train_uri, name)
                      for name in os.listdir(train_uri)]

# Create a `TFRecordDataset` to read these files
transformed_dataset = tf.data.TFRecordDataset(tfrecord_filenames, compression_type="GZIP")
```


```python
# import helper function to get examples from the dataset
from util import get_records

# Get 3 records from the dataset
sample_records_xf = get_records(transformed_dataset, 3)

# Print the output
pp.pprint(sample_records_xf)
```

    [{'features': {'feature': {'Cover_Type': {'int64List': {'value': ['4']}},
                               'Elevation_xf': {'floatList': {'value': [-1.2982628]}},
                               'Hillshade_9am_xf': {'floatList': {'value': [0.87007874]}},
                               'Hillshade_Noon_xf': {'floatList': {'value': [0.9133858]}},
                               'Horizontal_Distance_To_Fire_Points_xf': {'floatList': {'value': [0.875366]}},
                               'Horizontal_Distance_To_Hydrology_xf': {'floatList': {'value': [0.18468146]}},
                               'Horizontal_Distance_To_Roadways_xf': {'floatList': {'value': [-1.1803539]}},
                               'Slope_xf': {'floatList': {'value': [-1.483387]}},
                               'Soil_Type_xf': {'int64List': {'value': ['4']}},
                               'Vertical_Distance_To_Hydrology_xf': {'floatList': {'value': [0.22351421]}},
                               'Wilderness_Area_xf': {'int64List': {'value': ['0']}}}}},
     {'features': {'feature': {'Cover_Type': {'int64List': {'value': ['4']}},
                               'Elevation_xf': {'floatList': {'value': [-1.3197033]}},
                               'Hillshade_9am_xf': {'floatList': {'value': [0.86614174]}},
                               'Hillshade_Noon_xf': {'floatList': {'value': [0.9251968]}},
                               'Horizontal_Distance_To_Fire_Points_xf': {'floatList': {'value': [0.8678377]}},
                               'Horizontal_Distance_To_Hydrology_xf': {'floatList': {'value': [0.15175375]}},
                               'Horizontal_Distance_To_Roadways_xf': {'floatList': {'value': [-1.2572862]}},
                               'Slope_xf': {'floatList': {'value': [-1.6169325]}},
                               'Soil_Type_xf': {'int64List': {'value': ['4']}},
                               'Vertical_Distance_To_Hydrology_xf': {'floatList': {'value': [0.21576227]}},
                               'Wilderness_Area_xf': {'int64List': {'value': ['0']}}}}},
     {'features': {'feature': {'Cover_Type': {'int64List': {'value': ['1']}},
                               'Elevation_xf': {'floatList': {'value': [-0.5549895]}},
                               'Hillshade_9am_xf': {'floatList': {'value': [0.9212598]}},
                               'Hillshade_Noon_xf': {'floatList': {'value': [0.93700784]}},
                               'Horizontal_Distance_To_Fire_Points_xf': {'floatList': {'value': [0.8533389]}},
                               'Horizontal_Distance_To_Hydrology_xf': {'floatList': {'value': [0.19183965]}},
                               'Horizontal_Distance_To_Roadways_xf': {'floatList': {'value': [0.53138816]}},
                               'Slope_xf': {'floatList': {'value': [-0.6821134]}},
                               'Soil_Type_xf': {'int64List': {'value': ['4']}},
                               'Vertical_Distance_To_Hydrology_xf': {'floatList': {'value': [0.30749354]}},
                               'Wilderness_Area_xf': {'int64List': {'value': ['0']}}}}}]


<a name='5'></a>
## 5 - ML Metadata

TFX uses [ML Metadata](https://www.tensorflow.org/tfx/guide/mlmd) under the hood to keep records of artifacts that each component uses. This makes it easier to track how the pipeline is run so you can troubleshoot if needed or want to reproduce results.

In this final section of the assignment, you will demonstrate going through this metadata store to retrieve related artifacts. This skill is useful for when you want to recall which inputs are fed to a particular stage of the pipeline. For example, you can know where to locate the schema used to perform feature transformation, or you can determine which set of examples were used to train a model.

You will start by importing the relevant modules and setting up the connection to the metadata store. We have also provided some helper functions for displaying artifact information and you can review its code in the external `util.py` module in your lab workspace.


```python
# Import mlmd and utilities
import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2
from util import display_types, display_artifacts, display_properties

# Get the connection config to connect to the metadata store
connection_config = context.metadata_connection_config

# Instantiate a MetadataStore instance with the connection config
store = mlmd.MetadataStore(connection_config)

# Declare the base directory where All TFX artifacts are stored
base_dir = connection_config.sqlite.filename_uri.split('metadata.sqlite')[0]
```

<a name='5-1'></a>
#### 5.1 -  Accessing stored artifacts

With the connection setup, you can now interact with the metadata store. For instance, you can retrieve all artifact types stored with the `get_artifact_types()` function. For reference, the API is documented [here](https://www.tensorflow.org/tfx/ml_metadata/api_docs/python/mlmd/MetadataStore).


```python
# Get the artifact types
types = store.get_artifact_types()

# Display the results
display_types(types)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>Examples</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>ExampleStatistics</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>Schema</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>ExampleAnomalies</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14</td>
      <td>TransformGraph</td>
    </tr>
    <tr>
      <th>5</th>
      <td>15</td>
      <td>TransformCache</td>
    </tr>
  </tbody>
</table>
</div>



You can also get a list of artifacts for a particular type to see if there are variations used in the pipeline. For example, you curated a schema in an earlier part of the assignment so this should appear in the records. Running the cell below should show at least two rows: one for the inferred schema, and another for the updated schema. If you ran this notebook before, then you might see more rows because of the different schema artifacts saved under the `./SchemaGen/schema` directory.


```python
# Retrieve the transform graph list
schema_list = store.get_artifacts_by_type('Schema')

# Display artifact properties from the results
display_artifacts(store, schema_list, base_dir)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artifact id</th>
      <th>type</th>
      <th>uri</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>Schema</td>
      <td>./SchemaGen/schema/3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>Schema</td>
      <td>./updated_schema</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>Schema</td>
      <td>./SchemaGen/schema/7</td>
    </tr>
  </tbody>
</table>
</div>



Moreover, you can also get the properties of a particular artifact. TFX declares some properties automatically for each of its components. You will most likely see `name`, `state` and `producer_component` for each artifact type. Additional properties are added where appropriate. For example, a `split_names` property is added in `ExampleStatistics` artifacts to indicate which splits the statistics are generated for.


```python
# Get the latest TransformGraph artifact
statistics_artifact = store.get_artifacts_by_type('ExampleStatistics')[-1]

# Display the properties of the retrieved artifact
display_properties(store, statistics_artifact)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>property</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>split_names</td>
      <td>["train", "eval"]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>name</td>
      <td>statistics</td>
    </tr>
    <tr>
      <th>2</th>
      <td>state</td>
      <td>published</td>
    </tr>
    <tr>
      <th>3</th>
      <td>producer_component</td>
      <td>StatisticsGen</td>
    </tr>
  </tbody>
</table>
</div>



<a name='5-2'></a>
#### 5.2 - Tracking artifacts

For this final exercise, you will build a function to return the parent artifacts of a given one. For example, this should be able to list the artifacts that were used to generate a particular `TransformGraph` instance. 

<a name='ex-12'></a>
##### Exercise 12: Get parent artifacts

Complete the code below to track the inputs of a particular artifact.

Tips:

* You may find [get_events_by_artifact_ids()](https://www.tensorflow.org/tfx/ml_metadata/api_docs/python/mlmd/MetadataStore#get_events_by_artifact_ids) and [get_events_by_execution_ids()](https://www.tensorflow.org/tfx/ml_metadata/api_docs/python/mlmd/MetadataStore#get_executions_by_id) useful here. 

* Some of the methods of the MetadataStore class (such as the two given above) only accepts iterables so remember to convert to a list (or set) if you only have an int (e.g. pass `[x]` instead of `x`).




```python
def get_parent_artifacts(store, artifact):

    ### START CODE HERE ###
    
    # Get the artifact id of the input artifact
    artifact_id = artifact.id
    #artifact_id = store.get_artifact_by_id(artifact).id
    
    # Get events associated with the artifact id
    artifact_id_events = store.get_events_by_artifact_ids([artifact_id])
    
    # From the `artifact_id_events`, get the execution ids of OUTPUT events.
    # Cast to a set to remove duplicates if any.
    execution_id = set( 
        event.execution_id
        for event in artifact_id_events
        if event.type == metadata_store_pb2.Event.OUTPUT
    )
    
    # Get the events associated with the execution_id
    execution_id_events = store.get_events_by_execution_ids(execution_id)

    # From execution_id_events, get the artifact ids of INPUT events.
    # Cast to a set to remove duplicates if any.
    parent_artifact_ids = set( 
        event.artifact_id
        for event in execution_id_events
        if event.type == metadata_store_pb2.Event.INPUT
    )
    
    
    # Get the list of artifacts associated with the parent_artifact_ids
    parent_artifact_list = store.get_artifacts_by_id(parent_artifact_ids)
    

    ### END CODE HERE ###
    
    return parent_artifact_list
```


```python
# Get an artifact instance from the metadata store
artifact_instance = store.get_artifacts_by_type('TransformGraph')[0]

# Retrieve the parent artifacts of the instance
parent_artifacts = get_parent_artifacts(store, artifact_instance)

# Display the results
display_artifacts(store, parent_artifacts, base_dir)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artifact id</th>
      <th>type</th>
      <th>uri</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>Schema</td>
      <td>./updated_schema</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>Examples</td>
      <td>./CsvExampleGen/examples/5</td>
    </tr>
  </tbody>
</table>
</div>



**Expected Output:**

*Note: The ID numbers may differ.*

| artifact id | type | uri |
| ----------- | ---- | --- |
| 1	| Examples | ./CsvExampleGen/examples/1 |
| 4	| Schema | ./updated_schema |

**Congratulations!** You have now completed the assignment for this week. You've demonstrated your skills in selecting features, performing a data pipeline, and retrieving information from the metadata store. Having the ability to put these all together will be critical when working with production grade machine learning projects. For next week, you will work on more data types and see how these can be prepared in an ML pipeline. **Keep it up!**
