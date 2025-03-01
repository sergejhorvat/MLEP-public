# Week 2 Assignment: Feature Engineering

For this week's assignment, you will build a data pipeline using using [Tensorflow Extended (TFX)](https://www.tensorflow.org/tfx) to prepare features from the [Metro Interstate Traffic Volume dataset](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume). Try to only use the documentation and code hints to accomplish the tasks but feel free to review the 2nd ungraded lab this week in case you get stuck.

Upon completion, you will have:

* created an InteractiveContext to run TFX components interactively
* used TFX ExampleGen component to split your dataset into training and evaluation datasets
* generated the statistics and the schema of your dataset using TFX StatisticsGen and SchemaGen components
* validated the evaluation dataset statistics using TFX ExampleValidator
* performed feature engineering using the TFX Transform component

Let's begin!

## Table of Contents

- [1 - Setup](#1)
  - [1.1 - Imports](#1-1)
  - [1.2 - Define Paths](#1-2)
  - [1.3 - Preview the Dataset](#1-3)
  - [1.4 - Create the InteractiveContext](#1-4)
- [2 - Run TFX components interactively](#2)
  - [2.1 - ExampleGen](#2-1)
    - [Exercise 1 - ExampleGen](#ex-1)
    - [Exercise 2 - get_records()](#ex-2)
  - [2.2 - StatisticsGen](#2-2)
    - [Exercise 3 - StatisticsGen](#ex-3)
  - [2.3 - SchemaGen](#2-3)
    - [Exercise 4 - SchemaGen](#ex-4)
  - [2.4 - ExampleValidator](#2-4)
    - [Exercise 5 - ExampleValidator](#ex-5)
  - [2.5 - Transform](#2-5)
    - [Exercise 6 - preprocessing_fn()](#ex-6)
    - [Exercise 7 - Transform](#ex-7)

<a name='1'></a>
## 1 - Setup
As usual, you will first need to import the necessary packages. For reference, the lab environment uses *TensorFlow version: 2.3.1* and *TFX version: 0.24.0*.

<a name='1-1'></a>
### 1.1 Imports


```python
import tensorflow as tf

from tfx.components import CsvExampleGen
from tfx.components import ExampleValidator
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Transform

from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from google.protobuf.json_format import MessageToDict

import os
import pprint

pp = pprint.PrettyPrinter()
```

<a name='1-2'></a>
### 1.2 - Define paths

You will define a few global variables to indicate paths in the local workspace.


```python
# location of the pipeline metadata store
_pipeline_root = 'metro_traffic_pipeline/'

# directory of the raw data files
_data_root = 'metro_traffic_pipeline/data'

# path to the raw training data
_data_filepath = os.path.join(_data_root, 'metro_traffic_volume.csv')
```

<a name='1-3'></a>
### 1.3 - Preview the  dataset

The [Metro Interstate Traffic Volume dataset](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume) contains hourly traffic volume of a road in Minnesota from 2012-2018. With this data, you can develop a model for predicting the traffic volume given the date, time, and weather conditions. The attributes are:

* **holiday** - US National holidays plus regional holiday, Minnesota State Fair
* **temp** - Average temp in Kelvin
* **rain_1h** - Amount in mm of rain that occurred in the hour
* **snow_1h** - Amount in mm of snow that occurred in the hour
* **clouds_all** - Percentage of cloud cover
* **weather_main** - Short textual description of the current weather
* **weather_description** - Longer textual description of the current weather
* **date_time** - DateTime Hour of the data collected in local CST time
* **traffic_volume** - Numeric Hourly I-94 ATR 301 reported westbound traffic volume
* **month** - taken from date_time
* **day** - taken from date_time
* **day_of_week** - taken from date_time
* **hour** - taken from date_time


*Disclaimer: We added the last four attributes shown above (i.e. month, day, day_of_week, hour) to the original dataset to increase the features you can transform later.*

Take a quick look at the first few rows of the CSV file.


```python
# Preview the dataset
!head {_data_filepath}
```

    holiday,temp,rain_1h,snow_1h,clouds_all,weather_main,weather_description,date_time,traffic_volume,month,day,day_of_week,hour
    None,288.28,0.0,0.0,40,Clouds,scattered clouds,2012-10-02 09:00:00,5545,10,2,1,9
    None,289.36,0.0,0.0,75,Clouds,broken clouds,2012-10-02 10:00:00,4516,10,2,1,10
    None,289.58,0.0,0.0,90,Clouds,overcast clouds,2012-10-02 11:00:00,4767,10,2,1,11
    None,290.13,0.0,0.0,90,Clouds,overcast clouds,2012-10-02 12:00:00,5026,10,2,1,12
    None,291.14,0.0,0.0,75,Clouds,broken clouds,2012-10-02 13:00:00,4918,10,2,1,13
    None,291.72,0.0,0.0,1,Clear,sky is clear,2012-10-02 14:00:00,5181,10,2,1,14
    None,293.17,0.0,0.0,1,Clear,sky is clear,2012-10-02 15:00:00,5584,10,2,1,15
    None,293.86,0.0,0.0,1,Clear,sky is clear,2012-10-02 16:00:00,6015,10,2,1,16
    None,294.14,0.0,0.0,20,Clouds,few clouds,2012-10-02 17:00:00,5791,10,2,1,17


<a name='1-4'></a>
### 1.4 - Create the InteractiveContext

You will need to initialize the `InteractiveContext` to enable running the TFX components interactively. As before, you will let it create the metadata store in the `_pipeline_root` directory. You can safely ignore the warning about the missing metadata config file.


```python
# Declare the InteractiveContext and use a local sqlite file as the metadata store.
# You can ignore the warning about the missing metadata config file
context = InteractiveContext(pipeline_root=_pipeline_root)
```

    WARNING:absl:InteractiveContext metadata_connection_config not provided: using SQLite ML Metadata database at metro_traffic_pipeline/metadata.sqlite.


<a name='2'></a>
## 2 - Run TFX components interactively

In the following exercises, you will create the data pipeline components one-by-one, run each of them, and visualize their output artifacts. Recall that we refer to the outputs of pipeline components as *artifacts* and these can be inputs to the next stage of the pipeline.

<a name='2-1'></a>
### 2.1 - ExampleGen

The pipeline starts with the [ExampleGen](https://www.tensorflow.org/tfx/guide/examplegen) component. It will:

*   split the data into training and evaluation sets (by default: 2/3 train, 1/3 eval).
*   convert each data row into `tf.train.Example` format. This [protocol buffer](https://developers.google.com/protocol-buffers) is designed for Tensorflow operations and is used by the TFX components.
*   compress and save the data collection under the `_pipeline_root` directory for other components to access. These examples are stored in `TFRecord` format. This optimizes read and write operations within Tensorflow especially if you have a large collection of data.

<a name='ex-1'></a>
#### Exercise 1: ExampleGen

Fill out the code below to ingest the data from the CSV file stored in the `_data_root` directory.


```python
### START CODE HERE

# Instantiate ExampleGen with the input CSV dataset
example_gen = CsvExampleGen(input_base=_data_root)

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
<div class="tfx-object expanded"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExecutionResult</span><span class="deemphasize"> at 0x7ff545c58070</span></div><table class="attr-table"><tr><td class="attr-name">.execution_id</td><td class = "attrvalue">8</td></tr><tr><td class="attr-name">.component</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">CsvExampleGen</span><span class="deemphasize"> at 0x7ff4ac8d42b0</span></div><table class="attr-table"><tr><td class="attr-name">.inputs</td><td class = "attrvalue">{}</td></tr><tr><td class="attr-name">.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7ff4ac8d42e0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: metro_traffic_pipeline/CsvExampleGen/examples/8)<span class="deemphasize"> at 0x7ff545c583d0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">metro_traffic_pipeline/CsvExampleGen/examples/8</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.exec_properties</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['input_base']</td><td class = "attrvalue">metro_traffic_pipeline/data</td></tr><tr><td class="attr-name">['input_config']</td><td class = "attrvalue">{
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
}</td></tr><tr><td class="attr-name">['output_data_format']</td><td class = "attrvalue">6</td></tr><tr><td class="attr-name">['custom_config']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['span']</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">['version']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['input_fingerprint']</td><td class = "attrvalue">split:single_split,num_files:1,total_bytes:3648458,xor_checksum:1613618207,sum_checksum:1613618207</td></tr><tr><td class="attr-name">['_beam_pipeline_args']</td><td class = "attrvalue">[]</td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">.component.inputs</td><td class = "attrvalue">{}</td></tr><tr><td class="attr-name">.component.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7ff4ac8d42e0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: metro_traffic_pipeline/CsvExampleGen/examples/8)<span class="deemphasize"> at 0x7ff545c583d0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">metro_traffic_pipeline/CsvExampleGen/examples/8</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr></table></div>



You should see the output cell of the `InteractiveContext` above showing the metadata associated with the component execution. You can expand the items under `.component.outputs` and see that an `Examples` artifact for the train and eval split is created in `metro_traffic_pipeline/CsvExampleGen/examples/{execution_id}`. 

You can also check that programmatically with the following snippet. You can focus on the `try` block. The `except` and `else` block is needed mainly for grading. `context.run()` yields no operation when executed in a non-interactive environment (such as the grader script that runs outside of this notebook). In such scenarios, the URI must be manually set to avoid errors.


```python
try:
    # get the artifact object
    artifact = example_gen.outputs['examples'].get()[0]
    
    # print split names and uri
    print(f'split names: {artifact.split_names}')
    print(f'artifact uri: {artifact.uri}')

# for grading since context.run() does not work outside the notebook
except IndexError:
    print("context.run() was no-op")
    examples_path = './metro_traffic_pipeline/CsvExampleGen/examples'
    dir_id = os.listdir(examples_path)[0]
    artifact_uri = f'{examples_path}/{dir_id}'

else:
    artifact_uri = artifact.uri
```

    split names: ["train", "eval"]
    artifact uri: metro_traffic_pipeline/CsvExampleGen/examples/8


The ingested data has been saved to the directory specified by the artifact Uniform Resource Identifier (URI). As a sanity check, you can take a look at some of the training examples. This requires working with Tensorflow data types, particularly `tf.train.Example` and `TFRecord` (you can read more about them [here](https://www.tensorflow.org/tutorials/load_data/tfrecord)). Let's first load the `TFRecord` into a variable:


```python
# Get the URI of the output artifact representing the training examples, which is a directory
train_uri = os.path.join(artifact_uri, 'train')

# Get the list of files in this directory (all compressed TFRecord files)
tfrecord_filenames = [os.path.join(train_uri, name)
                      for name in os.listdir(train_uri)]

# Create a `TFRecordDataset` to read these files
dataset = tf.data.TFRecordDataset(tfrecord_filenames, compression_type="GZIP")
```

<a name='ex-2'></a>
#### Exercise 2: get_records()

Complete the helper function below to return a specified number of examples.

*Hints: You may find the [MessageToDict](https://googleapis.dev/python/protobuf/latest/google/protobuf/json_format.html#google.protobuf.json_format.MessageToDict) helper function and tf.train.Example's [ParseFromString()](https://googleapis.dev/python/protobuf/latest/google/protobuf/message.html#google.protobuf.message.Message.ParseFromString) method useful here. You can also refer [here](https://www.tensorflow.org/tutorials/load_data/tfrecord) for a refresher on TFRecord and tf.train.Example()*


```python
def get_records(dataset, num_records):
    '''Extracts records from the given dataset.
    Args:
        dataset (TFRecordDataset): dataset saved by ExampleGen
        num_records (int): number of records to preview
    '''
    
    # initialize an empty list
    records = []

    ### START CODE HERE
    # Use the `take()` method to specify how many records to get
    for tfrecord in dataset.take(num_records):
        
        # Get the numpy property of the tensor
        serialized_example = tfrecord.numpy()
        
        # Initialize a `tf.train.Example()` to read the serialized data
        example = tf.train.Example()
        
        # Read the example data (output is a protocol buffer message)
        example.ParseFromString(serialized_example)
        
        # convert the protocol bufffer message to a Python dictionary
        example_dict = MessageToDict(example)
        
        # append to the records list
        records.append(example_dict)
        
    ### END CODE HERE
    return records
```


```python
# Get 3 records from the dataset
sample_records = get_records(dataset, 3)

# Print the output
pp.pprint(sample_records)
```

    [{'features': {'feature': {'clouds_all': {'int64List': {'value': ['40']}},
                               'date_time': {'bytesList': {'value': ['MjAxMi0xMC0wMiAwOTowMDowMA==']}},
                               'day': {'int64List': {'value': ['2']}},
                               'day_of_week': {'int64List': {'value': ['1']}},
                               'holiday': {'bytesList': {'value': ['Tm9uZQ==']}},
                               'hour': {'int64List': {'value': ['9']}},
                               'month': {'int64List': {'value': ['10']}},
                               'rain_1h': {'floatList': {'value': [0.0]}},
                               'snow_1h': {'floatList': {'value': [0.0]}},
                               'temp': {'floatList': {'value': [288.28]}},
                               'traffic_volume': {'int64List': {'value': ['5545']}},
                               'weather_description': {'bytesList': {'value': ['c2NhdHRlcmVkIGNsb3Vkcw==']}},
                               'weather_main': {'bytesList': {'value': ['Q2xvdWRz']}}}}},
     {'features': {'feature': {'clouds_all': {'int64List': {'value': ['75']}},
                               'date_time': {'bytesList': {'value': ['MjAxMi0xMC0wMiAxMDowMDowMA==']}},
                               'day': {'int64List': {'value': ['2']}},
                               'day_of_week': {'int64List': {'value': ['1']}},
                               'holiday': {'bytesList': {'value': ['Tm9uZQ==']}},
                               'hour': {'int64List': {'value': ['10']}},
                               'month': {'int64List': {'value': ['10']}},
                               'rain_1h': {'floatList': {'value': [0.0]}},
                               'snow_1h': {'floatList': {'value': [0.0]}},
                               'temp': {'floatList': {'value': [289.36]}},
                               'traffic_volume': {'int64List': {'value': ['4516']}},
                               'weather_description': {'bytesList': {'value': ['YnJva2VuIGNsb3Vkcw==']}},
                               'weather_main': {'bytesList': {'value': ['Q2xvdWRz']}}}}},
     {'features': {'feature': {'clouds_all': {'int64List': {'value': ['90']}},
                               'date_time': {'bytesList': {'value': ['MjAxMi0xMC0wMiAxMTowMDowMA==']}},
                               'day': {'int64List': {'value': ['2']}},
                               'day_of_week': {'int64List': {'value': ['1']}},
                               'holiday': {'bytesList': {'value': ['Tm9uZQ==']}},
                               'hour': {'int64List': {'value': ['11']}},
                               'month': {'int64List': {'value': ['10']}},
                               'rain_1h': {'floatList': {'value': [0.0]}},
                               'snow_1h': {'floatList': {'value': [0.0]}},
                               'temp': {'floatList': {'value': [289.58]}},
                               'traffic_volume': {'int64List': {'value': ['4767']}},
                               'weather_description': {'bytesList': {'value': ['b3ZlcmNhc3QgY2xvdWRz']}},
                               'weather_main': {'bytesList': {'value': ['Q2xvdWRz']}}}}}]


You should see three of the examples printed above. Now that `ExampleGen` has finished ingesting the data, the next step is data analysis.

<a name='2-2'></a>
### 2.2 - StatisticsGen
The [StatisticsGen](https://www.tensorflow.org/tfx/guide/statsgen) component computes statistics over your dataset for data analysis, as well as for use in downstream components. It uses the [TensorFlow Data Validation](https://www.tensorflow.org/tfx/data_validation/get_started) library.

`StatisticsGen` takes as input the dataset ingested using `CsvExampleGen`.

<a name='ex-3'></a>
#### Exercise 3: StatisticsGen

Fill the code below to generate statistics from the output examples of `CsvExampleGen`.


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
<div class="tfx-object expanded"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExecutionResult</span><span class="deemphasize"> at 0x7ff4ac8d4eb0</span></div><table class="attr-table"><tr><td class="attr-name">.execution_id</td><td class = "attrvalue">9</td></tr><tr><td class="attr-name">.component</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">StatisticsGen</span><span class="deemphasize"> at 0x7ff5458453a0</span></div><table class="attr-table"><tr><td class="attr-name">.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7ff4ac8d42e0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: metro_traffic_pipeline/CsvExampleGen/examples/8)<span class="deemphasize"> at 0x7ff545c583d0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">metro_traffic_pipeline/CsvExampleGen/examples/8</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['statistics']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7ff545845be0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: metro_traffic_pipeline/StatisticsGen/statistics/9)<span class="deemphasize"> at 0x7ff55cc00c10</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">metro_traffic_pipeline/StatisticsGen/statistics/9</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.exec_properties</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['stats_options_json']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['exclude_splits']</td><td class = "attrvalue">[]</td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">.component.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7ff4ac8d42e0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: metro_traffic_pipeline/CsvExampleGen/examples/8)<span class="deemphasize"> at 0x7ff545c583d0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">metro_traffic_pipeline/CsvExampleGen/examples/8</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.component.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['statistics']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7ff545845be0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: metro_traffic_pipeline/StatisticsGen/statistics/9)<span class="deemphasize"> at 0x7ff55cc00c10</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">metro_traffic_pipeline/StatisticsGen/statistics/9</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr></table></div>




```python
# Plot the statistics generated
context.show(statistics_gen.outputs['statistics'])
```


<b>Artifact at metro_traffic_pipeline/StatisticsGen/statistics/9</b><br/><br/>



<div><b>'train' split:</b></div><br/>


    WARNING:tensorflow:From /opt/conda/lib/python3.8/site-packages/tensorflow_data_validation/utils/stats_util.py:229: tf_record_iterator (from tensorflow.python.lib.io.tf_record) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use eager execution and: 
    `tf.data.TFRecordDataset(path)`



<iframe id='facets-iframe' width="100%" height="500px"></iframe>
        <script>
        facets_iframe = document.getElementById('facets-iframe');
        facets_html = '<script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"><\/script><link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/master/facets-dist/facets-jupyter.html"><facets-overview proto-input="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CBUQFSITMjAxNy0wOC0xOCAxMzowMDowMCkAAAAAAAAQQAoiCBYQFiITMjAxNy0wNi0xNCAwMTowMDowMCkAAAAAAAAQQAoiCBcQFyITMjAxNy0wNS0xNyAwMjowMDowMCkAAAAAAAAQQAoiCBgQGCITMjAxNy0wNC0xNSAwNzowMDowMCkAAAAAAAAQQAoiCBkQGSITMjAxNy0wNC0xNSAwNTowMDowMCkAAAAAAAAQQAoiCBoQGiITMjAxNy0wMS0zMCAyMzowMDowMCkAAAAAAAAQQAoiCBsQGyITMjAxNy0wMS0wMiAxNjowMDowMCkAAAAAAAAQQAoiCBwQHCITMjAxNi0xMi0yNSAwMjowMDowMCkAAAAAAAAQQAoiCB0QHSITMjAxNi0xMi0yMyAxNjowMDowMCkAAAAAAAAQQAoiCB4QHiITMjAxNi0xMi0xNyAwMTowMDowMCkAAAAAAAAQQAoiCB8QHyITMjAxNi0xMi0xNiAxODowMDowMCkAAAAAAAAQQAoiCCAQICITMjAxNi0xMS0yMiAxNTowMDowMCkAAAAAAAAQQAoiCCEQISITMjAxNi0xMS0xOCAxNTowMDowMCkAAAAAAAAQQAoiCCIQIiITMjAxNi0wNS0zMSAxMDowMDowMCkAAAAAAAAQQAoiCCMQIyITMjAxNi0wMy0xNCAwODowMDowMCkAAAAAAAAQQAoiCCQQJCITMjAxNi0wMy0xNCAwNTowMDowMCkAAAAAAAAQQAoiCCUQJSITMjAxNS0xMi0xNiAxMDowMDowMCkAAAAAAAAQQAoiCCYQJiITMjAxNS0xMi0xNCAxMTowMDowMCkAAAAAAAAQQAoiCCcQJyITMjAxNS0xMC0yOCAxNzowMDowMCkAAAAAAAAQQAoiCCgQKCITMjAxNS0xMC0yOCAxNTowMDowMCkAAAAAAAAQQAoiCCkQKSITMjAxNS0wOC0yMiAyMTowMDowMCkAAAAAAAAQQAoiCCoQKiITMjAxNS0wNy0wNiAxNjowMDowMCkAAAAAAAAQQAoiCCsQKyITMjAxNS0wNy0wNiAxNDowMDowMCkAAAAAAAAQQAoiCCwQLCITMjAxNC0wNy0xMiAxNDowMDowMCkAAAAAAAAQQAoiCC0QLSITMjAxNC0wNi0xNSAwMTowMDowMCkAAAAAAAAQQAoiCC4QLiITMjAxNC0wNi0wMSAwNjowMDowMCkAAAAAAAAQQAoiCC8QLyITMjAxMy0wNy0yMiAwOTowMDowMCkAAAAAAAAQQAoiCDAQMCITMjAxMy0wNS0yMSAyMzowMDowMCkAAAAAAAAQQAoiCDEQMSITMjAxMy0wNS0yMCAxODowMDowMCkAAAAAAAAQQAoiCDIQMiITMjAxMy0wNS0yMCAxNzowMDowMCkAAAAAAAAQQAoiCDMQMyITMjAxMy0wNS0xOSAwOTowMDowMCkAAAAAAAAQQAoiCDQQNCITMjAxMy0wNS0wNCAwODowMDowMCkAAAAAAAAQQAoiCDUQNSITMjAxMy0wNC0yMyAxNzowMDowMCkAAAAAAAAQQAoiCDYQNiITMjAxMy0wNC0xOSAxNTowMDowMCkAAAAAAAAQQAoiCDcQNyITMjAxMy0wNC0xOCAyMzowMDowMCkAAAAAAAAQQAoiCDgQOCITMjAxMy0wNC0wOSAwMTowMDowMCkAAAAAAAAQQAoiCDkQOSITMjAxMy0wMy0xNiAwOTowMDowMCkAAAAAAAAQQAoiCDoQOiITMjAxMy0wMi0xMSAxMDowMDowMCkAAAAAAAAQQAoiCDsQOyITMjAxMi0xMi0xNiAxMDowMDowMCkAAAAAAAAQQAoiCDwQPCITMjAxMi0xMS0xMSAwNjowMDowMCkAAAAAAAAQQAoiCD0QPSITMjAxMi0xMS0xMSAwNDowMDowMCkAAAAAAAAQQAoiCD4QPiITMjAxMi0xMS0xMSAwMzowMDowMCkAAAAAAAAQQAoiCD8QPyITMjAxMi0xMC0yNiAxMDowMDowMCkAAAAAAAAQQAoiCEAQQCITMjAxMi0xMC0yNSAxODowMDowMCkAAAAAAAAQQAoiCEEQQSITMjAxMi0xMC0yMCAxMDowMDowMCkAAAAAAAAQQAoiCEIQQiITMjAxOC0wOS0yNCAyMzowMDowMCkAAAAAAAAIQAoiCEMQQyITMjAxOC0wOS0yMSAwNDowMDowMCkAAAAAAAAIQAoiCEQQRCITMjAxOC0wOS0yMCAxNjowMDowMCkAAAAAAAAIQAoiCEUQRSITMjAxOC0wOS0yMCAxNDowMDowMCkAAAAAAAAIQAoiCEYQRiITMjAxOC0wOS0xOSAxMzowMDowMCkAAAAAAAAIQAoiCEcQRyITMjAxOC0wOS0xOSAxMDowMDowMCkAAAAAAAAIQAoiCEgQSCITMjAxOC0wOS0xOCAxMDowMDowMCkAAAAAAAAIQAoiCEkQSSITMjAxOC0wOS0xOCAwNTowMDowMCkAAAAAAAAIQAoiCEoQSiITMjAxOC0wOS0xOCAwNDowMDowMCkAAAAAAAAIQAoiCEsQSyITMjAxOC0wOS0xOCAwMzowMDowMCkAAAAAAAAIQAoiCEwQTCITMjAxOC0wOS0xOCAwMTowMDowMCkAAAAAAAAIQAoiCE0QTSITMjAxOC0wOS0xNyAxNjowMDowMCkAAAAAAAAIQAoiCE4QTiITMjAxOC0wOS0wNCAyMTowMDowMCkAAAAAAAAIQAoiCE8QTyITMjAxOC0wOS0wNCAxNzowMDowMCkAAAAAAAAIQAoiCFAQUCITMjAxOC0wOS0wNCAwNTowMDowMCkAAAAAAAAIQAoiCFEQUSITMjAxOC0wOS0wNCAwMTowMDowMCkAAAAAAAAIQAoiCFIQUiITMjAxOC0wOS0wMyAxOTowMDowMCkAAAAAAAAIQAoiCFMQUyITMjAxOC0wOS0wMyAwMTowMDowMCkAAAAAAAAIQAoiCFQQVCITMjAxOC0wOC0zMSAwNDowMDowMCkAAAAAAAAIQAoiCFUQVSITMjAxOC0wOC0zMSAwMzowMDowMCkAAAAAAAAIQAoiCFYQViITMjAxOC0wOC0yOCAxMTowMDowMCkAAAAAAAAIQAoiCFcQVyITMjAxOC0wOC0yOCAxMDowMDowMCkAAAAAAAAIQAoiCFgQWCITMjAxOC0wOC0yNiAwNzowMDowMCkAAAAAAAAIQAoiCFkQWSITMjAxOC0wOC0yNiAwNjowMDowMCkAAAAAAAAIQAoiCFoQWiITMjAxOC0wOC0yNiAwMzowMDowMCkAAAAAAAAIQAoiCFsQWyITMjAxOC0wOC0yNiAwMTowMDowMCkAAAAAAAAIQAoiCFwQXCITMjAxOC0wOC0yNSAwOTowMDowMCkAAAAAAAAIQAoiCF0QXSITMjAxOC0wOC0yNSAwMjowMDowMCkAAAAAAAAIQAoiCF4QXiITMjAxOC0wOC0yNCAxNjowMDowMCkAAAAAAAAIQAoiCF8QXyITMjAxOC0wOC0yNCAwOTowMDowMCkAAAAAAAAIQAoiCGAQYCITMjAxOC0wOC0yNCAwODowMDowMCkAAAAAAAAIQAoiCGEQYSITMjAxOC0wOC0yNCAwNjowMDowMCkAAAAAAAAIQAoiCGIQYiITMjAxOC0wNy0yOSAxNjowMDowMCkAAAAAAAAIQAoiCGMQYyITMjAxOC0wNy0yMCAwNzowMDowMCkAAAAAAAAIQAoiCGQQZCITMjAxOC0wNy0yMCAwNjowMDowMCkAAAAAAAAIQAoiCGUQZSITMjAxOC0wNy0yMCAwNTowMDowMCkAAAAAAAAIQAoiCGYQZiITMjAxOC0wNy0yMCAwNDowMDowMCkAAAAAAAAIQAoiCGcQZyITMjAxOC0wNy0yMCAwMjowMDowMCkAAAAAAAAIQAoiCGgQaCITMjAxOC0wNy0xMyAwNjowMDowMCkAAAAAAAAIQAoiCGkQaSITMjAxOC0wNy0xMyAwMzowMDowMCkAAAAAAAAIQAoiCGoQaiITMjAxOC0wNy0xMiAxOTowMDowMCkAAAAAAAAIQAoiCGsQayITMjAxOC0wNy0xMiAxODowMDowMCkAAAAAAAAIQAoiCGwQbCITMjAxOC0wNi0yNiAwNDowMDowMCkAAAAAAAAIQAoiCG0QbSITMjAxOC0wNi0yNiAwMjowMDowMCkAAAAAAAAIQAoiCG4QbiITMjAxOC0wNi0yMyAwNjowMDowMCkAAAAAAAAIQAoiCG8QbyITMjAxOC0wNi0xOSAxOTowMDowMCkAAAAAAAAIQAoiCHAQcCITMjAxOC0wNi0xOSAxMjowMDowMCkAAAAAAAAIQAoiCHEQcSITMjAxOC0wNi0xOSAxMTowMDowMCkAAAAAAAAIQAoiCHIQciITMjAxOC0wNi0xOCAxMzowMDowMCkAAAAAAAAIQAoiCHMQcyITMjAxOC0wNi0xOCAxMDowMDowMCkAAAAAAAAIQAoiCHQQdCITMjAxOC0wNi0xOCAwMzowMDowMCkAAAAAAAAIQAoiCHUQdSITMjAxOC0wNi0wOSAxODowMDowMCkAAAAAAAAIQAoiCHYQdiITMjAxOC0wNi0wOSAxNDowMDowMCkAAAAAAAAIQAoiCHcQdyITMjAxOC0wNi0wOSAxMzowMDowMCkAAAAAAAAIQAoiCHgQeCITMjAxOC0wNi0wOSAxMjowMDowMCkAAAAAAAAIQAoi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AEAKJAjABhDABiITMjAxOC0wNS0yNCAyMTowMDowMCkAAAAAAAAAQAokCMEGEMEGIhMyMDE4LTA1LTIyIDA2OjAwOjAwKQAAAAAAAABACiQIwgYQwgYiEzIwMTgtMDUtMjIgMDU6MDA6MDApAAAAAAAAAEAKJAjDBhDDBiITMjAxOC0wNS0yMSAwODowMDowMCkAAAAAAAAAQAokCMQGEMQGIhMyMDE4LTA1LTE5IDEwOjAwOjAwKQAAAAAAAABACiQIxQYQxQYiEzIwMTgtMDUtMTkgMDk6MDA6MDApAAAAAAAAAEAKJAjGBhDGBiITMjAxOC0wNS0xOCAyMjowMDowMCkAAAAAAAAAQAokCMcGEMcGIhMyMDE4LTA1LTE1IDAzOjAwOjAwKQAAAAAAAABACiQIyAYQyAYiEzIwMTgtMDUtMTUgMDI6MDA6MDApAAAAAAAAAEAKJAjJBhDJBiITMjAxOC0wNS0xNCAyMTowMDowMCkAAAAAAAAAQAokCMoGEMoGIhMyMDE4LTA1LTE0IDE5OjAwOjAwKQAAAAAAAABACiQIywYQywYiEzIwMTgtMDUtMTQgMTU6MDA6MDApAAAAAAAAAEAKJAjMBhDMBiITMjAxOC0wNS0xNCAwNjowMDowMCkAAAAAAAAAQAokCM0GEM0GIhMyMDE4LTA1LTEyIDE2OjAwOjAwKQAAAAAAAABACiQIzgYQzgYiEzIwMTgtMDUtMTEgMTk6MDA6MDApAAAAAAAAAEAKJAjPBhDPBiITMjAxOC0wNS0wOSAxNjowMDowMCkAAAAAAAAAQAokCNAGENAGIhMyMDE4LTA1LTA5IDAzOjAwOjAwKQAAAAAAAABACiQI0QYQ0QYiEzIwMTgtMDUtMDkgMDA6MDA6MDApAAAAAAAAAEAKJAjSBhDSBiITMjAxOC0wNS0wOCAyMjowMDowMCkAAAAAAAAAQAokCNMGENMGIhMyMDE4LTA1LTA4IDEzOjAwOjAwKQAAAAAAAABACiQI1AYQ1AYiEzIwMTgtMDUtMDYgMDE6MDA6MDApAAAAAAAAAEAKJAjVBhDVBiITMjAxOC0wNS0wNSAyMDowMDowMCkAAAAAAAAAQAokCNYGENYGIhMyMDE4LTA1LTA1IDE4OjAwOjAwKQAAAAAAAABACiQI1wYQ1wYiEzIwMTgtMDUtMDIgMDI6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"></facets-overview>';
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
        facets_html = '<script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"><\/script><link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/master/facets-dist/facets-jupyter.html"><facets-overview proto-input="CvyLAwoObGhzX3N0YXRpc3RpY3MQwX4axK4CEAIisa4CCrYCCMF+GAEgAS0AAIA/MqQCGhsJAAAAAAAA8D8RAAAAAAAA8D8hMzMzMzNNmUAaGwkAAAAAAADwPxEAAAAAAADwPyEzMzMzM02ZQBobCQAAAAAAAPA/EQAAAAAAAPA/ITMzMzMzTZlAGhsJAAAAAAAA8D8RAAAAAAAA8D8hMzMzMzNNmUAaGwkAAAAAAADwPxEAAAAAAADwPyEzMzMzM02ZQBobCQAAAAAAAPA/EQAAAAAAAPA/ITMzMzMzTZlAGhsJAAAAAAAA8D8RAAAAAAAA8D8hMzMzMzNNmUAaGwkAAAAAAADwPxEAAAAAAADwPyEzMzMzM02ZQBobCQAAAAAAAPA/EQAAAAAAAPA/ITMzMzMzTZlAGhsJAAAAAAAA8D8RAAAAAAAA8D8hMzMzMzNNmUAgAUDBfhCVdhoeEhMyMDE4LTA5LTIwIDE4OjAwOjAwGQAAAAAAABBAGh4SEzIwMTgtMDYtMDkgMTE6MDA6MDAZAAAAAAAAEEAaHhITMjAxNi0wNS0yNyAxODowMDowMBkAAAAAAAAQQBoeEhMyMDE2LTA1LTI3IDEwOjAwOjAwGQAAAAAAABBAGh4SEzIwMTYtMDItMTkgMTE6MDA6MDAZAAAAAAAAEEAaHhITMjAxMy0wMy0xMCAxOTowMDowMBkAAAAAAAAQQBoeEhMyMDEzLTAyLTExIDEzOjAwOjAwGQAAAAAAABBAGh4SEzIwMTItMTItMTYgMDk6MDA6MDAZAAAAAAAAEEAaHhITMjAxMi0xMC0xOSAwNDowMDowMBkAAAAAAAAQQBoeEhMyMDE4LTA5LTIwIDE5OjAwOjAwGQAAAAAAAAhAGh4SEzIwMTgtMDktMjAgMDY6MDA6MDAZAAAAAAAACEAaHhITMjAxOC0wOS0xOSAwOTowMDowMBkAAAAAAAAIQBoeEhMyMDE4LTA5LTA0IDA4OjAwOjAwGQAAAAAAAAhAGh4SEzIwMTgtMDgtMjQgMTQ6MDA6MDAZAAAAAAAACEAaHhITMjAxOC0wOC0yNCAxMzowMDowMBkAAAAAAAAIQBoeEhMyMDE4LTA4LTIwIDA1OjAwOjAwGQAAAAAAAAhAGh4SEzIwMTgtMDctMTIgMjA6MDA6MDAZAAAAAAAACEAaHhITMjAxOC0wNi0wOCAwNDowMDowMBkAAAAAAAAIQBoeEhMyMDE4LTA0LTE1IDA5OjAwOjAwGQAAAAAAAAhAGh4SEzIwMTgtMDQtMTUgMDM6MDA6MDAZAAAAAAAACEAlAACYQSrspgIKHiITMjAxOC0wOS0yMCAxODowMDowMCkAAAAAAAAQQAoiCAEQASITMjAxOC0wNi0wOSAxMTowMDowMCkAAAAAAAAQQAoiCAIQAiITMjAxNi0wNS0yNyAxODowMDowMCkAAAAAAAAQQAoiCAMQAyITMjAxNi0wNS0yNyAxMDowMDowMCkAAAAAAAAQQAoiCAQQBCITMjAxNi0wMi0xOSAxMTowMDowMCkAAAAAAAAQQAoiCAUQBSITMjAxMy0wMy0xMCAxOTowMDowMCkAAAAAAAAQQAoiCAYQBiITMjAxMy0wMi0xMSAxMzowMDowMCkAAAAAAAAQQAoiCAcQByITMjAxMi0xMi0xNiAwOTowMDowMCkAAAAAAAAQQAoiCAgQCCITMjAxMi0xMC0xOSAwNDowMDowMCkAAAAAAAAQQAoiCAkQCSITMjAxOC0wOS0yMCAxOTowMDowMCkAAAAAAAAIQAoiCAoQCiITMjAxOC0wOS0yMCAwNjowMDowMCkAAAAAAAAIQAoiCAsQCyITMjAxOC0wOS0xOSAwOTowMDowMCkAAAAAAAAIQAoiCAwQDCITMjAxOC0wOS0wNCAwODowMDowMCkAAAAAAAAIQAoiCA0QDSITMjAxOC0wOC0yNCAxNDowMDowMCkAAAAAAAAIQAoiCA4QDiITMjAxOC0wOC0yNCAxMzowMDowMCkAAAAAAAAIQAoiCA8QDyITMjAxOC0wOC0yMCAwNTowMDowMCkAAAAAAAAIQAoiCBAQECITMjAxOC0wNy0xMiAyMDowMDowMCkAAAAAAAAIQAoiCBEQESITMjAxOC0wNi0wOCAwNDowMDowMCkAAAAAAAAIQAoiCBIQEiITMjAxOC0wNC0xNSAwOTowMDowMCkAAAAAAAAIQAoiCBMQEyITMjAxOC0wNC0xNSAwMzowMDowMCkAAAAAAAAIQAoiCBQQFCITMjAxOC0wNC0xNCAxMzowMDowMCkAAAAAAAAIQAoiCBUQFSITMjAxOC0wMy0yNiAyMzowMDowMCkAAAAAAAAIQAoiCBYQFiITMjAxOC0wMy0yMCAyMTowMDowMCkAAAAAAAAIQAoiCBcQFyITMjAxOC0wMi0yNSAwMzowMDowMCkAAAAAAAAIQAoiCBgQGCITMjAxOC0wMi0yMiAyMzowMDowMCkAAAAAAAAIQAoiCBkQGSITMjAxOC0wMS0yMiAxMjowMDowMCkAAAAAAAAIQAoiCBoQGiITMjAxOC0wMS0xNSAwMDowMDowMCkAAAAAAAAIQAoiCBsQGyITMjAxOC0wMS0xMSAwMjowMDowMCkAAAAAAAAIQAoiCBwQHCITMjAxNy0xMi0yOCAwNzowMDowMCkAAAAAAAAIQAoiCB0QHSITMjAxNy0xMi0wNCAxNDowMDowMCkAAAAAAAAIQAoiCB4QHiITMjAxNy0xMS0wMSAxODowMDowMCkAAAAAAAAIQAoiCB8QHyITMjAxNy0xMC0yNyAxNDowMDowMCkAAAAAAAAIQAoiCCAQICITMjAxNy0xMC0yNyAwNjowMDowMCkAAAAAAAAIQAoiCCEQISITMjAxNy0xMC0yNiAyMzowMDowMCkAAAAAAAAIQAoiCCIQIiITMjAxNy0xMC0yMSAxMDowMDowMCkAAAAAAAAIQAoiCCMQIyITMjAxNy0xMC0wNiAyMjowMDowMCkAAAAAAAAIQAoiCCQQJCITMjAxNy0xMC0wMiAxODowMDowMCkAAAAAAAAIQAoiCCUQJSITMjAxNy0xMC0wMiAxNjowMDowMCkAAAAAAAAIQAoiCCYQJiITMjAxNy0wOC0yNiAwMjowMDowMCkAAAAAAAAIQAoiCCcQJyITMjAxNy0wOC0yNiAwMDowMDowMCkAAAAAAAAIQAoiCCgQKCITMjAxNy0wOC0yMSAxNjowMDowMCkAAAAAAAAIQAoiCCkQKSITMjAxNy0wOC0xNyAwNDowMDowMCkAAAAAAAAIQAoiCCoQKiITMjAxNy0wOC0xNCAwNjowMDowMCkAAAAAAAAIQAoiCCsQKyITMjAxNy0wOC0wOSAxODowMDowMCkAAAAAAAAIQAoiCCwQLCITMjAxNy0wNy0xOCAxNTowMDowMCkAAAAAAAAIQAoiCC0QLSITMjAxNy0wNi0yMiAxNTowMDowMCkAAAAAAAAIQAoiCC4QLiITMjAxNy0wNS0xOCAwNTowMDowMCkAAAAAAAAIQAoiCC8QLyITMjAxNy0wNS0xNyAxODowMDowMCkAAAAAAAAIQAoiCDAQMCITMjAxNy0wNC0yNiAwODowMDowMCkAAAAAAAAIQAoiCDEQMSITMjAxNy0wNC0yMCAwMzowMDowMCkAAAAAAAAIQAoiCDIQMiITMjAxNy0wNC0xOSAxOTowMDowMCkAAAAAAAAIQAoiCDMQMyITMjAxNy0wNC0xMCAxODowMDowMCkAAAAAAAAIQAoiCDQQNCITMjAxNy0wMS0yMyAwODowMDowMCkAAAAAAAAIQAoiCDUQNSITMjAxNy0wMS0yMCAwODowMDowMCkAAAAAAAAIQAoiCDYQNiITMjAxNy0wMS0yMCAwNjowMDowMCkAAAAAAAAIQAoiCDcQNyITMjAxNy0wMS0xNyAxMDowMDowMCkAAAAAAAAIQAoiCDgQOCITMjAxNi0xMS0yNCAxODowMDowMCkAAAAAAAAIQAoiCDkQOSITMjAxNi0xMS0yMiAyMzowMDowMCkAAAAAAAAIQAoiCDoQOiITMjAxNi0xMS0yMiAyMDowMDowMCkAAAAAAAAIQAoiCDsQOyITMjAxNi0wOS0wNiAwMzowMDowMCkAAAAAAAAIQAoiCDwQPCITMjAxNi0wOC0yNCAwMDowMDowMCkAAAAAAAAIQAoiCD0QPSITMjAxNi0wNS0xMSAxNDowMDowMCkAAAAAAAAIQAoiCD4QPiITMjAxNi0wNC0yNCAxMDowMDowMCkAAAAAAAAIQAoiCD8QPyITMjAxNi0wNC0yMSAwNDowMDowMCkAAAAAAAAIQAoiCEAQQCITMjAxNi0wMy0zMCAwODowMDowMCkAAAAAAAAIQAoiCEEQQSITMjAxNi0wMy0wOSAwODowMDowMCkAAAAAAAAIQAoiCEIQQiITMjAxNi0wMi0yMyAwNzowMDowMCkAAAAAAAAIQAoiCEMQQyITMjAxNi0wMi0wNiAyMDowMDowMCkAAAAAAAAIQAoiCEQQRCITMjAxNi0wMS0yMSAwOTowMDowMCkAAAAAAAAIQAoiCEUQRSITMjAxNi0wMS0xMSAxMDowMDowMCkAAAAAAAAIQAoiCEYQRiITMjAxNS0xMi0yMyAxMjowMDowMCkAAAAAAAAIQAoiCEcQRyITMjAxNS0xMi0xNiAxNDowMDowMCkAAAAAAAAIQAoiCEgQSCITMjAxNS0xMC0yOCAwMDowMDowMCkAAAAAAAAIQAoiCEkQSSITMjAxNS0xMC0wOCAxODowMDowMCkAAAAAAAAIQAoiCEoQSiITMjAxNS0wOS0yNCAxNjowMDowMCkAAAAAAAAIQAoiCEsQSyITMjAxNS0wOS0xMCAwNjowMDowMCkAAAAAAAAIQAoiCEwQTCITMjAxNS0wOS0wOCAwNTowMDowMCkAAAAAAAAIQAoiCE0QTSITMjAxNS0wOC0xOSAwMTowMDowMCkAAAAAAAAIQAoiCE4QTiITMjAxNS0wOC0xNiAxOTowMDowMCkAAAAAAAAIQAoiCE8QTyITMjAxNS0wOC0wNiAyMDowMDowMCkAAAAAAAAIQAoiCFAQUCITMjAxNS0wNy0wNiAxMzowMDowMCkAAAAAAAAIQAoiCFEQUSITMjAxNS0wNy0wNiAwODowMDowMCkAAAAAAAAIQAoiCFIQUiITMjAxNS0wNy0wNCAwMzowMDowMCkAAAAAAAAIQAoiCFMQUyITMjAxNS0wNy0wNCAwMjowMDowMCkAAAAAAAAIQAoiCFQQVCITMjAxNS0wNi0yOSAwNzowMDowMCkAAAAAAAAIQAoiCFUQVSITMjAxNC0wNy0xMSAwODowMDowMCkAAAAAAAAIQAoiCFYQViITMjAxNC0wNi0wMiAwMzowMDowMCkAAAAAAAAIQAoiCFcQVyITMjAxNC0wNS0yMCAxMTowMDowMCkAAAAAAAAIQAoiCFgQWCITMjAxNC0wNS0xOSAxNDowMDowMCkAAAAAAAAIQAoiCFkQWSITMjAxMy0xMi0wMyAxMzowMDowMCkAAAAAAAAIQAoiCFoQWiITMjAxMy0xMS0xNiAxMzowMDowMCkAAAAAAAAIQAoiCFsQWyITMjAxMy0wOS0xNCAyMjowMDowMCkAAAAAAAAIQAoiCFwQXCITMjAxMy0wNy0xNCAwNjowMDowMCkAAAAAAAAIQAoiCF0QXSITMjAxMy0wNi0yMiAwNDowMDowMCkAAAAAAAAIQAoiCF4QXiITMjAxMy0wNi0wMSAwMjowMDowMCkAAAAAAAAIQAoiCF8QXyITMjAxMy0wNS0yNSAyMjowMDowMCkAAAAAAAAIQAoiCGAQYCITMjAxMy0wNS0yMiAyMDowMDowMCkAAAAAAAAIQAoiCGEQYSITMjAxMy0wNS0xOSAwODowMDowMCkAAAAAAAAIQAoiCGIQYiITMjAxMy0wNS0xOSAwNjowMDowMCkAAAAAAAAIQAoiCGMQYyITMjAxMy0wNS0wNCAxODowMDowMCkAAAAAAAAIQAoiCGQQZCITMjAxMy0wNC0yMyAyMjowMDowMCkAAAAAAAAIQAoiCGUQZSITMjAxMy0wNC0yMyAxNTowMDowMCkAAAAAAAAIQAoiCGYQZiITMjAxMy0wNC0xOSAyMDowMDowMCkAAAAAAAAIQAoiCGcQZyITMjAxMy0wNC0xOSAxNzowMDowMCkAAAAAAAAIQAoiCGgQaCITMjAxMy0wNC0xMiAxNjowMDowMCkAAAAAAAAIQAoiCGkQaSITMjAxMy0wNC0wOSAwMzowMDowMCkAAAAAAAAIQAoiCGoQaiITMjAxMy0wMy0zMSAwNjowMDowMCkAAAAAAAAIQAoiCGsQayITMjAxMy0wMy0wNiAwNTowMDowMCkAAAAAAAAIQAoiCGwQbCITMjAxMy0wMi0xNCAyMjowMDowMCkAAAAAAAAIQAoiCG0QbSITMjAxMy0wMS0xMiAxODowMDowMCkAAAAAAAAIQAoiCG4QbiITMjAxMi0xMi0xNiAxNDowMDowMCkAAAAAAAAIQAoiCG8QbyITMjAxMi0xMi0xMCAwOTowMDowMCkAAAAAAAAIQAoiCHAQcCITMjAxMi0xMC0yNiAxMTowMDowMCkAAAAAAAAIQAoiCHEQcSITMjAxMi0xMC0yNSAxNTowMDowMCkAAAAAAAAIQAoiCHIQciITMjAxMi0xMC0yMCAwOTowMDowMCkAAAAAAAAIQAoiCHMQcyITMjAxOC0wOS0yOSAxOTowMDowMCkAAAAAAAAAQAoiCHQQdCITMjAxOC0wOS0yNSAxNjowMDowMCkAAAAAAAAAQAoiCHUQdSITMjAxOC0wOS0yNSAxMjowMDowMCkAAAAAAAAAQAoiCHYQdiITMjAxOC0wOS0yNSAwMDowMDowMCkAAAAAAAAAQAoiCHcQdyITMjAxOC0wOS0yMSAwODowMDowMCkAAAAAAAAAQAoiCHgQeCITMjAxOC0wOS0yMCAxMjowMDowMCkAAAAAAAAAQAoiCHkQeSITMjAxOC0wOS0yMCAxMDowMDowMCkAAAAAAAAAQAoiCHoQeiITMjAxOC0w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IhMyMDEzLTA0LTI0IDAzOjAwOjAwKQAAAAAAAABACiQIwgYQwgYiEzIwMTMtMDQtMjMgMjE6MDA6MDApAAAAAAAAAEAKJAjDBhDDBiITMjAxMy0wNC0yMyAxOTowMDowMCkAAAAAAAAAQAokCMQGEMQGIhMyMDEzLTA0LTIyIDE5OjAwOjAwKQAAAAAAAABACiQIxQYQxQYiEzIwMTMtMDQtMjAgMDk6MDA6MDApAAAAAAAAAEAKJAjGBhDGBiITMjAxMy0wNC0yMCAwNDowMDowMCkAAAAAAAAAQAokCMcGEMcGIhMyMDEzLTA0LTE5IDIyOjAwOjAwKQAAAAAAAABACiQIyAYQyAYiEzIwMTMtMDQtMTkgMTk6MDA6MDApAAAAAAAAAEAKJAjJBhDJBiITMjAxMy0wNC0xOSAxNDowMDowMCkAAAAAAAAAQAokCMoGEMoGIhMyMDEzLTA0LTE5IDA2OjAwOjAwKQAAAAAAAABACiQIywYQywYiEzIwMTMtMDQtMTkgMDM6MDA6MDApAAAAAAAAAEAKJAjMBhDMBiITMjAxMy0wNC0xOSAwMTowMDowMCkAAAAAAAAAQAokCM0GEM0GIhMyMDEzLTA0LTE5IDAwOjAwOjAwKQAAAAAAAABACiQIzgYQzgYiEzIwMTMtMDQtMTggMTg6MDA6MDApAAAAAAAAAEAKJAjPBhDPBiITMjAxMy0wNC0xMiAyMTowMDowMCkAAAAAAAAAQAokCNAGENAGIhMyMDEzLTA0LTEyIDE5OjAwOjAwKQAAAAAAAABACiQI0QYQ0QYiEzIwMTMtMDQtMTIgMTA6MDA6MDApAAAAAAAAAEAKJAjSBhDSBiITMjAxMy0wNC0xMiAwOTowMDowMCkAAAAAAAAAQAokCNMGENMGIhMyMDEzLTA0LTEyIDA3OjAwOjAwKQAAAAAAAABACiQI1AYQ1AYiEzIwMTMtMDQtMTEgMDE6MDA6MDApAAAAAAAAAEAKJAjVBhDVBiITMjAxMy0wNC0xMCAxMjowMDowMCkAAAAAAAAAQAokCNYGENYGIhMyMDEzLTA0LTA5IDE1OjAwOjAwKQAAAAAAAABACiQI1wYQ1wYiEzIwMTMtMDQtMDkgMDc6MDA6MDApAAAAAAAAAEAKJAjYBhDYBiITMjAxMy0wNC0wNyAxNjowMDowMCkAAAAA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"></facets-overview>';
        facets_iframe.srcdoc = facets_html;
         facets_iframe.id = "";
         setTimeout(() => {
           facets_iframe.setAttribute('height', facets_iframe.contentWindow.document.body.offsetHeight + 'px')
         }, 1500)
         </script>


<a name='2-3'></a>
### 2.3 - SchemaGen

The [SchemaGen](https://www.tensorflow.org/tfx/guide/schemagen) component also uses TFDV to generate a schema based on your data statistics. As you've learned previously, a schema defines the expected bounds, types, and properties of the features in your dataset.

`SchemaGen` will take as input the statistics that we generated with `StatisticsGen`, looking at the training split by default.

<a name='ex-4'></a>
#### Exercise 4: SchemaGen


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
<div class="tfx-object expanded"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExecutionResult</span><span class="deemphasize"> at 0x7ff54456bdf0</span></div><table class="attr-table"><tr><td class="attr-name">.execution_id</td><td class = "attrvalue">10</td></tr><tr><td class="attr-name">.component</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">SchemaGen</span><span class="deemphasize"> at 0x7ff544566d30</span></div><table class="attr-table"><tr><td class="attr-name">.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['statistics']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7ff545845be0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: metro_traffic_pipeline/StatisticsGen/statistics/9)<span class="deemphasize"> at 0x7ff55cc00c10</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">metro_traffic_pipeline/StatisticsGen/statistics/9</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['schema']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7ff544566a30</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: metro_traffic_pipeline/SchemaGen/schema/10)<span class="deemphasize"> at 0x7ff544572790</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">metro_traffic_pipeline/SchemaGen/schema/10</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.exec_properties</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['infer_feature_shape']</td><td class = "attrvalue">False</td></tr><tr><td class="attr-name">['exclude_splits']</td><td class = "attrvalue">[]</td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">.component.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['statistics']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7ff545845be0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: metro_traffic_pipeline/StatisticsGen/statistics/9)<span class="deemphasize"> at 0x7ff55cc00c10</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">metro_traffic_pipeline/StatisticsGen/statistics/9</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.component.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['schema']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7ff544566a30</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: metro_traffic_pipeline/SchemaGen/schema/10)<span class="deemphasize"> at 0x7ff544572790</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">metro_traffic_pipeline/SchemaGen/schema/10</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr></table></div>



If all went well, you can now visualize the generated schema as a table.


```python
# Visualize the output
context.show(schema_gen.outputs['schema'])
```


<b>Artifact at metro_traffic_pipeline/SchemaGen/schema/10</b><br/><br/>



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
      <th>'clouds_all'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'date_time'</th>
      <td>BYTES</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'day'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'day_of_week'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'holiday'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'holiday'</td>
    </tr>
    <tr>
      <th>'hour'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'month'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'rain_1h'</th>
      <td>FLOAT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'snow_1h'</th>
      <td>FLOAT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'temp'</th>
      <td>FLOAT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'traffic_volume'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'weather_description'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'weather_description'</td>
    </tr>
    <tr>
      <th>'weather_main'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'weather_main'</td>
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
      <th>'holiday'</th>
      <td>'Christmas Day', 'Columbus Day', 'Independence Day', 'Labor Day', 'Martin Luther King Jr Day', 'Memorial Day', 'New Years Day', 'None', 'State Fair', 'Thanksgiving Day', 'Veterans Day', 'Washingtons Birthday'</td>
    </tr>
    <tr>
      <th>'weather_description'</th>
      <td>'SQUALLS', 'Sky is Clear', 'broken clouds', 'drizzle', 'few clouds', 'fog', 'freezing rain', 'haze', 'heavy intensity drizzle', 'heavy intensity rain', 'heavy snow', 'light intensity drizzle', 'light intensity shower rain', 'light rain', 'light rain and snow', 'light shower snow', 'light snow', 'mist', 'moderate rain', 'overcast clouds', 'proximity shower rain', 'proximity thunderstorm', 'proximity thunderstorm with drizzle', 'proximity thunderstorm with rain', 'scattered clouds', 'shower drizzle', 'sky is clear', 'sleet', 'smoke', 'snow', 'thunderstorm', 'thunderstorm with heavy rain', 'thunderstorm with light drizzle', 'thunderstorm with light rain', 'thunderstorm with rain', 'very heavy rain', 'shower snow', 'thunderstorm with drizzle'</td>
    </tr>
    <tr>
      <th>'weather_main'</th>
      <td>'Clear', 'Clouds', 'Drizzle', 'Fog', 'Haze', 'Mist', 'Rain', 'Smoke', 'Snow', 'Squall', 'Thunderstorm'</td>
    </tr>
  </tbody>
</table>
</div>


Each attribute in your dataset shows up as a row in the schema table, alongside its properties. The schema also captures all the values that a categorical feature takes on, denoted as its domain.

This schema will be used to detect anomalies in the next step.

<a name='2-4'></a>
### 2.4 - ExampleValidator

The [ExampleValidator](https://www.tensorflow.org/tfx/guide/exampleval) component detects anomalies in your data based on the generated schema from the previous step. Like the previous two components, it also uses TFDV under the hood. 

`ExampleValidator` will take as input the statistics from `StatisticsGen` and the schema from `SchemaGen`. By default, it compares the statistics from the evaluation split to the schema from the training split.

<a name='2-4'></a>
#### Exercise 5: ExampleValidator

Fill the code below to detect anomalies in your datasets.


```python
### START CODE HERE
# Instantiate ExampleValidator with the statistics and schema from the previous steps
example_validator = ExampleValidator(
                                    statistics=statistics_gen.outputs['statistics'],
                                    schema=schema_gen.outputs['schema'])
    
    

# Run the component
context.run(example_validator)
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
<div class="tfx-object expanded"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExecutionResult</span><span class="deemphasize"> at 0x7ff54459f5e0</span></div><table class="attr-table"><tr><td class="attr-name">.execution_id</td><td class = "attrvalue">11</td></tr><tr><td class="attr-name">.component</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExampleValidator</span><span class="deemphasize"> at 0x7ff544566700</span></div><table class="attr-table"><tr><td class="attr-name">.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['statistics']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7ff545845be0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: metro_traffic_pipeline/StatisticsGen/statistics/9)<span class="deemphasize"> at 0x7ff55cc00c10</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">metro_traffic_pipeline/StatisticsGen/statistics/9</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['schema']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7ff544566a30</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: metro_traffic_pipeline/SchemaGen/schema/10)<span class="deemphasize"> at 0x7ff544572790</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">metro_traffic_pipeline/SchemaGen/schema/10</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['anomalies']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleAnomalies'</span> (1 artifact)<span class="deemphasize"> at 0x7ff544566070</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleAnomalies</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleAnomalies'</span> (uri: metro_traffic_pipeline/ExampleValidator/anomalies/11)<span class="deemphasize"> at 0x7ff54456bcd0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleAnomalies&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">metro_traffic_pipeline/ExampleValidator/anomalies/11</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.exec_properties</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['exclude_splits']</td><td class = "attrvalue">[]</td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">.component.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['statistics']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7ff545845be0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: metro_traffic_pipeline/StatisticsGen/statistics/9)<span class="deemphasize"> at 0x7ff55cc00c10</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">metro_traffic_pipeline/StatisticsGen/statistics/9</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['schema']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7ff544566a30</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: metro_traffic_pipeline/SchemaGen/schema/10)<span class="deemphasize"> at 0x7ff544572790</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">metro_traffic_pipeline/SchemaGen/schema/10</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.component.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['anomalies']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleAnomalies'</span> (1 artifact)<span class="deemphasize"> at 0x7ff544566070</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleAnomalies</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleAnomalies'</span> (uri: metro_traffic_pipeline/ExampleValidator/anomalies/11)<span class="deemphasize"> at 0x7ff54456bcd0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleAnomalies&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">metro_traffic_pipeline/ExampleValidator/anomalies/11</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr></table></div>



As with the previous steps, you can visualize the anomalies as a table.


```python
# Visualize the output
context.show(example_validator.outputs['anomalies'])
```


<b>Artifact at metro_traffic_pipeline/ExampleValidator/anomalies/11</b><br/><br/>



<div><b>'train' split:</b></div><br/>



<h4 style="color:green;">No anomalies found.</h4>



<div><b>'eval' split:</b></div><br/>



<h4 style="color:green;">No anomalies found.</h4>


If there are anomalies detected, you should examine how you should handle it. For example, you can relax distribution constraints or modify the domain of some features. You've had some practice with this last week when you used TFDV and you can also do that here. 

For this particular case, there should be no anomalies detected and we can proceed to the next step.

<a name='2-5'></a>
### 2.5 - Transform

In this section, you will use the [Transform](https://www.tensorflow.org/tfx/guide/transform) component to perform feature engineering.

`Transform` will take as input the data from `ExampleGen`, the schema from `SchemaGen`, as well as a module containing the preprocessing function.

The component expects an external module for your Transform code so you need to use the magic command `%% writefile` to save the file to disk. We have defined a few constants that group the data's attributes according to the transforms you will perform later. This file will also be saved locally.


```python
# Set the constants module filename
_traffic_constants_module_file = 'traffic_constants.py'
```


```python
%%writefile {_traffic_constants_module_file}

# Features to be scaled to the z-score
DENSE_FLOAT_FEATURE_KEYS = ['temp', 'snow_1h']

# Features to bucketize
BUCKET_FEATURE_KEYS = ['rain_1h']

# Number of buckets used by tf.transform for encoding each feature.
FEATURE_BUCKET_COUNT = {'rain_1h': 3}

# Feature to scale from 0 to 1
RANGE_FEATURE_KEYS = ['clouds_all']

# Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
VOCAB_SIZE = 1000

# Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
OOV_SIZE = 10

# Features with string data types that will be converted to indices
VOCAB_FEATURE_KEYS = [
    'holiday',
    'weather_main',
    'weather_description'
]

# Features with int data type that will be kept as is
CATEGORICAL_FEATURE_KEYS = [
    'hour', 'day', 'day_of_week', 'month'
]

# Feature to predict
VOLUME_KEY = 'traffic_volume'

def transformed_name(key):
    return key + '_xf'
```

    Overwriting traffic_constants.py


<a name='ex-6'></a>
#### Exercise 6

Next, you will fill out the transform module. As mentioned, this will also be saved to disk. Specifically, you will complete the `preprocessing_fn` which defines the transformations. See the code comments for instructions and refer to the [tft module documentation](https://www.tensorflow.org/tfx/transform/api_docs/python/tft) to look up which function to use for a given group of keys.

For the label (i.e. `VOLUME_KEY`), you will transform it to indicate if it is greater than the mean of the entire dataset. For the transform to work, you will need to convert a [SparseTensor](https://www.tensorflow.org/api_docs/python/tf/sparse/SparseTensor) to a dense one. We've provided a `_fill_in_missing()` helper function for you to use.


```python
# Set the transform module filename
_traffic_transform_module_file = 'traffic_transform.py'
```


```python
%%writefile {_traffic_transform_module_file}

import tensorflow as tf
import tensorflow_transform as tft

import traffic_constants

# Unpack the contents of the constants module
_DENSE_FLOAT_FEATURE_KEYS = traffic_constants.DENSE_FLOAT_FEATURE_KEYS
_RANGE_FEATURE_KEYS = traffic_constants.RANGE_FEATURE_KEYS
_VOCAB_FEATURE_KEYS = traffic_constants.VOCAB_FEATURE_KEYS
_VOCAB_SIZE = traffic_constants.VOCAB_SIZE
_OOV_SIZE = traffic_constants.OOV_SIZE
_CATEGORICAL_FEATURE_KEYS = traffic_constants.CATEGORICAL_FEATURE_KEYS
_BUCKET_FEATURE_KEYS = traffic_constants.BUCKET_FEATURE_KEYS
_FEATURE_BUCKET_COUNT = traffic_constants.FEATURE_BUCKET_COUNT
_VOLUME_KEY = traffic_constants.VOLUME_KEY
_transformed_name = traffic_constants.transformed_name


def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
    Args:
    inputs: map from feature keys to raw not-yet-transformed features.
    Returns:
    Map from string feature key to transformed feature operations.
    """
    outputs = {}

    ### START CODE HERE
    
    # Scale these features to the z-score.
    for key in _DENSE_FLOAT_FEATURE_KEYS:
        # Scale these features to the z-score.
        outputs[_transformed_name(key)] = tft.scale_to_z_score(inputs[key])
            

    # Scale these feature/s from 0 to 1
    for key in _RANGE_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.scale_to_0_1(inputs[key])
            

    # Transform the strings into indices 
    # hint: use the VOCAB_SIZE and OOV_SIZE to define the top_k and num_oov parameters
    for key in _VOCAB_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.compute_and_apply_vocabulary(inputs[key],
                                                                           top_k=_VOCAB_SIZE)
            
            
            

    # Bucketize the feature
    for key in _BUCKET_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.bucketize(inputs[key],
                                                       _FEATURE_BUCKET_COUNT[key])
            
            

    # Keep as is. No tft function needed.
    for key in _CATEGORICAL_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.compute_and_apply_vocabulary(inputs[key])

        
    # Use `tf.cast` to cast the label key to float32 and fill in the missing values.
    traffic_volume = _fill_in_missing(tf.cast(inputs['traffic_volume'], tf.float32))
  
    
    # Create a feature that shows if the traffic volume is greater than the mean and cast to an int
    outputs[_transformed_name(_VOLUME_KEY)] = tf.cast(  
        
        # Use `tf.greater` to check if the traffic volume in a row is greater than the mean of the entire traffic volumn column
        tf.greater(traffic_volume, tft.mean(tf.cast(inputs[_VOLUME_KEY], tf.float32))),
        
        tf.int64)                                        

    ### END CODE HERE
    return outputs


def _fill_in_missing(x):
    """Replace missing values in a SparseTensor and convert to a dense tensor.
    Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
    Args:
        x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
          in the second dimension.
    Returns:
        A rank 1 tensor where missing values of `x` have been filled in.
    """
    default_value = '' if x.dtype == tf.string else 0
    
    return tf.squeeze(
      tf.sparse.to_dense(
          tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
          default_value),
      axis=1)
```

    Overwriting traffic_transform.py


<a name='ex-7'></a>
#### Exercise 7

With the transform module defined, complete the code below to perform feature engineering on the raw data.


```python
# ignore tf warning messages
tf.get_logger().setLevel('ERROR')


### START CODE HERE
# Instantiate the Transform component
transform = Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file=os.path.abspath( _traffic_transform_module_file)
                      )
    
    
    

# Run the component.
# The `enable_cache` flag is disabled in case you need to update your transform module file.
context.run(transform, enable_cache=False)
### END CODE HERE
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
<div class="tfx-object expanded"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExecutionResult</span><span class="deemphasize"> at 0x7ff560510bb0</span></div><table class="attr-table"><tr><td class="attr-name">.execution_id</td><td class = "attrvalue">20</td></tr><tr><td class="attr-name">.component</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Transform</span><span class="deemphasize"> at 0x7ff544595940</span></div><table class="attr-table"><tr><td class="attr-name">.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7ff4ac8d42e0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: metro_traffic_pipeline/CsvExampleGen/examples/8)<span class="deemphasize"> at 0x7ff545c583d0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">metro_traffic_pipeline/CsvExampleGen/examples/8</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['schema']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7ff544566a30</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: metro_traffic_pipeline/SchemaGen/schema/10)<span class="deemphasize"> at 0x7ff544572790</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">metro_traffic_pipeline/SchemaGen/schema/10</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['transform_graph']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'TransformGraph'</span> (1 artifact)<span class="deemphasize"> at 0x7ff5445957f0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">TransformGraph</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'TransformGraph'</span> (uri: metro_traffic_pipeline/Transform/transform_graph/20)<span class="deemphasize"> at 0x7ff53bd61a60</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.TransformGraph&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">metro_traffic_pipeline/Transform/transform_graph/20</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['transformed_examples']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7ff53bd61400</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: metro_traffic_pipeline/Transform/transformed_examples/20)<span class="deemphasize"> at 0x7ff53bd61a00</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">metro_traffic_pipeline/Transform/transformed_examples/20</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['updated_analyzer_cache']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'TransformCache'</span> (1 artifact)<span class="deemphasize"> at 0x7ff53bd61520</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">TransformCache</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'TransformCache'</span> (uri: metro_traffic_pipeline/Transform/updated_analyzer_cache/20)<span class="deemphasize"> at 0x7ff53bd61af0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.TransformCache&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">metro_traffic_pipeline/Transform/updated_analyzer_cache/20</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.exec_properties</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['module_file']</td><td class = "attrvalue">/home/jovyan/work/traffic_transform.py</td></tr><tr><td class="attr-name">['preprocessing_fn']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['custom_config']</td><td class = "attrvalue">null</td></tr><tr><td class="attr-name">['splits_config']</td><td class = "attrvalue">None</td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">.component.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7ff4ac8d42e0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: metro_traffic_pipeline/CsvExampleGen/examples/8)<span class="deemphasize"> at 0x7ff545c583d0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">metro_traffic_pipeline/CsvExampleGen/examples/8</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['schema']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7ff544566a30</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: metro_traffic_pipeline/SchemaGen/schema/10)<span class="deemphasize"> at 0x7ff544572790</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">metro_traffic_pipeline/SchemaGen/schema/10</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.component.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['transform_graph']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'TransformGraph'</span> (1 artifact)<span class="deemphasize"> at 0x7ff5445957f0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">TransformGraph</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'TransformGraph'</span> (uri: metro_traffic_pipeline/Transform/transform_graph/20)<span class="deemphasize"> at 0x7ff53bd61a60</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.TransformGraph&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">metro_traffic_pipeline/Transform/transform_graph/20</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['transformed_examples']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7ff53bd61400</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: metro_traffic_pipeline/Transform/transformed_examples/20)<span class="deemphasize"> at 0x7ff53bd61a00</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">metro_traffic_pipeline/Transform/transformed_examples/20</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['updated_analyzer_cache']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'TransformCache'</span> (1 artifact)<span class="deemphasize"> at 0x7ff53bd61520</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">TransformCache</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'TransformCache'</span> (uri: metro_traffic_pipeline/Transform/updated_analyzer_cache/20)<span class="deemphasize"> at 0x7ff53bd61af0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.TransformCache&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">metro_traffic_pipeline/Transform/updated_analyzer_cache/20</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr></table></div>



You should see the output cell by `InteractiveContext` above and see the three artifacts in `.component.outputs`:

* `transform_graph` is the graph that performs the preprocessing operations. This will be included during training and serving to ensure consistent transformations of incoming data.
* `transformed_examples` points to the preprocessed training and evaluation data.
* `updated_analyzer_cache` are stored calculations from previous runs.

The `transform_graph` artifact URI should point to a directory containing:

* The `metadata` subdirectory containing the schema of the original data.
* The `transformed_metadata` subdirectory containing the schema of the preprocessed data. 
* The `transform_fn` subdirectory containing the actual preprocessing graph.

Again, for grading purposes, we inserted an `except` and `else` below to handle checking the output outside the notebook environment.


```python
try:
    # Get the uri of the transform graph
    transform_graph_uri = transform.outputs['transform_graph'].get()[0].uri

except IndexError:
    print("context.run() was no-op")
    transform_path = './metro_traffic_pipeline/Transform/transformed_examples'
    dir_id = os.listdir(transform_path)[0]
    transform_graph_uri = f'{transform_path}/{dir_id}'
    
else:
    # List the subdirectories under the uri
    os.listdir(transform_graph_uri)
```

Lastly, you can also take a look at a few of the transformed examples.


```python
try:
    # Get the URI of the output artifact representing the transformed examples
    train_uri = os.path.join(transform.outputs['transformed_examples'].get()[0].uri, 'train')
    
except IndexError:
    print("context.run() was no-op")
    train_uri = os.path.join(transform_graph_uri, 'train')
```


```python
# Get the list of files in this directory (all compressed TFRecord files)
tfrecord_filenames = [os.path.join(train_uri, name)
                      for name in os.listdir(train_uri)]

# Create a `TFRecordDataset` to read these files
transformed_dataset = tf.data.TFRecordDataset(tfrecord_filenames, compression_type="GZIP")
```


```python
# Get 3 records from the dataset
sample_records_xf = get_records(transformed_dataset, 3)

# Print the output
pp.pprint(sample_records_xf)
```

    [{'features': {'feature': {'clouds_all_xf': {'floatList': {'value': [0.39999998]}},
                               'day_of_week_xf': {'int64List': {'value': ['2']}},
                               'day_xf': {'int64List': {'value': ['26']}},
                               'holiday_xf': {'int64List': {'value': ['0']}},
                               'hour_xf': {'int64List': {'value': ['5']}},
                               'month_xf': {'int64List': {'value': ['11']}},
                               'rain_1h_xf': {'int64List': {'value': ['2']}},
                               'snow_1h_xf': {'floatList': {'value': [-0.027424417]}},
                               'temp_xf': {'floatList': {'value': [0.53368527]}},
                               'traffic_volume_xf': {'int64List': {'value': ['1']}},
                               'weather_description_xf': {'int64List': {'value': ['4']}},
                               'weather_main_xf': {'int64List': {'value': ['0']}}}}},
     {'features': {'feature': {'clouds_all_xf': {'floatList': {'value': [0.75]}},
                               'day_of_week_xf': {'int64List': {'value': ['2']}},
                               'day_xf': {'int64List': {'value': ['26']}},
                               'holiday_xf': {'int64List': {'value': ['0']}},
                               'hour_xf': {'int64List': {'value': ['1']}},
                               'month_xf': {'int64List': {'value': ['11']}},
                               'rain_1h_xf': {'int64List': {'value': ['2']}},
                               'snow_1h_xf': {'floatList': {'value': [-0.027424417]}},
                               'temp_xf': {'floatList': {'value': [0.6156978]}},
                               'traffic_volume_xf': {'int64List': {'value': ['1']}},
                               'weather_description_xf': {'int64List': {'value': ['3']}},
                               'weather_main_xf': {'int64List': {'value': ['0']}}}}},
     {'features': {'feature': {'clouds_all_xf': {'floatList': {'value': [0.9]}},
                               'day_of_week_xf': {'int64List': {'value': ['2']}},
                               'day_xf': {'int64List': {'value': ['26']}},
                               'holiday_xf': {'int64List': {'value': ['0']}},
                               'hour_xf': {'int64List': {'value': ['16']}},
                               'month_xf': {'int64List': {'value': ['11']}},
                               'rain_1h_xf': {'int64List': {'value': ['2']}},
                               'snow_1h_xf': {'floatList': {'value': [-0.027424417]}},
                               'temp_xf': {'floatList': {'value': [0.6324043]}},
                               'traffic_volume_xf': {'int64List': {'value': ['1']}},
                               'weather_description_xf': {'int64List': {'value': ['2']}},
                               'weather_main_xf': {'int64List': {'value': ['0']}}}}}]


**Congratulations on completing this week's assignment!** You've just demonstrated how to build a data pipeline and do feature engineering. You will build upon these concepts in the next weeks where you will deal with more complex datasets and also access the metadata store. Keep up the good work!
