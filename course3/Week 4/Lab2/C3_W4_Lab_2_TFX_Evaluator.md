<a href="https://colab.research.google.com/github/sergejhorvat/MLEP-public/blob/main/course3/Week%204/Lab2/C3_W4_Lab_2_TFX_Evaluator.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Ungraded Lab: Model Analysis with TFX Evaluator

Now that you've used TFMA as a standalone library in the previous lab, you will now see how it is used by TFX with its [Evaluator](https://www.tensorflow.org/tfx/guide/evaluator) component. This component comes after your `Trainer` run and it checks if your trained model meets the minimum required metrics and also compares it with previously generated models.

You will go through a TFX pipeline that prepares and trains the same model architecture you used in the previous lab. As a reminder, this is a binary classifier to be trained on the [Census Income dataset](http://archive.ics.uci.edu/ml/datasets/Census+Income). Since you're already familiar with the earlier TFX components, we will just go over them quickly but we've placed notes on where you can modify code if you want to practice or produce a better result.

Let's begin!

*Credits: Some of the code and discussions are based on the TensorFlow team's [official tutorial](https://www.tensorflow.org/tfx/tutorials/tfx/components_keras).*

## Setup

### Install TFX


```python
!pip install tfx==1.2
```

*Note: In Google Colab, you need to restart the runtime at this point to finalize updating the packages you just installed. You can do so by clicking the `Restart Runtime` at the end of the output cell above (after installation), or by selecting `Runtime > Restart Runtime` in the Menu bar. **Please do not proceed to the next section without restarting.** You can also ignore the errors about version incompatibility of some of the bundled packages because we won't be using those in this notebook.*

### Imports


```python
import os
import pprint

from tfx.components import CsvExampleGen
from tfx.components import ExampleValidator
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Transform
from tfx.components import Tuner
from tfx.components import Trainer
from tfx.components import Evaluator

import tensorflow as tf
import tensorflow_model_analysis as tfma

from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext

tf.get_logger().propagate = False
tf.get_logger().setLevel('ERROR')
pp = pprint.PrettyPrinter()
```

### Set up pipeline paths


```python
# Location of the pipeline metadata store
_pipeline_root = './pipeline/'

# Directory of the raw data files
_data_root = './data/census'

_data_filepath = os.path.join(_data_root, "data.csv")
```


```python
# Create the TFX pipeline files directory
!mkdir {_pipeline_root}

# Create the dataset directory
!mkdir -p {_data_root}
```

    mkdir: cannot create directory ‘./pipeline/’: File exists


### Download and prepare the dataset

Here, you will download the training split of the [Census Income Dataset](http://archive.ics.uci.edu/ml/datasets/Census+Income). This is twice as large as the test dataset you used in the previous lab.


```python
# Define filename and URL
TAR_NAME = 'lab_2_data.tar.gz'
DATA_PATH = f'https://github.com/https-deeplearning-ai/MLEP-public/raw/main/course3/week4-ungraded-lab/{TAR_NAME}'

# Download dataset
!wget -nc {DATA_PATH}

# Extract archive
!tar xvzf {TAR_NAME}

# Delete archive
!rm {TAR_NAME}
```

    --2021-09-03 08:51:13--  https://github.com/https-deeplearning-ai/MLEP-public/raw/main/course3/week4-ungraded-lab/lab_2_data.tar.gz
    Resolving github.com (github.com)... 140.82.121.4
    Connecting to github.com (github.com)|140.82.121.4|:443... connected.
    HTTP request sent, awaiting response... 301 Moved Permanently
    Location: https://github.com/https-deeplearning-ai/machine-learning-engineering-for-production-public/raw/main/course3/week4-ungraded-lab/lab_2_data.tar.gz [following]
    --2021-09-03 08:51:13--  https://github.com/https-deeplearning-ai/machine-learning-engineering-for-production-public/raw/main/course3/week4-ungraded-lab/lab_2_data.tar.gz
    Reusing existing connection to github.com:443.
    HTTP request sent, awaiting response... 302 Found
    Location: https://raw.githubusercontent.com/https-deeplearning-ai/machine-learning-engineering-for-production-public/main/course3/week4-ungraded-lab/lab_2_data.tar.gz [following]
    --2021-09-03 08:51:13--  https://raw.githubusercontent.com/https-deeplearning-ai/machine-learning-engineering-for-production-public/main/course3/week4-ungraded-lab/lab_2_data.tar.gz
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.111.133, 185.199.108.133, 185.199.110.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.111.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 418898 (409K) [application/octet-stream]
    Saving to: ‘lab_2_data.tar.gz’
    
    lab_2_data.tar.gz   100%[===================>] 409.08K  --.-KB/s    in 0.01s   
    
    2021-09-03 08:51:13 (31.2 MB/s) - ‘lab_2_data.tar.gz’ saved [418898/418898]
    
    ./data/census/data.csv


Take a quick look at the first few rows.


```python
# Preview dataset
!head {_data_filepath}
```

    age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,label
    39,State-gov,77516,Bachelors,13,Never-married,Adm-clerical,Not-in-family,White,Male,2174,0,40,United-States,0
    50,Self-emp-not-inc,83311,Bachelors,13,Married-civ-spouse,Exec-managerial,Husband,White,Male,0,0,13,United-States,0
    38,Private,215646,HS-grad,9,Divorced,Handlers-cleaners,Not-in-family,White,Male,0,0,40,United-States,0
    53,Private,234721,11th,7,Married-civ-spouse,Handlers-cleaners,Husband,Black,Male,0,0,40,United-States,0
    28,Private,338409,Bachelors,13,Married-civ-spouse,Prof-specialty,Wife,Black,Female,0,0,40,Cuba,0
    37,Private,284582,Masters,14,Married-civ-spouse,Exec-managerial,Wife,White,Female,0,0,40,United-States,0
    49,Private,160187,9th,5,Married-spouse-absent,Other-service,Not-in-family,Black,Female,0,0,16,Jamaica,0
    52,Self-emp-not-inc,209642,HS-grad,9,Married-civ-spouse,Exec-managerial,Husband,White,Male,0,0,45,United-States,1
    31,Private,45781,Masters,14,Never-married,Prof-specialty,Not-in-family,White,Female,14084,0,50,United-States,1


## TFX Pipeline

### Create the InteractiveContext

As usual, you will initialize the pipeline and use a local SQLite file for the metadata store.


```python
# Initialize InteractiveContext
context = InteractiveContext(pipeline_root=_pipeline_root)
```

    WARNING:absl:InteractiveContext metadata_connection_config not provided: using SQLite ML Metadata database at ./pipeline/metadata.sqlite.


### ExampleGen

You will start by ingesting the data through `CsvExampleGen`. The code below uses the default 2:1 train-eval split (i.e. 33% of the data goes to eval) but feel free to modify if you want. You can review splitting techniques [here](https://www.tensorflow.org/tfx/guide/examplegen#splitting_method).


```python
# Run CsvExampleGen
example_gen = CsvExampleGen(input_base=_data_root)
context.run(example_gen)
```

    WARNING:apache_beam.runners.interactive.interactive_environment:Dependencies required for Interactive Beam PCollection visualization are not available, please use: `pip install apache-beam[interactive]` to install necessary dependencies to enable all data visualization features.




    WARNING:root:Make sure that locally built Python SDK docker image has Python 3.7 interpreter.
    WARNING:apache_beam.io.tfrecordio:Couldn't find python-snappy so the implementation of _TFRecordUtil._masked_crc32c is not as fast as it could be.





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
<div class="tfx-object expanded"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExecutionResult</span><span class="deemphasize"> at 0x7f0b4a161310</span></div><table class="attr-table"><tr><td class="attr-name">.execution_id</td><td class = "attrvalue">13</td></tr><tr><td class="attr-name">.component</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">CsvExampleGen</span><span class="deemphasize"> at 0x7f0ac9d8da50</span></div><table class="attr-table"><tr><td class="attr-name">.inputs</td><td class = "attrvalue">{}</td></tr><tr><td class="attr-name">.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7f0acd962fd0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/CsvExampleGen/examples/13)<span class="deemphasize"> at 0x7f0acc5d8c10</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/CsvExampleGen/examples/13</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.exec_properties</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['input_base']</td><td class = "attrvalue">./data/census</td></tr><tr><td class="attr-name">['input_config']</td><td class = "attrvalue">{
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
}</td></tr><tr><td class="attr-name">['output_data_format']</td><td class = "attrvalue">6</td></tr><tr><td class="attr-name">['output_file_format']</td><td class = "attrvalue">5</td></tr><tr><td class="attr-name">['custom_config']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['range_config']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['span']</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">['version']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['input_fingerprint']</td><td class = "attrvalue">split:single_split,num_files:1,total_bytes:3396202,xor_checksum:1624628026,sum_checksum:1624628026</td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">.component.inputs</td><td class = "attrvalue">{}</td></tr><tr><td class="attr-name">.component.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7f0acd962fd0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/CsvExampleGen/examples/13)<span class="deemphasize"> at 0x7f0acc5d8c10</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/CsvExampleGen/examples/13</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr></table></div>




```python
# Print split names and URI
artifact = example_gen.outputs['examples'].get()[0]
print(artifact.split_names, artifact.uri)
```

    ["train", "eval"] ./pipeline/CsvExampleGen/examples/13


### StatisticsGen
You will then compute the statistics so it can be used by the next components.


```python
# Run StatisticsGen
statistics_gen = StatisticsGen(
    examples=example_gen.outputs['examples'])
context.run(statistics_gen)
```

    WARNING:root:Make sure that locally built Python SDK docker image has Python 3.7 interpreter.





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
<div class="tfx-object expanded"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExecutionResult</span><span class="deemphasize"> at 0x7f0ac7cc6110</span></div><table class="attr-table"><tr><td class="attr-name">.execution_id</td><td class = "attrvalue">14</td></tr><tr><td class="attr-name">.component</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">StatisticsGen</span><span class="deemphasize"> at 0x7f0ac7cc6dd0</span></div><table class="attr-table"><tr><td class="attr-name">.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7f0acd962fd0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/CsvExampleGen/examples/13)<span class="deemphasize"> at 0x7f0acc5d8c10</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/CsvExampleGen/examples/13</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['statistics']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac7cc6f10</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: ./pipeline/StatisticsGen/statistics/14)<span class="deemphasize"> at 0x7f0ac908f7d0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/StatisticsGen/statistics/14</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.exec_properties</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['stats_options_json']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['exclude_splits']</td><td class = "attrvalue">[]</td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">.component.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7f0acd962fd0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/CsvExampleGen/examples/13)<span class="deemphasize"> at 0x7f0acc5d8c10</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/CsvExampleGen/examples/13</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.component.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['statistics']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac7cc6f10</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: ./pipeline/StatisticsGen/statistics/14)<span class="deemphasize"> at 0x7f0ac908f7d0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/StatisticsGen/statistics/14</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr></table></div>



You can look at the visualizations below if you want to explore the data some more.


```python
# Visualize statistics
context.show(statistics_gen.outputs['statistics'])
```


<b>Artifact at ./pipeline/StatisticsGen/statistics/14</b><br/><br/>



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


### SchemaGen

You can then infer the dataset schema with [SchemaGen](https://www.tensorflow.org/tfx/guide/schemagen). This will be used to validate incoming data to ensure that it is formatted correctly.


```python
# Run SchemaGen
schema_gen = SchemaGen(
    statistics=statistics_gen.outputs['statistics'])
context.run(schema_gen)
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
<div class="tfx-object expanded"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExecutionResult</span><span class="deemphasize"> at 0x7f0ac85bc810</span></div><table class="attr-table"><tr><td class="attr-name">.execution_id</td><td class = "attrvalue">15</td></tr><tr><td class="attr-name">.component</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">SchemaGen</span><span class="deemphasize"> at 0x7f0ac85c4d90</span></div><table class="attr-table"><tr><td class="attr-name">.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['statistics']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac7cc6f10</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: ./pipeline/StatisticsGen/statistics/14)<span class="deemphasize"> at 0x7f0ac908f7d0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/StatisticsGen/statistics/14</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['schema']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac85a2f50</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline/SchemaGen/schema/15)<span class="deemphasize"> at 0x7f0ac85bc450</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/SchemaGen/schema/15</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.exec_properties</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['infer_feature_shape']</td><td class = "attrvalue">1</td></tr><tr><td class="attr-name">['exclude_splits']</td><td class = "attrvalue">[]</td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">.component.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['statistics']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac7cc6f10</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: ./pipeline/StatisticsGen/statistics/14)<span class="deemphasize"> at 0x7f0ac908f7d0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/StatisticsGen/statistics/14</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.component.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['schema']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac85a2f50</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline/SchemaGen/schema/15)<span class="deemphasize"> at 0x7f0ac85bc450</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/SchemaGen/schema/15</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr></table></div>



For simplicity, you will just accept the inferred schema but feel free to modify with the [TFDV API](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv) if you want.


```python
# Visualize the inferred Schema
context.show(schema_gen.outputs['schema'])
```


<b>Artifact at ./pipeline/SchemaGen/schema/15</b><br/><br/>



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
      <th>'education'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'education'</td>
    </tr>
    <tr>
      <th>'marital-status'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'marital-status'</td>
    </tr>
    <tr>
      <th>'native-country'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'native-country'</td>
    </tr>
    <tr>
      <th>'occupation'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'occupation'</td>
    </tr>
    <tr>
      <th>'race'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'race'</td>
    </tr>
    <tr>
      <th>'relationship'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'relationship'</td>
    </tr>
    <tr>
      <th>'sex'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'sex'</td>
    </tr>
    <tr>
      <th>'workclass'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'workclass'</td>
    </tr>
    <tr>
      <th>'age'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'capital-gain'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'capital-loss'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'education-num'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'fnlwgt'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'hours-per-week'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'label'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
  </tbody>
</table>
</div>


    /usr/local/lib/python3.7/dist-packages/tensorflow_data_validation/utils/display_util.py:180: FutureWarning: Passing a negative integer is deprecated in version 1.0 and will not be supported in future version. Instead, use None to not limit the column width.
      pd.set_option('max_colwidth', -1)



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
      <th>'education'</th>
      <td>'10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th', 'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Preschool', 'Prof-school', 'Some-college'</td>
    </tr>
    <tr>
      <th>'marital-status'</th>
      <td>'Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'</td>
    </tr>
    <tr>
      <th>'native-country'</th>
      <td>'?', 'Cambodia', 'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic', 'Ecuador', 'El-Salvador', 'England', 'France', 'Germany', 'Greece', 'Guatemala', 'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary', 'India', 'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto-Rico', 'Scotland', 'South', 'Taiwan', 'Thailand', 'Trinadad&amp;Tobago', 'United-States', 'Vietnam', 'Yugoslavia'</td>
    </tr>
    <tr>
      <th>'occupation'</th>
      <td>'?', 'Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving'</td>
    </tr>
    <tr>
      <th>'race'</th>
      <td>'Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'</td>
    </tr>
    <tr>
      <th>'relationship'</th>
      <td>'Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife'</td>
    </tr>
    <tr>
      <th>'sex'</th>
      <td>'Female', 'Male'</td>
    </tr>
    <tr>
      <th>'workclass'</th>
      <td>'?', 'Federal-gov', 'Local-gov', 'Never-worked', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay'</td>
    </tr>
  </tbody>
</table>
</div>


### ExampleValidator

Next, run `ExampleValidator` to check if there are anomalies in the data.


```python
# Run ExampleValidator
example_validator = ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=schema_gen.outputs['schema'])
context.run(example_validator)
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
<div class="tfx-object expanded"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExecutionResult</span><span class="deemphasize"> at 0x7f0ac85abf50</span></div><table class="attr-table"><tr><td class="attr-name">.execution_id</td><td class = "attrvalue">16</td></tr><tr><td class="attr-name">.component</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExampleValidator</span><span class="deemphasize"> at 0x7f0ac85bc610</span></div><table class="attr-table"><tr><td class="attr-name">.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['statistics']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac7cc6f10</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: ./pipeline/StatisticsGen/statistics/14)<span class="deemphasize"> at 0x7f0ac908f7d0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/StatisticsGen/statistics/14</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['schema']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac85a2f50</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline/SchemaGen/schema/15)<span class="deemphasize"> at 0x7f0ac85bc450</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/SchemaGen/schema/15</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['anomalies']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleAnomalies'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac85bc990</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleAnomalies</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleAnomalies'</span> (uri: ./pipeline/ExampleValidator/anomalies/16)<span class="deemphasize"> at 0x7f0ac8600ed0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleAnomalies&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/ExampleValidator/anomalies/16</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.exec_properties</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['exclude_splits']</td><td class = "attrvalue">[]</td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">.component.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['statistics']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac7cc6f10</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: ./pipeline/StatisticsGen/statistics/14)<span class="deemphasize"> at 0x7f0ac908f7d0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/StatisticsGen/statistics/14</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['schema']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac85a2f50</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline/SchemaGen/schema/15)<span class="deemphasize"> at 0x7f0ac85bc450</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/SchemaGen/schema/15</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.component.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['anomalies']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleAnomalies'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac85bc990</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleAnomalies</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleAnomalies'</span> (uri: ./pipeline/ExampleValidator/anomalies/16)<span class="deemphasize"> at 0x7f0ac8600ed0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleAnomalies&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/ExampleValidator/anomalies/16</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr></table></div>



If you just used the inferred schema, then there should not be any anomalies detected. If you modified the schema, then there might be some results here and you can again use TFDV to modify or relax constraints as needed. 

In actual deployments, this component will also help you understand how your data evolves over time and identify data errors. For example, the first batches of data that you get from your users might conform to the schema but it might not be the case after several months. This component will detect that and let you know that your model might need to be updated.


```python
# Check results
context.show(example_validator.outputs['anomalies'])
```


<b>Artifact at ./pipeline/ExampleValidator/anomalies/16</b><br/><br/>



<div><b>'train' split:</b></div><br/>


    /usr/local/lib/python3.7/dist-packages/tensorflow_data_validation/utils/display_util.py:217: FutureWarning: Passing a negative integer is deprecated in version 1.0 and will not be supported in future version. Instead, use None to not limit the column width.
      pd.set_option('max_colwidth', -1)



<h4 style="color:green;">No anomalies found.</h4>



<div><b>'eval' split:</b></div><br/>



<h4 style="color:green;">No anomalies found.</h4>


### Transform

Now you will perform feature engineering on the training data. As shown when you previewed the CSV earlier, the data is still in raw format and cannot be consumed by the model just yet. The transform code in the following cells will take care of scaling your numeric features and one-hot encoding your categorical variables.

*Note: If you're running this exercise for the first time, we advise that you leave the transformation code as is. After you've gone through the entire notebook, then you can modify these for practice and see what results you get. Just make sure that your model builder code in the `Trainer` component will also reflect those changes if needed. For example, removing a feature here should also remove an input layer for that feature in the model.*




```python
# Set the constants module filename
_census_constants_module_file = 'census_constants.py'
```


```python
%%writefile {_census_constants_module_file}

# Features with string data types that will be converted to indices
VOCAB_FEATURE_DICT = {
    'education': 16, 'marital-status': 7, 'occupation': 15, 'race': 5, 
    'relationship': 6, 'workclass': 9, 'sex': 2, 'native-country': 42
}

# Numerical features that are marked as continuous
NUMERIC_FEATURE_KEYS = ['fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

# Feature that can be grouped into buckets
BUCKET_FEATURE_DICT = {'age': 4}

# Number of out-of-vocabulary buckets
NUM_OOV_BUCKETS = 5

# Feature that the model will predict
LABEL_KEY = 'label'

# Utility function for renaming the feature
def transformed_name(key):
    return key + '_xf'
```

    Overwriting census_constants.py



```python
# Set the transform module filename
_census_transform_module_file = 'census_transform.py'
```


```python
%%writefile {_census_transform_module_file}

import tensorflow as tf
import tensorflow_transform as tft

# import constants from cells above
import census_constants

# Unpack the contents of the constants module
_NUMERIC_FEATURE_KEYS = census_constants.NUMERIC_FEATURE_KEYS
_VOCAB_FEATURE_DICT = census_constants.VOCAB_FEATURE_DICT
_BUCKET_FEATURE_DICT = census_constants.BUCKET_FEATURE_DICT
_NUM_OOV_BUCKETS = census_constants.NUM_OOV_BUCKETS
_LABEL_KEY = census_constants.LABEL_KEY
_transformed_name = census_constants.transformed_name

# Define the transformations
def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
    Args:
        inputs: map from feature keys to raw not-yet-transformed features.
    Returns:
        Map from string feature key to transformed feature operations.
    """

    # Initialize outputs dictionary
    outputs = {}

    # Scale these features to the range [0,1]
    for key in _NUMERIC_FEATURE_KEYS:
        scaled = tft.scale_to_0_1(inputs[key])
        outputs[_transformed_name(key)] = tf.reshape(scaled, [-1])

    # Convert strings to indices and convert to one-hot vectors
    for key, vocab_size in _VOCAB_FEATURE_DICT.items():
        indices = tft.compute_and_apply_vocabulary(inputs[key], num_oov_buckets=_NUM_OOV_BUCKETS)
        one_hot = tf.one_hot(indices, vocab_size + _NUM_OOV_BUCKETS)
        outputs[_transformed_name(key)] = tf.reshape(one_hot, [-1, vocab_size + _NUM_OOV_BUCKETS])

    # Bucketize this feature and convert to one-hot vectors
    for key, num_buckets in _BUCKET_FEATURE_DICT.items():
        indices = tft.bucketize(inputs[key], num_buckets)
        one_hot = tf.one_hot(indices, num_buckets)
        outputs[_transformed_name(key)] = tf.reshape(one_hot, [-1, num_buckets])

    # Cast label to float
    outputs[_transformed_name(_LABEL_KEY)] = tf.cast(inputs[_LABEL_KEY], tf.float32)

    return outputs
```

    Overwriting census_transform.py


Now, we pass in this feature engineering code to the `Transform` component and run it to transform your data.


```python
# Run the Transform component
transform = Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file=os.path.abspath(_census_transform_module_file))
context.run(transform, enable_cache=False)
```

    WARNING:root:This output type hint will be ignored and not used for type-checking purposes. Typically, output type hints for a PTransform are single (or nested) types wrapped by a PCollection, PDone, or None. Got: Tuple[Dict[str, Union[NoneType, _Dataset]], Union[Dict[str, Dict[str, PCollection]], NoneType]] instead.
    WARNING:absl:Tables initialized inside a tf.function  will be re-initialized on every invocation of the function. This  re-initialization can have significant impact on performance. Consider lifting  them out of the graph context using  `tf.init_scope`.: compute_and_apply_vocabulary/apply_vocab/text_file_init/InitializeTableFromTextFileV2
    WARNING:absl:Tables initialized inside a tf.function  will be re-initialized on every invocation of the function. This  re-initialization can have significant impact on performance. Consider lifting  them out of the graph context using  `tf.init_scope`.: compute_and_apply_vocabulary_1/apply_vocab/text_file_init/InitializeTableFromTextFileV2
    WARNING:absl:Tables initialized inside a tf.function  will be re-initialized on every invocation of the function. This  re-initialization can have significant impact on performance. Consider lifting  them out of the graph context using  `tf.init_scope`.: compute_and_apply_vocabulary_2/apply_vocab/text_file_init/InitializeTableFromTextFileV2
    WARNING:absl:Tables initialized inside a tf.function  will be re-initialized on every invocation of the function. This  re-initialization can have significant impact on performance. Consider lifting  them out of the graph context using  `tf.init_scope`.: compute_and_apply_vocabulary_3/apply_vocab/text_file_init/InitializeTableFromTextFileV2
    WARNING:absl:Tables initialized inside a tf.function  will be re-initialized on every invocation of the function. This  re-initialization can have significant impact on performance. Consider lifting  them out of the graph context using  `tf.init_scope`.: compute_and_apply_vocabulary_4/apply_vocab/text_file_init/InitializeTableFromTextFileV2
    WARNING:absl:Tables initialized inside a tf.function  will be re-initialized on every invocation of the function. This  re-initialization can have significant impact on performance. Consider lifting  them out of the graph context using  `tf.init_scope`.: compute_and_apply_vocabulary_5/apply_vocab/text_file_init/InitializeTableFromTextFileV2
    WARNING:absl:Tables initialized inside a tf.function  will be re-initialized on every invocation of the function. This  re-initialization can have significant impact on performance. Consider lifting  them out of the graph context using  `tf.init_scope`.: compute_and_apply_vocabulary_6/apply_vocab/text_file_init/InitializeTableFromTextFileV2
    WARNING:absl:Tables initialized inside a tf.function  will be re-initialized on every invocation of the function. This  re-initialization can have significant impact on performance. Consider lifting  them out of the graph context using  `tf.init_scope`.: compute_and_apply_vocabulary_7/apply_vocab/text_file_init/InitializeTableFromTextFileV2
    WARNING:absl:Tables initialized inside a tf.function  will be re-initialized on every invocation of the function. This  re-initialization can have significant impact on performance. Consider lifting  them out of the graph context using  `tf.init_scope`.: compute_and_apply_vocabulary/apply_vocab/text_file_init/InitializeTableFromTextFileV2
    WARNING:absl:Tables initialized inside a tf.function  will be re-initialized on every invocation of the function. This  re-initialization can have significant impact on performance. Consider lifting  them out of the graph context using  `tf.init_scope`.: compute_and_apply_vocabulary_1/apply_vocab/text_file_init/InitializeTableFromTextFileV2
    WARNING:absl:Tables initialized inside a tf.function  will be re-initialized on every invocation of the function. This  re-initialization can have significant impact on performance. Consider lifting  them out of the graph context using  `tf.init_scope`.: compute_and_apply_vocabulary_2/apply_vocab/text_file_init/InitializeTableFromTextFileV2
    WARNING:absl:Tables initialized inside a tf.function  will be re-initialized on every invocation of the function. This  re-initialization can have significant impact on performance. Consider lifting  them out of the graph context using  `tf.init_scope`.: compute_and_apply_vocabulary_3/apply_vocab/text_file_init/InitializeTableFromTextFileV2
    WARNING:absl:Tables initialized inside a tf.function  will be re-initialized on every invocation of the function. This  re-initialization can have significant impact on performance. Consider lifting  them out of the graph context using  `tf.init_scope`.: compute_and_apply_vocabulary_4/apply_vocab/text_file_init/InitializeTableFromTextFileV2
    WARNING:absl:Tables initialized inside a tf.function  will be re-initialized on every invocation of the function. This  re-initialization can have significant impact on performance. Consider lifting  them out of the graph context using  `tf.init_scope`.: compute_and_apply_vocabulary_5/apply_vocab/text_file_init/InitializeTableFromTextFileV2
    WARNING:absl:Tables initialized inside a tf.function  will be re-initialized on every invocation of the function. This  re-initialization can have significant impact on performance. Consider lifting  them out of the graph context using  `tf.init_scope`.: compute_and_apply_vocabulary_6/apply_vocab/text_file_init/InitializeTableFromTextFileV2
    WARNING:absl:Tables initialized inside a tf.function  will be re-initialized on every invocation of the function. This  re-initialization can have significant impact on performance. Consider lifting  them out of the graph context using  `tf.init_scope`.: compute_and_apply_vocabulary_7/apply_vocab/text_file_init/InitializeTableFromTextFileV2
    WARNING:root:This output type hint will be ignored and not used for type-checking purposes. Typically, output type hints for a PTransform are single (or nested) types wrapped by a PCollection, PDone, or None. Got: Tuple[Dict[str, Union[NoneType, _Dataset]], Union[Dict[str, Dict[str, PCollection]], NoneType]] instead.
    WARNING:root:Make sure that locally built Python SDK docker image has Python 3.7 interpreter.





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
<div class="tfx-object expanded"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExecutionResult</span><span class="deemphasize"> at 0x7f0ac8690d10</span></div><table class="attr-table"><tr><td class="attr-name">.execution_id</td><td class = "attrvalue">17</td></tr><tr><td class="attr-name">.component</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Transform</span><span class="deemphasize"> at 0x7f0ac8615150</span></div><table class="attr-table"><tr><td class="attr-name">.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7f0acd962fd0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/CsvExampleGen/examples/13)<span class="deemphasize"> at 0x7f0acc5d8c10</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/CsvExampleGen/examples/13</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['schema']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac85a2f50</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline/SchemaGen/schema/15)<span class="deemphasize"> at 0x7f0ac85bc450</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/SchemaGen/schema/15</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['transform_graph']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'TransformGraph'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac86151d0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">TransformGraph</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'TransformGraph'</span> (uri: ./pipeline/Transform/transform_graph/17)<span class="deemphasize"> at 0x7f0ac85c6090</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.TransformGraph&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/transform_graph/17</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['transformed_examples']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac8615510</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/Transform/transformed_examples/17)<span class="deemphasize"> at 0x7f0ac8615a50</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/transformed_examples/17</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['updated_analyzer_cache']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'TransformCache'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac8615250</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">TransformCache</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'TransformCache'</span> (uri: ./pipeline/Transform/updated_analyzer_cache/17)<span class="deemphasize"> at 0x7f0ac85bc850</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.TransformCache&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/updated_analyzer_cache/17</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['pre_transform_schema']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac8615310</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline/Transform/pre_transform_schema/17)<span class="deemphasize"> at 0x7f0ac85bcbd0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/pre_transform_schema/17</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['pre_transform_stats']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac8615dd0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: ./pipeline/Transform/pre_transform_stats/17)<span class="deemphasize"> at 0x7f0ac90cc9d0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/pre_transform_stats/17</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue"></td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['post_transform_schema']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac8615790</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline/Transform/post_transform_schema/17)<span class="deemphasize"> at 0x7f0ac7cc6a50</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/post_transform_schema/17</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['post_transform_stats']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac8615f10</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: ./pipeline/Transform/post_transform_stats/17)<span class="deemphasize"> at 0x7f0ac860ce10</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/post_transform_stats/17</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue"></td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['post_transform_anomalies']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleAnomalies'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac8615390</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleAnomalies</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleAnomalies'</span> (uri: ./pipeline/Transform/post_transform_anomalies/17)<span class="deemphasize"> at 0x7f0ac860ca10</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleAnomalies&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/post_transform_anomalies/17</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue"></td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.exec_properties</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['module_file']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['preprocessing_fn']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['stats_options_updater_fn']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['force_tf_compat_v1']</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">['custom_config']</td><td class = "attrvalue">null</td></tr><tr><td class="attr-name">['splits_config']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['disable_statistics']</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">['module_path']</td><td class = "attrvalue">census_transform@./pipeline/_wheels/tfx_user_code_Transform-0.0+cc29a9c825e35b0142dabcb732b412ff69124c1b5f5c6eee2546a102dcbf15c9-py3-none-any.whl</td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">.component.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7f0acd962fd0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/CsvExampleGen/examples/13)<span class="deemphasize"> at 0x7f0acc5d8c10</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/CsvExampleGen/examples/13</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['schema']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac85a2f50</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline/SchemaGen/schema/15)<span class="deemphasize"> at 0x7f0ac85bc450</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/SchemaGen/schema/15</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.component.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['transform_graph']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'TransformGraph'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac86151d0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">TransformGraph</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'TransformGraph'</span> (uri: ./pipeline/Transform/transform_graph/17)<span class="deemphasize"> at 0x7f0ac85c6090</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.TransformGraph&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/transform_graph/17</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['transformed_examples']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac8615510</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/Transform/transformed_examples/17)<span class="deemphasize"> at 0x7f0ac8615a50</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/transformed_examples/17</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['updated_analyzer_cache']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'TransformCache'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac8615250</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">TransformCache</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'TransformCache'</span> (uri: ./pipeline/Transform/updated_analyzer_cache/17)<span class="deemphasize"> at 0x7f0ac85bc850</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.TransformCache&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/updated_analyzer_cache/17</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['pre_transform_schema']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac8615310</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline/Transform/pre_transform_schema/17)<span class="deemphasize"> at 0x7f0ac85bcbd0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/pre_transform_schema/17</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['pre_transform_stats']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac8615dd0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: ./pipeline/Transform/pre_transform_stats/17)<span class="deemphasize"> at 0x7f0ac90cc9d0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/pre_transform_stats/17</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue"></td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['post_transform_schema']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac8615790</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline/Transform/post_transform_schema/17)<span class="deemphasize"> at 0x7f0ac7cc6a50</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/post_transform_schema/17</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['post_transform_stats']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac8615f10</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: ./pipeline/Transform/post_transform_stats/17)<span class="deemphasize"> at 0x7f0ac860ce10</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/post_transform_stats/17</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue"></td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['post_transform_anomalies']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleAnomalies'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac8615390</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleAnomalies</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleAnomalies'</span> (uri: ./pipeline/Transform/post_transform_anomalies/17)<span class="deemphasize"> at 0x7f0ac860ca10</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleAnomalies&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/post_transform_anomalies/17</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue"></td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr></table></div>



You can see a sample result for one row with the code below. Notice that the numeric features are indeed scaled and the categorical features are now one-hot encoded.


```python
# Get the URI of the output artifact representing the transformed examples
train_uri = os.path.join(transform.outputs['transformed_examples'].get()[0].uri, 'Split-train')

# Get the list of files in this directory (all compressed TFRecord files)
tfrecord_filenames = [os.path.join(train_uri, name)
                      for name in os.listdir(train_uri)]

# Create a `TFRecordDataset` to read these files
dataset = tf.data.TFRecordDataset(tfrecord_filenames, compression_type="GZIP")

# Decode the first record and print output
for tfrecord in dataset.take(1):
  serialized_example = tfrecord.numpy()
  example = tf.train.Example()
  example.ParseFromString(serialized_example)
  pp.pprint(example)
```

    features {
      feature {
        key: "age_xf"
        value {
          float_list {
            value: 0.0
            value: 0.0
            value: 1.0
            value: 0.0
          }
        }
      }
      feature {
        key: "capital-gain_xf"
        value {
          float_list {
            value: 0.02174021676182747
          }
        }
      }
      feature {
        key: "capital-loss_xf"
        value {
          float_list {
            value: 0.0
          }
        }
      }
      feature {
        key: "education-num_xf"
        value {
          float_list {
            value: 0.800000011920929
          }
        }
      }
      feature {
        key: "education_xf"
        value {
          float_list {
            value: 0.0
            value: 0.0
            value: 1.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
          }
        }
      }
      feature {
        key: "fnlwgt_xf"
        value {
          float_list {
            value: 0.044301897287368774
          }
        }
      }
      feature {
        key: "hours-per-week_xf"
        value {
          float_list {
            value: 0.3979591727256775
          }
        }
      }
      feature {
        key: "label_xf"
        value {
          float_list {
            value: 0.0
          }
        }
      }
      feature {
        key: "marital-status_xf"
        value {
          float_list {
            value: 0.0
            value: 1.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
          }
        }
      }
      feature {
        key: "native-country_xf"
        value {
          float_list {
            value: 1.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
          }
        }
      }
      feature {
        key: "occupation_xf"
        value {
          float_list {
            value: 0.0
            value: 0.0
            value: 0.0
            value: 1.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
          }
        }
      }
      feature {
        key: "race_xf"
        value {
          float_list {
            value: 1.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
          }
        }
      }
      feature {
        key: "relationship_xf"
        value {
          float_list {
            value: 0.0
            value: 1.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
          }
        }
      }
      feature {
        key: "sex_xf"
        value {
          float_list {
            value: 1.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
          }
        }
      }
      feature {
        key: "workclass_xf"
        value {
          float_list {
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 1.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
            value: 0.0
          }
        }
      }
    }
    


As you already know, the `Transform` component not only outputs the transformed examples but also the transformation graph. This should be used on all inputs when your model is deployed to ensure that it is transformed the same way as your training data. Else, it can produce training-serving skew which leads to noisy predictions.

The `Transform` component stores related files in its `transform_graph` output and it would be good to quickly review its contents before we move on to the next component. As shown below, the URI of this output points to a directory containing three subdirectories.


```python
# Get URI and list subdirectories
graph_uri = transform.outputs['transform_graph'].get()[0].uri
os.listdir(graph_uri)
```




    ['metadata', 'transformed_metadata', 'transform_fn']



* The `transformed_metadata` subdirectory contains the schema of the preprocessed data. 
* The `transform_fn` subdirectory contains the actual preprocessing graph. 
* The `metadata` subdirectory contains the schema of the original data.

### Trainer

Next, you will now build the model to make your predictions. As mentioned earlier, this is a binary classifier where the label is 1 if a person earns more than 50k USD and 0 if less than or equal. The model used here uses the [wide and deep model](https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html) as reference but feel free to modify after you've completed the exercise. Also for simplicity, the hyperparameters (e.g. number of hidden units) have been hardcoded but feel free to use a `Tuner` component as you did in Week 1 if you want to get some practice.

As a reminder, it is best to start from `run_fn()` when you're reviewing the module file below. The `Trainer` component looks for that function first. All other functions defined in the module are just utility functions for `run_fn()`.

Another thing you will notice below is the `_get_serve_tf_examples_fn()` function. This is tied to the `serving_default` [signature](https://www.tensorflow.org/guide/saved_model#specifying_signatures_during_export) which makes it possible for you to just pass in raw features when the model is served for inference. You have seen this in action in the previous lab. This is done by decorating the enclosing function, `serve_tf_examples_fn()`, with [tf.function](https://www.tensorflow.org/guide/intro_to_graphs). This indicates that the computation will be done by first tracing a [Tensorflow graph](https://www.tensorflow.org/api_docs/python/tf/Graph). You will notice that this function uses `model.tft_layer` which comes from `transform_graph` output. Now when you call the `.get_concrete_function()` method on this tf.function in `run_fn()`, you are creating the graph that will be used in later computations. This graph is used whenever you pass in an `examples` argument pointing to a Tensor with `tf.string` dtype. That matches the format of the serialized batches of data you used in the previous lab.




```python
# Declare trainer module file
_census_trainer_module_file = 'census_trainer.py'
```


```python
%%writefile {_census_trainer_module_file}

from typing import List, Text

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils

from tfx.components.trainer.fn_args_utils import DataAccessor, FnArgs
from tfx_bsl.public.tfxio import TensorFlowDatasetOptions

# import same constants from transform module
import census_constants

# Unpack the contents of the constants module
_NUMERIC_FEATURE_KEYS = census_constants.NUMERIC_FEATURE_KEYS
_VOCAB_FEATURE_DICT = census_constants.VOCAB_FEATURE_DICT
_BUCKET_FEATURE_DICT = census_constants.BUCKET_FEATURE_DICT
_NUM_OOV_BUCKETS = census_constants.NUM_OOV_BUCKETS
_LABEL_KEY = census_constants.LABEL_KEY
_transformed_name = census_constants.transformed_name


def _gzip_reader_fn(filenames):
  '''Load compressed dataset
  
  Args:
    filenames - filenames of TFRecords to load

  Returns:
    TFRecordDataset loaded from the filenames
  '''

  # Load the dataset. Specify the compression type since it is saved as `.gz`
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _input_fn(file_pattern,
              tf_transform_output,
              num_epochs=None,
              batch_size=32) -> tf.data.Dataset:
  '''Create batches of features and labels from TF Records

  Args:
    file_pattern - List of files or patterns of file paths containing Example records.
    tf_transform_output - transform output graph
    num_epochs - Integer specifying the number of times to read through the dataset. 
            If None, cycles through the dataset forever.
    batch_size - An int representing the number of records to combine in a single batch.

  Returns:
    A dataset of dict elements, (or a tuple of dict elements and label). 
    Each dict maps feature keys to Tensor or SparseTensor objects.
  '''

  # Get post-transfrom feature spec
  transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy())
  
  # Create batches of data
  dataset = tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=batch_size,
      features=transformed_feature_spec,
      reader=_gzip_reader_fn,
      num_epochs=num_epochs,
      label_key=_transformed_name(_LABEL_KEY)
      )
  
  return dataset


def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Returns a function that parses a serialized tf.Example and applies TFT."""

  # Get transformation graph
  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    # Get pre-transform feature spec
    feature_spec = tf_transform_output.raw_feature_spec()

    # Pop label since serving inputs do not include the label
    feature_spec.pop(_LABEL_KEY)

    # Parse raw examples into a dictionary of tensors matching the feature spec
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

    # Transform the raw examples using the transform graph
    transformed_features = model.tft_layer(parsed_features)

    # Get predictions using the transformed features
    return model(transformed_features)

  return serve_tf_examples_fn


def _build_keras_model(hidden_units: List[int] = None) -> tf.keras.Model:
  """Creates a DNN Keras model for classifying income data.

  Args:
    hidden_units: [int], the layer sizes of the DNN (input layer first).

  Returns:
    A keras Model.
  """

  # Use helper function to create the model
  model = _wide_and_deep_classifier(
      dnn_hidden_units=hidden_units or [100, 70, 50, 25])
  
  return model


def _wide_and_deep_classifier(dnn_hidden_units):
  """Build a simple keras wide and deep model using the Functional API.

  Args:
    wide_columns: Feature columns wrapped in indicator_column for wide (linear)
      part of the model.
    deep_columns: Feature columns for deep part of the model.
    dnn_hidden_units: [int], the layer sizes of the hidden DNN.

  Returns:
    A Wide and Deep Keras model
  """

  # Define input layers for numeric keys
  input_numeric = [
      tf.keras.layers.Input(name=_transformed_name(colname), shape=(1,), dtype=tf.float32)
      for colname in _NUMERIC_FEATURE_KEYS
  ]

  # Define input layers for vocab keys
  input_categorical = [
      tf.keras.layers.Input(name=_transformed_name(colname), shape=(vocab_size + _NUM_OOV_BUCKETS,), dtype=tf.float32)
      for colname, vocab_size in _VOCAB_FEATURE_DICT.items()
  ]

  # Define input layers for bucket key
  input_categorical += [
      tf.keras.layers.Input(name=_transformed_name(colname), shape=(num_buckets,), dtype=tf.float32)
      for colname, num_buckets in _BUCKET_FEATURE_DICT.items()
  ]

  # Concatenate numeric inputs
  deep = tf.keras.layers.concatenate(input_numeric)

  # Create deep dense network for numeric inputs
  for numnodes in dnn_hidden_units:
    deep = tf.keras.layers.Dense(numnodes)(deep)

  # Concatenate categorical inputs
  wide = tf.keras.layers.concatenate(input_categorical)

  # Create shallow dense network for categorical inputs
  wide = tf.keras.layers.Dense(128, activation='relu')(wide)

  # Combine wide and deep networks
  combined = tf.keras.layers.concatenate([deep, wide])
                                              
  # Define output of binary classifier
  output = tf.keras.layers.Dense(
      1, activation='sigmoid')(combined)

  # Setup combined input
  input_layers = input_numeric + input_categorical

  # Create the Keras model
  model = tf.keras.Model(input_layers, output)

  # Define training parameters
  model.compile(
      loss='binary_crossentropy',
      optimizer=tf.keras.optimizers.Adam(lr=0.001),
      metrics='binary_accuracy')
  
  # Print model summary
  model.summary()

  return model


# TFX Trainer will call this function.
def run_fn(fn_args: FnArgs):
  """Defines and trains the model.
  
  Args:
    fn_args: Holds args as name/value pairs. Refer here for the complete attributes: 
    https://www.tensorflow.org/tfx/api_docs/python/tfx/components/trainer/fn_args_utils/FnArgs#attributes
  """

  # Number of nodes in the first layer of the DNN
  first_dnn_layer_size = 100
  num_dnn_layers = 4
  dnn_decay_factor = 0.7

  # Get transform output (i.e. transform graph) wrapper
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  # Create batches of train and eval sets
  train_dataset = _input_fn(fn_args.train_files[0], tf_transform_output, 10)
  eval_dataset = _input_fn(fn_args.eval_files[0], tf_transform_output, 10)

  # Build the model
  model = _build_keras_model(
      # Construct layers sizes with exponential decay
      hidden_units=[
          max(2, int(first_dnn_layer_size * dnn_decay_factor**i))
          for i in range(num_dnn_layers)
      ])
  
  # Callback for TensorBoard
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=fn_args.model_run_dir, update_freq='batch')


  # Train the model
  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      callbacks=[tensorboard_callback])
  

  # Define default serving signature
  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model,
                                    tf_transform_output).get_concrete_function(
                                        tf.TensorSpec(
                                            shape=[None],
                                            dtype=tf.string,
                                            name='examples')),
  }
  

  # Save model with signature
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
```

    Overwriting census_trainer.py


Now, we pass in this model code to the `Trainer` component and run it to train the model.

*Note: You can ignore the `Exception ignored in: <function CapturableResource.__del__>` prompt which generates a long traceback. This might be an issue with the underlying TFMA version used and has been flagged to the TFX team so it can be suppressed. This might pop up here and in the Evaluator component.*


```python
from tfx.proto import trainer_pb2

trainer = Trainer(
    module_file=os.path.abspath(_census_trainer_module_file),
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(num_steps=50),
    eval_args=trainer_pb2.EvalArgs(num_steps=50))
context.run(trainer, enable_cache=False)
```

    WARNING:absl:Examples artifact does not have payload_format custom property. Falling back to FORMAT_TF_EXAMPLE
    WARNING:absl:Examples artifact does not have payload_format custom property. Falling back to FORMAT_TF_EXAMPLE
    WARNING:absl:Examples artifact does not have payload_format custom property. Falling back to FORMAT_TF_EXAMPLE
    /usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/optimizer_v2/optimizer_v2.py:375: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
      "The `lr` argument is deprecated, use `learning_rate` instead.")


    Model: "model"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    fnlwgt_xf (InputLayer)          [(None, 1)]          0                                            
    __________________________________________________________________________________________________
    education-num_xf (InputLayer)   [(None, 1)]          0                                            
    __________________________________________________________________________________________________
    capital-gain_xf (InputLayer)    [(None, 1)]          0                                            
    __________________________________________________________________________________________________
    capital-loss_xf (InputLayer)    [(None, 1)]          0                                            
    __________________________________________________________________________________________________
    hours-per-week_xf (InputLayer)  [(None, 1)]          0                                            
    __________________________________________________________________________________________________
    concatenate (Concatenate)       (None, 5)            0           fnlwgt_xf[0][0]                  
                                                                     education-num_xf[0][0]           
                                                                     capital-gain_xf[0][0]            
                                                                     capital-loss_xf[0][0]            
                                                                     hours-per-week_xf[0][0]          
    __________________________________________________________________________________________________
    dense (Dense)                   (None, 100)          600         concatenate[0][0]                
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 70)           7070        dense[0][0]                      
    __________________________________________________________________________________________________
    education_xf (InputLayer)       [(None, 21)]         0                                            
    __________________________________________________________________________________________________
    marital-status_xf (InputLayer)  [(None, 12)]         0                                            
    __________________________________________________________________________________________________
    occupation_xf (InputLayer)      [(None, 20)]         0                                            
    __________________________________________________________________________________________________
    race_xf (InputLayer)            [(None, 10)]         0                                            
    __________________________________________________________________________________________________
    relationship_xf (InputLayer)    [(None, 11)]         0                                            
    __________________________________________________________________________________________________
    workclass_xf (InputLayer)       [(None, 14)]         0                                            
    __________________________________________________________________________________________________
    sex_xf (InputLayer)             [(None, 7)]          0                                            
    __________________________________________________________________________________________________
    native-country_xf (InputLayer)  [(None, 47)]         0                                            
    __________________________________________________________________________________________________
    age_xf (InputLayer)             [(None, 4)]          0                                            
    __________________________________________________________________________________________________
    dense_2 (Dense)                 (None, 48)           3408        dense_1[0][0]                    
    __________________________________________________________________________________________________
    concatenate_1 (Concatenate)     (None, 146)          0           education_xf[0][0]               
                                                                     marital-status_xf[0][0]          
                                                                     occupation_xf[0][0]              
                                                                     race_xf[0][0]                    
                                                                     relationship_xf[0][0]            
                                                                     workclass_xf[0][0]               
                                                                     sex_xf[0][0]                     
                                                                     native-country_xf[0][0]          
                                                                     age_xf[0][0]                     
    __________________________________________________________________________________________________
    dense_3 (Dense)                 (None, 34)           1666        dense_2[0][0]                    
    __________________________________________________________________________________________________
    dense_4 (Dense)                 (None, 128)          18816       concatenate_1[0][0]              
    __________________________________________________________________________________________________
    concatenate_2 (Concatenate)     (None, 162)          0           dense_3[0][0]                    
                                                                     dense_4[0][0]                    
    __________________________________________________________________________________________________
    dense_5 (Dense)                 (None, 1)            163         concatenate_2[0][0]              
    ==================================================================================================
    Total params: 31,723
    Trainable params: 31,723
    Non-trainable params: 0
    __________________________________________________________________________________________________
    50/50 [==============================] - 2s 15ms/step - loss: 0.5082 - binary_accuracy: 0.7625 - val_loss: 0.4058 - val_binary_accuracy: 0.8075





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
<div class="tfx-object expanded"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExecutionResult</span><span class="deemphasize"> at 0x7f0ac845d150</span></div><table class="attr-table"><tr><td class="attr-name">.execution_id</td><td class = "attrvalue">18</td></tr><tr><td class="attr-name">.component</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Trainer</span><span class="deemphasize"> at 0x7f0ac7731f50</span></div><table class="attr-table"><tr><td class="attr-name">.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac8615510</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/Transform/transformed_examples/17)<span class="deemphasize"> at 0x7f0ac8615a50</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/transformed_examples/17</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['transform_graph']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'TransformGraph'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac86151d0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">TransformGraph</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'TransformGraph'</span> (uri: ./pipeline/Transform/transform_graph/17)<span class="deemphasize"> at 0x7f0ac85c6090</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.TransformGraph&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/transform_graph/17</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['schema']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac85a2f50</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline/SchemaGen/schema/15)<span class="deemphasize"> at 0x7f0ac85bc450</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/SchemaGen/schema/15</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['model']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Model'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac7731a90</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Model</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Model'</span> (uri: ./pipeline/Trainer/model/18)<span class="deemphasize"> at 0x7f0ac7762290</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Model&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Trainer/model/18</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['model_run']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ModelRun'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac7731510</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ModelRun</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ModelRun'</span> (uri: ./pipeline/Trainer/model_run/18)<span class="deemphasize"> at 0x7f0ac7762210</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ModelRun&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Trainer/model_run/18</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.exec_properties</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['train_args']</td><td class = "attrvalue">{
  &quot;num_steps&quot;: 50
}</td></tr><tr><td class="attr-name">['eval_args']</td><td class = "attrvalue">{
  &quot;num_steps&quot;: 50
}</td></tr><tr><td class="attr-name">['module_file']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['run_fn']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['trainer_fn']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['custom_config']</td><td class = "attrvalue">null</td></tr><tr><td class="attr-name">['module_path']</td><td class = "attrvalue">census_trainer@./pipeline/_wheels/tfx_user_code_Trainer-0.0+cc29a9c825e35b0142dabcb732b412ff69124c1b5f5c6eee2546a102dcbf15c9-py3-none-any.whl</td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">.component.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac8615510</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/Transform/transformed_examples/17)<span class="deemphasize"> at 0x7f0ac8615a50</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/transformed_examples/17</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['transform_graph']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'TransformGraph'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac86151d0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">TransformGraph</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'TransformGraph'</span> (uri: ./pipeline/Transform/transform_graph/17)<span class="deemphasize"> at 0x7f0ac85c6090</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.TransformGraph&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/transform_graph/17</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['schema']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac85a2f50</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline/SchemaGen/schema/15)<span class="deemphasize"> at 0x7f0ac85bc450</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/SchemaGen/schema/15</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.component.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['model']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Model'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac7731a90</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Model</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Model'</span> (uri: ./pipeline/Trainer/model/18)<span class="deemphasize"> at 0x7f0ac7762290</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Model&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Trainer/model/18</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['model_run']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ModelRun'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac7731510</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ModelRun</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ModelRun'</span> (uri: ./pipeline/Trainer/model_run/18)<span class="deemphasize"> at 0x7f0ac7762210</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ModelRun&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Trainer/model_run/18</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr></table></div>



Let's review the outputs of this component. The `model` output points to the model output itself.


```python
# Get `model` output of the component
model_artifact_dir = trainer.outputs['model'].get()[0].uri

# List top-level directory
pp.pprint(os.listdir(model_artifact_dir))

# Get model directory
model_dir = os.path.join(model_artifact_dir, 'Format-Serving')

# List subdirectories
pp.pprint(os.listdir(model_dir))
```

    ['Format-Serving']
    ['keras_metadata.pb', 'saved_model.pb', 'assets', 'variables']


The `model_run` output acts as the working directory and can be used to output non-model related output (e.g., TensorBoard logs).


```python
# Get `model_run` output URI
model_run_artifact_dir = trainer.outputs['model_run'].get()[0].uri

# Load results to Tensorboard
%load_ext tensorboard
%tensorboard --logdir {model_run_artifact_dir}
```


    <IPython.core.display.Javascript object>


### Evaluator

The `Evaluator` component computes model performance metrics over the evaluation set using the [TensorFlow Model Analysis](https://www.tensorflow.org/tfx/model_analysis/get_started) library. The `Evaluator` can also optionally validate that a newly trained model is better than the previous model. This is useful in a production pipeline setting where you may automatically train and validate a model every day.

There's a few steps needed to setup this component and you will do it in the next cells.

#### Define EvalConfig

First, you will define the `EvalConfig` message as you did in the previous lab. You can also set thresholds so you can compare subsequent models to it. The module below should look familiar. One minor difference is you don't have to define the candidate and baseline model names in the `model_specs`. That is automatically detected.


```python
import tensorflow_model_analysis as tfma
from google.protobuf import text_format

eval_config = text_format.Parse("""
  ## Model information
  model_specs {
    # This assumes a serving model with signature 'serving_default'.
    label_key: "label"
  }

  ## Post training metric information
  metrics_specs {
    metrics { class_name: "ExampleCount" }
    metrics {
      class_name: "BinaryAccuracy"
      threshold {
        # Ensure that metric is always > 0.5
        value_threshold {
          lower_bound { value: 0.5 }
        }
        # Ensure that metric does not drop by more than a small epsilon
        # e.g. (candidate - baseline) > -1e-10 or candidate > baseline - 1e-10
        change_threshold {
          direction: HIGHER_IS_BETTER
          absolute { value: -1e-10 }
        }
      }
    }
    metrics { class_name: "BinaryCrossentropy" }
    metrics { class_name: "AUC" }
    metrics { class_name: "AUCPrecisionRecall" }
    metrics { class_name: "Precision" }
    metrics { class_name: "Recall" }
    metrics { class_name: "MeanLabel" }
    metrics { class_name: "MeanPrediction" }
    metrics { class_name: "Calibration" }
    metrics { class_name: "CalibrationPlot" }
    metrics { class_name: "ConfusionMatrixPlot" }
    # ... add additional metrics and plots ...
  }

  ## Slicing information
  slicing_specs {}  # overall slice
  slicing_specs {
    feature_keys: ["race"]
  }
  slicing_specs {
    feature_keys: ["sex"]
  }
""", tfma.EvalConfig())
```

#### Resolve latest blessed model

If you remember in the last lab, you were able to validate a candidate model against a baseline by modifying the `EvalConfig` and `EvalSharedModel` definitions. That is also possible using the `Evaluator` component and you will see how it is done in this section.

First thing to note is that the `Evaluator` marks a model as `BLESSED` if it meets the metrics thresholds you set in the eval config module. You can load the latest blessed model by using the [`LatestBlessedModelStrategy`](https://www.tensorflow.org/tfx/api_docs/python/tfx/dsl/input_resolution/strategies/latest_blessed_model_strategy/LatestBlessedModelStrategy) with the [Resolver](https://www.tensorflow.org/tfx/api_docs/python/tfx/dsl/components/common/resolver/Resolver) component. This component takes care of finding the latest blessed model for you so you don't have to remember it manually. The syntax is shown below. 


```python
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import LatestBlessedModelStrategy
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing

# Setup the Resolver node to find the latest blessed model
model_resolver = Resolver(
      strategy_class=LatestBlessedModelStrategy,
      model=Channel(type=Model),
      model_blessing=Channel(
          type=ModelBlessing)).with_id(
              'latest_blessed_model_resolver')

# Run the Resolver node
context.run(model_resolver)
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
<div class="tfx-object expanded"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExecutionResult</span><span class="deemphasize"> at 0x7f0ac86e80d0</span></div><table class="attr-table"><tr><td class="attr-name">.execution_id</td><td class = "attrvalue">19</td></tr><tr><td class="attr-name">.component</td><td class = "attrvalue">&lt;tfx.dsl.components.common.resolver.Resolver object at 0x7f0ac811b550&gt;</td></tr><tr><td class="attr-name">.component.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['model']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Model'</span> (0 artifacts)<span class="deemphasize"> at 0x7f0ac814fad0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Model</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue">[]</td></tr></table></div></td></tr><tr><td class="attr-name">['model_blessing']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ModelBlessing'</span> (0 artifacts)<span class="deemphasize"> at 0x7f0ac817fc90</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ModelBlessing</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue">[]</td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.component.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['model']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Model'</span> (0 artifacts)<span class="deemphasize"> at 0x7f0ac40b2a50</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Model</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue">[]</td></tr></table></div></td></tr><tr><td class="attr-name">['model_blessing']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ModelBlessing'</span> (0 artifacts)<span class="deemphasize"> at 0x7f0ac075bf50</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ModelBlessing</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue">[]</td></tr></table></div></td></tr></table></td></tr></table></div>



As expected, the search yielded 0 artifacts because you haven't evaluated any models yet. You will run this component again in later parts of this notebook and you will see a different result.


```python
# Load Resolver outputs
model_resolver.outputs['model']
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
<div class="tfx-object expanded"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Model'</span> (0 artifacts)<span class="deemphasize"> at 0x7f0ac40b2a50</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Model</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue">[]</td></tr></table></div>



#### Run the Evaluator component

With the `EvalConfig` defined and code to load the latest blessed model available, you can proceed to run the `Evaluator` component. 

You will notice that two models are passed into the component. The `Trainer` output will serve as the candidate model while the output of the `Resolver` will be the baseline model. While you can indeed run the `Evaluator` without comparing two models, it would likely be required in production environments so it's best to include it. Since the `Resolver` doesn't have any results yet, the `Evaluator` will just mark the candidate model as `BLESSED` in this run.

Aside from the eval config and models (i.e. Trainer and Resolver output), you will also pass in the *raw* examples from `ExampleGen`. By default, the component will look for the `eval` split of these examples and since you've defined the serving signature, these will be transformed internally before feeding to the model inputs.

*Note: You can ignore the `Exception ignored in: <function CapturableResource.__del__>` prompt which generates a long traceback. This might be an issue with the underlying TFMA version used and has been flagged to the TFX team so it can be suppressed.*


```python
# Setup and run the Evaluator component
evaluator = Evaluator(
    examples=example_gen.outputs['examples'],
    model=trainer.outputs['model'],
    baseline_model=model_resolver.outputs['model'],
    eval_config=eval_config)
context.run(evaluator, enable_cache=False)
```

    ERROR:absl:There are change thresholds, but the baseline is missing. This is allowed only when rubber stamping (first run).
    Exception ignored in: <function CapturableResource.__del__ at 0x7f0af1d10ef0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/tracking.py", line 277, in __del__
        self._destroy_resource()
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 889, in __call__
        result = self._call(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 924, in _call
        results = self._stateful_fn(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3022, in __call__
        filtered_flat_args) = self._maybe_define_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3444, in _maybe_define_function
        graph_function = self._create_graph_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3289, in _create_graph_function
        capture_by_value=self._capture_by_value),
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py", line 999, in func_graph_from_py_func
        func_outputs = python_func(*func_args, **func_kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 672, in wrapped_fn
        out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    AttributeError: 'NoneType' object has no attribute '__wrapped__'
    Exception ignored in: <function CapturableResource.__del__ at 0x7f0af1d10ef0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/tracking.py", line 277, in __del__
        self._destroy_resource()
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 889, in __call__
        result = self._call(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 924, in _call
        results = self._stateful_fn(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3022, in __call__
        filtered_flat_args) = self._maybe_define_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3444, in _maybe_define_function
        graph_function = self._create_graph_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3289, in _create_graph_function
        capture_by_value=self._capture_by_value),
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py", line 999, in func_graph_from_py_func
        func_outputs = python_func(*func_args, **func_kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 672, in wrapped_fn
        out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    AttributeError: 'NoneType' object has no attribute '__wrapped__'
    Exception ignored in: <function CapturableResource.__del__ at 0x7f0af1d10ef0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/tracking.py", line 277, in __del__
        self._destroy_resource()
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 889, in __call__
        result = self._call(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 924, in _call
        results = self._stateful_fn(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3022, in __call__
        filtered_flat_args) = self._maybe_define_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3444, in _maybe_define_function
        graph_function = self._create_graph_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3289, in _create_graph_function
        capture_by_value=self._capture_by_value),
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py", line 999, in func_graph_from_py_func
        func_outputs = python_func(*func_args, **func_kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 672, in wrapped_fn
        out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    AttributeError: 'NoneType' object has no attribute '__wrapped__'
    Exception ignored in: <function CapturableResource.__del__ at 0x7f0af1d10ef0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/tracking.py", line 277, in __del__
        self._destroy_resource()
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 889, in __call__
        result = self._call(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 924, in _call
        results = self._stateful_fn(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3022, in __call__
        filtered_flat_args) = self._maybe_define_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3444, in _maybe_define_function
        graph_function = self._create_graph_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3289, in _create_graph_function
        capture_by_value=self._capture_by_value),
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py", line 999, in func_graph_from_py_func
        func_outputs = python_func(*func_args, **func_kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 672, in wrapped_fn
        out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    AttributeError: 'NoneType' object has no attribute '__wrapped__'
    Exception ignored in: <function CapturableResource.__del__ at 0x7f0af1d10ef0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/tracking.py", line 277, in __del__
        self._destroy_resource()
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 889, in __call__
        result = self._call(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 924, in _call
        results = self._stateful_fn(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3022, in __call__
        filtered_flat_args) = self._maybe_define_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3444, in _maybe_define_function
        graph_function = self._create_graph_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3289, in _create_graph_function
        capture_by_value=self._capture_by_value),
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py", line 999, in func_graph_from_py_func
        func_outputs = python_func(*func_args, **func_kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 672, in wrapped_fn
        out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    AttributeError: 'NoneType' object has no attribute '__wrapped__'
    Exception ignored in: <function CapturableResource.__del__ at 0x7f0af1d10ef0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/tracking.py", line 277, in __del__
        self._destroy_resource()
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 889, in __call__
        result = self._call(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 924, in _call
        results = self._stateful_fn(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3022, in __call__
        filtered_flat_args) = self._maybe_define_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3444, in _maybe_define_function
        graph_function = self._create_graph_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3289, in _create_graph_function
        capture_by_value=self._capture_by_value),
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py", line 999, in func_graph_from_py_func
        func_outputs = python_func(*func_args, **func_kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 672, in wrapped_fn
        out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    AttributeError: 'NoneType' object has no attribute '__wrapped__'
    Exception ignored in: <function CapturableResource.__del__ at 0x7f0af1d10ef0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/tracking.py", line 277, in __del__
        self._destroy_resource()
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 889, in __call__
        result = self._call(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 924, in _call
        results = self._stateful_fn(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3022, in __call__
        filtered_flat_args) = self._maybe_define_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3444, in _maybe_define_function
        graph_function = self._create_graph_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3289, in _create_graph_function
        capture_by_value=self._capture_by_value),
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py", line 999, in func_graph_from_py_func
        func_outputs = python_func(*func_args, **func_kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 672, in wrapped_fn
        out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    AttributeError: 'NoneType' object has no attribute '__wrapped__'
    Exception ignored in: <function CapturableResource.__del__ at 0x7f0af1d10ef0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/tracking.py", line 277, in __del__
        self._destroy_resource()
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 889, in __call__
        result = self._call(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 924, in _call
        results = self._stateful_fn(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3022, in __call__
        filtered_flat_args) = self._maybe_define_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3444, in _maybe_define_function
        graph_function = self._create_graph_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3289, in _create_graph_function
        capture_by_value=self._capture_by_value),
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py", line 999, in func_graph_from_py_func
        func_outputs = python_func(*func_args, **func_kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 672, in wrapped_fn
        out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    AttributeError: 'NoneType' object has no attribute '__wrapped__'
    Exception ignored in: <function CapturableResource.__del__ at 0x7f0af1d10ef0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/tracking.py", line 277, in __del__
        self._destroy_resource()
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 889, in __call__
        result = self._call(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 924, in _call
        results = self._stateful_fn(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3022, in __call__
        filtered_flat_args) = self._maybe_define_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3444, in _maybe_define_function
        graph_function = self._create_graph_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3289, in _create_graph_function
        capture_by_value=self._capture_by_value),
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py", line 999, in func_graph_from_py_func
        func_outputs = python_func(*func_args, **func_kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 672, in wrapped_fn
        out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    AttributeError: 'NoneType' object has no attribute '__wrapped__'
    Exception ignored in: <function CapturableResource.__del__ at 0x7f0af1d10ef0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/tracking.py", line 277, in __del__
        self._destroy_resource()
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 889, in __call__
        result = self._call(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 924, in _call
        results = self._stateful_fn(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3022, in __call__
        filtered_flat_args) = self._maybe_define_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3444, in _maybe_define_function
        graph_function = self._create_graph_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3289, in _create_graph_function
        capture_by_value=self._capture_by_value),
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py", line 999, in func_graph_from_py_func
        func_outputs = python_func(*func_args, **func_kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 672, in wrapped_fn
        out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    AttributeError: 'NoneType' object has no attribute '__wrapped__'
    Exception ignored in: <function CapturableResource.__del__ at 0x7f0af1d10ef0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/tracking.py", line 277, in __del__
        self._destroy_resource()
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 889, in __call__
        result = self._call(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 924, in _call
        results = self._stateful_fn(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3022, in __call__
        filtered_flat_args) = self._maybe_define_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3444, in _maybe_define_function
        graph_function = self._create_graph_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3289, in _create_graph_function
        capture_by_value=self._capture_by_value),
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py", line 999, in func_graph_from_py_func
        func_outputs = python_func(*func_args, **func_kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 672, in wrapped_fn
        out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    AttributeError: 'NoneType' object has no attribute '__wrapped__'
    Exception ignored in: <function CapturableResource.__del__ at 0x7f0af1d10ef0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/tracking.py", line 277, in __del__
        self._destroy_resource()
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 889, in __call__
        result = self._call(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 924, in _call
        results = self._stateful_fn(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3022, in __call__
        filtered_flat_args) = self._maybe_define_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3444, in _maybe_define_function
        graph_function = self._create_graph_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3289, in _create_graph_function
        capture_by_value=self._capture_by_value),
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py", line 999, in func_graph_from_py_func
        func_outputs = python_func(*func_args, **func_kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 672, in wrapped_fn
        out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    AttributeError: 'NoneType' object has no attribute '__wrapped__'
    Exception ignored in: <function CapturableResource.__del__ at 0x7f0af1d10ef0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/tracking.py", line 277, in __del__
        self._destroy_resource()
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 889, in __call__
        result = self._call(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 924, in _call
        results = self._stateful_fn(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3022, in __call__
        filtered_flat_args) = self._maybe_define_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3444, in _maybe_define_function
        graph_function = self._create_graph_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3289, in _create_graph_function
        capture_by_value=self._capture_by_value),
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py", line 999, in func_graph_from_py_func
        func_outputs = python_func(*func_args, **func_kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 672, in wrapped_fn
        out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    AttributeError: 'NoneType' object has no attribute '__wrapped__'
    Exception ignored in: <function CapturableResource.__del__ at 0x7f0af1d10ef0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/tracking.py", line 277, in __del__
        self._destroy_resource()
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 889, in __call__
        result = self._call(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 924, in _call
        results = self._stateful_fn(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3022, in __call__
        filtered_flat_args) = self._maybe_define_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3444, in _maybe_define_function
        graph_function = self._create_graph_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3289, in _create_graph_function
        capture_by_value=self._capture_by_value),
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py", line 999, in func_graph_from_py_func
        func_outputs = python_func(*func_args, **func_kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 672, in wrapped_fn
        out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    AttributeError: 'NoneType' object has no attribute '__wrapped__'
    Exception ignored in: <function CapturableResource.__del__ at 0x7f0af1d10ef0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/tracking.py", line 277, in __del__
        self._destroy_resource()
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 889, in __call__
        result = self._call(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 924, in _call
        results = self._stateful_fn(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3022, in __call__
        filtered_flat_args) = self._maybe_define_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3444, in _maybe_define_function
        graph_function = self._create_graph_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3289, in _create_graph_function
        capture_by_value=self._capture_by_value),
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py", line 999, in func_graph_from_py_func
        func_outputs = python_func(*func_args, **func_kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 672, in wrapped_fn
        out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    AttributeError: 'NoneType' object has no attribute '__wrapped__'
    Exception ignored in: <function CapturableResource.__del__ at 0x7f0af1d10ef0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/tracking.py", line 277, in __del__
        self._destroy_resource()
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 889, in __call__
        result = self._call(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 924, in _call
        results = self._stateful_fn(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3022, in __call__
        filtered_flat_args) = self._maybe_define_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3444, in _maybe_define_function
        graph_function = self._create_graph_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3289, in _create_graph_function
        capture_by_value=self._capture_by_value),
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py", line 999, in func_graph_from_py_func
        func_outputs = python_func(*func_args, **func_kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 672, in wrapped_fn
        out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    AttributeError: 'NoneType' object has no attribute '__wrapped__'
    WARNING:root:Make sure that locally built Python SDK docker image has Python 3.7 interpreter.





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
<div class="tfx-object expanded"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExecutionResult</span><span class="deemphasize"> at 0x7f0ac85d5e50</span></div><table class="attr-table"><tr><td class="attr-name">.execution_id</td><td class = "attrvalue">20</td></tr><tr><td class="attr-name">.component</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Evaluator</span><span class="deemphasize"> at 0x7f0ac02117d0</span></div><table class="attr-table"><tr><td class="attr-name">.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7f0acd962fd0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/CsvExampleGen/examples/13)<span class="deemphasize"> at 0x7f0acc5d8c10</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/CsvExampleGen/examples/13</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['model']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Model'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac7731a90</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Model</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Model'</span> (uri: ./pipeline/Trainer/model/18)<span class="deemphasize"> at 0x7f0ac7762290</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Model&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Trainer/model/18</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['baseline_model']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Model'</span> (0 artifacts)<span class="deemphasize"> at 0x7f0ac40b2a50</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Model</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue">[]</td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['evaluation']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ModelEvaluation'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac0211d90</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ModelEvaluation</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ModelEvaluation'</span> (uri: ./pipeline/Evaluator/evaluation/20)<span class="deemphasize"> at 0x7f0ac2a3be90</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ModelEvaluation&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Evaluator/evaluation/20</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['blessing']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ModelBlessing'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac0211f50</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ModelBlessing</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ModelBlessing'</span> (uri: ./pipeline/Evaluator/blessing/20)<span class="deemphasize"> at 0x7f0ac2a3bb90</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ModelBlessing&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Evaluator/blessing/20</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.exec_properties</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['eval_config']</td><td class = "attrvalue">{
  &quot;metrics_specs&quot;: [
    {
      &quot;metrics&quot;: [
        {
          &quot;class_name&quot;: &quot;ExampleCount&quot;
        },
        {
          &quot;class_name&quot;: &quot;BinaryAccuracy&quot;,
          &quot;threshold&quot;: {
            &quot;change_threshold&quot;: {
              &quot;absolute&quot;: -1e-10,
              &quot;direction&quot;: &quot;HIGHER_IS_BETTER&quot;
            },
            &quot;value_threshold&quot;: {
              &quot;lower_bound&quot;: 0.5
            }
          }
        },
        {
          &quot;class_name&quot;: &quot;BinaryCrossentropy&quot;
        },
        {
          &quot;class_name&quot;: &quot;AUC&quot;
        },
        {
          &quot;class_name&quot;: &quot;AUCPrecisionRecall&quot;
        },
        {
          &quot;class_name&quot;: &quot;Precision&quot;
        },
        {
          &quot;class_name&quot;: &quot;Recall&quot;
        },
        {
          &quot;class_name&quot;: &quot;MeanLabel&quot;
        },
        {
          &quot;class_name&quot;: &quot;MeanPrediction&quot;
        },
        {
          &quot;class_name&quot;: &quot;Calibration&quot;
        },
        {
          &quot;class_name&quot;: &quot;CalibrationPlot&quot;
        },
        {
          &quot;class_name&quot;: &quot;ConfusionMatrixPlot&quot;
        }
      ]
    }
  ],
  &quot;model_specs&quot;: [
    {
      &quot;label_key&quot;: &quot;label&quot;
    }
  ],
  &quot;slicing_specs&quot;: [
    {},
    {
      &quot;feature_keys&quot;: [
        &quot;race&quot;
      ]
    },
    {
      &quot;feature_keys&quot;: [
        &quot;sex&quot;
      ]
    }
  ]
}</td></tr><tr><td class="attr-name">['feature_slicing_spec']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['fairness_indicator_thresholds']</td><td class = "attrvalue">null</td></tr><tr><td class="attr-name">['example_splits']</td><td class = "attrvalue">null</td></tr><tr><td class="attr-name">['module_file']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['module_path']</td><td class = "attrvalue">None</td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">.component.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7f0acd962fd0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/CsvExampleGen/examples/13)<span class="deemphasize"> at 0x7f0acc5d8c10</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/CsvExampleGen/examples/13</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['model']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Model'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac7731a90</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Model</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Model'</span> (uri: ./pipeline/Trainer/model/18)<span class="deemphasize"> at 0x7f0ac7762290</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Model&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Trainer/model/18</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['baseline_model']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Model'</span> (0 artifacts)<span class="deemphasize"> at 0x7f0ac40b2a50</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Model</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue">[]</td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.component.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['evaluation']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ModelEvaluation'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac0211d90</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ModelEvaluation</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ModelEvaluation'</span> (uri: ./pipeline/Evaluator/evaluation/20)<span class="deemphasize"> at 0x7f0ac2a3be90</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ModelEvaluation&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Evaluator/evaluation/20</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['blessing']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ModelBlessing'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac0211f50</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ModelBlessing</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ModelBlessing'</span> (uri: ./pipeline/Evaluator/blessing/20)<span class="deemphasize"> at 0x7f0ac2a3bb90</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ModelBlessing&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Evaluator/blessing/20</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr></table></div>



Now let's examine the output artifacts of `Evaluator`. 


```python
# Print component output keys
evaluator.outputs.keys()
```




    dict_keys(['evaluation', 'blessing'])



The `blessing` output simply states if the candidate model was blessed. The artifact URI will have a `BLESSED` or `NOT_BLESSED` file depending on the result. As mentioned earlier, this first run will pass the evaluation because there is no baseline model yet.


```python
# Get `Evaluator` blessing output URI
blessing_uri = evaluator.outputs['blessing'].get()[0].uri

# List files under URI
os.listdir(blessing_uri)
```




    ['BLESSED']



The `evaluation` output, on the other hand, contains the evaluation logs and can be used to visualize the global metrics on the entire evaluation set.


```python
# Visualize the evaluation results
context.show(evaluator.outputs['evaluation'])
```


<b>Artifact at ./pipeline/Evaluator/evaluation/20</b><br/><br/>





<script src="/nbextensions/tensorflow_model_analysis/vulcanized_tfma.js"></script>

  <tfma-nb-slicing-metrics id="component"></tfma-nb-slicing-metrics>
  <script>
  const element = document.getElementById('component');

  /** No event handlers needed. */

  const json = JSON.parse(atob('eyJjb25maWciOiB7IndlaWdodGVkRXhhbXBsZXNDb2x1bW4iOiAiZXhhbXBsZV9jb3VudCJ9LCAiZGF0YSI6IFt7InNsaWNlIjogIk92ZXJhbGwiLCAibWV0cmljcyI6IHsiIjogeyIiOiB7ImJpbmFyeV9hY2N1cmFjeSI6IHsiZG91YmxlVmFsdWUiOiAwLjgwMTAxODA1OTI1MzY5MjZ9LCAibG9zcyI6IHsiZG91YmxlVmFsdWUiOiAwLjQxNTY0NjIyNTIxNDAwNDV9LCAiYmluYXJ5X2Nyb3NzZW50cm9weSI6IHsiZG91YmxlVmFsdWUiOiAwLjQxNTY0NjIyNTIxNDAwNDV9LCAiZXhhbXBsZV9jb3VudCI6IHsiZG91YmxlVmFsdWUiOiAxMDgwNS4wfSwgImF1YyI6IHsiZG91YmxlVmFsdWUiOiAwLjg1NzY1NjU5ODA5MTEyNTV9LCAiYXVjX3ByZWNpc2lvbl9yZWNhbGwiOiB7ImRvdWJsZVZhbHVlIjogMC42NTY5NjA0ODczNjU3MjI3fSwgInByZWNpc2lvbiI6IHsiZG91YmxlVmFsdWUiOiAwLjgwMzU3MTQwMzAyNjU4MDh9LCAicmVjYWxsIjogeyJkb3VibGVWYWx1ZSI6IDAuMjI1Njk0NDQ3NzU1ODEzNn0sICJtZWFuX2xhYmVsIjogeyJkb3VibGVWYWx1ZSI6IDAuMjM5ODg4OTQwMzA1NDE0MTZ9LCAibWVhbl9wcmVkaWN0aW9uIjogeyJkb3VibGVWYWx1ZSI6IDAuMjQxNzg1Nzk5Mzg0MDYxOTd9LCAiY2FsaWJyYXRpb24iOiB7ImRvdWJsZVZhbHVlIjogMS4wMDc5MDcyMzg1NTg5NDY1fX19fX1dfQ=='));
  element.config = json.config;
  element.data = json.data;
  </script>



To see the individual slices, you will need to import TFMA and use the commands you learned in the previous lab.


```python
import tensorflow_model_analysis as tfma

# Get the TFMA output result path and load the result.
PATH_TO_RESULT = evaluator.outputs['evaluation'].get()[0].uri
tfma_result = tfma.load_eval_result(PATH_TO_RESULT)

# Show data sliced along feature column trip_start_hour.
tfma.view.render_slicing_metrics(
    tfma_result, slicing_column='sex')
```




<script src="/nbextensions/tensorflow_model_analysis/vulcanized_tfma.js"></script>

  <tfma-nb-slicing-metrics id="component"></tfma-nb-slicing-metrics>
  <script>
  const element = document.getElementById('component');

  /** No event handlers needed. */

  const json = JSON.parse(atob('eyJjb25maWciOiB7IndlaWdodGVkRXhhbXBsZXNDb2x1bW4iOiAiZXhhbXBsZV9jb3VudCJ9LCAiZGF0YSI6IFt7InNsaWNlIjogInNleDpNYWxlIiwgIm1ldHJpY3MiOiB7IiI6IHsiIjogeyJiaW5hcnlfYWNjdXJhY3kiOiB7ImRvdWJsZVZhbHVlIjogMC43NTY0OTg4NzMyMzM3OTUyfSwgImxvc3MiOiB7ImRvdWJsZVZhbHVlIjogMC40ODQ5MjQ0OTUyMjAxODQzfSwgImJpbmFyeV9jcm9zc2VudHJvcHkiOiB7ImRvdWJsZVZhbHVlIjogMC40ODQ5MjQ0OTUyMjAxODQzfSwgImV4YW1wbGVfY291bnQiOiB7ImRvdWJsZVZhbHVlIjogNzIzMi4wfSwgImF1YyI6IHsiZG91YmxlVmFsdWUiOiAwLjgzMDg1NDU5NDcwNzQ4OX0sICJhdWNfcHJlY2lzaW9uX3JlY2FsbCI6IHsiZG91YmxlVmFsdWUiOiAwLjY4NTcyNDY3NTY1NTM2NX0sICJwcmVjaXNpb24iOiB7ImRvdWJsZVZhbHVlIjogMC44MDE5MzkwNzAyMjQ3NjJ9LCAicmVjYWxsIjogeyJkb3VibGVWYWx1ZSI6IDAuMjYzNTQxMTkxODE2MzI5OTZ9LCAibWVhbl9sYWJlbCI6IHsiZG91YmxlVmFsdWUiOiAwLjMwMzc4ODcxNjgxNDE1OTN9LCAibWVhbl9wcmVkaWN0aW9uIjogeyJkb3VibGVWYWx1ZSI6IDAuMjk2Mzk3ODY0MDIzMjR9LCAiY2FsaWJyYXRpb24iOiB7ImRvdWJsZVZhbHVlIjogMC45NzU2NzEwNzUzODI4Mjc0fX19fX0sIHsic2xpY2UiOiAic2V4OkZlbWFsZSIsICJtZXRyaWNzIjogeyIiOiB7IiI6IHsiYmluYXJ5X2FjY3VyYWN5IjogeyJkb3VibGVWYWx1ZSI6IDAuODkxMTI3ODg0Mzg3OTd9LCAibG9zcyI6IHsiZG91YmxlVmFsdWUiOiAwLjI3NTQyMjAzNjY0Nzc5NjYzfSwgImJpbmFyeV9jcm9zc2VudHJvcHkiOiB7ImRvdWJsZVZhbHVlIjogMC4yNzU0MjIwMzY2NDc3OTY2M30sICJleGFtcGxlX2NvdW50IjogeyJkb3VibGVWYWx1ZSI6IDM1NzMuMH0sICJhdWMiOiB7ImRvdWJsZVZhbHVlIjogMC44OTU4ODgyNjg5NDc2MDEzfSwgImF1Y19wcmVjaXNpb25fcmVjYWxsIjogeyJkb3VibGVWYWx1ZSI6IDAuNTkzMzg0OTIxNTUwNzUwN30sICJwcmVjaXNpb24iOiB7ImRvdWJsZVZhbHVlIjogMS4wfSwgInJlY2FsbCI6IHsiZG91YmxlVmFsdWUiOiAwLjAxNTE4OTg3MzA1NDYyMzYwNH0sICJtZWFuX2xhYmVsIjogeyJkb3VibGVWYWx1ZSI6IDAuMTEwNTUxMzU3NDAyNzQyNzl9LCAibWVhbl9wcmVkaWN0aW9uIjogeyJkb3VibGVWYWx1ZSI6IDAuMTMxMjQ3MTg5OTYwNDU4NH0sICJjYWxpYnJhdGlvbiI6IHsiZG91YmxlVmFsdWUiOiAxLjE4NzIwNTU5NDI0OTkxODR9fX19fV19'));
  element.config = json.config;
  element.data = json.data;
  </script>



You can also use TFMA to load the validation results as before by specifying the output URI of the evaluation output. This would be more useful if your model was not blessed because you can see the metric failure prompts. Try to simulate this later by training with fewer epochs (or raising the threshold) and see the results you get here.


```python
# Get `evaluation` output URI
PATH_TO_RESULT = evaluator.outputs['evaluation'].get()[0].uri

# Print validation result
print(tfma.load_validation_result(PATH_TO_RESULT))
```

    validation_ok: true
    validation_details {
      slicing_details {
        slicing_spec {
        }
        num_matching_slices: 8
      }
    }
    


Now that your `Evaluator` has finished running, the `Resolver` component should be able to detect the latest blessed model. Let's run the component again.


```python
# Re-run the Resolver component
context.run(model_resolver)
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
<div class="tfx-object expanded"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExecutionResult</span><span class="deemphasize"> at 0x7f0abfc93850</span></div><table class="attr-table"><tr><td class="attr-name">.execution_id</td><td class = "attrvalue">21</td></tr><tr><td class="attr-name">.component</td><td class = "attrvalue">&lt;tfx.dsl.components.common.resolver.Resolver object at 0x7f0ac811b550&gt;</td></tr><tr><td class="attr-name">.component.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['model']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Model'</span> (0 artifacts)<span class="deemphasize"> at 0x7f0ac814fad0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Model</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue">[]</td></tr></table></div></td></tr><tr><td class="attr-name">['model_blessing']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ModelBlessing'</span> (0 artifacts)<span class="deemphasize"> at 0x7f0ac817fc90</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ModelBlessing</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue">[]</td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.component.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['model']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Model'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ab9bb1f50</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Model</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Model'</span> (uri: ./pipeline/Trainer/model/18)<span class="deemphasize"> at 0x7f0ab9bb1bd0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Model&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Trainer/model/18</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['model_blessing']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ModelBlessing'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ab9bb14d0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ModelBlessing</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ModelBlessing'</span> (uri: ./pipeline/Evaluator/blessing/20)<span class="deemphasize"> at 0x7f0abfc72710</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ModelBlessing&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Evaluator/blessing/20</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr></table></div>



You should now see an artifact in the component outputs. You can also get it programmatically as shown below.


```python
# Get path to latest blessed model
model_resolver.outputs['model'].get()[0].uri
```




    './pipeline/Trainer/model/18'



#### Comparing two models

Now let's see how `Evaluator` compares two models. You will train the same model with more epochs and this should hopefully result in higher accuracy and overall metrics.


```python
from tfx.proto import trainer_pb2

# Setup trainer to train with more epochs
trainer = Trainer(
    module_file=os.path.abspath(_census_trainer_module_file),
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(num_steps=500),
    eval_args=trainer_pb2.EvalArgs(num_steps=200))

# Run trainer
context.run(trainer, enable_cache=False)
```

    WARNING:absl:Examples artifact does not have payload_format custom property. Falling back to FORMAT_TF_EXAMPLE
    WARNING:absl:Examples artifact does not have payload_format custom property. Falling back to FORMAT_TF_EXAMPLE
    WARNING:absl:Examples artifact does not have payload_format custom property. Falling back to FORMAT_TF_EXAMPLE
    /usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/optimizer_v2/optimizer_v2.py:375: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
      "The `lr` argument is deprecated, use `learning_rate` instead.")


    Model: "model_1"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    fnlwgt_xf (InputLayer)          [(None, 1)]          0                                            
    __________________________________________________________________________________________________
    education-num_xf (InputLayer)   [(None, 1)]          0                                            
    __________________________________________________________________________________________________
    capital-gain_xf (InputLayer)    [(None, 1)]          0                                            
    __________________________________________________________________________________________________
    capital-loss_xf (InputLayer)    [(None, 1)]          0                                            
    __________________________________________________________________________________________________
    hours-per-week_xf (InputLayer)  [(None, 1)]          0                                            
    __________________________________________________________________________________________________
    concatenate_3 (Concatenate)     (None, 5)            0           fnlwgt_xf[0][0]                  
                                                                     education-num_xf[0][0]           
                                                                     capital-gain_xf[0][0]            
                                                                     capital-loss_xf[0][0]            
                                                                     hours-per-week_xf[0][0]          
    __________________________________________________________________________________________________
    dense_6 (Dense)                 (None, 100)          600         concatenate_3[0][0]              
    __________________________________________________________________________________________________
    dense_7 (Dense)                 (None, 70)           7070        dense_6[0][0]                    
    __________________________________________________________________________________________________
    education_xf (InputLayer)       [(None, 21)]         0                                            
    __________________________________________________________________________________________________
    marital-status_xf (InputLayer)  [(None, 12)]         0                                            
    __________________________________________________________________________________________________
    occupation_xf (InputLayer)      [(None, 20)]         0                                            
    __________________________________________________________________________________________________
    race_xf (InputLayer)            [(None, 10)]         0                                            
    __________________________________________________________________________________________________
    relationship_xf (InputLayer)    [(None, 11)]         0                                            
    __________________________________________________________________________________________________
    workclass_xf (InputLayer)       [(None, 14)]         0                                            
    __________________________________________________________________________________________________
    sex_xf (InputLayer)             [(None, 7)]          0                                            
    __________________________________________________________________________________________________
    native-country_xf (InputLayer)  [(None, 47)]         0                                            
    __________________________________________________________________________________________________
    age_xf (InputLayer)             [(None, 4)]          0                                            
    __________________________________________________________________________________________________
    dense_8 (Dense)                 (None, 48)           3408        dense_7[0][0]                    
    __________________________________________________________________________________________________
    concatenate_4 (Concatenate)     (None, 146)          0           education_xf[0][0]               
                                                                     marital-status_xf[0][0]          
                                                                     occupation_xf[0][0]              
                                                                     race_xf[0][0]                    
                                                                     relationship_xf[0][0]            
                                                                     workclass_xf[0][0]               
                                                                     sex_xf[0][0]                     
                                                                     native-country_xf[0][0]          
                                                                     age_xf[0][0]                     
    __________________________________________________________________________________________________
    dense_9 (Dense)                 (None, 34)           1666        dense_8[0][0]                    
    __________________________________________________________________________________________________
    dense_10 (Dense)                (None, 128)          18816       concatenate_4[0][0]              
    __________________________________________________________________________________________________
    concatenate_5 (Concatenate)     (None, 162)          0           dense_9[0][0]                    
                                                                     dense_10[0][0]                   
    __________________________________________________________________________________________________
    dense_11 (Dense)                (None, 1)            163         concatenate_5[0][0]              
    ==================================================================================================
    Total params: 31,723
    Trainable params: 31,723
    Non-trainable params: 0
    __________________________________________________________________________________________________
    500/500 [==============================] - 4s 6ms/step - loss: 0.3535 - binary_accuracy: 0.8353 - val_loss: 0.3160 - val_binary_accuracy: 0.8486





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
<div class="tfx-object expanded"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExecutionResult</span><span class="deemphasize"> at 0x7f0abaacfd90</span></div><table class="attr-table"><tr><td class="attr-name">.execution_id</td><td class = "attrvalue">22</td></tr><tr><td class="attr-name">.component</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Trainer</span><span class="deemphasize"> at 0x7f0ac860c910</span></div><table class="attr-table"><tr><td class="attr-name">.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac8615510</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/Transform/transformed_examples/17)<span class="deemphasize"> at 0x7f0ac8615a50</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/transformed_examples/17</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['transform_graph']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'TransformGraph'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac86151d0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">TransformGraph</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'TransformGraph'</span> (uri: ./pipeline/Transform/transform_graph/17)<span class="deemphasize"> at 0x7f0ac85c6090</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.TransformGraph&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/transform_graph/17</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['schema']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac85a2f50</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline/SchemaGen/schema/15)<span class="deemphasize"> at 0x7f0ac85bc450</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/SchemaGen/schema/15</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['model']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Model'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac860ced0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Model</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Model'</span> (uri: ./pipeline/Trainer/model/22)<span class="deemphasize"> at 0x7f0ab9bb1910</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Model&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Trainer/model/22</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['model_run']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ModelRun'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac851d550</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ModelRun</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ModelRun'</span> (uri: ./pipeline/Trainer/model_run/22)<span class="deemphasize"> at 0x7f0abfd2b8d0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ModelRun&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Trainer/model_run/22</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.exec_properties</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['train_args']</td><td class = "attrvalue">{
  &quot;num_steps&quot;: 500
}</td></tr><tr><td class="attr-name">['eval_args']</td><td class = "attrvalue">{
  &quot;num_steps&quot;: 200
}</td></tr><tr><td class="attr-name">['module_file']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['run_fn']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['trainer_fn']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['custom_config']</td><td class = "attrvalue">null</td></tr><tr><td class="attr-name">['module_path']</td><td class = "attrvalue">census_trainer@./pipeline/_wheels/tfx_user_code_Trainer-0.0+cc29a9c825e35b0142dabcb732b412ff69124c1b5f5c6eee2546a102dcbf15c9-py3-none-any.whl</td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">.component.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac8615510</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/Transform/transformed_examples/17)<span class="deemphasize"> at 0x7f0ac8615a50</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/transformed_examples/17</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['transform_graph']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'TransformGraph'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac86151d0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">TransformGraph</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'TransformGraph'</span> (uri: ./pipeline/Transform/transform_graph/17)<span class="deemphasize"> at 0x7f0ac85c6090</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.TransformGraph&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/transform_graph/17</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['schema']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac85a2f50</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline/SchemaGen/schema/15)<span class="deemphasize"> at 0x7f0ac85bc450</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/SchemaGen/schema/15</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.component.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['model']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Model'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac860ced0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Model</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Model'</span> (uri: ./pipeline/Trainer/model/22)<span class="deemphasize"> at 0x7f0ab9bb1910</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Model&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Trainer/model/22</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['model_run']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ModelRun'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac851d550</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ModelRun</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ModelRun'</span> (uri: ./pipeline/Trainer/model_run/22)<span class="deemphasize"> at 0x7f0abfd2b8d0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ModelRun&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Trainer/model_run/22</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr></table></div>



You will re-run the evaluator but you will specify the latest trainer output as the candidate model. The baseline is automatically found with the Resolver node.


```python
# Re-run the evaluator
evaluator = Evaluator(
    examples=example_gen.outputs['examples'],
    model=trainer.outputs['model'],
    baseline_model=model_resolver.outputs['model'],
    eval_config=eval_config)

context.run(evaluator, enable_cache=False)
```

    Exception ignored in: <function CapturableResource.__del__ at 0x7f0af1d10ef0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/tracking.py", line 277, in __del__
        self._destroy_resource()
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 889, in __call__
        result = self._call(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 924, in _call
        results = self._stateful_fn(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3022, in __call__
        filtered_flat_args) = self._maybe_define_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3444, in _maybe_define_function
        graph_function = self._create_graph_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3289, in _create_graph_function
        capture_by_value=self._capture_by_value),
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py", line 999, in func_graph_from_py_func
        func_outputs = python_func(*func_args, **func_kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 672, in wrapped_fn
        out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    AttributeError: 'NoneType' object has no attribute '__wrapped__'
    Exception ignored in: <function CapturableResource.__del__ at 0x7f0af1d10ef0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/tracking.py", line 277, in __del__
        self._destroy_resource()
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 889, in __call__
        result = self._call(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 924, in _call
        results = self._stateful_fn(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3022, in __call__
        filtered_flat_args) = self._maybe_define_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3444, in _maybe_define_function
        graph_function = self._create_graph_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3289, in _create_graph_function
        capture_by_value=self._capture_by_value),
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py", line 999, in func_graph_from_py_func
        func_outputs = python_func(*func_args, **func_kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 672, in wrapped_fn
        out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    AttributeError: 'NoneType' object has no attribute '__wrapped__'
    Exception ignored in: <function CapturableResource.__del__ at 0x7f0af1d10ef0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/tracking.py", line 277, in __del__
        self._destroy_resource()
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 889, in __call__
        result = self._call(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 924, in _call
        results = self._stateful_fn(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3022, in __call__
        filtered_flat_args) = self._maybe_define_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3444, in _maybe_define_function
        graph_function = self._create_graph_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3289, in _create_graph_function
        capture_by_value=self._capture_by_value),
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py", line 999, in func_graph_from_py_func
        func_outputs = python_func(*func_args, **func_kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 672, in wrapped_fn
        out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    AttributeError: 'NoneType' object has no attribute '__wrapped__'
    Exception ignored in: <function CapturableResource.__del__ at 0x7f0af1d10ef0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/tracking.py", line 277, in __del__
        self._destroy_resource()
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 889, in __call__
        result = self._call(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 924, in _call
        results = self._stateful_fn(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3022, in __call__
        filtered_flat_args) = self._maybe_define_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3444, in _maybe_define_function
        graph_function = self._create_graph_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3289, in _create_graph_function
        capture_by_value=self._capture_by_value),
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py", line 999, in func_graph_from_py_func
        func_outputs = python_func(*func_args, **func_kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 672, in wrapped_fn
        out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    AttributeError: 'NoneType' object has no attribute '__wrapped__'
    Exception ignored in: <function CapturableResource.__del__ at 0x7f0af1d10ef0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/tracking.py", line 277, in __del__
        self._destroy_resource()
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 889, in __call__
        result = self._call(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 924, in _call
        results = self._stateful_fn(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3022, in __call__
        filtered_flat_args) = self._maybe_define_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3444, in _maybe_define_function
        graph_function = self._create_graph_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3289, in _create_graph_function
        capture_by_value=self._capture_by_value),
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py", line 999, in func_graph_from_py_func
        func_outputs = python_func(*func_args, **func_kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 672, in wrapped_fn
        out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    AttributeError: 'NoneType' object has no attribute '__wrapped__'
    Exception ignored in: <function CapturableResource.__del__ at 0x7f0af1d10ef0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/tracking.py", line 277, in __del__
        self._destroy_resource()
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 889, in __call__
        result = self._call(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 924, in _call
        results = self._stateful_fn(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3022, in __call__
        filtered_flat_args) = self._maybe_define_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3444, in _maybe_define_function
        graph_function = self._create_graph_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3289, in _create_graph_function
        capture_by_value=self._capture_by_value),
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py", line 999, in func_graph_from_py_func
        func_outputs = python_func(*func_args, **func_kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 672, in wrapped_fn
        out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    AttributeError: 'NoneType' object has no attribute '__wrapped__'
    Exception ignored in: <function CapturableResource.__del__ at 0x7f0af1d10ef0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/tracking.py", line 277, in __del__
        self._destroy_resource()
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 889, in __call__
        result = self._call(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 924, in _call
        results = self._stateful_fn(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3022, in __call__
        filtered_flat_args) = self._maybe_define_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3444, in _maybe_define_function
        graph_function = self._create_graph_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3289, in _create_graph_function
        capture_by_value=self._capture_by_value),
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py", line 999, in func_graph_from_py_func
        func_outputs = python_func(*func_args, **func_kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 672, in wrapped_fn
        out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    AttributeError: 'NoneType' object has no attribute '__wrapped__'
    Exception ignored in: <function CapturableResource.__del__ at 0x7f0af1d10ef0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/tracking.py", line 277, in __del__
        self._destroy_resource()
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 889, in __call__
        result = self._call(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 924, in _call
        results = self._stateful_fn(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3022, in __call__
        filtered_flat_args) = self._maybe_define_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3444, in _maybe_define_function
        graph_function = self._create_graph_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3289, in _create_graph_function
        capture_by_value=self._capture_by_value),
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py", line 999, in func_graph_from_py_func
        func_outputs = python_func(*func_args, **func_kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 672, in wrapped_fn
        out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    AttributeError: 'NoneType' object has no attribute '__wrapped__'
    Exception ignored in: <function CapturableResource.__del__ at 0x7f0af1d10ef0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/tracking.py", line 277, in __del__
        self._destroy_resource()
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 889, in __call__
        result = self._call(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 924, in _call
        results = self._stateful_fn(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3022, in __call__
        filtered_flat_args) = self._maybe_define_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3444, in _maybe_define_function
        graph_function = self._create_graph_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3289, in _create_graph_function
        capture_by_value=self._capture_by_value),
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py", line 999, in func_graph_from_py_func
        func_outputs = python_func(*func_args, **func_kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 672, in wrapped_fn
        out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    AttributeError: 'NoneType' object has no attribute '__wrapped__'
    Exception ignored in: <function CapturableResource.__del__ at 0x7f0af1d10ef0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/tracking.py", line 277, in __del__
        self._destroy_resource()
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 889, in __call__
        result = self._call(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 924, in _call
        results = self._stateful_fn(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3022, in __call__
        filtered_flat_args) = self._maybe_define_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3444, in _maybe_define_function
        graph_function = self._create_graph_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3289, in _create_graph_function
        capture_by_value=self._capture_by_value),
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py", line 999, in func_graph_from_py_func
        func_outputs = python_func(*func_args, **func_kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 672, in wrapped_fn
        out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    AttributeError: 'NoneType' object has no attribute '__wrapped__'
    Exception ignored in: <function CapturableResource.__del__ at 0x7f0af1d10ef0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/tracking.py", line 277, in __del__
        self._destroy_resource()
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 889, in __call__
        result = self._call(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 924, in _call
        results = self._stateful_fn(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3022, in __call__
        filtered_flat_args) = self._maybe_define_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3444, in _maybe_define_function
        graph_function = self._create_graph_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3289, in _create_graph_function
        capture_by_value=self._capture_by_value),
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py", line 999, in func_graph_from_py_func
        func_outputs = python_func(*func_args, **func_kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 672, in wrapped_fn
        out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    AttributeError: 'NoneType' object has no attribute '__wrapped__'
    Exception ignored in: <function CapturableResource.__del__ at 0x7f0af1d10ef0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/tracking.py", line 277, in __del__
        self._destroy_resource()
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 889, in __call__
        result = self._call(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 924, in _call
        results = self._stateful_fn(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3022, in __call__
        filtered_flat_args) = self._maybe_define_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3444, in _maybe_define_function
        graph_function = self._create_graph_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3289, in _create_graph_function
        capture_by_value=self._capture_by_value),
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py", line 999, in func_graph_from_py_func
        func_outputs = python_func(*func_args, **func_kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 672, in wrapped_fn
        out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    AttributeError: 'NoneType' object has no attribute '__wrapped__'
    Exception ignored in: <function CapturableResource.__del__ at 0x7f0af1d10ef0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/tracking.py", line 277, in __del__
        self._destroy_resource()
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 889, in __call__
        result = self._call(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 924, in _call
        results = self._stateful_fn(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3022, in __call__
        filtered_flat_args) = self._maybe_define_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3444, in _maybe_define_function
        graph_function = self._create_graph_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3289, in _create_graph_function
        capture_by_value=self._capture_by_value),
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py", line 999, in func_graph_from_py_func
        func_outputs = python_func(*func_args, **func_kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 672, in wrapped_fn
        out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    AttributeError: 'NoneType' object has no attribute '__wrapped__'
    Exception ignored in: <function CapturableResource.__del__ at 0x7f0af1d10ef0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/tracking.py", line 277, in __del__
        self._destroy_resource()
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 889, in __call__
        result = self._call(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 924, in _call
        results = self._stateful_fn(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3022, in __call__
        filtered_flat_args) = self._maybe_define_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3444, in _maybe_define_function
        graph_function = self._create_graph_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3289, in _create_graph_function
        capture_by_value=self._capture_by_value),
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py", line 999, in func_graph_from_py_func
        func_outputs = python_func(*func_args, **func_kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 672, in wrapped_fn
        out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    AttributeError: 'NoneType' object has no attribute '__wrapped__'
    Exception ignored in: <function CapturableResource.__del__ at 0x7f0af1d10ef0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/tracking.py", line 277, in __del__
        self._destroy_resource()
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 889, in __call__
        result = self._call(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 924, in _call
        results = self._stateful_fn(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3022, in __call__
        filtered_flat_args) = self._maybe_define_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3444, in _maybe_define_function
        graph_function = self._create_graph_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3289, in _create_graph_function
        capture_by_value=self._capture_by_value),
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py", line 999, in func_graph_from_py_func
        func_outputs = python_func(*func_args, **func_kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 672, in wrapped_fn
        out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    AttributeError: 'NoneType' object has no attribute '__wrapped__'
    Exception ignored in: <function CapturableResource.__del__ at 0x7f0af1d10ef0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/tracking.py", line 277, in __del__
        self._destroy_resource()
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 889, in __call__
        result = self._call(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 924, in _call
        results = self._stateful_fn(*args, **kwds)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3022, in __call__
        filtered_flat_args) = self._maybe_define_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3444, in _maybe_define_function
        graph_function = self._create_graph_function(args, kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 3289, in _create_graph_function
        capture_by_value=self._capture_by_value),
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py", line 999, in func_graph_from_py_func
        func_outputs = python_func(*func_args, **func_kwargs)
      File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 672, in wrapped_fn
        out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    AttributeError: 'NoneType' object has no attribute '__wrapped__'
    WARNING:root:Make sure that locally built Python SDK docker image has Python 3.7 interpreter.





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
<div class="tfx-object expanded"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExecutionResult</span><span class="deemphasize"> at 0x7f0ab9b32090</span></div><table class="attr-table"><tr><td class="attr-name">.execution_id</td><td class = "attrvalue">23</td></tr><tr><td class="attr-name">.component</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Evaluator</span><span class="deemphasize"> at 0x7f0ab9b3c3d0</span></div><table class="attr-table"><tr><td class="attr-name">.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7f0acd962fd0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/CsvExampleGen/examples/13)<span class="deemphasize"> at 0x7f0acc5d8c10</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/CsvExampleGen/examples/13</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['model']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Model'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac860ced0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Model</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Model'</span> (uri: ./pipeline/Trainer/model/22)<span class="deemphasize"> at 0x7f0ab9bb1910</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Model&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Trainer/model/22</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['baseline_model']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Model'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ab9bb1f50</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Model</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Model'</span> (uri: ./pipeline/Trainer/model/18)<span class="deemphasize"> at 0x7f0ab9bb1bd0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Model&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Trainer/model/18</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['evaluation']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ModelEvaluation'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ab9b3c410</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ModelEvaluation</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ModelEvaluation'</span> (uri: ./pipeline/Evaluator/evaluation/23)<span class="deemphasize"> at 0x7f0ab9b32a50</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ModelEvaluation&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Evaluator/evaluation/23</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['blessing']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ModelBlessing'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ab9b3c550</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ModelBlessing</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ModelBlessing'</span> (uri: ./pipeline/Evaluator/blessing/23)<span class="deemphasize"> at 0x7f0ab9b32490</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ModelBlessing&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Evaluator/blessing/23</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.exec_properties</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['eval_config']</td><td class = "attrvalue">{
  &quot;metrics_specs&quot;: [
    {
      &quot;metrics&quot;: [
        {
          &quot;class_name&quot;: &quot;ExampleCount&quot;
        },
        {
          &quot;class_name&quot;: &quot;BinaryAccuracy&quot;,
          &quot;threshold&quot;: {
            &quot;change_threshold&quot;: {
              &quot;absolute&quot;: -1e-10,
              &quot;direction&quot;: &quot;HIGHER_IS_BETTER&quot;
            },
            &quot;value_threshold&quot;: {
              &quot;lower_bound&quot;: 0.5
            }
          }
        },
        {
          &quot;class_name&quot;: &quot;BinaryCrossentropy&quot;
        },
        {
          &quot;class_name&quot;: &quot;AUC&quot;
        },
        {
          &quot;class_name&quot;: &quot;AUCPrecisionRecall&quot;
        },
        {
          &quot;class_name&quot;: &quot;Precision&quot;
        },
        {
          &quot;class_name&quot;: &quot;Recall&quot;
        },
        {
          &quot;class_name&quot;: &quot;MeanLabel&quot;
        },
        {
          &quot;class_name&quot;: &quot;MeanPrediction&quot;
        },
        {
          &quot;class_name&quot;: &quot;Calibration&quot;
        },
        {
          &quot;class_name&quot;: &quot;CalibrationPlot&quot;
        },
        {
          &quot;class_name&quot;: &quot;ConfusionMatrixPlot&quot;
        }
      ]
    }
  ],
  &quot;model_specs&quot;: [
    {
      &quot;label_key&quot;: &quot;label&quot;
    }
  ],
  &quot;slicing_specs&quot;: [
    {},
    {
      &quot;feature_keys&quot;: [
        &quot;race&quot;
      ]
    },
    {
      &quot;feature_keys&quot;: [
        &quot;sex&quot;
      ]
    }
  ]
}</td></tr><tr><td class="attr-name">['feature_slicing_spec']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['fairness_indicator_thresholds']</td><td class = "attrvalue">null</td></tr><tr><td class="attr-name">['example_splits']</td><td class = "attrvalue">null</td></tr><tr><td class="attr-name">['module_file']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['module_path']</td><td class = "attrvalue">None</td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">.component.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7f0acd962fd0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/CsvExampleGen/examples/13)<span class="deemphasize"> at 0x7f0acc5d8c10</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/CsvExampleGen/examples/13</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['model']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Model'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac860ced0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Model</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Model'</span> (uri: ./pipeline/Trainer/model/22)<span class="deemphasize"> at 0x7f0ab9bb1910</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Model&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Trainer/model/22</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['baseline_model']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Model'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ab9bb1f50</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Model</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Model'</span> (uri: ./pipeline/Trainer/model/18)<span class="deemphasize"> at 0x7f0ab9bb1bd0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Model&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Trainer/model/18</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.component.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['evaluation']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ModelEvaluation'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ab9b3c410</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ModelEvaluation</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ModelEvaluation'</span> (uri: ./pipeline/Evaluator/evaluation/23)<span class="deemphasize"> at 0x7f0ab9b32a50</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ModelEvaluation&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Evaluator/evaluation/23</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['blessing']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ModelBlessing'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ab9b3c550</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ModelBlessing</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ModelBlessing'</span> (uri: ./pipeline/Evaluator/blessing/23)<span class="deemphasize"> at 0x7f0ab9b32490</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ModelBlessing&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Evaluator/blessing/23</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr></table></div>



Depending on the result, the Resolver should reflect the latest blessed model. Since you trained with more epochs, it is most likely that your candidate model will pass the thresholds you set in the eval config. If so, the artifact URI should be different here compared to your earlier runs.


```python
# Re-run the resolver
context.run(model_resolver, enable_cache=False)
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
<div class="tfx-object expanded"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExecutionResult</span><span class="deemphasize"> at 0x7f0ab9b36910</span></div><table class="attr-table"><tr><td class="attr-name">.execution_id</td><td class = "attrvalue">24</td></tr><tr><td class="attr-name">.component</td><td class = "attrvalue">&lt;tfx.dsl.components.common.resolver.Resolver object at 0x7f0ac811b550&gt;</td></tr><tr><td class="attr-name">.component.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['model']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Model'</span> (0 artifacts)<span class="deemphasize"> at 0x7f0ac814fad0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Model</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue">[]</td></tr></table></div></td></tr><tr><td class="attr-name">['model_blessing']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ModelBlessing'</span> (0 artifacts)<span class="deemphasize"> at 0x7f0ac817fc90</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ModelBlessing</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue">[]</td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.component.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['model']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Model'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac2155a10</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Model</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Model'</span> (uri: ./pipeline/Trainer/model/22)<span class="deemphasize"> at 0x7f0ac20c5390</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Model&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Trainer/model/22</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['model_blessing']</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ModelBlessing'</span> (1 artifact)<span class="deemphasize"> at 0x7f0ac20d3bd0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ModelBlessing</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
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
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ModelBlessing'</span> (uri: ./pipeline/Evaluator/blessing/23)<span class="deemphasize"> at 0x7f0ab9537390</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ModelBlessing&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Evaluator/blessing/23</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr></table></div>




```python
# Get path to latest blessed model
model_resolver.outputs['model'].get()[0].uri
```




    './pipeline/Trainer/model/22'



Finally, the `evaluation` output of the `Evaluator` component will now be able to produce the `diff` results you saw in the previous lab. This will signify if the metrics from the candidate model has indeed improved compared to the baseline. Unlike when using TFMA as a standalone library, visualizing this will just show the results for the candidate (i.e. only one row instead of two in the tabular output in the graph below).

*Note: You can ignore the warning about failing to find plots.*


```python
context.show(evaluator.outputs['evaluation'])
```


<b>Artifact at ./pipeline/Evaluator/evaluation/23</b><br/><br/>


    WARNING:absl:Fail to find plots for model name: None . Available model names are [candidate, baseline]
    WARNING:absl:Fail to find plots for model name: None . Available model names are [candidate, baseline]
    WARNING:absl:Fail to find plots for model name: None . Available model names are [candidate, baseline]
    WARNING:absl:Fail to find plots for model name: None . Available model names are [candidate, baseline]
    WARNING:absl:Fail to find plots for model name: None . Available model names are [candidate, baseline]
    WARNING:absl:Fail to find plots for model name: None . Available model names are [candidate, baseline]
    WARNING:absl:Fail to find plots for model name: None . Available model names are [candidate, baseline]
    WARNING:absl:Fail to find plots for model name: None . Available model names are [candidate, baseline]





<script src="/nbextensions/tensorflow_model_analysis/vulcanized_tfma.js"></script>

  <tfma-nb-slicing-metrics id="component"></tfma-nb-slicing-metrics>
  <script>
  const element = document.getElementById('component');

  /** No event handlers needed. */

  const json = JSON.parse(atob('eyJjb25maWciOiB7IndlaWdodGVkRXhhbXBsZXNDb2x1bW4iOiAiZXhhbXBsZV9jb3VudCJ9LCAiZGF0YSI6IFt7InNsaWNlIjogIk92ZXJhbGwiLCAibWV0cmljcyI6IHsiIjogeyIiOiB7ImJpbmFyeV9hY2N1cmFjeSI6IHsiZG91YmxlVmFsdWUiOiAwLjg0OTMyODk5NDc1MDk3NjZ9LCAibG9zcyI6IHsiZG91YmxlVmFsdWUiOiAwLjMxNTYxMDU4NzU5Njg5MzN9LCAiYmluYXJ5X2Nyb3NzZW50cm9weSI6IHsiZG91YmxlVmFsdWUiOiAwLjMxNTYxMDU4NzU5Njg5MzN9LCAiZXhhbXBsZV9jb3VudCI6IHsiZG91YmxlVmFsdWUiOiAxMDgwNS4wfSwgImF1YyI6IHsiZG91YmxlVmFsdWUiOiAwLjkwOTMwMTg3NzAyMTc4OTZ9LCAiYXVjX3ByZWNpc2lvbl9yZWNhbGwiOiB7ImRvdWJsZVZhbHVlIjogMC43NjQzOTIyNTY3MzY3NTU0fSwgInByZWNpc2lvbiI6IHsiZG91YmxlVmFsdWUiOiAwLjczMzc1MzYyMTU3ODIxNjZ9LCAicmVjYWxsIjogeyJkb3VibGVWYWx1ZSI6IDAuNTgzNzE5MTM0MzMwNzQ5NX0sICJtZWFuX2xhYmVsIjogeyJkb3VibGVWYWx1ZSI6IDAuMjM5ODg4OTQwMzA1NDE0MTZ9LCAibWVhbl9wcmVkaWN0aW9uIjogeyJkb3VibGVWYWx1ZSI6IDAuMjMxNzY0NDk4NTAwNTY4MDJ9LCAiY2FsaWJyYXRpb24iOiB7ImRvdWJsZVZhbHVlIjogMC45NjYxMzI0ODY5OTc5MzExfSwgImJpbmFyeV9hY2N1cmFjeV9kaWZmIjogeyJkb3VibGVWYWx1ZSI6IDAuMDQ4MzEwOTM1NDk3MjgzOTM2fSwgImxvc3NfZGlmZiI6IHsiZG91YmxlVmFsdWUiOiAtMC4xMDAwMzU2Mzc2MTcxMTEyfSwgImJpbmFyeV9jcm9zc2VudHJvcHlfZGlmZiI6IHsiZG91YmxlVmFsdWUiOiAtMC4xMDAwMzU2Mzc2MTcxMTEyfSwgImV4YW1wbGVfY291bnRfZGlmZiI6IHsiZG91YmxlVmFsdWUiOiAwLjB9LCAiYXVjX2RpZmYiOiB7ImRvdWJsZVZhbHVlIjogMC4wNTE2NDUyNzg5MzA2NjQwNn0sICJhdWNfcHJlY2lzaW9uX3JlY2FsbF9kaWZmIjogeyJkb3VibGVWYWx1ZSI6IDAuMTA3NDMxNzY5MzcxMDMyNzF9LCAicHJlY2lzaW9uX2RpZmYiOiB7ImRvdWJsZVZhbHVlIjogLTAuMDY5ODE3NzgxNDQ4MzY0MjZ9LCAicmVjYWxsX2RpZmYiOiB7ImRvdWJsZVZhbHVlIjogMC4zNTgwMjQ2ODY1NzQ5MzU5fSwgIm1lYW5fbGFiZWxfZGlmZiI6IHsiZG91YmxlVmFsdWUiOiAwLjB9LCAibWVhbl9wcmVkaWN0aW9uX2RpZmYiOiB7ImRvdWJsZVZhbHVlIjogLTAuMDEwMDIxMzAwODgzNDkzOTU1fSwgImNhbGlicmF0aW9uX2RpZmYiOiB7ImRvdWJsZVZhbHVlIjogLTAuMDQxNzc0NzUxNTYxMDE1NH19fX19XX0='));
  element.config = json.config;
  element.data = json.data;
  </script>



**Congratulations! You can now successfully evaluate your models in a TFX pipeline! This is a critical part of production ML because you want to make sure that subsequent deployments are indeed improving your results. Moreover, you can extract the evaluation results from your pipeline directory for further investigation with TFMA. In the next sections, you will continue to study techniques related to model evaluation and ensuring fairness.**
