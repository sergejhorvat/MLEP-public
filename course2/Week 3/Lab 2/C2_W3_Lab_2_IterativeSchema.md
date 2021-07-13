# Ungraded Lab: Iterative Schema with TFX and ML Metadata


In this notebook, you will get to review how to update an inferred schema and save the result to the metadata store used by TFX. As mentioned before, the TFX components get information from this database before running executions. Thus, if you will be curating a schema, you will need to save this as an artifact in the metadata store. You will get to see how that is done in the following exercise.

Afterwards, you will also practice accessing the TFX metadata store and see how you can track the lineage of an artifact.

## Setup

### Imports


```python
import tensorflow as tf
import tensorflow_data_validation as tfdv

from tfx.components import CsvExampleGen
from tfx.components import ExampleValidator
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import ImporterNode
from tfx.types import standard_artifacts

from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from google.protobuf.json_format import MessageToDict
from tensorflow_metadata.proto.v0 import schema_pb2

import os
import pprint
pp = pprint.PrettyPrinter()
```

### Define paths

For familiarity, you will again be using the [Census Income dataset](https://archive.ics.uci.edu/ml/datasets/Adult) from the previous weeks' ungraded labs. You will use the same paths to your raw data and pipeline files as shown below.


```python
# location of the pipeline metadata store
_pipeline_root = './pipeline/'

# directory of the raw data files
_data_root = './data/census_data'

# path to the raw training data
_data_filepath = os.path.join(_data_root, 'adult.data')
```

## Data Pipeline

Each TFX component you use accepts and generates artifacts which are instances of the different artifact types TFX has configured in the metadata store. The properties of these instances are shown neatly in a table in the outputs of `context.run()`. TFX does all of these for you so you only need to inspect the output of each component to know which property of the artifact you can pass on to the next component (e.g. the `outputs['examples']` of `ExampleGen` can be passed to `StatisticsGen`).

Since you've already used this dataset before, we will just quickly go over `ExampleGen`, `StatisticsGen`, and `SchemaGen`. The new concepts will be discussed after the said components.

### Create the Interactive Context


```python
# Initialize the InteractiveContext.
# If you leave `_pipeline_root` blank, then the db will be created in a temporary directory.
context = InteractiveContext(pipeline_root=_pipeline_root)
```

    WARNING:absl:InteractiveContext metadata_connection_config not provided: using SQLite ML Metadata database at ./pipeline/metadata.sqlite.


### ExampleGen


```python
# Instantiate ExampleGen with the input CSV dataset
example_gen = CsvExampleGen(input_base=_data_root)

# Execute the component
context.run(example_gen)
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
<div class="tfx-object expanded"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExecutionResult</span><span class="deemphasize"> at 0x7fa8383bdf70</span></div><table class="attr-table"><tr><td class="attr-name">.execution_id</td><td class = "attrvalue">1</td></tr><tr><td class="attr-name">.component</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">CsvExampleGen</span><span class="deemphasize"> at 0x7fa8e86fee50</span></div><table class="attr-table"><tr><td class="attr-name">.inputs</td><td class = "attrvalue">{}</td></tr><tr><td class="attr-name">.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7fa8e86fe970</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/CsvExampleGen/examples/1)<span class="deemphasize"> at 0x7fa8383595e0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/CsvExampleGen/examples/1</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.exec_properties</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['input_base']</td><td class = "attrvalue">./data/census_data</td></tr><tr><td class="attr-name">['input_config']</td><td class = "attrvalue">{
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
}</td></tr><tr><td class="attr-name">['output_data_format']</td><td class = "attrvalue">6</td></tr><tr><td class="attr-name">['custom_config']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['span']</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">['version']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['input_fingerprint']</td><td class = "attrvalue">split:single_split,num_files:1,total_bytes:3974460,xor_checksum:1618242085,sum_checksum:1618242085</td></tr><tr><td class="attr-name">['_beam_pipeline_args']</td><td class = "attrvalue">[]</td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">.component.inputs</td><td class = "attrvalue">{}</td></tr><tr><td class="attr-name">.component.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7fa8e86fe970</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/CsvExampleGen/examples/1)<span class="deemphasize"> at 0x7fa8383595e0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/CsvExampleGen/examples/1</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr></table></div>



### StatisticsGen


```python
# Instantiate StatisticsGen with the ExampleGen ingested dataset
statistics_gen = StatisticsGen(
    examples=example_gen.outputs['examples'])

# Execute the component
context.run(statistics_gen)
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
<div class="tfx-object expanded"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExecutionResult</span><span class="deemphasize"> at 0x7fa8383bde50</span></div><table class="attr-table"><tr><td class="attr-name">.execution_id</td><td class = "attrvalue">2</td></tr><tr><td class="attr-name">.component</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">StatisticsGen</span><span class="deemphasize"> at 0x7fa8d163ae50</span></div><table class="attr-table"><tr><td class="attr-name">.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7fa8e86fe970</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/CsvExampleGen/examples/1)<span class="deemphasize"> at 0x7fa8383595e0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/CsvExampleGen/examples/1</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['statistics']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7fa8d163a370</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: ./pipeline/StatisticsGen/statistics/2)<span class="deemphasize"> at 0x7fa8eaf66790</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/StatisticsGen/statistics/2</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.exec_properties</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['stats_options_json']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['exclude_splits']</td><td class = "attrvalue">[]</td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">.component.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7fa8e86fe970</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/CsvExampleGen/examples/1)<span class="deemphasize"> at 0x7fa8383595e0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/CsvExampleGen/examples/1</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.component.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['statistics']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7fa8d163a370</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: ./pipeline/StatisticsGen/statistics/2)<span class="deemphasize"> at 0x7fa8eaf66790</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/StatisticsGen/statistics/2</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr></table></div>



### SchemaGen


```python
# Instantiate SchemaGen with the StatisticsGen ingested dataset
schema_gen = SchemaGen(
    statistics=statistics_gen.outputs['statistics'],
    )

# Run the component
context.run(schema_gen)
```

    WARNING:tensorflow:From /opt/conda/lib/python3.8/site-packages/tensorflow_data_validation/utils/stats_util.py:229: tf_record_iterator (from tensorflow.python.lib.io.tf_record) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use eager execution and: 
    `tf.data.TFRecordDataset(path)`





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
<div class="tfx-object expanded"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExecutionResult</span><span class="deemphasize"> at 0x7fa8383bdfd0</span></div><table class="attr-table"><tr><td class="attr-name">.execution_id</td><td class = "attrvalue">3</td></tr><tr><td class="attr-name">.component</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">SchemaGen</span><span class="deemphasize"> at 0x7fa8d15daf40</span></div><table class="attr-table"><tr><td class="attr-name">.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['statistics']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7fa8d163a370</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: ./pipeline/StatisticsGen/statistics/2)<span class="deemphasize"> at 0x7fa8eaf66790</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/StatisticsGen/statistics/2</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['schema']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7fa8d15da5e0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline/SchemaGen/schema/3)<span class="deemphasize"> at 0x7fa8d163a5b0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/SchemaGen/schema/3</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.exec_properties</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['infer_feature_shape']</td><td class = "attrvalue">False</td></tr><tr><td class="attr-name">['exclude_splits']</td><td class = "attrvalue">[]</td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">.component.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['statistics']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7fa8d163a370</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: ./pipeline/StatisticsGen/statistics/2)<span class="deemphasize"> at 0x7fa8eaf66790</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/StatisticsGen/statistics/2</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.component.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['schema']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7fa8d15da5e0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline/SchemaGen/schema/3)<span class="deemphasize"> at 0x7fa8d163a5b0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/SchemaGen/schema/3</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr></table></div>




```python
# Visualize the schema
context.show(schema_gen.outputs['schema'])
```


<b>Artifact at ./pipeline/SchemaGen/schema/3</b><br/><br/>



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
      <th>'age'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'capital-gain'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'capital-loss'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'education'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'education'</td>
    </tr>
    <tr>
      <th>'education-num'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'fnlwgt'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'hours-per-week'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'label'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'label'</td>
    </tr>
    <tr>
      <th>'marital-status'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'marital-status'</td>
    </tr>
    <tr>
      <th>'native-country'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'native-country'</td>
    </tr>
    <tr>
      <th>'occupation'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'occupation'</td>
    </tr>
    <tr>
      <th>'race'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'race'</td>
    </tr>
    <tr>
      <th>'relationship'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'relationship'</td>
    </tr>
    <tr>
      <th>'sex'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'sex'</td>
    </tr>
    <tr>
      <th>'workclass'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'workclass'</td>
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
      <th>'education'</th>
      <td>' 10th', ' 11th', ' 12th', ' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' Assoc-acdm', ' Assoc-voc', ' Bachelors', ' Doctorate', ' HS-grad', ' Masters', ' Preschool', ' Prof-school', ' Some-college'</td>
    </tr>
    <tr>
      <th>'label'</th>
      <td>' &lt;=50K', ' &gt;50K'</td>
    </tr>
    <tr>
      <th>'marital-status'</th>
      <td>' Divorced', ' Married-AF-spouse', ' Married-civ-spouse', ' Married-spouse-absent', ' Never-married', ' Separated', ' Widowed'</td>
    </tr>
    <tr>
      <th>'native-country'</th>
      <td>' ?', ' Cambodia', ' Canada', ' China', ' Columbia', ' Cuba', ' Dominican-Republic', ' Ecuador', ' El-Salvador', ' England', ' France', ' Germany', ' Greece', ' Guatemala', ' Haiti', ' Honduras', ' Hong', ' Hungary', ' India', ' Iran', ' Ireland', ' Italy', ' Jamaica', ' Japan', ' Laos', ' Mexico', ' Nicaragua', ' Outlying-US(Guam-USVI-etc)', ' Peru', ' Philippines', ' Poland', ' Portugal', ' Puerto-Rico', ' Scotland', ' South', ' Taiwan', ' Thailand', ' Trinadad&amp;Tobago', ' United-States', ' Vietnam', ' Yugoslavia', ' Holand-Netherlands'</td>
    </tr>
    <tr>
      <th>'occupation'</th>
      <td>' ?', ' Adm-clerical', ' Armed-Forces', ' Craft-repair', ' Exec-managerial', ' Farming-fishing', ' Handlers-cleaners', ' Machine-op-inspct', ' Other-service', ' Priv-house-serv', ' Prof-specialty', ' Protective-serv', ' Sales', ' Tech-support', ' Transport-moving'</td>
    </tr>
    <tr>
      <th>'race'</th>
      <td>' Amer-Indian-Eskimo', ' Asian-Pac-Islander', ' Black', ' Other', ' White'</td>
    </tr>
    <tr>
      <th>'relationship'</th>
      <td>' Husband', ' Not-in-family', ' Other-relative', ' Own-child', ' Unmarried', ' Wife'</td>
    </tr>
    <tr>
      <th>'sex'</th>
      <td>' Female', ' Male'</td>
    </tr>
    <tr>
      <th>'workclass'</th>
      <td>' ?', ' Federal-gov', ' Local-gov', ' Never-worked', ' Private', ' Self-emp-inc', ' Self-emp-not-inc', ' State-gov', ' Without-pay'</td>
    </tr>
  </tbody>
</table>
</div>


### Curating the Schema

Now that you have the inferred schema, you can proceed to revising it to be more robust. For instance, you can restrict the age as you did in Week 1. First, you have to load the `Schema` protocol buffer from the metadata store. You can do this by getting the schema uri from the output of `SchemaGen` then use TFDV's `load_schema_text()` method.


```python
# Get the schema uri
schema_uri = schema_gen.outputs['schema']._artifacts[0].uri

# Get the schema pbtxt file from the SchemaGen output
schema = tfdv.load_schema_text(os.path.join(schema_uri, 'schema.pbtxt'))
```

With that, you can now make changes to the schema as before. For the purpose of this exercise, you will only modify the age domain but feel free to add more if you want.


```python
# Restrict the range of the `age` feature
tfdv.set_domain(schema, 'age', schema_pb2.IntDomain(name='age', min=17, max=90))

# Display the modified schema. Notice the `Domain` column of `age`.
tfdv.display_schema(schema)
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
      <th>'age'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>[17,90]</td>
    </tr>
    <tr>
      <th>'capital-gain'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'capital-loss'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'education'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'education'</td>
    </tr>
    <tr>
      <th>'education-num'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'fnlwgt'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'hours-per-week'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'label'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'label'</td>
    </tr>
    <tr>
      <th>'marital-status'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'marital-status'</td>
    </tr>
    <tr>
      <th>'native-country'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'native-country'</td>
    </tr>
    <tr>
      <th>'occupation'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'occupation'</td>
    </tr>
    <tr>
      <th>'race'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'race'</td>
    </tr>
    <tr>
      <th>'relationship'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'relationship'</td>
    </tr>
    <tr>
      <th>'sex'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'sex'</td>
    </tr>
    <tr>
      <th>'workclass'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'workclass'</td>
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
      <th>'education'</th>
      <td>' 10th', ' 11th', ' 12th', ' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' Assoc-acdm', ' Assoc-voc', ' Bachelors', ' Doctorate', ' HS-grad', ' Masters', ' Preschool', ' Prof-school', ' Some-college'</td>
    </tr>
    <tr>
      <th>'label'</th>
      <td>' &lt;=50K', ' &gt;50K'</td>
    </tr>
    <tr>
      <th>'marital-status'</th>
      <td>' Divorced', ' Married-AF-spouse', ' Married-civ-spouse', ' Married-spouse-absent', ' Never-married', ' Separated', ' Widowed'</td>
    </tr>
    <tr>
      <th>'native-country'</th>
      <td>' ?', ' Cambodia', ' Canada', ' China', ' Columbia', ' Cuba', ' Dominican-Republic', ' Ecuador', ' El-Salvador', ' England', ' France', ' Germany', ' Greece', ' Guatemala', ' Haiti', ' Honduras', ' Hong', ' Hungary', ' India', ' Iran', ' Ireland', ' Italy', ' Jamaica', ' Japan', ' Laos', ' Mexico', ' Nicaragua', ' Outlying-US(Guam-USVI-etc)', ' Peru', ' Philippines', ' Poland', ' Portugal', ' Puerto-Rico', ' Scotland', ' South', ' Taiwan', ' Thailand', ' Trinadad&amp;Tobago', ' United-States', ' Vietnam', ' Yugoslavia', ' Holand-Netherlands'</td>
    </tr>
    <tr>
      <th>'occupation'</th>
      <td>' ?', ' Adm-clerical', ' Armed-Forces', ' Craft-repair', ' Exec-managerial', ' Farming-fishing', ' Handlers-cleaners', ' Machine-op-inspct', ' Other-service', ' Priv-house-serv', ' Prof-specialty', ' Protective-serv', ' Sales', ' Tech-support', ' Transport-moving'</td>
    </tr>
    <tr>
      <th>'race'</th>
      <td>' Amer-Indian-Eskimo', ' Asian-Pac-Islander', ' Black', ' Other', ' White'</td>
    </tr>
    <tr>
      <th>'relationship'</th>
      <td>' Husband', ' Not-in-family', ' Other-relative', ' Own-child', ' Unmarried', ' Wife'</td>
    </tr>
    <tr>
      <th>'sex'</th>
      <td>' Female', ' Male'</td>
    </tr>
    <tr>
      <th>'workclass'</th>
      <td>' ?', ' Federal-gov', ' Local-gov', ' Never-worked', ' Private', ' Self-emp-inc', ' Self-emp-not-inc', ' State-gov', ' Without-pay'</td>
    </tr>
  </tbody>
</table>
</div>


### Schema Environments

By default, your schema will watch for all the features declared above including the label. However, when the model is served for inference, it will get datasets that will not have the label because that is the feature that the model will be trying to predict. You need to configure the pipeline to not raise an alarm when this kind of dataset is received.

You can do that with [schema environments](https://www.tensorflow.org/tfx/tutorials/data_validation/tfdv_basic#schema_environments). First, you will need to declare training and serving environments, then configure the serving schema to not watch for the presence of labels. See how it is implemented below.


```python
# Create schema environments for training and serving
schema.default_environment.append('TRAINING')
schema.default_environment.append('SERVING')

# Omit label from the serving environment
tfdv.get_feature(schema, 'label').not_in_environment.append('SERVING')
```

You can now freeze the curated schema and save to a local directory.


```python
# Declare the path to the updated schema directory
_updated_schema_dir = f'{_pipeline_root}/updated_schema'

# Create the said directory
!mkdir -p {_updated_schema_dir}

# Declare the path to the schema file
schema_file = os.path.join(_updated_schema_dir, 'schema.pbtxt')

# Save the curated schema to the said file
tfdv.write_schema_text(schema, schema_file)
```

### ImporterNode

Now that the schema has been saved, you need to create an artifact in the metadata store that will point to it. TFX provides the [ImporterNode](https://www.tensorflow.org/tfx/guide/statsgen#using_the_statsgen_component_with_a_schema) component used to import external objects to ML Metadata. You will need to pass in the URI of the object and what type of artifact it is. See the syntax below.


```python
# Use an ImporterNode to put the curated schema to ML Metadata
user_schema_importer = ImporterNode(
    instance_name='import_user_schema',
    source_uri=_updated_schema_dir,
    artifact_type=standard_artifacts.Schema
)

# Run the component
context.run(user_schema_importer, enable_cache=False)
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
<div class="tfx-object expanded"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExecutionResult</span><span class="deemphasize"> at 0x7fa8eaf66fd0</span></div><table class="attr-table"><tr><td class="attr-name">.execution_id</td><td class = "attrvalue">4</td></tr><tr><td class="attr-name">.component</td><td class = "attrvalue">&lt;tfx.components.common_nodes.importer_node.ImporterNode object at 0x7fa8d088aa90&gt;</td></tr><tr><td class="attr-name">.component.inputs</td><td class = "attrvalue">{}</td></tr><tr><td class="attr-name">.component.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['result']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7fa8d088a5e0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline//updated_schema)<span class="deemphasize"> at 0x7fa8d088a100</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline//updated_schema</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr></table></div>



If you pass in the component output to `context.show()`, then you should see the schema.


```python
# See the result
context.show(user_schema_importer.outputs['result'])
```


<b>Artifact at ./pipeline//updated_schema</b><br/><br/>



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
      <th>'age'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>[17,90]</td>
    </tr>
    <tr>
      <th>'capital-gain'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'capital-loss'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'education'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'education'</td>
    </tr>
    <tr>
      <th>'education-num'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'fnlwgt'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'hours-per-week'</th>
      <td>INT</td>
      <td>required</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'label'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'label'</td>
    </tr>
    <tr>
      <th>'marital-status'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'marital-status'</td>
    </tr>
    <tr>
      <th>'native-country'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'native-country'</td>
    </tr>
    <tr>
      <th>'occupation'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'occupation'</td>
    </tr>
    <tr>
      <th>'race'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'race'</td>
    </tr>
    <tr>
      <th>'relationship'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'relationship'</td>
    </tr>
    <tr>
      <th>'sex'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'sex'</td>
    </tr>
    <tr>
      <th>'workclass'</th>
      <td>STRING</td>
      <td>required</td>
      <td>single</td>
      <td>'workclass'</td>
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
      <th>'education'</th>
      <td>' 10th', ' 11th', ' 12th', ' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' Assoc-acdm', ' Assoc-voc', ' Bachelors', ' Doctorate', ' HS-grad', ' Masters', ' Preschool', ' Prof-school', ' Some-college'</td>
    </tr>
    <tr>
      <th>'label'</th>
      <td>' &lt;=50K', ' &gt;50K'</td>
    </tr>
    <tr>
      <th>'marital-status'</th>
      <td>' Divorced', ' Married-AF-spouse', ' Married-civ-spouse', ' Married-spouse-absent', ' Never-married', ' Separated', ' Widowed'</td>
    </tr>
    <tr>
      <th>'native-country'</th>
      <td>' ?', ' Cambodia', ' Canada', ' China', ' Columbia', ' Cuba', ' Dominican-Republic', ' Ecuador', ' El-Salvador', ' England', ' France', ' Germany', ' Greece', ' Guatemala', ' Haiti', ' Honduras', ' Hong', ' Hungary', ' India', ' Iran', ' Ireland', ' Italy', ' Jamaica', ' Japan', ' Laos', ' Mexico', ' Nicaragua', ' Outlying-US(Guam-USVI-etc)', ' Peru', ' Philippines', ' Poland', ' Portugal', ' Puerto-Rico', ' Scotland', ' South', ' Taiwan', ' Thailand', ' Trinadad&amp;Tobago', ' United-States', ' Vietnam', ' Yugoslavia', ' Holand-Netherlands'</td>
    </tr>
    <tr>
      <th>'occupation'</th>
      <td>' ?', ' Adm-clerical', ' Armed-Forces', ' Craft-repair', ' Exec-managerial', ' Farming-fishing', ' Handlers-cleaners', ' Machine-op-inspct', ' Other-service', ' Priv-house-serv', ' Prof-specialty', ' Protective-serv', ' Sales', ' Tech-support', ' Transport-moving'</td>
    </tr>
    <tr>
      <th>'race'</th>
      <td>' Amer-Indian-Eskimo', ' Asian-Pac-Islander', ' Black', ' Other', ' White'</td>
    </tr>
    <tr>
      <th>'relationship'</th>
      <td>' Husband', ' Not-in-family', ' Other-relative', ' Own-child', ' Unmarried', ' Wife'</td>
    </tr>
    <tr>
      <th>'sex'</th>
      <td>' Female', ' Male'</td>
    </tr>
    <tr>
      <th>'workclass'</th>
      <td>' ?', ' Federal-gov', ' Local-gov', ' Never-worked', ' Private', ' Self-emp-inc', ' Self-emp-not-inc', ' State-gov', ' Without-pay'</td>
    </tr>
  </tbody>
</table>
</div>


### ExampleValidator

You can then use this new artifact as input to the other components of the pipeline. See how it is used as the `schema` argument in `ExampleValidator` below.


```python
# Instantiate ExampleValidator with the StatisticsGen and SchemaGen ingested data
example_validator = ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=user_schema_importer.outputs['result'])

# Run the component.
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
<div class="tfx-object expanded"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExecutionResult</span><span class="deemphasize"> at 0x7fa8d15dab80</span></div><table class="attr-table"><tr><td class="attr-name">.execution_id</td><td class = "attrvalue">5</td></tr><tr><td class="attr-name">.component</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExampleValidator</span><span class="deemphasize"> at 0x7fa8d088aac0</span></div><table class="attr-table"><tr><td class="attr-name">.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['statistics']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7fa8d163a370</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: ./pipeline/StatisticsGen/statistics/2)<span class="deemphasize"> at 0x7fa8eaf66790</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/StatisticsGen/statistics/2</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['schema']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7fa8d088a5e0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline//updated_schema)<span class="deemphasize"> at 0x7fa8d088a100</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline//updated_schema</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['anomalies']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleAnomalies'</span> (1 artifact)<span class="deemphasize"> at 0x7fa8d088aa00</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleAnomalies</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleAnomalies'</span> (uri: ./pipeline/ExampleValidator/anomalies/5)<span class="deemphasize"> at 0x7fa8d15dabe0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleAnomalies&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/ExampleValidator/anomalies/5</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.exec_properties</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['exclude_splits']</td><td class = "attrvalue">[]</td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">.component.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['statistics']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7fa8d163a370</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: ./pipeline/StatisticsGen/statistics/2)<span class="deemphasize"> at 0x7fa8eaf66790</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/StatisticsGen/statistics/2</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['schema']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7fa8d088a5e0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline//updated_schema)<span class="deemphasize"> at 0x7fa8d088a100</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline//updated_schema</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.component.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['anomalies']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleAnomalies'</span> (1 artifact)<span class="deemphasize"> at 0x7fa8d088aa00</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleAnomalies</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleAnomalies'</span> (uri: ./pipeline/ExampleValidator/anomalies/5)<span class="deemphasize"> at 0x7fa8d15dabe0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleAnomalies&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/ExampleValidator/anomalies/5</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr></table></div>




```python
# Visualize the results
context.show(example_validator.outputs['anomalies'])
```


<b>Artifact at ./pipeline/ExampleValidator/anomalies/5</b><br/><br/>



<div><b>'train' split:</b></div><br/>



<h4 style="color:green;">No anomalies found.</h4>



<div><b>'eval' split:</b></div><br/>



<h4 style="color:green;">No anomalies found.</h4>


### Practice with ML Metadata

At this point, you should now take some time exploring the contents of the metadata store saved by your component runs. This will let you practice tracking artifacts and how they are related to each other. This involves looking at artifacts, executions, and events. This skill will let you recover related artifacts even without seeing the code of the training run. All you need is access to the metadata store. 

See how the input artifact IDs to an instance of `ExampleAnomalies` are tracked in the following cells. If you have this notebook, then you will already know that it uses the output of StatisticsGen for this run and also the curated schema you imported. However, if you already have hundreds of training runs and parameter iterations, you may find it hard to track which is which. That's where the metadata store will be useful. Since it records information about a specific pipeline run, you will be able to track the inputs and outputs of a particular artifact.

You will start by setting the connection config to the metadata store. 


```python
# Import mlmd and utilities
import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2

# Get the connection config to connect to the context's metadata store
connection_config = context.metadata_connection_config

# Instantiate a MetadataStore instance with the connection config
store = mlmd.MetadataStore(connection_config)
```

Next, let's see what artifact types are available in the metadata store.


```python
# Get artifact types
artifact_types = store.get_artifact_types()

# Print the results
[artifact_type.name for artifact_type in artifact_types]
```




    ['Examples', 'ExampleStatistics', 'Schema', 'ExampleAnomalies']



If you get the artifacts of type `Schema`, you will see that there are two entries. One is the inferred and the other is the one you imported. At the end of this exercise, you can verify that the curated schema is the one used for the `ExampleValidator` run we will be investigating.


```python
# Get artifact types
schema_list = store.get_artifacts_by_type('Schema')

[(f'schema uri: {schema.uri}', f'schema id:{schema.id}') for schema in schema_list]
```




    [('schema uri: ./pipeline/SchemaGen/schema/3', 'schema id:3'),
     ('schema uri: ./pipeline//updated_schema', 'schema id:4')]



Let's get the first instance of `ExampleAnomalies` to get the output of `ExampleValidator`.


```python
# Get 1st instance of ExampleAnomalies
example_anomalies = store.get_artifacts_by_type('ExampleAnomalies')[0]

# Print the artifact id
print(f'Artifact id: {example_anomalies.id}')
```

    Artifact id: 5


You will use the artifact ID to get events related to it. Let's just get the first instance.


```python
# Get first event related to the ID
anomalies_id_event = store.get_events_by_artifact_ids([example_anomalies.id])[0]

# Print results
print(anomalies_id_event)
```

    artifact_id: 5
    execution_id: 5
    path {
      steps {
        key: "anomalies"
      }
      steps {
        index: 0
      }
    }
    type: OUTPUT
    milliseconds_since_epoch: 1625830637413
    


As expected, the event type will be an `OUTPUT` because this is the output of the `ExampleValidator` component. Since we want to get the inputs, we can track it through the execution id.


```python
# Get execution ID
anomalies_execution_id = anomalies_id_event.execution_id

# Get events by the execution ID
events_execution = store.get_events_by_execution_ids([anomalies_execution_id])

# Print results
print(events_execution)
```

    [artifact_id: 2
    execution_id: 5
    path {
      steps {
        key: "statistics"
      }
      steps {
        index: 0
      }
    }
    type: INPUT
    milliseconds_since_epoch: 1625830637021
    , artifact_id: 4
    execution_id: 5
    path {
      steps {
        key: "schema"
      }
      steps {
        index: 0
      }
    }
    type: INPUT
    milliseconds_since_epoch: 1625830637021
    , artifact_id: 5
    execution_id: 5
    path {
      steps {
        key: "anomalies"
      }
      steps {
        index: 0
      }
    }
    type: OUTPUT
    milliseconds_since_epoch: 1625830637413
    ]


We see the artifacts which are marked as `INPUT` above representing the statistics and schema inputs. We can extract their IDs programmatically like this. You will see that you will get the artifact ID of the curated schema you printed out earlier.


```python
# Filter INPUT type events
inputs_to_exval = [event.artifact_id for event in events_execution 
                       if event.type == metadata_store_pb2.Event.INPUT]

# Print results
print(inputs_to_exval)
```

    [2, 4]


**Congratulations!** You have now completed this notebook on iterative schemas and saw how it can be used in a TFX pipeline. You were also able to track an artifact's lineage by looking at the artifacts, events, and executions in the metadata store. These will come in handy in this week's assignment!
