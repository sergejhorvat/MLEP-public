¶
ź
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
h
All	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.6.02v2.6.0-rc2-32-g919f693420e8ų
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ’
L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ’
L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  ’
L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *  ’
L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *  ’
L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *  ’
L
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *  ’
L
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *  ’

NoOpNoOp
Ņ
Const_8Const"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB By
k
created_variables
	resources
trackable_objects
initializers

assets

signatures
 
 
 
 
 
 
q
serving_default_inputsPlaceholder*#
_output_shapes
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’
s
serving_default_inputs_1Placeholder*#
_output_shapes
:’’’’’’’’’*
dtype0	*
shape:’’’’’’’’’
s
serving_default_inputs_2Placeholder*#
_output_shapes
:’’’’’’’’’*
dtype0	*
shape:’’’’’’’’’
s
serving_default_inputs_3Placeholder*#
_output_shapes
:’’’’’’’’’*
dtype0	*
shape:’’’’’’’’’
s
serving_default_inputs_4Placeholder*#
_output_shapes
:’’’’’’’’’*
dtype0	*
shape:’’’’’’’’’
£
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputsserving_default_inputs_1serving_default_inputs_2serving_default_inputs_3serving_default_inputs_4ConstConst_1Const_2Const_3Const_4Const_5Const_6Const_7*
Tin
2				*
Tout	
2*
_collective_manager_ids
 *_
_output_shapesM
K:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_1078
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameConst_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *&
f!R
__inference__traced_save_1117

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_restore_1127¼ć
ę“
²
__inference_pruned_1043

inputs
inputs_1	
inputs_2	
inputs_3	
inputs_4	-
)scale_to_0_1_min_and_max_identity_2_input-
)scale_to_0_1_min_and_max_identity_3_input/
+scale_to_0_1_1_min_and_max_identity_2_input/
+scale_to_0_1_1_min_and_max_identity_3_input/
+scale_to_0_1_2_min_and_max_identity_2_input/
+scale_to_0_1_2_min_and_max_identity_3_input/
+scale_to_0_1_3_min_and_max_identity_2_input/
+scale_to_0_1_3_min_and_max_identity_3_input
identity

identity_1

identity_2

identity_3

identity_4\
inputs_copyIdentityinputs*
T0*#
_output_shapes
:’’’’’’’’’2
inputs_copy
 scale_to_0_1_3/min_and_max/ShapeConst*
_output_shapes
: *
dtype0*
valueB 2"
 scale_to_0_1_3/min_and_max/Shape
"scale_to_0_1_3/min_and_max/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 2$
"scale_to_0_1_3/min_and_max/Shape_1ę
/scale_to_0_1_3/min_and_max/assert_equal_1/EqualEqual)scale_to_0_1_3/min_and_max/Shape:output:0+scale_to_0_1_3/min_and_max/Shape_1:output:0*
T0*
_output_shapes
: 21
/scale_to_0_1_3/min_and_max/assert_equal_1/Equal¬
/scale_to_0_1_3/min_and_max/assert_equal_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/scale_to_0_1_3/min_and_max/assert_equal_1/Constģ
-scale_to_0_1_3/min_and_max/assert_equal_1/AllAll3scale_to_0_1_3/min_and_max/assert_equal_1/Equal:z:08scale_to_0_1_3/min_and_max/assert_equal_1/Const:output:0*
_output_shapes
: 2/
-scale_to_0_1_3/min_and_max/assert_equal_1/Allģ
>scale_to_0_1_3/min_and_max/assert_equal_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:2@
>scale_to_0_1_3/min_and_max/assert_equal_1/Assert/Assert/data_0ź
>scale_to_0_1_3/min_and_max/assert_equal_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*:
value1B/ B)x (scale_to_0_1_3/min_and_max/Shape:0) = 2@
>scale_to_0_1_3/min_and_max/assert_equal_1/Assert/Assert/data_1ģ
>scale_to_0_1_3/min_and_max/assert_equal_1/Assert/Assert/data_3Const*
_output_shapes
: *
dtype0*<
value3B1 B+y (scale_to_0_1_3/min_and_max/Shape_1:0) = 2@
>scale_to_0_1_3/min_and_max/assert_equal_1/Assert/Assert/data_3
 scale_to_0_1_2/min_and_max/ShapeConst*
_output_shapes
: *
dtype0*
valueB 2"
 scale_to_0_1_2/min_and_max/Shape
"scale_to_0_1_2/min_and_max/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 2$
"scale_to_0_1_2/min_and_max/Shape_1ę
/scale_to_0_1_2/min_and_max/assert_equal_1/EqualEqual)scale_to_0_1_2/min_and_max/Shape:output:0+scale_to_0_1_2/min_and_max/Shape_1:output:0*
T0*
_output_shapes
: 21
/scale_to_0_1_2/min_and_max/assert_equal_1/Equal¬
/scale_to_0_1_2/min_and_max/assert_equal_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/scale_to_0_1_2/min_and_max/assert_equal_1/Constģ
-scale_to_0_1_2/min_and_max/assert_equal_1/AllAll3scale_to_0_1_2/min_and_max/assert_equal_1/Equal:z:08scale_to_0_1_2/min_and_max/assert_equal_1/Const:output:0*
_output_shapes
: 2/
-scale_to_0_1_2/min_and_max/assert_equal_1/Allģ
>scale_to_0_1_2/min_and_max/assert_equal_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:2@
>scale_to_0_1_2/min_and_max/assert_equal_1/Assert/Assert/data_0ź
>scale_to_0_1_2/min_and_max/assert_equal_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*:
value1B/ B)x (scale_to_0_1_2/min_and_max/Shape:0) = 2@
>scale_to_0_1_2/min_and_max/assert_equal_1/Assert/Assert/data_1ģ
>scale_to_0_1_2/min_and_max/assert_equal_1/Assert/Assert/data_3Const*
_output_shapes
: *
dtype0*<
value3B1 B+y (scale_to_0_1_2/min_and_max/Shape_1:0) = 2@
>scale_to_0_1_2/min_and_max/assert_equal_1/Assert/Assert/data_3
 scale_to_0_1_1/min_and_max/ShapeConst*
_output_shapes
: *
dtype0*
valueB 2"
 scale_to_0_1_1/min_and_max/Shape
"scale_to_0_1_1/min_and_max/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 2$
"scale_to_0_1_1/min_and_max/Shape_1ę
/scale_to_0_1_1/min_and_max/assert_equal_1/EqualEqual)scale_to_0_1_1/min_and_max/Shape:output:0+scale_to_0_1_1/min_and_max/Shape_1:output:0*
T0*
_output_shapes
: 21
/scale_to_0_1_1/min_and_max/assert_equal_1/Equal¬
/scale_to_0_1_1/min_and_max/assert_equal_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/scale_to_0_1_1/min_and_max/assert_equal_1/Constģ
-scale_to_0_1_1/min_and_max/assert_equal_1/AllAll3scale_to_0_1_1/min_and_max/assert_equal_1/Equal:z:08scale_to_0_1_1/min_and_max/assert_equal_1/Const:output:0*
_output_shapes
: 2/
-scale_to_0_1_1/min_and_max/assert_equal_1/Allģ
>scale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:2@
>scale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert/data_0ź
>scale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*:
value1B/ B)x (scale_to_0_1_1/min_and_max/Shape:0) = 2@
>scale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert/data_1ģ
>scale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert/data_3Const*
_output_shapes
: *
dtype0*<
value3B1 B+y (scale_to_0_1_1/min_and_max/Shape_1:0) = 2@
>scale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert/data_3
scale_to_0_1/min_and_max/ShapeConst*
_output_shapes
: *
dtype0*
valueB 2 
scale_to_0_1/min_and_max/Shape
 scale_to_0_1/min_and_max/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 2"
 scale_to_0_1/min_and_max/Shape_1Ž
-scale_to_0_1/min_and_max/assert_equal_1/EqualEqual'scale_to_0_1/min_and_max/Shape:output:0)scale_to_0_1/min_and_max/Shape_1:output:0*
T0*
_output_shapes
: 2/
-scale_to_0_1/min_and_max/assert_equal_1/EqualØ
-scale_to_0_1/min_and_max/assert_equal_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-scale_to_0_1/min_and_max/assert_equal_1/Constä
+scale_to_0_1/min_and_max/assert_equal_1/AllAll1scale_to_0_1/min_and_max/assert_equal_1/Equal:z:06scale_to_0_1/min_and_max/assert_equal_1/Const:output:0*
_output_shapes
: 2-
+scale_to_0_1/min_and_max/assert_equal_1/Allč
<scale_to_0_1/min_and_max/assert_equal_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:2>
<scale_to_0_1/min_and_max/assert_equal_1/Assert/Assert/data_0ä
<scale_to_0_1/min_and_max/assert_equal_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*8
value/B- B'x (scale_to_0_1/min_and_max/Shape:0) = 2>
<scale_to_0_1/min_and_max/assert_equal_1/Assert/Assert/data_1ę
<scale_to_0_1/min_and_max/assert_equal_1/Assert/Assert/data_3Const*
_output_shapes
: *
dtype0*:
value1B/ B)y (scale_to_0_1/min_and_max/Shape_1:0) = 2>
<scale_to_0_1/min_and_max/assert_equal_1/Assert/Assert/data_3ż
5scale_to_0_1/min_and_max/assert_equal_1/Assert/AssertAssert4scale_to_0_1/min_and_max/assert_equal_1/All:output:0Escale_to_0_1/min_and_max/assert_equal_1/Assert/Assert/data_0:output:0Escale_to_0_1/min_and_max/assert_equal_1/Assert/Assert/data_1:output:0'scale_to_0_1/min_and_max/Shape:output:0Escale_to_0_1/min_and_max/assert_equal_1/Assert/Assert/data_3:output:0)scale_to_0_1/min_and_max/Shape_1:output:0*
T	
2*
_output_shapes
 27
5scale_to_0_1/min_and_max/assert_equal_1/Assert/AssertÅ
7scale_to_0_1_1/min_and_max/assert_equal_1/Assert/AssertAssert6scale_to_0_1_1/min_and_max/assert_equal_1/All:output:0Gscale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert/data_0:output:0Gscale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert/data_1:output:0)scale_to_0_1_1/min_and_max/Shape:output:0Gscale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert/data_3:output:0+scale_to_0_1_1/min_and_max/Shape_1:output:06^scale_to_0_1/min_and_max/assert_equal_1/Assert/Assert*
T	
2*
_output_shapes
 29
7scale_to_0_1_1/min_and_max/assert_equal_1/Assert/AssertĒ
7scale_to_0_1_2/min_and_max/assert_equal_1/Assert/AssertAssert6scale_to_0_1_2/min_and_max/assert_equal_1/All:output:0Gscale_to_0_1_2/min_and_max/assert_equal_1/Assert/Assert/data_0:output:0Gscale_to_0_1_2/min_and_max/assert_equal_1/Assert/Assert/data_1:output:0)scale_to_0_1_2/min_and_max/Shape:output:0Gscale_to_0_1_2/min_and_max/assert_equal_1/Assert/Assert/data_3:output:0+scale_to_0_1_2/min_and_max/Shape_1:output:08^scale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert*
T	
2*
_output_shapes
 29
7scale_to_0_1_2/min_and_max/assert_equal_1/Assert/AssertĒ
7scale_to_0_1_3/min_and_max/assert_equal_1/Assert/AssertAssert6scale_to_0_1_3/min_and_max/assert_equal_1/All:output:0Gscale_to_0_1_3/min_and_max/assert_equal_1/Assert/Assert/data_0:output:0Gscale_to_0_1_3/min_and_max/assert_equal_1/Assert/Assert/data_1:output:0)scale_to_0_1_3/min_and_max/Shape:output:0Gscale_to_0_1_3/min_and_max/assert_equal_1/Assert/Assert/data_3:output:0+scale_to_0_1_3/min_and_max/Shape_1:output:08^scale_to_0_1_2/min_and_max/assert_equal_1/Assert/Assert*
T	
2*
_output_shapes
 29
7scale_to_0_1_3/min_and_max/assert_equal_1/Assert/Assert“
NoOpNoOp6^scale_to_0_1/min_and_max/assert_equal_1/Assert/Assert8^scale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert8^scale_to_0_1_2/min_and_max/assert_equal_1/Assert/Assert8^scale_to_0_1_3/min_and_max/assert_equal_1/Assert/Assert*"
_acd_function_control_output(*
_output_shapes
 2
NoOpk
IdentityIdentityinputs_copy:output:0^NoOp*
T0*#
_output_shapes
:’’’’’’’’’2

Identityb
inputs_1_copyIdentityinputs_1*
T0	*#
_output_shapes
:’’’’’’’’’2
inputs_1_copy
scale_to_0_1/CastCastinputs_1_copy:output:0*

DstT0*

SrcT0	*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1/Cast
 scale_to_0_1/min_and_max/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 scale_to_0_1/min_and_max/sub_1/x¢
#scale_to_0_1/min_and_max/Identity_2Identity)scale_to_0_1_min_and_max_identity_2_input*
T0*
_output_shapes
: 2%
#scale_to_0_1/min_and_max/Identity_2Į
scale_to_0_1/min_and_max/sub_1Sub)scale_to_0_1/min_and_max/sub_1/x:output:0,scale_to_0_1/min_and_max/Identity_2:output:0*
T0*
_output_shapes
: 2 
scale_to_0_1/min_and_max/sub_1
scale_to_0_1/subSubscale_to_0_1/Cast:y:0"scale_to_0_1/min_and_max/sub_1:z:0*
T0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1/sub
scale_to_0_1/zeros_like	ZerosLikescale_to_0_1/sub:z:0*
T0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1/zeros_like¢
#scale_to_0_1/min_and_max/Identity_3Identity)scale_to_0_1_min_and_max_identity_3_input*
T0*
_output_shapes
: 2%
#scale_to_0_1/min_and_max/Identity_3”
scale_to_0_1/LessLess"scale_to_0_1/min_and_max/sub_1:z:0,scale_to_0_1/min_and_max/Identity_3:output:0*
T0*
_output_shapes
: 2
scale_to_0_1/Lessy
scale_to_0_1/Cast_1Castscale_to_0_1/Less:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2
scale_to_0_1/Cast_1
scale_to_0_1/addAddV2scale_to_0_1/zeros_like:y:0scale_to_0_1/Cast_1:y:0*
T0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1/add
scale_to_0_1/Cast_2Castscale_to_0_1/add:z:0*

DstT0
*

SrcT0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1/Cast_2¢
scale_to_0_1/sub_1Sub,scale_to_0_1/min_and_max/Identity_3:output:0"scale_to_0_1/min_and_max/sub_1:z:0*
T0*
_output_shapes
: 2
scale_to_0_1/sub_1
scale_to_0_1/truedivRealDivscale_to_0_1/sub:z:0scale_to_0_1/sub_1:z:0*
T0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1/truediv|
scale_to_0_1/SigmoidSigmoidscale_to_0_1/Cast:y:0*
T0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1/Sigmoidµ
scale_to_0_1/SelectV2SelectV2scale_to_0_1/Cast_2:y:0scale_to_0_1/truediv:z:0scale_to_0_1/Sigmoid:y:0*
T0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1/SelectV2m
scale_to_0_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
scale_to_0_1/mul/y
scale_to_0_1/mulMulscale_to_0_1/SelectV2:output:0scale_to_0_1/mul/y:output:0*
T0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1/mulq
scale_to_0_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
scale_to_0_1/add_1/y
scale_to_0_1/add_1AddV2scale_to_0_1/mul:z:0scale_to_0_1/add_1/y:output:0*
T0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1/add_1q

Identity_1Identityscale_to_0_1/add_1:z:0^NoOp*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_1b
inputs_2_copyIdentityinputs_2*
T0	*#
_output_shapes
:’’’’’’’’’2
inputs_2_copy
scale_to_0_1_1/CastCastinputs_2_copy:output:0*

DstT0*

SrcT0	*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1_1/Cast
"scale_to_0_1_1/min_and_max/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"scale_to_0_1_1/min_and_max/sub_1/xØ
%scale_to_0_1_1/min_and_max/Identity_2Identity+scale_to_0_1_1_min_and_max_identity_2_input*
T0*
_output_shapes
: 2'
%scale_to_0_1_1/min_and_max/Identity_2É
 scale_to_0_1_1/min_and_max/sub_1Sub+scale_to_0_1_1/min_and_max/sub_1/x:output:0.scale_to_0_1_1/min_and_max/Identity_2:output:0*
T0*
_output_shapes
: 2"
 scale_to_0_1_1/min_and_max/sub_1
scale_to_0_1_1/subSubscale_to_0_1_1/Cast:y:0$scale_to_0_1_1/min_and_max/sub_1:z:0*
T0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1_1/sub
scale_to_0_1_1/zeros_like	ZerosLikescale_to_0_1_1/sub:z:0*
T0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1_1/zeros_likeØ
%scale_to_0_1_1/min_and_max/Identity_3Identity+scale_to_0_1_1_min_and_max_identity_3_input*
T0*
_output_shapes
: 2'
%scale_to_0_1_1/min_and_max/Identity_3©
scale_to_0_1_1/LessLess$scale_to_0_1_1/min_and_max/sub_1:z:0.scale_to_0_1_1/min_and_max/Identity_3:output:0*
T0*
_output_shapes
: 2
scale_to_0_1_1/Less
scale_to_0_1_1/Cast_1Castscale_to_0_1_1/Less:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2
scale_to_0_1_1/Cast_1
scale_to_0_1_1/addAddV2scale_to_0_1_1/zeros_like:y:0scale_to_0_1_1/Cast_1:y:0*
T0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1_1/add
scale_to_0_1_1/Cast_2Castscale_to_0_1_1/add:z:0*

DstT0
*

SrcT0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1_1/Cast_2Ŗ
scale_to_0_1_1/sub_1Sub.scale_to_0_1_1/min_and_max/Identity_3:output:0$scale_to_0_1_1/min_and_max/sub_1:z:0*
T0*
_output_shapes
: 2
scale_to_0_1_1/sub_1
scale_to_0_1_1/truedivRealDivscale_to_0_1_1/sub:z:0scale_to_0_1_1/sub_1:z:0*
T0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1_1/truediv
scale_to_0_1_1/SigmoidSigmoidscale_to_0_1_1/Cast:y:0*
T0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1_1/Sigmoidæ
scale_to_0_1_1/SelectV2SelectV2scale_to_0_1_1/Cast_2:y:0scale_to_0_1_1/truediv:z:0scale_to_0_1_1/Sigmoid:y:0*
T0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1_1/SelectV2q
scale_to_0_1_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
scale_to_0_1_1/mul/y
scale_to_0_1_1/mulMul scale_to_0_1_1/SelectV2:output:0scale_to_0_1_1/mul/y:output:0*
T0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1_1/mulu
scale_to_0_1_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
scale_to_0_1_1/add_1/y
scale_to_0_1_1/add_1AddV2scale_to_0_1_1/mul:z:0scale_to_0_1_1/add_1/y:output:0*
T0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1_1/add_1s

Identity_2Identityscale_to_0_1_1/add_1:z:0^NoOp*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_2b
inputs_3_copyIdentityinputs_3*
T0	*#
_output_shapes
:’’’’’’’’’2
inputs_3_copy
scale_to_0_1_3/CastCastinputs_3_copy:output:0*

DstT0*

SrcT0	*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1_3/Cast
"scale_to_0_1_3/min_and_max/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"scale_to_0_1_3/min_and_max/sub_1/xØ
%scale_to_0_1_3/min_and_max/Identity_2Identity+scale_to_0_1_3_min_and_max_identity_2_input*
T0*
_output_shapes
: 2'
%scale_to_0_1_3/min_and_max/Identity_2É
 scale_to_0_1_3/min_and_max/sub_1Sub+scale_to_0_1_3/min_and_max/sub_1/x:output:0.scale_to_0_1_3/min_and_max/Identity_2:output:0*
T0*
_output_shapes
: 2"
 scale_to_0_1_3/min_and_max/sub_1
scale_to_0_1_3/subSubscale_to_0_1_3/Cast:y:0$scale_to_0_1_3/min_and_max/sub_1:z:0*
T0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1_3/sub
scale_to_0_1_3/zeros_like	ZerosLikescale_to_0_1_3/sub:z:0*
T0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1_3/zeros_likeØ
%scale_to_0_1_3/min_and_max/Identity_3Identity+scale_to_0_1_3_min_and_max_identity_3_input*
T0*
_output_shapes
: 2'
%scale_to_0_1_3/min_and_max/Identity_3©
scale_to_0_1_3/LessLess$scale_to_0_1_3/min_and_max/sub_1:z:0.scale_to_0_1_3/min_and_max/Identity_3:output:0*
T0*
_output_shapes
: 2
scale_to_0_1_3/Less
scale_to_0_1_3/Cast_1Castscale_to_0_1_3/Less:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2
scale_to_0_1_3/Cast_1
scale_to_0_1_3/addAddV2scale_to_0_1_3/zeros_like:y:0scale_to_0_1_3/Cast_1:y:0*
T0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1_3/add
scale_to_0_1_3/Cast_2Castscale_to_0_1_3/add:z:0*

DstT0
*

SrcT0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1_3/Cast_2Ŗ
scale_to_0_1_3/sub_1Sub.scale_to_0_1_3/min_and_max/Identity_3:output:0$scale_to_0_1_3/min_and_max/sub_1:z:0*
T0*
_output_shapes
: 2
scale_to_0_1_3/sub_1
scale_to_0_1_3/truedivRealDivscale_to_0_1_3/sub:z:0scale_to_0_1_3/sub_1:z:0*
T0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1_3/truediv
scale_to_0_1_3/SigmoidSigmoidscale_to_0_1_3/Cast:y:0*
T0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1_3/Sigmoidæ
scale_to_0_1_3/SelectV2SelectV2scale_to_0_1_3/Cast_2:y:0scale_to_0_1_3/truediv:z:0scale_to_0_1_3/Sigmoid:y:0*
T0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1_3/SelectV2q
scale_to_0_1_3/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
scale_to_0_1_3/mul/y
scale_to_0_1_3/mulMul scale_to_0_1_3/SelectV2:output:0scale_to_0_1_3/mul/y:output:0*
T0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1_3/mulu
scale_to_0_1_3/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
scale_to_0_1_3/add_1/y
scale_to_0_1_3/add_1AddV2scale_to_0_1_3/mul:z:0scale_to_0_1_3/add_1/y:output:0*
T0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1_3/add_1s

Identity_3Identityscale_to_0_1_3/add_1:z:0^NoOp*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_3b
inputs_4_copyIdentityinputs_4*
T0	*#
_output_shapes
:’’’’’’’’’2
inputs_4_copy
scale_to_0_1_2/CastCastinputs_4_copy:output:0*

DstT0*

SrcT0	*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1_2/Cast
"scale_to_0_1_2/min_and_max/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"scale_to_0_1_2/min_and_max/sub_1/xØ
%scale_to_0_1_2/min_and_max/Identity_2Identity+scale_to_0_1_2_min_and_max_identity_2_input*
T0*
_output_shapes
: 2'
%scale_to_0_1_2/min_and_max/Identity_2É
 scale_to_0_1_2/min_and_max/sub_1Sub+scale_to_0_1_2/min_and_max/sub_1/x:output:0.scale_to_0_1_2/min_and_max/Identity_2:output:0*
T0*
_output_shapes
: 2"
 scale_to_0_1_2/min_and_max/sub_1
scale_to_0_1_2/subSubscale_to_0_1_2/Cast:y:0$scale_to_0_1_2/min_and_max/sub_1:z:0*
T0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1_2/sub
scale_to_0_1_2/zeros_like	ZerosLikescale_to_0_1_2/sub:z:0*
T0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1_2/zeros_likeØ
%scale_to_0_1_2/min_and_max/Identity_3Identity+scale_to_0_1_2_min_and_max_identity_3_input*
T0*
_output_shapes
: 2'
%scale_to_0_1_2/min_and_max/Identity_3©
scale_to_0_1_2/LessLess$scale_to_0_1_2/min_and_max/sub_1:z:0.scale_to_0_1_2/min_and_max/Identity_3:output:0*
T0*
_output_shapes
: 2
scale_to_0_1_2/Less
scale_to_0_1_2/Cast_1Castscale_to_0_1_2/Less:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2
scale_to_0_1_2/Cast_1
scale_to_0_1_2/addAddV2scale_to_0_1_2/zeros_like:y:0scale_to_0_1_2/Cast_1:y:0*
T0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1_2/add
scale_to_0_1_2/Cast_2Castscale_to_0_1_2/add:z:0*

DstT0
*

SrcT0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1_2/Cast_2Ŗ
scale_to_0_1_2/sub_1Sub.scale_to_0_1_2/min_and_max/Identity_3:output:0$scale_to_0_1_2/min_and_max/sub_1:z:0*
T0*
_output_shapes
: 2
scale_to_0_1_2/sub_1
scale_to_0_1_2/truedivRealDivscale_to_0_1_2/sub:z:0scale_to_0_1_2/sub_1:z:0*
T0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1_2/truediv
scale_to_0_1_2/SigmoidSigmoidscale_to_0_1_2/Cast:y:0*
T0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1_2/Sigmoidæ
scale_to_0_1_2/SelectV2SelectV2scale_to_0_1_2/Cast_2:y:0scale_to_0_1_2/truediv:z:0scale_to_0_1_2/Sigmoid:y:0*
T0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1_2/SelectV2q
scale_to_0_1_2/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
scale_to_0_1_2/mul/y
scale_to_0_1_2/mulMul scale_to_0_1_2/SelectV2:output:0scale_to_0_1_2/mul/y:output:0*
T0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1_2/mulu
scale_to_0_1_2/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
scale_to_0_1_2/add_1/y
scale_to_0_1_2/add_1AddV2scale_to_0_1_2/mul:z:0scale_to_0_1_2/add_1/y:output:0*
T0*#
_output_shapes
:’’’’’’’’’2
scale_to_0_1_2/add_1s

Identity_4Identityscale_to_0_1_2/add_1:z:0^NoOp*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : : : : : : :) %
#
_output_shapes
:’’’’’’’’’:)%
#
_output_shapes
:’’’’’’’’’:)%
#
_output_shapes
:’’’’’’’’’:)%
#
_output_shapes
:’’’’’’’’’:)%
#
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
®
l
__inference__traced_save_1117
file_prefix
savev2_const_8

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2/shape_and_slices¼
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_const_8"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2ŗ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes”
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*
_input_shapes
: : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: 
ģ
É
"__inference_signature_wrapper_1078

inputs
inputs_1	
inputs_2	
inputs_3	
inputs_4	
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity

identity_1

identity_2

identity_3

identity_4¢StatefulPartitionedCallŌ
StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2				*
Tout	
2*_
_output_shapesM
K:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 * 
fR
__inference_pruned_10432
StatefulPartitionedCallw
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:’’’’’’’’’2

Identity{

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_1{

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_2{

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_3{

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_4h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:MI
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_1:MI
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_2:MI
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_3:MI
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Æ
F
 __inference__traced_restore_1127
file_prefix

identity_1¤
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2/shape_and_slices°
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
:*
dtypes
22
	RestoreV29
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpd
IdentityIdentityfile_prefix^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"ØL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*’
serving_defaultė
5
inputs+
serving_default_inputs:0’’’’’’’’’
9
inputs_1-
serving_default_inputs_1:0	’’’’’’’’’
9
inputs_2-
serving_default_inputs_2:0	’’’’’’’’’
9
inputs_3-
serving_default_inputs_3:0	’’’’’’’’’
9
inputs_4-
serving_default_inputs_4:0	’’’’’’’’’6
Energy,
StatefulPartitionedCall:0’’’’’’’’’;
NormalizedC,
StatefulPartitionedCall:1’’’’’’’’’;
NormalizedH,
StatefulPartitionedCall:2’’’’’’’’’;
NormalizedN,
StatefulPartitionedCall:3’’’’’’’’’;
NormalizedO,
StatefulPartitionedCall:4’’’’’’’’’tensorflow/serving/predict:×

created_variables
	resources
trackable_objects
initializers

assets

signatures
transform_fn"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
,
serving_default"
signature_map
MBK
__inference_pruned_1043inputsinputs_1inputs_2inputs_3inputs_4
īBė
"__inference_signature_wrapper_1078inputsinputs_1inputs_2inputs_3inputs_4"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
J	
Const_5
J	
Const_6
J	
Const_7¢
__inference_pruned_1043	
¢ž
ö¢ņ
ļŖė
-
Energy# 
inputs/Energy’’’’’’’’’
-
TotalC# 
inputs/TotalC’’’’’’’’’	
-
TotalH# 
inputs/TotalH’’’’’’’’’	
-
TotalN# 
inputs/TotalN’’’’’’’’’	
-
TotalO# 
inputs/TotalO’’’’’’’’’	
Ŗ "ōŖš
&
Energy
Energy’’’’’’’’’
0
NormalizedC!
NormalizedC’’’’’’’’’
0
NormalizedH!
NormalizedH’’’’’’’’’
0
NormalizedN!
NormalizedN’’’’’’’’’
0
NormalizedO!
NormalizedO’’’’’’’’’
"__inference_signature_wrapper_1078ģ	
č¢ä
¢ 
ÜŖŲ
&
inputs
inputs’’’’’’’’’
*
inputs_1
inputs_1’’’’’’’’’	
*
inputs_2
inputs_2’’’’’’’’’	
*
inputs_3
inputs_3’’’’’’’’’	
*
inputs_4
inputs_4’’’’’’’’’	"ōŖš
&
Energy
Energy’’’’’’’’’
0
NormalizedC!
NormalizedC’’’’’’’’’
0
NormalizedH!
NormalizedH’’’’’’’’’
0
NormalizedN!
NormalizedN’’’’’’’’’
0
NormalizedO!
NormalizedO’’’’’’’’’