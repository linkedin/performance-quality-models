˲
?4?3
:
Add
x"T
y"T
z"T"
Ttype:
2	
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
?
ArgMax

input"T
	dimension"Tidx
output"output_type"!
Ttype:
2	
"
Tidxtype0:
2	"
output_typetype0	:
2	
?
AsString

input"T

output"
Ttype:
2		
"
	precisionint?????????"

scientificbool( "
shortestbool( "
widthint?????????"
fillstring 
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%??L>"
Ttype0:
2
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
2
LookupTableSizeV2
table_handle
size	?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
ParseExampleV2

serialized	
names
sparse_keys

dense_keys
ragged_keys
dense_defaults2Tdense
sparse_indices	*
num_sparse
sparse_values2sparse_types
sparse_shapes	*
num_sparse
dense_values2Tdense#
ragged_values2ragged_value_types'
ragged_row_splits2ragged_split_types"
Tdense
list(type)(:
2	"

num_sparseint("%
sparse_types
list(type)(:
2	"+
ragged_value_types
list(type)(:
2	"*
ragged_split_types
list(type)(:
2	"
dense_shapeslist(shape)(
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
b
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:

2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
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
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
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
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
SparseFillEmptyRows
indices	
values"T
dense_shape	
default_value"T
output_indices	
output_values"T
empty_row_indicator

reverse_index_map	"	
Ttype
h
SparseReshape
input_indices	
input_shape	
	new_shape	
output_indices	
output_shape	
?
SparseSegmentMean	
data"T
indices"Tidx
segment_ids"Tsegmentids
output"T"
Ttype:
2"
Tidxtype0:
2	"
Tsegmentidstype0:
2	
?
StatelessIf
cond"Tcond
input2Tin
output2Tout"
Tcondtype"
Tin
list(type)("
Tout
list(type)("
then_branchfunc"
else_branchfunc" 
output_shapeslist(shape)
 
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
G
StringToHashBucketFast	
input

output	"
num_bucketsint(0
;
Sub
x"T
y"T
z"T"
Ttype:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
?
TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
P
Unique
x"T
y"T
idx"out_idx"	
Ttype"
out_idxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
9
VarIsInitializedOp
resource
is_initialized
?
E
Where

input"T	
index	"%
Ttype0
:
2	

&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8??

global_step/Initializer/zerosConst*
_class
loc:@global_step*
_output_shapes
: *
dtype0	*
value	B	 R 
?
global_stepVarHandleOp*
_class
loc:@global_step*
_output_shapes
: *
dtype0	*
shape: *
shared_nameglobal_step
g
,global_step/IsInitialized/VarIsInitializedOpVarIsInitializedOpglobal_step*
_output_shapes
: 
_
global_step/AssignAssignVariableOpglobal_stepglobal_step/Initializer/zeros*
dtype0	
c
global_step/Read/ReadVariableOpReadVariableOpglobal_step*
_output_shapes
: *
dtype0	
o
input_example_tensorPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
U
ParseExample/ConstConst*
_output_shapes
: *
dtype0*
valueB 
W
ParseExample/Const_1Const*
_output_shapes
: *
dtype0*
valueB 
d
!ParseExample/ParseExampleV2/namesConst*
_output_shapes
: *
dtype0*
valueB 
?
'ParseExample/ParseExampleV2/sparse_keysConst*
_output_shapes
:*
dtype0*k
valuebB`B
asn_numberBbrowser_major_version_naBbrowser_nameBcountry_codeBosfamilyB
osmajor_na
?
&ParseExample/ParseExampleV2/dense_keysConst*
_output_shapes
:*
dtype0*3
value*B(Bbrowser_major_versionBosmajor
j
'ParseExample/ParseExampleV2/ragged_keysConst*
_output_shapes
: *
dtype0*
valueB 
?
ParseExample/ParseExampleV2ParseExampleV2input_example_tensor!ParseExample/ParseExampleV2/names'ParseExample/ParseExampleV2/sparse_keys&ParseExample/ParseExampleV2/dense_keys'ParseExample/ParseExampleV2/ragged_keysParseExample/ConstParseExample/Const_1*
Tdense
2*?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:::::::?????????:?????????*
dense_shapes
::*

num_sparse*
ragged_split_types
 *
ragged_value_types
 *
sparse_types

2
?
tdnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*d
_classZ
XVloc:@dnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights*
_output_shapes
:*
dtype0*
valueB"?  2   
?
sdnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights/Initializer/truncated_normal/meanConst*d
_classZ
XVloc:@dnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights*
_output_shapes
: *
dtype0*
valueB
 *    
?
udnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*d
_classZ
XVloc:@dnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights*
_output_shapes
: *
dtype0*
valueB
 *??>
?
~dnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormaltdnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights/Initializer/truncated_normal/shape*
T0*d
_classZ
XVloc:@dnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights*
_output_shapes
:	?2*
dtype0*
seed????
?
rdnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights/Initializer/truncated_normal/mulMul~dnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormaludnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights/Initializer/truncated_normal/stddev*
T0*d
_classZ
XVloc:@dnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights*
_output_shapes
:	?2
?
ndnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights/Initializer/truncated_normalAddrdnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights/Initializer/truncated_normal/mulsdnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights/Initializer/truncated_normal/mean*
T0*d
_classZ
XVloc:@dnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights*
_output_shapes
:	?2
?
Qdnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weightsVarHandleOp*d
_classZ
XVloc:@dnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights*
_output_shapes
: *
dtype0*
shape:	?2*b
shared_nameSQdnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights
?
rdnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights/IsInitialized/VarIsInitializedOpVarIsInitializedOpQdnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights*
_output_shapes
: 
?
Xdnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights/AssignAssignVariableOpQdnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weightsndnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights/Initializer/truncated_normal*
dtype0
?
ednn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights/Read/ReadVariableOpReadVariableOpQdnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights*
_output_shapes
:	?2*
dtype0
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*r
_classh
fdloc:@dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights*
_output_shapes
:*
dtype0*
valueB"      
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights/Initializer/truncated_normal/meanConst*r
_classh
fdloc:@dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights*
_output_shapes
: *
dtype0*
valueB
 *    
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*r
_classh
fdloc:@dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights*
_output_shapes
: *
dtype0*
valueB
 *?5?
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights/Initializer/truncated_normal/shape*
T0*r
_classh
fdloc:@dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights*
_output_shapes

:*
dtype0*
seed????*
seed2
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights/Initializer/truncated_normal/mulMul?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormal?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights/Initializer/truncated_normal/stddev*
T0*r
_classh
fdloc:@dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights*
_output_shapes

:
?
|dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights/Initializer/truncated_normalAdd?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights/Initializer/truncated_normal/mul?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights/Initializer/truncated_normal/mean*
T0*r
_classh
fdloc:@dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights*
_output_shapes

:
?
_dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weightsVarHandleOp*r
_classh
fdloc:@dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights*
_output_shapes
: *
dtype0*
shape
:*p
shared_namea_dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp_dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights*
_output_shapes
: 
?
fdnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights/AssignAssignVariableOp_dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights|dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights/Initializer/truncated_normal*
dtype0
?
sdnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights/Read/ReadVariableOpReadVariableOp_dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights*
_output_shapes

:*
dtype0
?
vdnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*f
_class\
ZXloc:@dnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights*
_output_shapes
:*
dtype0*
valueB"	      
?
udnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights/Initializer/truncated_normal/meanConst*f
_class\
ZXloc:@dnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights*
_output_shapes
: *
dtype0*
valueB
 *    
?
wdnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*f
_class\
ZXloc:@dnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights*
_output_shapes
: *
dtype0*
valueB
 *??>
?
?dnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalvdnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights/Initializer/truncated_normal/shape*
T0*f
_class\
ZXloc:@dnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights*
_output_shapes

:	*
dtype0*
seed????*
seed2
?
tdnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights/Initializer/truncated_normal/mulMul?dnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalwdnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights/Initializer/truncated_normal/stddev*
T0*f
_class\
ZXloc:@dnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights*
_output_shapes

:	
?
pdnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights/Initializer/truncated_normalAddtdnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights/Initializer/truncated_normal/muludnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights/Initializer/truncated_normal/mean*
T0*f
_class\
ZXloc:@dnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights*
_output_shapes

:	
?
Sdnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weightsVarHandleOp*f
_class\
ZXloc:@dnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights*
_output_shapes
: *
dtype0*
shape
:	*d
shared_nameUSdnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights
?
tdnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights/IsInitialized/VarIsInitializedOpVarIsInitializedOpSdnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights*
_output_shapes
: 
?
Zdnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights/AssignAssignVariableOpSdnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weightspdnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights/Initializer/truncated_normal*
dtype0
?
gdnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights/Read/ReadVariableOpReadVariableOpSdnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights*
_output_shapes

:	*
dtype0
?
vdnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*f
_class\
ZXloc:@dnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights*
_output_shapes
:*
dtype0*
valueB"?   2   
?
udnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights/Initializer/truncated_normal/meanConst*f
_class\
ZXloc:@dnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights*
_output_shapes
: *
dtype0*
valueB
 *    
?
wdnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*f
_class\
ZXloc:@dnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights*
_output_shapes
: *
dtype0*
valueB
 *??>
?
?dnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalvdnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights/Initializer/truncated_normal/shape*
T0*f
_class\
ZXloc:@dnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights*
_output_shapes
:	?2*
dtype0*
seed????*
seed2
?
tdnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights/Initializer/truncated_normal/mulMul?dnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalwdnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights/Initializer/truncated_normal/stddev*
T0*f
_class\
ZXloc:@dnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights*
_output_shapes
:	?2
?
pdnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights/Initializer/truncated_normalAddtdnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights/Initializer/truncated_normal/muludnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights/Initializer/truncated_normal/mean*
T0*f
_class\
ZXloc:@dnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights*
_output_shapes
:	?2
?
Sdnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weightsVarHandleOp*f
_class\
ZXloc:@dnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights*
_output_shapes
: *
dtype0*
shape:	?2*d
shared_nameUSdnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights
?
tdnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights/IsInitialized/VarIsInitializedOpVarIsInitializedOpSdnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights*
_output_shapes
: 
?
Zdnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights/AssignAssignVariableOpSdnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weightspdnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights/Initializer/truncated_normal*
dtype0
?
gdnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights/Read/ReadVariableOpReadVariableOpSdnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights*
_output_shapes
:	?2*
dtype0
?
rdnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*b
_classX
VTloc:@dnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights*
_output_shapes
:*
dtype0*
valueB"      
?
qdnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights/Initializer/truncated_normal/meanConst*b
_classX
VTloc:@dnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights*
_output_shapes
: *
dtype0*
valueB
 *    
?
sdnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*b
_classX
VTloc:@dnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights*
_output_shapes
: *
dtype0*
valueB
 *?_?>
?
|dnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalrdnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights/Initializer/truncated_normal/shape*
T0*b
_classX
VTloc:@dnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights*
_output_shapes

:*
dtype0*
seed????*
seed2
?
pdnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights/Initializer/truncated_normal/mulMul|dnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalsdnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights/Initializer/truncated_normal/stddev*
T0*b
_classX
VTloc:@dnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights*
_output_shapes

:
?
ldnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights/Initializer/truncated_normalAddpdnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights/Initializer/truncated_normal/mulqdnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights/Initializer/truncated_normal/mean*
T0*b
_classX
VTloc:@dnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights*
_output_shapes

:
?
Odnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weightsVarHandleOp*b
_classX
VTloc:@dnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights*
_output_shapes
: *
dtype0*
shape
:*`
shared_nameQOdnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights
?
pdnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights/IsInitialized/VarIsInitializedOpVarIsInitializedOpOdnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights*
_output_shapes
: 
?
Vdnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights/AssignAssignVariableOpOdnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weightsldnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights/Initializer/truncated_normal*
dtype0
?
cdnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights/Read/ReadVariableOpReadVariableOpOdnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights*
_output_shapes

:*
dtype0
?
tdnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*d
_classZ
XVloc:@dnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights*
_output_shapes
:*
dtype0*
valueB"      
?
sdnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights/Initializer/truncated_normal/meanConst*d
_classZ
XVloc:@dnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights*
_output_shapes
: *
dtype0*
valueB
 *    
?
udnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*d
_classZ
XVloc:@dnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights*
_output_shapes
: *
dtype0*
valueB
 *?5?
?
~dnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormaltdnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights/Initializer/truncated_normal/shape*
T0*d
_classZ
XVloc:@dnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights*
_output_shapes

:*
dtype0*
seed????*
seed2
?
rdnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights/Initializer/truncated_normal/mulMul~dnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormaludnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights/Initializer/truncated_normal/stddev*
T0*d
_classZ
XVloc:@dnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights*
_output_shapes

:
?
ndnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights/Initializer/truncated_normalAddrdnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights/Initializer/truncated_normal/mulsdnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights/Initializer/truncated_normal/mean*
T0*d
_classZ
XVloc:@dnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights*
_output_shapes

:
?
Qdnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weightsVarHandleOp*d
_classZ
XVloc:@dnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights*
_output_shapes
: *
dtype0*
shape
:*b
shared_nameSQdnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights
?
rdnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights/IsInitialized/VarIsInitializedOpVarIsInitializedOpQdnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights*
_output_shapes
: 
?
Xdnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights/AssignAssignVariableOpQdnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weightsndnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights/Initializer/truncated_normal*
dtype0
?
ednn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights/Read/ReadVariableOpReadVariableOpQdnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights*
_output_shapes

:*
dtype0
?
Hdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/lookupStringToHashBucketFastParseExample/ParseExampleV2:6*#
_output_shapes
:?????????*
num_buckets?
?
jdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 
?
idnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?
ddnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/SliceSliceParseExample/ParseExampleV2:12jdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Slice/beginidnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Slice/size*
Index0*
T0	*
_output_shapes
:
?
ddnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
cdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/ProdProdddnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Sliceddnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Const*
T0	*
_output_shapes
: 
?
odnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :
?
ldnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
gdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/GatherV2GatherV2ParseExample/ParseExampleV2:12odnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/GatherV2/indicesldnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 
?
ednn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Cast/xPackcdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Prodgdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/GatherV2*
N*
T0	*
_output_shapes
:
?
ldnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/SparseReshapeSparseReshapeParseExample/ParseExampleV2ParseExample/ParseExampleV2:12ednn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Cast/x*-
_output_shapes
:?????????:
?
udnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/SparseReshape/IdentityIdentityHdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/lookup*
T0	*#
_output_shapes
:?????????
?
mdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
kdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/GreaterEqualGreaterEqualudnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/SparseReshape/Identitymdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:?????????
?
ddnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/WhereWherekdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/GreaterEqual*'
_output_shapes
:?????????
?
ldnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
fdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/ReshapeReshapeddnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Whereldnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:?????????
?
ndnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
idnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/GatherV2_1GatherV2ldnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/SparseReshapefdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Reshapendnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/GatherV2_1/axis*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:?????????
?
ndnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
idnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/GatherV2_2GatherV2udnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/SparseReshape/Identityfdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Reshapendnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/GatherV2_2/axis*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????
?
gdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/IdentityIdentityndnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
?
xdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
?dnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsidnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/GatherV2_1idnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/GatherV2_2gdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Identityxdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:?????????
?
?dnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
?
?dnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
?
?dnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
?
?dnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlice?dnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows?dnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/embedding_lookup_sparse/strided_slice/stack?dnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1?dnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask
?
}dnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/embedding_lookup_sparse/UniqueUnique?dnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:?????????:?????????
?
?dnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGatherQdnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights}dnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/embedding_lookup_sparse/Unique*
Tindices0	*d
_classZ
XVloc:@dnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights*'
_output_shapes
:?????????2*
dtype0
?
?dnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity?dnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*d
_classZ
XVloc:@dnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights*'
_output_shapes
:?????????2
?
?dnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1Identity?dnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity*
T0*'
_output_shapes
:?????????2
?
vdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/embedding_lookup_sparseSparseSegmentMean?dnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1dnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/embedding_lookup_sparse/Unique:1?dnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/embedding_lookup_sparse/strided_slice*
T0*
Tsegmentids0	*'
_output_shapes
:?????????2
?
ndnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   
?
hdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Reshape_1Reshape?dnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2ndnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:?????????
?
ddnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/ShapeShapevdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
?
rdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?
tdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
tdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
ldnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/strided_sliceStridedSliceddnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Shaperdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/strided_slice/stacktdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/strided_slice/stack_1tdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
fdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :
?
ddnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/stackPackfdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/stack/0ldnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/strided_slice*
N*
T0*
_output_shapes
:
?
cdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/TileTilehdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Reshape_1ddnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/stack*
T0
*0
_output_shapes
:??????????????????
?
idnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/zeros_like	ZerosLikevdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????2
?
^dnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weightsSelectcdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Tileidnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/zeros_likevdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????2
?
ednn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Cast_1CastParseExample/ParseExampleV2:12*

DstT0*

SrcT0	*
_output_shapes
:
?
ldnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 
?
kdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?
fdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Slice_1Sliceednn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Cast_1ldnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Slice_1/beginkdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
?
fdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Shape_1Shape^dnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights*
T0*
_output_shapes
:
?
ldnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:
?
kdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
fdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Slice_2Slicefdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Shape_1ldnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Slice_2/beginkdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
?
jdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
ednn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/concatConcatV2fdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Slice_1fdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Slice_2jdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/concat/axis*
N*
T0*
_output_shapes
:
?
hdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Reshape_2Reshape^dnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weightsednn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/concat*
T0*'
_output_shapes
:?????????2
?
Gdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/ShapeShapehdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Reshape_2*
T0*
_output_shapes
:
?
Udnn/input_from_feature_columns/input_layer/asn_number_embedding_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
Wdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
Wdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Odnn/input_from_feature_columns/input_layer/asn_number_embedding_1/strided_sliceStridedSliceGdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/ShapeUdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/strided_slice/stackWdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/strided_slice/stack_1Wdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Qdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
?
Odnn/input_from_feature_columns/input_layer/asn_number_embedding_1/Reshape/shapePackOdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/strided_sliceQdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/Reshape/shape/1*
N*
T0*
_output_shapes
:
?
Idnn/input_from_feature_columns/input_layer/asn_number_embedding_1/ReshapeReshapehdnn/input_from_feature_columns/input_layer/asn_number_embedding_1/asn_number_embedding_weights/Reshape_2Odnn/input_from_feature_columns/input_layer/asn_number_embedding_1/Reshape/shape*
T0*'
_output_shapes
:?????????2
?
Hdnn/input_from_feature_columns/input_layer/browser_major_version_1/ShapeShapeParseExample/ParseExampleV2:18*
T0*
_output_shapes
:
?
Vdnn/input_from_feature_columns/input_layer/browser_major_version_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
Xdnn/input_from_feature_columns/input_layer/browser_major_version_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
Xdnn/input_from_feature_columns/input_layer/browser_major_version_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Pdnn/input_from_feature_columns/input_layer/browser_major_version_1/strided_sliceStridedSliceHdnn/input_from_feature_columns/input_layer/browser_major_version_1/ShapeVdnn/input_from_feature_columns/input_layer/browser_major_version_1/strided_slice/stackXdnn/input_from_feature_columns/input_layer/browser_major_version_1/strided_slice/stack_1Xdnn/input_from_feature_columns/input_layer/browser_major_version_1/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Rdnn/input_from_feature_columns/input_layer/browser_major_version_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
?
Pdnn/input_from_feature_columns/input_layer/browser_major_version_1/Reshape/shapePackPdnn/input_from_feature_columns/input_layer/browser_major_version_1/strided_sliceRdnn/input_from_feature_columns/input_layer/browser_major_version_1/Reshape/shape/1*
N*
T0*
_output_shapes
:
?
Jdnn/input_from_feature_columns/input_layer/browser_major_version_1/ReshapeReshapeParseExample/ParseExampleV2:18Pdnn/input_from_feature_columns/input_layer/browser_major_version_1/Reshape/shape*
T0*'
_output_shapes
:?????????
?
udnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_lookup/ConstConst*
_output_shapes
:*
dtype0* 
valueBBFalseBTrue
?
tdnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_lookup/SizeConst*
_output_shapes
: *
dtype0*
value	B :
?
{dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_lookup/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
?
{dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_lookup/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
?
udnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_lookup/rangeRange{dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_lookup/range/starttdnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_lookup/Size{dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_lookup/range/delta*
_output_shapes
:
?
tdnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_lookup/CastCastudnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_lookup/range*

DstT0	*

SrcT0*
_output_shapes
:
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_lookup/hash_table/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_lookup/hash_table/hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*@
shared_name1/hash_table_f24a85a5-aa16-4ca9-bc06-55af4c7537e8*
value_dtype0	
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_lookup/hash_table/table_init/LookupTableImportV2LookupTableImportV2?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_lookup/hash_table/hash_tableudnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_lookup/Consttdnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_lookup/Cast*	
Tin0*

Tout0	
?
mdnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/hash_table_Lookup/hash_bucketStringToHashBucketFastParseExample/ParseExampleV2:7*#
_output_shapes
:?????????*
num_buckets
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/hash_table_Lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_lookup/hash_table/hash_tableParseExample/ParseExampleV2:7?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_lookup/hash_table/Const*	
Tin0*

Tout0	*#
_output_shapes
:?????????
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/hash_table_Lookup/hash_table_Size/LookupTableSizeV2LookupTableSizeV2?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_lookup/hash_table/hash_table*
_output_shapes
: 
?
ednn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/hash_table_Lookup/AddAddmdnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/hash_table_Lookup/hash_bucket?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/hash_table_Lookup/hash_table_Size/LookupTableSizeV2*
T0	*#
_output_shapes
:?????????
?
jdnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/hash_table_Lookup/NotEqualNotEqual?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/hash_table_Lookup/hash_table_Lookup/LookupTableFindV2?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_lookup/hash_table/Const*
T0	*#
_output_shapes
:?????????
?
jdnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/hash_table_Lookup/SelectV2SelectV2jdnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/hash_table_Lookup/NotEqual?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/hash_table_Lookup/hash_table_Lookup/LookupTableFindV2ednn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/hash_table_Lookup/Add*
T0	*#
_output_shapes
:?????????
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/SliceSliceParseExample/ParseExampleV2:13?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Slice/begin?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Slice/size*
Index0*
T0	*
_output_shapes
:
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/ProdProd?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Slice?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Const*
T0	*
_output_shapes
: 
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/GatherV2GatherV2ParseExample/ParseExampleV2:13?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/GatherV2/indices?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Cast/xPackdnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Prod?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/GatherV2*
N*
T0	*
_output_shapes
:
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/SparseReshapeSparseReshapeParseExample/ParseExampleV2:1ParseExample/ParseExampleV2:13?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Cast/x*-
_output_shapes
:?????????:
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/SparseReshape/IdentityIdentityjdnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/hash_table_Lookup/SelectV2*
T0	*#
_output_shapes
:?????????
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/GreaterEqualGreaterEqual?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/SparseReshape/Identity?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:?????????
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/WhereWhere?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/GreaterEqual*'
_output_shapes
:?????????
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/ReshapeReshape?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Where?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:?????????
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/GatherV2_1GatherV2?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/SparseReshape?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Reshape?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/GatherV2_1/axis*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:?????????
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/GatherV2_2GatherV2?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/SparseReshape/Identity?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Reshape?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/GatherV2_2/axis*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/IdentityIdentity?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/GatherV2_1?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/GatherV2_2?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Identity?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:?????????
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlice?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/embedding_lookup_sparse/strided_slice/stack?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/embedding_lookup_sparse/UniqueUnique?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:?????????:?????????
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGather_dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/embedding_lookup_sparse/Unique*
Tindices0	*r
_classh
fdloc:@dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights*'
_output_shapes
:?????????*
dtype0
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*r
_classh
fdloc:@dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights*'
_output_shapes
:?????????
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1Identity?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity*
T0*'
_output_shapes
:?????????
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/embedding_lookup_sparseSparseSegmentMean?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/embedding_lookup_sparse/Unique:1?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/embedding_lookup_sparse/strided_slice*
T0*
Tsegmentids0	*'
_output_shapes
:?????????
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Reshape_1Reshape?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:?????????
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/ShapeShape?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/strided_sliceStridedSlice?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Shape?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/strided_slice/stack?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/strided_slice/stack_1?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/stackPack?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/stack/0?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/strided_slice*
N*
T0*
_output_shapes
:
?
dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/TileTile?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Reshape_1?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/stack*
T0
*0
_output_shapes
:??????????????????
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/zeros_like	ZerosLike?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
zdnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weightsSelectdnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Tile?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/zeros_like?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Cast_1CastParseExample/ParseExampleV2:13*

DstT0*

SrcT0	*
_output_shapes
:
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Slice_1Slice?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Cast_1?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Slice_1/begin?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Shape_1Shapezdnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights*
T0*
_output_shapes
:
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Slice_2Slice?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Shape_1?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Slice_2/begin?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/concatConcatV2?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Slice_1?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Slice_2?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/concat/axis*
N*
T0*
_output_shapes
:
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Reshape_2Reshapezdnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/concat*
T0*'
_output_shapes
:?????????
?
Udnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/ShapeShape?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Reshape_2*
T0*
_output_shapes
:
?
cdnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
ednn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
ednn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
]dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/strided_sliceStridedSliceUdnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/Shapecdnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/strided_slice/stackednn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/strided_slice/stack_1ednn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
_dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
?
]dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/Reshape/shapePack]dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/strided_slice_dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/Reshape/shape/1*
N*
T0*
_output_shapes
:
?
Wdnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/ReshapeReshape?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_embedding_weights/Reshape_2]dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/Reshape/shape*
T0*'
_output_shapes
:?????????
?
]dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_lookup/ConstConst*
_output_shapes
:*
dtype0*`
valueWBUBchromeBfirefoxBibrowseBiemobileBmobile iphoneBmozillaBsafariBunknown
?
\dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_lookup/SizeConst*
_output_shapes
: *
dtype0*
value	B :
?
cdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_lookup/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
?
cdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_lookup/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
?
]dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_lookup/rangeRangecdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_lookup/range/start\dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_lookup/Sizecdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_lookup/range/delta*
_output_shapes
:
?
\dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_lookup/CastCast]dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_lookup/range*

DstT0	*

SrcT0*
_output_shapes
:
?
hdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_lookup/hash_table/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
?
mdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_lookup/hash_table/hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*@
shared_name1/hash_table_cec3afeb-cf2d-4c0c-9536-878dc88b71e6*
value_dtype0	
?
?dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_lookup/hash_table/table_init/LookupTableImportV2LookupTableImportV2mdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_lookup/hash_table/hash_table]dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_lookup/Const\dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_lookup/Cast*	
Tin0*

Tout0	
?
adnn/input_from_feature_columns/input_layer/browser_name_embedding_1/hash_table_Lookup/hash_bucketStringToHashBucketFastParseExample/ParseExampleV2:8*#
_output_shapes
:?????????*
num_buckets
?
ydnn/input_from_feature_columns/input_layer/browser_name_embedding_1/hash_table_Lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2mdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_lookup/hash_table/hash_tableParseExample/ParseExampleV2:8hdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_lookup/hash_table/Const*	
Tin0*

Tout0	*#
_output_shapes
:?????????
?
wdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/hash_table_Lookup/hash_table_Size/LookupTableSizeV2LookupTableSizeV2mdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_lookup/hash_table/hash_table*
_output_shapes
: 
?
Ydnn/input_from_feature_columns/input_layer/browser_name_embedding_1/hash_table_Lookup/AddAddadnn/input_from_feature_columns/input_layer/browser_name_embedding_1/hash_table_Lookup/hash_bucketwdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/hash_table_Lookup/hash_table_Size/LookupTableSizeV2*
T0	*#
_output_shapes
:?????????
?
^dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/hash_table_Lookup/NotEqualNotEqualydnn/input_from_feature_columns/input_layer/browser_name_embedding_1/hash_table_Lookup/hash_table_Lookup/LookupTableFindV2hdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_lookup/hash_table/Const*
T0	*#
_output_shapes
:?????????
?
^dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/hash_table_Lookup/SelectV2SelectV2^dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/hash_table_Lookup/NotEqualydnn/input_from_feature_columns/input_layer/browser_name_embedding_1/hash_table_Lookup/hash_table_Lookup/LookupTableFindV2Ydnn/input_from_feature_columns/input_layer/browser_name_embedding_1/hash_table_Lookup/Add*
T0	*#
_output_shapes
:?????????
?
ndnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 
?
mdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?
hdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/SliceSliceParseExample/ParseExampleV2:14ndnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Slice/beginmdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Slice/size*
Index0*
T0	*
_output_shapes
:
?
hdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
gdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/ProdProdhdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Slicehdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Const*
T0	*
_output_shapes
: 
?
sdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :
?
pdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
kdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/GatherV2GatherV2ParseExample/ParseExampleV2:14sdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/GatherV2/indicespdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 
?
idnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Cast/xPackgdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Prodkdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/GatherV2*
N*
T0	*
_output_shapes
:
?
pdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/SparseReshapeSparseReshapeParseExample/ParseExampleV2:2ParseExample/ParseExampleV2:14idnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Cast/x*-
_output_shapes
:?????????:
?
ydnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/SparseReshape/IdentityIdentity^dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/hash_table_Lookup/SelectV2*
T0	*#
_output_shapes
:?????????
?
qdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
odnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/GreaterEqualGreaterEqualydnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/SparseReshape/Identityqdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:?????????
?
hdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/WhereWhereodnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/GreaterEqual*'
_output_shapes
:?????????
?
pdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
jdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/ReshapeReshapehdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Wherepdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:?????????
?
rdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
mdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/GatherV2_1GatherV2pdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/SparseReshapejdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Reshaperdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/GatherV2_1/axis*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:?????????
?
rdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
mdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/GatherV2_2GatherV2ydnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/SparseReshape/Identityjdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Reshaperdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/GatherV2_2/axis*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????
?
kdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/IdentityIdentityrdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
?
|dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
?dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsmdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/GatherV2_1mdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/GatherV2_2kdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Identity|dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:?????????
?
?dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
?
?dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
?
?dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
?
?dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlice?dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows?dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/embedding_lookup_sparse/strided_slice/stack?dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1?dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask
?
?dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/embedding_lookup_sparse/UniqueUnique?dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:?????????:?????????
?
?dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGatherSdnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights?dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/embedding_lookup_sparse/Unique*
Tindices0	*f
_class\
ZXloc:@dnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights*'
_output_shapes
:?????????*
dtype0
?
?dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity?dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*f
_class\
ZXloc:@dnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights*'
_output_shapes
:?????????
?
?dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1Identity?dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity*
T0*'
_output_shapes
:?????????
?
zdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/embedding_lookup_sparseSparseSegmentMean?dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1?dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/embedding_lookup_sparse/Unique:1?dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/embedding_lookup_sparse/strided_slice*
T0*
Tsegmentids0	*'
_output_shapes
:?????????
?
rdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   
?
ldnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Reshape_1Reshape?dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2rdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:?????????
?
hdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/ShapeShapezdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
?
vdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?
xdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
xdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
pdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/strided_sliceStridedSlicehdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Shapevdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/strided_slice/stackxdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/strided_slice/stack_1xdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
jdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :
?
hdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/stackPackjdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/stack/0pdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/strided_slice*
N*
T0*
_output_shapes
:
?
gdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/TileTileldnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Reshape_1hdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/stack*
T0
*0
_output_shapes
:??????????????????
?
mdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/zeros_like	ZerosLikezdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
bdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weightsSelectgdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Tilemdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/zeros_likezdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
idnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Cast_1CastParseExample/ParseExampleV2:14*

DstT0*

SrcT0	*
_output_shapes
:
?
pdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 
?
odnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?
jdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Slice_1Sliceidnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Cast_1pdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Slice_1/beginodnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
?
jdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Shape_1Shapebdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights*
T0*
_output_shapes
:
?
pdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:
?
odnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
jdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Slice_2Slicejdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Shape_1pdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Slice_2/beginodnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
?
ndnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
idnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/concatConcatV2jdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Slice_1jdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Slice_2ndnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/concat/axis*
N*
T0*
_output_shapes
:
?
ldnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Reshape_2Reshapebdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weightsidnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/concat*
T0*'
_output_shapes
:?????????
?
Idnn/input_from_feature_columns/input_layer/browser_name_embedding_1/ShapeShapeldnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Reshape_2*
T0*
_output_shapes
:
?
Wdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
Ydnn/input_from_feature_columns/input_layer/browser_name_embedding_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
Ydnn/input_from_feature_columns/input_layer/browser_name_embedding_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Qdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/strided_sliceStridedSliceIdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/ShapeWdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/strided_slice/stackYdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/strided_slice/stack_1Ydnn/input_from_feature_columns/input_layer/browser_name_embedding_1/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Sdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
?
Qdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/Reshape/shapePackQdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/strided_sliceSdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/Reshape/shape/1*
N*
T0*
_output_shapes
:
?
Kdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/ReshapeReshapeldnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_embedding_weights/Reshape_2Qdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/Reshape/shape*
T0*'
_output_shapes
:?????????
?
]dnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_lookup/ConstConst*
_output_shapes	
:?*
dtype0*?
value?B??B**BadBaeBafBagBaiBalBamBaoBarBatBauBawBaxBazBbaBbbBbdBbeBbfBbgBbhBbiBbjBblBbmBbnBboBbqBbrBbsBbtBbwBbyBbzBcaBcdBcfBcgBchBciBckBclBcmBcnBcoBcrBcuBcvBcwBcxBcyBczBdeBdjBdkBdmBdoBdzBecBeeBegBerBesBetBeuBfiBfjBfkBfmBfoBfrBgaBgbBgdBgeBgfBggBghBgiBglBgmBgnBgpBgqBgrBgtBguBgwBgyBhkBhnBhrBhtBhuBidBieBilBimBinBioBiqBirBisBitBjeBjmBjoBjpBkeBkgBkhBkiBkmBknBkrBkwBkyBkzBlaBlbBlcBliBlkBlrBlsBltBluBlvBlyBmaBmcBmdBmeBmfBmgBmhBmkBmlBmmBmnBmoBmpBmqBmrBmsBmtBmuBmvBmwBmxBmyBmzBnaBncBneBnfBngBniBnlBnoBnpBnrBnuBnzBomBpaBpeBpfBpgBphBpkBplBpmBprBpsBptBpwBpyBqaBreBroBrsBruBrwBsaBsbBscBsdBseBsgBsiBskBslBsmBsnBsoBsrBssBstBsvBsxBsyBszBtcBtdBtgBthBtjBtlBtmBtnBtoBtrBttBtvBtwBtzBuaBugBusBuyBuzBvaBvcBveBvgBviBvnBvuBwfBwsByeBytBzaBzmBzw
?
\dnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_lookup/SizeConst*
_output_shapes
: *
dtype0*
value
B :?
?
cdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_lookup/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
?
cdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_lookup/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
?
]dnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_lookup/rangeRangecdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_lookup/range/start\dnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_lookup/Sizecdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_lookup/range/delta*
_output_shapes	
:?
?
\dnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_lookup/CastCast]dnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_lookup/range*

DstT0	*

SrcT0*
_output_shapes	
:?
?
hdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_lookup/hash_table/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
?
mdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_lookup/hash_table/hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*@
shared_name1/hash_table_e4036e51-c08b-475a-b929-e07f2df57317*
value_dtype0	
?
?dnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_lookup/hash_table/table_init/LookupTableImportV2LookupTableImportV2mdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_lookup/hash_table/hash_table]dnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_lookup/Const\dnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_lookup/Cast*	
Tin0*

Tout0	
?
adnn/input_from_feature_columns/input_layer/country_code_embedding_1/hash_table_Lookup/hash_bucketStringToHashBucketFastParseExample/ParseExampleV2:9*#
_output_shapes
:?????????*
num_buckets
?
ydnn/input_from_feature_columns/input_layer/country_code_embedding_1/hash_table_Lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2mdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_lookup/hash_table/hash_tableParseExample/ParseExampleV2:9hdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_lookup/hash_table/Const*	
Tin0*

Tout0	*#
_output_shapes
:?????????
?
wdnn/input_from_feature_columns/input_layer/country_code_embedding_1/hash_table_Lookup/hash_table_Size/LookupTableSizeV2LookupTableSizeV2mdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_lookup/hash_table/hash_table*
_output_shapes
: 
?
Ydnn/input_from_feature_columns/input_layer/country_code_embedding_1/hash_table_Lookup/AddAddadnn/input_from_feature_columns/input_layer/country_code_embedding_1/hash_table_Lookup/hash_bucketwdnn/input_from_feature_columns/input_layer/country_code_embedding_1/hash_table_Lookup/hash_table_Size/LookupTableSizeV2*
T0	*#
_output_shapes
:?????????
?
^dnn/input_from_feature_columns/input_layer/country_code_embedding_1/hash_table_Lookup/NotEqualNotEqualydnn/input_from_feature_columns/input_layer/country_code_embedding_1/hash_table_Lookup/hash_table_Lookup/LookupTableFindV2hdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_lookup/hash_table/Const*
T0	*#
_output_shapes
:?????????
?
^dnn/input_from_feature_columns/input_layer/country_code_embedding_1/hash_table_Lookup/SelectV2SelectV2^dnn/input_from_feature_columns/input_layer/country_code_embedding_1/hash_table_Lookup/NotEqualydnn/input_from_feature_columns/input_layer/country_code_embedding_1/hash_table_Lookup/hash_table_Lookup/LookupTableFindV2Ydnn/input_from_feature_columns/input_layer/country_code_embedding_1/hash_table_Lookup/Add*
T0	*#
_output_shapes
:?????????
?
ndnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 
?
mdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?
hdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/SliceSliceParseExample/ParseExampleV2:15ndnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Slice/beginmdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Slice/size*
Index0*
T0	*
_output_shapes
:
?
hdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
gdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/ProdProdhdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Slicehdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Const*
T0	*
_output_shapes
: 
?
sdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :
?
pdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
kdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/GatherV2GatherV2ParseExample/ParseExampleV2:15sdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/GatherV2/indicespdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 
?
idnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Cast/xPackgdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Prodkdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/GatherV2*
N*
T0	*
_output_shapes
:
?
pdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/SparseReshapeSparseReshapeParseExample/ParseExampleV2:3ParseExample/ParseExampleV2:15idnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Cast/x*-
_output_shapes
:?????????:
?
ydnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/SparseReshape/IdentityIdentity^dnn/input_from_feature_columns/input_layer/country_code_embedding_1/hash_table_Lookup/SelectV2*
T0	*#
_output_shapes
:?????????
?
qdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
odnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/GreaterEqualGreaterEqualydnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/SparseReshape/Identityqdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:?????????
?
hdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/WhereWhereodnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/GreaterEqual*'
_output_shapes
:?????????
?
pdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
jdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/ReshapeReshapehdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Wherepdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:?????????
?
rdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
mdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/GatherV2_1GatherV2pdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/SparseReshapejdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Reshaperdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/GatherV2_1/axis*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:?????????
?
rdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
mdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/GatherV2_2GatherV2ydnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/SparseReshape/Identityjdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Reshaperdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/GatherV2_2/axis*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????
?
kdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/IdentityIdentityrdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
?
|dnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
?dnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsmdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/GatherV2_1mdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/GatherV2_2kdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Identity|dnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:?????????
?
?dnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
?
?dnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
?
?dnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
?
?dnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlice?dnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows?dnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/embedding_lookup_sparse/strided_slice/stack?dnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1?dnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask
?
?dnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/embedding_lookup_sparse/UniqueUnique?dnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:?????????:?????????
?
?dnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGatherSdnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights?dnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/embedding_lookup_sparse/Unique*
Tindices0	*f
_class\
ZXloc:@dnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights*'
_output_shapes
:?????????2*
dtype0
?
?dnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity?dnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*f
_class\
ZXloc:@dnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights*'
_output_shapes
:?????????2
?
?dnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1Identity?dnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity*
T0*'
_output_shapes
:?????????2
?
zdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/embedding_lookup_sparseSparseSegmentMean?dnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1?dnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/embedding_lookup_sparse/Unique:1?dnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/embedding_lookup_sparse/strided_slice*
T0*
Tsegmentids0	*'
_output_shapes
:?????????2
?
rdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   
?
ldnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Reshape_1Reshape?dnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2rdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:?????????
?
hdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/ShapeShapezdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
?
vdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?
xdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
xdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
pdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/strided_sliceStridedSlicehdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Shapevdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/strided_slice/stackxdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/strided_slice/stack_1xdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
jdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :
?
hdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/stackPackjdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/stack/0pdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/strided_slice*
N*
T0*
_output_shapes
:
?
gdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/TileTileldnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Reshape_1hdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/stack*
T0
*0
_output_shapes
:??????????????????
?
mdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/zeros_like	ZerosLikezdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????2
?
bdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weightsSelectgdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Tilemdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/zeros_likezdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????2
?
idnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Cast_1CastParseExample/ParseExampleV2:15*

DstT0*

SrcT0	*
_output_shapes
:
?
pdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 
?
odnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?
jdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Slice_1Sliceidnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Cast_1pdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Slice_1/beginodnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
?
jdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Shape_1Shapebdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights*
T0*
_output_shapes
:
?
pdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:
?
odnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
jdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Slice_2Slicejdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Shape_1pdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Slice_2/beginodnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
?
ndnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
idnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/concatConcatV2jdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Slice_1jdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Slice_2ndnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/concat/axis*
N*
T0*
_output_shapes
:
?
ldnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Reshape_2Reshapebdnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weightsidnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/concat*
T0*'
_output_shapes
:?????????2
?
Idnn/input_from_feature_columns/input_layer/country_code_embedding_1/ShapeShapeldnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Reshape_2*
T0*
_output_shapes
:
?
Wdnn/input_from_feature_columns/input_layer/country_code_embedding_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
Ydnn/input_from_feature_columns/input_layer/country_code_embedding_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
Ydnn/input_from_feature_columns/input_layer/country_code_embedding_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Qdnn/input_from_feature_columns/input_layer/country_code_embedding_1/strided_sliceStridedSliceIdnn/input_from_feature_columns/input_layer/country_code_embedding_1/ShapeWdnn/input_from_feature_columns/input_layer/country_code_embedding_1/strided_slice/stackYdnn/input_from_feature_columns/input_layer/country_code_embedding_1/strided_slice/stack_1Ydnn/input_from_feature_columns/input_layer/country_code_embedding_1/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Sdnn/input_from_feature_columns/input_layer/country_code_embedding_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
?
Qdnn/input_from_feature_columns/input_layer/country_code_embedding_1/Reshape/shapePackQdnn/input_from_feature_columns/input_layer/country_code_embedding_1/strided_sliceSdnn/input_from_feature_columns/input_layer/country_code_embedding_1/Reshape/shape/1*
N*
T0*
_output_shapes
:
?
Kdnn/input_from_feature_columns/input_layer/country_code_embedding_1/ReshapeReshapeldnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_embedding_weights/Reshape_2Qdnn/input_from_feature_columns/input_layer/country_code_embedding_1/Reshape/shape*
T0*'
_output_shapes
:?????????2
?
Udnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_lookup/ConstConst*
_output_shapes
:*
dtype0*?
value|BzBAndroidBLinuxBMac OS XB
Symbian OSBUbuntuB
Windows 10B	Windows 7BWindows 8.1BWindows PhoneB
Windows XPBiOS
?
Tdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_lookup/SizeConst*
_output_shapes
: *
dtype0*
value	B :
?
[dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_lookup/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
?
[dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_lookup/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
?
Udnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_lookup/rangeRange[dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_lookup/range/startTdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_lookup/Size[dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_lookup/range/delta*
_output_shapes
:
?
Tdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_lookup/CastCastUdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_lookup/range*

DstT0	*

SrcT0*
_output_shapes
:
?
`dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_lookup/hash_table/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
?
ednn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_lookup/hash_table/hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*@
shared_name1/hash_table_40660f9c-8322-4fb8-a098-75ea91bf8d1a*
value_dtype0	
?
ydnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_lookup/hash_table/table_init/LookupTableImportV2LookupTableImportV2ednn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_lookup/hash_table/hash_tableUdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_lookup/ConstTdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_lookup/Cast*	
Tin0*

Tout0	
?
]dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/hash_table_Lookup/hash_bucketStringToHashBucketFastParseExample/ParseExampleV2:10*#
_output_shapes
:?????????*
num_buckets
?
udnn/input_from_feature_columns/input_layer/osfamily_embedding_1/hash_table_Lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2ednn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_lookup/hash_table/hash_tableParseExample/ParseExampleV2:10`dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_lookup/hash_table/Const*	
Tin0*

Tout0	*#
_output_shapes
:?????????
?
sdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/hash_table_Lookup/hash_table_Size/LookupTableSizeV2LookupTableSizeV2ednn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_lookup/hash_table/hash_table*
_output_shapes
: 
?
Udnn/input_from_feature_columns/input_layer/osfamily_embedding_1/hash_table_Lookup/AddAdd]dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/hash_table_Lookup/hash_bucketsdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/hash_table_Lookup/hash_table_Size/LookupTableSizeV2*
T0	*#
_output_shapes
:?????????
?
Zdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/hash_table_Lookup/NotEqualNotEqualudnn/input_from_feature_columns/input_layer/osfamily_embedding_1/hash_table_Lookup/hash_table_Lookup/LookupTableFindV2`dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_lookup/hash_table/Const*
T0	*#
_output_shapes
:?????????
?
Zdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/hash_table_Lookup/SelectV2SelectV2Zdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/hash_table_Lookup/NotEqualudnn/input_from_feature_columns/input_layer/osfamily_embedding_1/hash_table_Lookup/hash_table_Lookup/LookupTableFindV2Udnn/input_from_feature_columns/input_layer/osfamily_embedding_1/hash_table_Lookup/Add*
T0	*#
_output_shapes
:?????????
?
fdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 
?
ednn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?
`dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/SliceSliceParseExample/ParseExampleV2:16fdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Slice/beginednn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Slice/size*
Index0*
T0	*
_output_shapes
:
?
`dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
_dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/ProdProd`dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Slice`dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Const*
T0	*
_output_shapes
: 
?
kdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :
?
hdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
cdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/GatherV2GatherV2ParseExample/ParseExampleV2:16kdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/GatherV2/indiceshdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 
?
adnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Cast/xPack_dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Prodcdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/GatherV2*
N*
T0	*
_output_shapes
:
?
hdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/SparseReshapeSparseReshapeParseExample/ParseExampleV2:4ParseExample/ParseExampleV2:16adnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Cast/x*-
_output_shapes
:?????????:
?
qdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/SparseReshape/IdentityIdentityZdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/hash_table_Lookup/SelectV2*
T0	*#
_output_shapes
:?????????
?
idnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
gdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/GreaterEqualGreaterEqualqdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/SparseReshape/Identityidnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:?????????
?
`dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/WhereWheregdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/GreaterEqual*'
_output_shapes
:?????????
?
hdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
bdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/ReshapeReshape`dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Wherehdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:?????????
?
jdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
ednn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/GatherV2_1GatherV2hdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/SparseReshapebdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Reshapejdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/GatherV2_1/axis*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:?????????
?
jdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
ednn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/GatherV2_2GatherV2qdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/SparseReshape/Identitybdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Reshapejdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/GatherV2_2/axis*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????
?
cdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/IdentityIdentityjdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
?
tdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
?dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsednn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/GatherV2_1ednn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/GatherV2_2cdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Identitytdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:?????????
?
?dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
?
?dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
?
?dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
?
?dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlice?dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows?dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/embedding_lookup_sparse/strided_slice/stack?dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1?dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask
?
ydnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/embedding_lookup_sparse/UniqueUnique?dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:?????????:?????????
?
?dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGatherOdnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weightsydnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/embedding_lookup_sparse/Unique*
Tindices0	*b
_classX
VTloc:@dnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights*'
_output_shapes
:?????????*
dtype0
?
?dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity?dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*b
_classX
VTloc:@dnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights*'
_output_shapes
:?????????
?
?dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1Identity?dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity*
T0*'
_output_shapes
:?????????
?
rdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/embedding_lookup_sparseSparseSegmentMean?dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1{dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/embedding_lookup_sparse/Unique:1?dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/embedding_lookup_sparse/strided_slice*
T0*
Tsegmentids0	*'
_output_shapes
:?????????
?
jdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   
?
ddnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Reshape_1Reshape?dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2jdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:?????????
?
`dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/ShapeShaperdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
?
ndnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?
pdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
pdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
hdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/strided_sliceStridedSlice`dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Shapendnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/strided_slice/stackpdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/strided_slice/stack_1pdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
bdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :
?
`dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/stackPackbdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/stack/0hdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/strided_slice*
N*
T0*
_output_shapes
:
?
_dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/TileTileddnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Reshape_1`dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/stack*
T0
*0
_output_shapes
:??????????????????
?
ednn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/zeros_like	ZerosLikerdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
Zdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weightsSelect_dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Tileednn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/zeros_likerdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
adnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Cast_1CastParseExample/ParseExampleV2:16*

DstT0*

SrcT0	*
_output_shapes
:
?
hdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 
?
gdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?
bdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Slice_1Sliceadnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Cast_1hdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Slice_1/begingdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
?
bdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Shape_1ShapeZdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights*
T0*
_output_shapes
:
?
hdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:
?
gdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
bdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Slice_2Slicebdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Shape_1hdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Slice_2/begingdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
?
fdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
adnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/concatConcatV2bdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Slice_1bdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Slice_2fdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/concat/axis*
N*
T0*
_output_shapes
:
?
ddnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Reshape_2ReshapeZdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weightsadnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/concat*
T0*'
_output_shapes
:?????????
?
Ednn/input_from_feature_columns/input_layer/osfamily_embedding_1/ShapeShapeddnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Reshape_2*
T0*
_output_shapes
:
?
Sdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
Udnn/input_from_feature_columns/input_layer/osfamily_embedding_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
Udnn/input_from_feature_columns/input_layer/osfamily_embedding_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Mdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/strided_sliceStridedSliceEdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/ShapeSdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/strided_slice/stackUdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/strided_slice/stack_1Udnn/input_from_feature_columns/input_layer/osfamily_embedding_1/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Odnn/input_from_feature_columns/input_layer/osfamily_embedding_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
?
Mdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/Reshape/shapePackMdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/strided_sliceOdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/Reshape/shape/1*
N*
T0*
_output_shapes
:
?
Gdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/ReshapeReshapeddnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_embedding_weights/Reshape_2Mdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/Reshape/shape*
T0*'
_output_shapes
:?????????
?
:dnn/input_from_feature_columns/input_layer/osmajor_1/ShapeShapeParseExample/ParseExampleV2:19*
T0*
_output_shapes
:
?
Hdnn/input_from_feature_columns/input_layer/osmajor_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
Jdnn/input_from_feature_columns/input_layer/osmajor_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
Jdnn/input_from_feature_columns/input_layer/osmajor_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Bdnn/input_from_feature_columns/input_layer/osmajor_1/strided_sliceStridedSlice:dnn/input_from_feature_columns/input_layer/osmajor_1/ShapeHdnn/input_from_feature_columns/input_layer/osmajor_1/strided_slice/stackJdnn/input_from_feature_columns/input_layer/osmajor_1/strided_slice/stack_1Jdnn/input_from_feature_columns/input_layer/osmajor_1/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Ddnn/input_from_feature_columns/input_layer/osmajor_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
?
Bdnn/input_from_feature_columns/input_layer/osmajor_1/Reshape/shapePackBdnn/input_from_feature_columns/input_layer/osmajor_1/strided_sliceDdnn/input_from_feature_columns/input_layer/osmajor_1/Reshape/shape/1*
N*
T0*
_output_shapes
:
?
<dnn/input_from_feature_columns/input_layer/osmajor_1/ReshapeReshapeParseExample/ParseExampleV2:19Bdnn/input_from_feature_columns/input_layer/osmajor_1/Reshape/shape*
T0*'
_output_shapes
:?????????
?
Ydnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_lookup/ConstConst*
_output_shapes
:*
dtype0* 
valueBBFalseBTrue
?
Xdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_lookup/SizeConst*
_output_shapes
: *
dtype0*
value	B :
?
_dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_lookup/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
?
_dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_lookup/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
?
Ydnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_lookup/rangeRange_dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_lookup/range/startXdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_lookup/Size_dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_lookup/range/delta*
_output_shapes
:
?
Xdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_lookup/CastCastYdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_lookup/range*

DstT0	*

SrcT0*
_output_shapes
:
?
ddnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_lookup/hash_table/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
?
idnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_lookup/hash_table/hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*@
shared_name1/hash_table_7e9d4e8c-4f97-4585-90e6-60962dd4bf1d*
value_dtype0	
?
}dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_lookup/hash_table/table_init/LookupTableImportV2LookupTableImportV2idnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_lookup/hash_table/hash_tableYdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_lookup/ConstXdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_lookup/Cast*	
Tin0*

Tout0	
?
_dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/hash_table_Lookup/hash_bucketStringToHashBucketFastParseExample/ParseExampleV2:11*#
_output_shapes
:?????????*
num_buckets
?
wdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/hash_table_Lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2idnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_lookup/hash_table/hash_tableParseExample/ParseExampleV2:11ddnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_lookup/hash_table/Const*	
Tin0*

Tout0	*#
_output_shapes
:?????????
?
udnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/hash_table_Lookup/hash_table_Size/LookupTableSizeV2LookupTableSizeV2idnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_lookup/hash_table/hash_table*
_output_shapes
: 
?
Wdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/hash_table_Lookup/AddAdd_dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/hash_table_Lookup/hash_bucketudnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/hash_table_Lookup/hash_table_Size/LookupTableSizeV2*
T0	*#
_output_shapes
:?????????
?
\dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/hash_table_Lookup/NotEqualNotEqualwdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/hash_table_Lookup/hash_table_Lookup/LookupTableFindV2ddnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_lookup/hash_table/Const*
T0	*#
_output_shapes
:?????????
?
\dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/hash_table_Lookup/SelectV2SelectV2\dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/hash_table_Lookup/NotEqualwdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/hash_table_Lookup/hash_table_Lookup/LookupTableFindV2Wdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/hash_table_Lookup/Add*
T0	*#
_output_shapes
:?????????
?
jdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 
?
idnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?
ddnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/SliceSliceParseExample/ParseExampleV2:17jdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Slice/beginidnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Slice/size*
Index0*
T0	*
_output_shapes
:
?
ddnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
cdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/ProdProdddnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Sliceddnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Const*
T0	*
_output_shapes
: 
?
odnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :
?
ldnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
gdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/GatherV2GatherV2ParseExample/ParseExampleV2:17odnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/GatherV2/indicesldnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 
?
ednn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Cast/xPackcdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Prodgdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/GatherV2*
N*
T0	*
_output_shapes
:
?
ldnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/SparseReshapeSparseReshapeParseExample/ParseExampleV2:5ParseExample/ParseExampleV2:17ednn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Cast/x*-
_output_shapes
:?????????:
?
udnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/SparseReshape/IdentityIdentity\dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/hash_table_Lookup/SelectV2*
T0	*#
_output_shapes
:?????????
?
mdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
kdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/GreaterEqualGreaterEqualudnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/SparseReshape/Identitymdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:?????????
?
ddnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/WhereWherekdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/GreaterEqual*'
_output_shapes
:?????????
?
ldnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
fdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/ReshapeReshapeddnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Whereldnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:?????????
?
ndnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
idnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/GatherV2_1GatherV2ldnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/SparseReshapefdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Reshapendnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/GatherV2_1/axis*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:?????????
?
ndnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
idnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/GatherV2_2GatherV2udnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/SparseReshape/Identityfdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Reshapendnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/GatherV2_2/axis*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????
?
gdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/IdentityIdentityndnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
?
xdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
?dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsidnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/GatherV2_1idnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/GatherV2_2gdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Identityxdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:?????????
?
?dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
?
?dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
?
?dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
?
?dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlice?dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows?dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/embedding_lookup_sparse/strided_slice/stack?dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1?dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask
?
}dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/embedding_lookup_sparse/UniqueUnique?dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:?????????:?????????
?
?dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGatherQdnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights}dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/embedding_lookup_sparse/Unique*
Tindices0	*d
_classZ
XVloc:@dnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights*'
_output_shapes
:?????????*
dtype0
?
?dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity?dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*d
_classZ
XVloc:@dnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights*'
_output_shapes
:?????????
?
?dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1Identity?dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity*
T0*'
_output_shapes
:?????????
?
vdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/embedding_lookup_sparseSparseSegmentMean?dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/embedding_lookup_sparse/Unique:1?dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/embedding_lookup_sparse/strided_slice*
T0*
Tsegmentids0	*'
_output_shapes
:?????????
?
ndnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   
?
hdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Reshape_1Reshape?dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2ndnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:?????????
?
ddnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/ShapeShapevdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
?
rdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?
tdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
tdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
ldnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/strided_sliceStridedSliceddnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Shaperdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/strided_slice/stacktdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/strided_slice/stack_1tdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
fdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :
?
ddnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/stackPackfdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/stack/0ldnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/strided_slice*
N*
T0*
_output_shapes
:
?
cdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/TileTilehdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Reshape_1ddnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/stack*
T0
*0
_output_shapes
:??????????????????
?
idnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/zeros_like	ZerosLikevdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
^dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weightsSelectcdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Tileidnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/zeros_likevdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
ednn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Cast_1CastParseExample/ParseExampleV2:17*

DstT0*

SrcT0	*
_output_shapes
:
?
ldnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 
?
kdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?
fdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Slice_1Sliceednn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Cast_1ldnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Slice_1/beginkdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
?
fdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Shape_1Shape^dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights*
T0*
_output_shapes
:
?
ldnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:
?
kdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
fdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Slice_2Slicefdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Shape_1ldnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Slice_2/beginkdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
?
jdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
ednn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/concatConcatV2fdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Slice_1fdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Slice_2jdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/concat/axis*
N*
T0*
_output_shapes
:
?
hdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Reshape_2Reshape^dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weightsednn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/concat*
T0*'
_output_shapes
:?????????
?
Gdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/ShapeShapehdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Reshape_2*
T0*
_output_shapes
:
?
Udnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
Wdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
Wdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Odnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/strided_sliceStridedSliceGdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/ShapeUdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/strided_slice/stackWdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/strided_slice/stack_1Wdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Qdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
?
Odnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/Reshape/shapePackOdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/strided_sliceQdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/Reshape/shape/1*
N*
T0*
_output_shapes
:
?
Idnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/ReshapeReshapehdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_embedding_weights/Reshape_2Odnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/Reshape/shape*
T0*'
_output_shapes
:?????????
?
6dnn/input_from_feature_columns/input_layer/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????
?
1dnn/input_from_feature_columns/input_layer/concatConcatV2Idnn/input_from_feature_columns/input_layer/asn_number_embedding_1/ReshapeJdnn/input_from_feature_columns/input_layer/browser_major_version_1/ReshapeWdnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/ReshapeKdnn/input_from_feature_columns/input_layer/browser_name_embedding_1/ReshapeKdnn/input_from_feature_columns/input_layer/country_code_embedding_1/ReshapeGdnn/input_from_feature_columns/input_layer/osfamily_embedding_1/Reshape<dnn/input_from_feature_columns/input_layer/osmajor_1/ReshapeIdnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/Reshape6dnn/input_from_feature_columns/input_layer/concat/axis*
N*
T0*'
_output_shapes
:?????????}
?
9dnn/hiddenlayer_0/kernel/Initializer/random_uniform/shapeConst*+
_class!
loc:@dnn/hiddenlayer_0/kernel*
_output_shapes
:*
dtype0*
valueB"}   ?   
?
7dnn/hiddenlayer_0/kernel/Initializer/random_uniform/minConst*+
_class!
loc:@dnn/hiddenlayer_0/kernel*
_output_shapes
: *
dtype0*
valueB
 *?
?
7dnn/hiddenlayer_0/kernel/Initializer/random_uniform/maxConst*+
_class!
loc:@dnn/hiddenlayer_0/kernel*
_output_shapes
: *
dtype0*
valueB
 *>
?
Adnn/hiddenlayer_0/kernel/Initializer/random_uniform/RandomUniformRandomUniform9dnn/hiddenlayer_0/kernel/Initializer/random_uniform/shape*
T0*+
_class!
loc:@dnn/hiddenlayer_0/kernel*
_output_shapes
:	}?*
dtype0*
seed????*
seed2
?
7dnn/hiddenlayer_0/kernel/Initializer/random_uniform/subSub7dnn/hiddenlayer_0/kernel/Initializer/random_uniform/max7dnn/hiddenlayer_0/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/hiddenlayer_0/kernel*
_output_shapes
: 
?
7dnn/hiddenlayer_0/kernel/Initializer/random_uniform/mulMulAdnn/hiddenlayer_0/kernel/Initializer/random_uniform/RandomUniform7dnn/hiddenlayer_0/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@dnn/hiddenlayer_0/kernel*
_output_shapes
:	}?
?
3dnn/hiddenlayer_0/kernel/Initializer/random_uniformAdd7dnn/hiddenlayer_0/kernel/Initializer/random_uniform/mul7dnn/hiddenlayer_0/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/hiddenlayer_0/kernel*
_output_shapes
:	}?
?
dnn/hiddenlayer_0/kernelVarHandleOp*+
_class!
loc:@dnn/hiddenlayer_0/kernel*
_output_shapes
: *
dtype0*
shape:	}?*)
shared_namednn/hiddenlayer_0/kernel
?
9dnn/hiddenlayer_0/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_0/kernel*
_output_shapes
: 
?
dnn/hiddenlayer_0/kernel/AssignAssignVariableOpdnn/hiddenlayer_0/kernel3dnn/hiddenlayer_0/kernel/Initializer/random_uniform*
dtype0
?
,dnn/hiddenlayer_0/kernel/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel*
_output_shapes
:	}?*
dtype0
?
(dnn/hiddenlayer_0/bias/Initializer/zerosConst*)
_class
loc:@dnn/hiddenlayer_0/bias*
_output_shapes	
:?*
dtype0*
valueB?*    
?
dnn/hiddenlayer_0/biasVarHandleOp*)
_class
loc:@dnn/hiddenlayer_0/bias*
_output_shapes
: *
dtype0*
shape:?*'
shared_namednn/hiddenlayer_0/bias
}
7dnn/hiddenlayer_0/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_0/bias*
_output_shapes
: 
?
dnn/hiddenlayer_0/bias/AssignAssignVariableOpdnn/hiddenlayer_0/bias(dnn/hiddenlayer_0/bias/Initializer/zeros*
dtype0
~
*dnn/hiddenlayer_0/bias/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias*
_output_shapes	
:?*
dtype0
?
'dnn/hiddenlayer_0/MatMul/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel*
_output_shapes
:	}?*
dtype0
?
dnn/hiddenlayer_0/MatMulMatMul1dnn/input_from_feature_columns/input_layer/concat'dnn/hiddenlayer_0/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????
|
(dnn/hiddenlayer_0/BiasAdd/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias*
_output_shapes	
:?*
dtype0
?
dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMul(dnn/hiddenlayer_0/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:??????????
m
dnn/hiddenlayer_0/LeakyRelu	LeakyReludnn/hiddenlayer_0/BiasAdd*(
_output_shapes
:??????????
?
4dnn/hiddenlayer_0/batchnorm_0/gamma/Initializer/onesConst*6
_class,
*(loc:@dnn/hiddenlayer_0/batchnorm_0/gamma*
_output_shapes	
:?*
dtype0*
valueB?*  ??
?
#dnn/hiddenlayer_0/batchnorm_0/gammaVarHandleOp*6
_class,
*(loc:@dnn/hiddenlayer_0/batchnorm_0/gamma*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#dnn/hiddenlayer_0/batchnorm_0/gamma
?
Ddnn/hiddenlayer_0/batchnorm_0/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp#dnn/hiddenlayer_0/batchnorm_0/gamma*
_output_shapes
: 
?
*dnn/hiddenlayer_0/batchnorm_0/gamma/AssignAssignVariableOp#dnn/hiddenlayer_0/batchnorm_0/gamma4dnn/hiddenlayer_0/batchnorm_0/gamma/Initializer/ones*
dtype0
?
7dnn/hiddenlayer_0/batchnorm_0/gamma/Read/ReadVariableOpReadVariableOp#dnn/hiddenlayer_0/batchnorm_0/gamma*
_output_shapes	
:?*
dtype0
?
4dnn/hiddenlayer_0/batchnorm_0/beta/Initializer/zerosConst*5
_class+
)'loc:@dnn/hiddenlayer_0/batchnorm_0/beta*
_output_shapes	
:?*
dtype0*
valueB?*    
?
"dnn/hiddenlayer_0/batchnorm_0/betaVarHandleOp*5
_class+
)'loc:@dnn/hiddenlayer_0/batchnorm_0/beta*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"dnn/hiddenlayer_0/batchnorm_0/beta
?
Cdnn/hiddenlayer_0/batchnorm_0/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp"dnn/hiddenlayer_0/batchnorm_0/beta*
_output_shapes
: 
?
)dnn/hiddenlayer_0/batchnorm_0/beta/AssignAssignVariableOp"dnn/hiddenlayer_0/batchnorm_0/beta4dnn/hiddenlayer_0/batchnorm_0/beta/Initializer/zeros*
dtype0
?
6dnn/hiddenlayer_0/batchnorm_0/beta/Read/ReadVariableOpReadVariableOp"dnn/hiddenlayer_0/batchnorm_0/beta*
_output_shapes	
:?*
dtype0
?
;dnn/hiddenlayer_0/batchnorm_0/moving_mean/Initializer/zerosConst*<
_class2
0.loc:@dnn/hiddenlayer_0/batchnorm_0/moving_mean*
_output_shapes	
:?*
dtype0*
valueB?*    
?
)dnn/hiddenlayer_0/batchnorm_0/moving_meanVarHandleOp*<
_class2
0.loc:@dnn/hiddenlayer_0/batchnorm_0/moving_mean*
_output_shapes
: *
dtype0*
shape:?*:
shared_name+)dnn/hiddenlayer_0/batchnorm_0/moving_mean
?
Jdnn/hiddenlayer_0/batchnorm_0/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp)dnn/hiddenlayer_0/batchnorm_0/moving_mean*
_output_shapes
: 
?
0dnn/hiddenlayer_0/batchnorm_0/moving_mean/AssignAssignVariableOp)dnn/hiddenlayer_0/batchnorm_0/moving_mean;dnn/hiddenlayer_0/batchnorm_0/moving_mean/Initializer/zeros*
dtype0
?
=dnn/hiddenlayer_0/batchnorm_0/moving_mean/Read/ReadVariableOpReadVariableOp)dnn/hiddenlayer_0/batchnorm_0/moving_mean*
_output_shapes	
:?*
dtype0
?
>dnn/hiddenlayer_0/batchnorm_0/moving_variance/Initializer/onesConst*@
_class6
42loc:@dnn/hiddenlayer_0/batchnorm_0/moving_variance*
_output_shapes	
:?*
dtype0*
valueB?*  ??
?
-dnn/hiddenlayer_0/batchnorm_0/moving_varianceVarHandleOp*@
_class6
42loc:@dnn/hiddenlayer_0/batchnorm_0/moving_variance*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-dnn/hiddenlayer_0/batchnorm_0/moving_variance
?
Ndnn/hiddenlayer_0/batchnorm_0/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp-dnn/hiddenlayer_0/batchnorm_0/moving_variance*
_output_shapes
: 
?
4dnn/hiddenlayer_0/batchnorm_0/moving_variance/AssignAssignVariableOp-dnn/hiddenlayer_0/batchnorm_0/moving_variance>dnn/hiddenlayer_0/batchnorm_0/moving_variance/Initializer/ones*
dtype0
?
Adnn/hiddenlayer_0/batchnorm_0/moving_variance/Read/ReadVariableOpReadVariableOp-dnn/hiddenlayer_0/batchnorm_0/moving_variance*
_output_shapes	
:?*
dtype0
?
6dnn/hiddenlayer_0/batchnorm_0/batchnorm/ReadVariableOpReadVariableOp-dnn/hiddenlayer_0/batchnorm_0/moving_variance*
_output_shapes	
:?*
dtype0
r
-dnn/hiddenlayer_0/batchnorm_0/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:
?
+dnn/hiddenlayer_0/batchnorm_0/batchnorm/addAddV26dnn/hiddenlayer_0/batchnorm_0/batchnorm/ReadVariableOp-dnn/hiddenlayer_0/batchnorm_0/batchnorm/add/y*
T0*
_output_shapes	
:?
?
-dnn/hiddenlayer_0/batchnorm_0/batchnorm/RsqrtRsqrt+dnn/hiddenlayer_0/batchnorm_0/batchnorm/add*
T0*
_output_shapes	
:?
?
:dnn/hiddenlayer_0/batchnorm_0/batchnorm/mul/ReadVariableOpReadVariableOp#dnn/hiddenlayer_0/batchnorm_0/gamma*
_output_shapes	
:?*
dtype0
?
+dnn/hiddenlayer_0/batchnorm_0/batchnorm/mulMul-dnn/hiddenlayer_0/batchnorm_0/batchnorm/Rsqrt:dnn/hiddenlayer_0/batchnorm_0/batchnorm/mul/ReadVariableOp*
T0*
_output_shapes	
:?
?
-dnn/hiddenlayer_0/batchnorm_0/batchnorm/mul_1Muldnn/hiddenlayer_0/LeakyRelu+dnn/hiddenlayer_0/batchnorm_0/batchnorm/mul*
T0*(
_output_shapes
:??????????
?
8dnn/hiddenlayer_0/batchnorm_0/batchnorm/ReadVariableOp_1ReadVariableOp)dnn/hiddenlayer_0/batchnorm_0/moving_mean*
_output_shapes	
:?*
dtype0
?
-dnn/hiddenlayer_0/batchnorm_0/batchnorm/mul_2Mul8dnn/hiddenlayer_0/batchnorm_0/batchnorm/ReadVariableOp_1+dnn/hiddenlayer_0/batchnorm_0/batchnorm/mul*
T0*
_output_shapes	
:?
?
8dnn/hiddenlayer_0/batchnorm_0/batchnorm/ReadVariableOp_2ReadVariableOp"dnn/hiddenlayer_0/batchnorm_0/beta*
_output_shapes	
:?*
dtype0
?
+dnn/hiddenlayer_0/batchnorm_0/batchnorm/subSub8dnn/hiddenlayer_0/batchnorm_0/batchnorm/ReadVariableOp_2-dnn/hiddenlayer_0/batchnorm_0/batchnorm/mul_2*
T0*
_output_shapes	
:?
?
-dnn/hiddenlayer_0/batchnorm_0/batchnorm/add_1AddV2-dnn/hiddenlayer_0/batchnorm_0/batchnorm/mul_1+dnn/hiddenlayer_0/batchnorm_0/batchnorm/sub*
T0*(
_output_shapes
:??????????
~
dnn/zero_fraction/SizeSize-dnn/hiddenlayer_0/batchnorm_0/batchnorm/add_1*
T0*
_output_shapes
: *
out_type0	
c
dnn/zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R????
?
dnn/zero_fraction/LessEqual	LessEqualdnn/zero_fraction/Sizednn/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
?
dnn/zero_fraction/condStatelessIfdnn/zero_fraction/LessEqual-dnn/hiddenlayer_0/batchnorm_0/batchnorm/add_1*
Tcond0
*
Tin
2*
Tout

2	*
_lower_using_switch_merge(* 
_output_shapes
: : : : : : * 
_read_only_resource_inputs
 *6
else_branch'R%
#dnn_zero_fraction_cond_false_511528*
output_shapes
: : : : : : *5
then_branch&R$
"dnn_zero_fraction_cond_true_511527
d
dnn/zero_fraction/cond/IdentityIdentitydnn/zero_fraction/cond*
T0	*
_output_shapes
: 
h
!dnn/zero_fraction/cond/Identity_1Identitydnn/zero_fraction/cond:1*
T0*
_output_shapes
: 
h
!dnn/zero_fraction/cond/Identity_2Identitydnn/zero_fraction/cond:2*
T0*
_output_shapes
: 
h
!dnn/zero_fraction/cond/Identity_3Identitydnn/zero_fraction/cond:3*
T0*
_output_shapes
: 
h
!dnn/zero_fraction/cond/Identity_4Identitydnn/zero_fraction/cond:4*
T0*
_output_shapes
: 
h
!dnn/zero_fraction/cond/Identity_5Identitydnn/zero_fraction/cond:5*
T0*
_output_shapes
: 
?
(dnn/zero_fraction/counts_to_fraction/subSubdnn/zero_fraction/Sizednn/zero_fraction/cond/Identity*
T0	*
_output_shapes
: 
?
)dnn/zero_fraction/counts_to_fraction/CastCast(dnn/zero_fraction/counts_to_fraction/sub*

DstT0*

SrcT0	*
_output_shapes
: 
{
+dnn/zero_fraction/counts_to_fraction/Cast_1Castdnn/zero_fraction/Size*

DstT0*

SrcT0	*
_output_shapes
: 
?
,dnn/zero_fraction/counts_to_fraction/truedivRealDiv)dnn/zero_fraction/counts_to_fraction/Cast+dnn/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
u
dnn/zero_fraction/fractionIdentity,dnn/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
.dnn/hiddenlayer_0/fraction_of_zero_values/tagsConst*
_output_shapes
: *
dtype0*:
value1B/ B)dnn/hiddenlayer_0/fraction_of_zero_values
?
)dnn/hiddenlayer_0/fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_0/fraction_of_zero_values/tagsdnn/zero_fraction/fraction*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_0/activation/tagConst*
_output_shapes
: *
dtype0*-
value$B" Bdnn/hiddenlayer_0/activation
?
dnn/hiddenlayer_0/activationHistogramSummary dnn/hiddenlayer_0/activation/tag-dnn/hiddenlayer_0/batchnorm_0/batchnorm/add_1*
_output_shapes
: 
?
9dnn/hiddenlayer_1/kernel/Initializer/random_uniform/shapeConst*+
_class!
loc:@dnn/hiddenlayer_1/kernel*
_output_shapes
:*
dtype0*
valueB"?   d   
?
7dnn/hiddenlayer_1/kernel/Initializer/random_uniform/minConst*+
_class!
loc:@dnn/hiddenlayer_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *d%?
?
7dnn/hiddenlayer_1/kernel/Initializer/random_uniform/maxConst*+
_class!
loc:@dnn/hiddenlayer_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *d%>
?
Adnn/hiddenlayer_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform9dnn/hiddenlayer_1/kernel/Initializer/random_uniform/shape*
T0*+
_class!
loc:@dnn/hiddenlayer_1/kernel*
_output_shapes
:	?d*
dtype0*
seed????*
seed2
?
7dnn/hiddenlayer_1/kernel/Initializer/random_uniform/subSub7dnn/hiddenlayer_1/kernel/Initializer/random_uniform/max7dnn/hiddenlayer_1/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/hiddenlayer_1/kernel*
_output_shapes
: 
?
7dnn/hiddenlayer_1/kernel/Initializer/random_uniform/mulMulAdnn/hiddenlayer_1/kernel/Initializer/random_uniform/RandomUniform7dnn/hiddenlayer_1/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@dnn/hiddenlayer_1/kernel*
_output_shapes
:	?d
?
3dnn/hiddenlayer_1/kernel/Initializer/random_uniformAdd7dnn/hiddenlayer_1/kernel/Initializer/random_uniform/mul7dnn/hiddenlayer_1/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/hiddenlayer_1/kernel*
_output_shapes
:	?d
?
dnn/hiddenlayer_1/kernelVarHandleOp*+
_class!
loc:@dnn/hiddenlayer_1/kernel*
_output_shapes
: *
dtype0*
shape:	?d*)
shared_namednn/hiddenlayer_1/kernel
?
9dnn/hiddenlayer_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_1/kernel*
_output_shapes
: 
?
dnn/hiddenlayer_1/kernel/AssignAssignVariableOpdnn/hiddenlayer_1/kernel3dnn/hiddenlayer_1/kernel/Initializer/random_uniform*
dtype0
?
,dnn/hiddenlayer_1/kernel/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel*
_output_shapes
:	?d*
dtype0
?
(dnn/hiddenlayer_1/bias/Initializer/zerosConst*)
_class
loc:@dnn/hiddenlayer_1/bias*
_output_shapes
:d*
dtype0*
valueBd*    
?
dnn/hiddenlayer_1/biasVarHandleOp*)
_class
loc:@dnn/hiddenlayer_1/bias*
_output_shapes
: *
dtype0*
shape:d*'
shared_namednn/hiddenlayer_1/bias
}
7dnn/hiddenlayer_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_1/bias*
_output_shapes
: 
?
dnn/hiddenlayer_1/bias/AssignAssignVariableOpdnn/hiddenlayer_1/bias(dnn/hiddenlayer_1/bias/Initializer/zeros*
dtype0
}
*dnn/hiddenlayer_1/bias/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias*
_output_shapes
:d*
dtype0
?
'dnn/hiddenlayer_1/MatMul/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel*
_output_shapes
:	?d*
dtype0
?
dnn/hiddenlayer_1/MatMulMatMul-dnn/hiddenlayer_0/batchnorm_0/batchnorm/add_1'dnn/hiddenlayer_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d
{
(dnn/hiddenlayer_1/BiasAdd/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias*
_output_shapes
:d*
dtype0
?
dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMul(dnn/hiddenlayer_1/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:?????????d
l
dnn/hiddenlayer_1/LeakyRelu	LeakyReludnn/hiddenlayer_1/BiasAdd*'
_output_shapes
:?????????d
?
4dnn/hiddenlayer_1/batchnorm_1/gamma/Initializer/onesConst*6
_class,
*(loc:@dnn/hiddenlayer_1/batchnorm_1/gamma*
_output_shapes
:d*
dtype0*
valueBd*  ??
?
#dnn/hiddenlayer_1/batchnorm_1/gammaVarHandleOp*6
_class,
*(loc:@dnn/hiddenlayer_1/batchnorm_1/gamma*
_output_shapes
: *
dtype0*
shape:d*4
shared_name%#dnn/hiddenlayer_1/batchnorm_1/gamma
?
Ddnn/hiddenlayer_1/batchnorm_1/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp#dnn/hiddenlayer_1/batchnorm_1/gamma*
_output_shapes
: 
?
*dnn/hiddenlayer_1/batchnorm_1/gamma/AssignAssignVariableOp#dnn/hiddenlayer_1/batchnorm_1/gamma4dnn/hiddenlayer_1/batchnorm_1/gamma/Initializer/ones*
dtype0
?
7dnn/hiddenlayer_1/batchnorm_1/gamma/Read/ReadVariableOpReadVariableOp#dnn/hiddenlayer_1/batchnorm_1/gamma*
_output_shapes
:d*
dtype0
?
4dnn/hiddenlayer_1/batchnorm_1/beta/Initializer/zerosConst*5
_class+
)'loc:@dnn/hiddenlayer_1/batchnorm_1/beta*
_output_shapes
:d*
dtype0*
valueBd*    
?
"dnn/hiddenlayer_1/batchnorm_1/betaVarHandleOp*5
_class+
)'loc:@dnn/hiddenlayer_1/batchnorm_1/beta*
_output_shapes
: *
dtype0*
shape:d*3
shared_name$"dnn/hiddenlayer_1/batchnorm_1/beta
?
Cdnn/hiddenlayer_1/batchnorm_1/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp"dnn/hiddenlayer_1/batchnorm_1/beta*
_output_shapes
: 
?
)dnn/hiddenlayer_1/batchnorm_1/beta/AssignAssignVariableOp"dnn/hiddenlayer_1/batchnorm_1/beta4dnn/hiddenlayer_1/batchnorm_1/beta/Initializer/zeros*
dtype0
?
6dnn/hiddenlayer_1/batchnorm_1/beta/Read/ReadVariableOpReadVariableOp"dnn/hiddenlayer_1/batchnorm_1/beta*
_output_shapes
:d*
dtype0
?
;dnn/hiddenlayer_1/batchnorm_1/moving_mean/Initializer/zerosConst*<
_class2
0.loc:@dnn/hiddenlayer_1/batchnorm_1/moving_mean*
_output_shapes
:d*
dtype0*
valueBd*    
?
)dnn/hiddenlayer_1/batchnorm_1/moving_meanVarHandleOp*<
_class2
0.loc:@dnn/hiddenlayer_1/batchnorm_1/moving_mean*
_output_shapes
: *
dtype0*
shape:d*:
shared_name+)dnn/hiddenlayer_1/batchnorm_1/moving_mean
?
Jdnn/hiddenlayer_1/batchnorm_1/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp)dnn/hiddenlayer_1/batchnorm_1/moving_mean*
_output_shapes
: 
?
0dnn/hiddenlayer_1/batchnorm_1/moving_mean/AssignAssignVariableOp)dnn/hiddenlayer_1/batchnorm_1/moving_mean;dnn/hiddenlayer_1/batchnorm_1/moving_mean/Initializer/zeros*
dtype0
?
=dnn/hiddenlayer_1/batchnorm_1/moving_mean/Read/ReadVariableOpReadVariableOp)dnn/hiddenlayer_1/batchnorm_1/moving_mean*
_output_shapes
:d*
dtype0
?
>dnn/hiddenlayer_1/batchnorm_1/moving_variance/Initializer/onesConst*@
_class6
42loc:@dnn/hiddenlayer_1/batchnorm_1/moving_variance*
_output_shapes
:d*
dtype0*
valueBd*  ??
?
-dnn/hiddenlayer_1/batchnorm_1/moving_varianceVarHandleOp*@
_class6
42loc:@dnn/hiddenlayer_1/batchnorm_1/moving_variance*
_output_shapes
: *
dtype0*
shape:d*>
shared_name/-dnn/hiddenlayer_1/batchnorm_1/moving_variance
?
Ndnn/hiddenlayer_1/batchnorm_1/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp-dnn/hiddenlayer_1/batchnorm_1/moving_variance*
_output_shapes
: 
?
4dnn/hiddenlayer_1/batchnorm_1/moving_variance/AssignAssignVariableOp-dnn/hiddenlayer_1/batchnorm_1/moving_variance>dnn/hiddenlayer_1/batchnorm_1/moving_variance/Initializer/ones*
dtype0
?
Adnn/hiddenlayer_1/batchnorm_1/moving_variance/Read/ReadVariableOpReadVariableOp-dnn/hiddenlayer_1/batchnorm_1/moving_variance*
_output_shapes
:d*
dtype0
?
6dnn/hiddenlayer_1/batchnorm_1/batchnorm/ReadVariableOpReadVariableOp-dnn/hiddenlayer_1/batchnorm_1/moving_variance*
_output_shapes
:d*
dtype0
r
-dnn/hiddenlayer_1/batchnorm_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:
?
+dnn/hiddenlayer_1/batchnorm_1/batchnorm/addAddV26dnn/hiddenlayer_1/batchnorm_1/batchnorm/ReadVariableOp-dnn/hiddenlayer_1/batchnorm_1/batchnorm/add/y*
T0*
_output_shapes
:d
?
-dnn/hiddenlayer_1/batchnorm_1/batchnorm/RsqrtRsqrt+dnn/hiddenlayer_1/batchnorm_1/batchnorm/add*
T0*
_output_shapes
:d
?
:dnn/hiddenlayer_1/batchnorm_1/batchnorm/mul/ReadVariableOpReadVariableOp#dnn/hiddenlayer_1/batchnorm_1/gamma*
_output_shapes
:d*
dtype0
?
+dnn/hiddenlayer_1/batchnorm_1/batchnorm/mulMul-dnn/hiddenlayer_1/batchnorm_1/batchnorm/Rsqrt:dnn/hiddenlayer_1/batchnorm_1/batchnorm/mul/ReadVariableOp*
T0*
_output_shapes
:d
?
-dnn/hiddenlayer_1/batchnorm_1/batchnorm/mul_1Muldnn/hiddenlayer_1/LeakyRelu+dnn/hiddenlayer_1/batchnorm_1/batchnorm/mul*
T0*'
_output_shapes
:?????????d
?
8dnn/hiddenlayer_1/batchnorm_1/batchnorm/ReadVariableOp_1ReadVariableOp)dnn/hiddenlayer_1/batchnorm_1/moving_mean*
_output_shapes
:d*
dtype0
?
-dnn/hiddenlayer_1/batchnorm_1/batchnorm/mul_2Mul8dnn/hiddenlayer_1/batchnorm_1/batchnorm/ReadVariableOp_1+dnn/hiddenlayer_1/batchnorm_1/batchnorm/mul*
T0*
_output_shapes
:d
?
8dnn/hiddenlayer_1/batchnorm_1/batchnorm/ReadVariableOp_2ReadVariableOp"dnn/hiddenlayer_1/batchnorm_1/beta*
_output_shapes
:d*
dtype0
?
+dnn/hiddenlayer_1/batchnorm_1/batchnorm/subSub8dnn/hiddenlayer_1/batchnorm_1/batchnorm/ReadVariableOp_2-dnn/hiddenlayer_1/batchnorm_1/batchnorm/mul_2*
T0*
_output_shapes
:d
?
-dnn/hiddenlayer_1/batchnorm_1/batchnorm/add_1AddV2-dnn/hiddenlayer_1/batchnorm_1/batchnorm/mul_1+dnn/hiddenlayer_1/batchnorm_1/batchnorm/sub*
T0*'
_output_shapes
:?????????d
?
dnn/zero_fraction_1/SizeSize-dnn/hiddenlayer_1/batchnorm_1/batchnorm/add_1*
T0*
_output_shapes
: *
out_type0	
e
dnn/zero_fraction_1/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R????
?
dnn/zero_fraction_1/LessEqual	LessEqualdnn/zero_fraction_1/Sizednn/zero_fraction_1/LessEqual/y*
T0	*
_output_shapes
: 
?
dnn/zero_fraction_1/condStatelessIfdnn/zero_fraction_1/LessEqual-dnn/hiddenlayer_1/batchnorm_1/batchnorm/add_1*
Tcond0
*
Tin
2*
Tout

2	*
_lower_using_switch_merge(* 
_output_shapes
: : : : : : * 
_read_only_resource_inputs
 *8
else_branch)R'
%dnn_zero_fraction_1_cond_false_511626*
output_shapes
: : : : : : *7
then_branch(R&
$dnn_zero_fraction_1_cond_true_511625
h
!dnn/zero_fraction_1/cond/IdentityIdentitydnn/zero_fraction_1/cond*
T0	*
_output_shapes
: 
l
#dnn/zero_fraction_1/cond/Identity_1Identitydnn/zero_fraction_1/cond:1*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_1/cond/Identity_2Identitydnn/zero_fraction_1/cond:2*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_1/cond/Identity_3Identitydnn/zero_fraction_1/cond:3*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_1/cond/Identity_4Identitydnn/zero_fraction_1/cond:4*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_1/cond/Identity_5Identitydnn/zero_fraction_1/cond:5*
T0*
_output_shapes
: 
?
*dnn/zero_fraction_1/counts_to_fraction/subSubdnn/zero_fraction_1/Size!dnn/zero_fraction_1/cond/Identity*
T0	*
_output_shapes
: 
?
+dnn/zero_fraction_1/counts_to_fraction/CastCast*dnn/zero_fraction_1/counts_to_fraction/sub*

DstT0*

SrcT0	*
_output_shapes
: 

-dnn/zero_fraction_1/counts_to_fraction/Cast_1Castdnn/zero_fraction_1/Size*

DstT0*

SrcT0	*
_output_shapes
: 
?
.dnn/zero_fraction_1/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_1/counts_to_fraction/Cast-dnn/zero_fraction_1/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
y
dnn/zero_fraction_1/fractionIdentity.dnn/zero_fraction_1/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
.dnn/hiddenlayer_1/fraction_of_zero_values/tagsConst*
_output_shapes
: *
dtype0*:
value1B/ B)dnn/hiddenlayer_1/fraction_of_zero_values
?
)dnn/hiddenlayer_1/fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_1/fraction_of_zero_values/tagsdnn/zero_fraction_1/fraction*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_1/activation/tagConst*
_output_shapes
: *
dtype0*-
value$B" Bdnn/hiddenlayer_1/activation
?
dnn/hiddenlayer_1/activationHistogramSummary dnn/hiddenlayer_1/activation/tag-dnn/hiddenlayer_1/batchnorm_1/batchnorm/add_1*
_output_shapes
: 
?
9dnn/hiddenlayer_2/kernel/Initializer/random_uniform/shapeConst*+
_class!
loc:@dnn/hiddenlayer_2/kernel*
_output_shapes
:*
dtype0*
valueB"d   d   
?
7dnn/hiddenlayer_2/kernel/Initializer/random_uniform/minConst*+
_class!
loc:@dnn/hiddenlayer_2/kernel*
_output_shapes
: *
dtype0*
valueB
 *?\1?
?
7dnn/hiddenlayer_2/kernel/Initializer/random_uniform/maxConst*+
_class!
loc:@dnn/hiddenlayer_2/kernel*
_output_shapes
: *
dtype0*
valueB
 *?\1>
?
Adnn/hiddenlayer_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform9dnn/hiddenlayer_2/kernel/Initializer/random_uniform/shape*
T0*+
_class!
loc:@dnn/hiddenlayer_2/kernel*
_output_shapes

:dd*
dtype0*
seed????*
seed2
?
7dnn/hiddenlayer_2/kernel/Initializer/random_uniform/subSub7dnn/hiddenlayer_2/kernel/Initializer/random_uniform/max7dnn/hiddenlayer_2/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/hiddenlayer_2/kernel*
_output_shapes
: 
?
7dnn/hiddenlayer_2/kernel/Initializer/random_uniform/mulMulAdnn/hiddenlayer_2/kernel/Initializer/random_uniform/RandomUniform7dnn/hiddenlayer_2/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@dnn/hiddenlayer_2/kernel*
_output_shapes

:dd
?
3dnn/hiddenlayer_2/kernel/Initializer/random_uniformAdd7dnn/hiddenlayer_2/kernel/Initializer/random_uniform/mul7dnn/hiddenlayer_2/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/hiddenlayer_2/kernel*
_output_shapes

:dd
?
dnn/hiddenlayer_2/kernelVarHandleOp*+
_class!
loc:@dnn/hiddenlayer_2/kernel*
_output_shapes
: *
dtype0*
shape
:dd*)
shared_namednn/hiddenlayer_2/kernel
?
9dnn/hiddenlayer_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_2/kernel*
_output_shapes
: 
?
dnn/hiddenlayer_2/kernel/AssignAssignVariableOpdnn/hiddenlayer_2/kernel3dnn/hiddenlayer_2/kernel/Initializer/random_uniform*
dtype0
?
,dnn/hiddenlayer_2/kernel/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_2/kernel*
_output_shapes

:dd*
dtype0
?
(dnn/hiddenlayer_2/bias/Initializer/zerosConst*)
_class
loc:@dnn/hiddenlayer_2/bias*
_output_shapes
:d*
dtype0*
valueBd*    
?
dnn/hiddenlayer_2/biasVarHandleOp*)
_class
loc:@dnn/hiddenlayer_2/bias*
_output_shapes
: *
dtype0*
shape:d*'
shared_namednn/hiddenlayer_2/bias
}
7dnn/hiddenlayer_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_2/bias*
_output_shapes
: 
?
dnn/hiddenlayer_2/bias/AssignAssignVariableOpdnn/hiddenlayer_2/bias(dnn/hiddenlayer_2/bias/Initializer/zeros*
dtype0
}
*dnn/hiddenlayer_2/bias/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_2/bias*
_output_shapes
:d*
dtype0
?
'dnn/hiddenlayer_2/MatMul/ReadVariableOpReadVariableOpdnn/hiddenlayer_2/kernel*
_output_shapes

:dd*
dtype0
?
dnn/hiddenlayer_2/MatMulMatMul-dnn/hiddenlayer_1/batchnorm_1/batchnorm/add_1'dnn/hiddenlayer_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d
{
(dnn/hiddenlayer_2/BiasAdd/ReadVariableOpReadVariableOpdnn/hiddenlayer_2/bias*
_output_shapes
:d*
dtype0
?
dnn/hiddenlayer_2/BiasAddBiasAdddnn/hiddenlayer_2/MatMul(dnn/hiddenlayer_2/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:?????????d
l
dnn/hiddenlayer_2/LeakyRelu	LeakyReludnn/hiddenlayer_2/BiasAdd*'
_output_shapes
:?????????d
?
4dnn/hiddenlayer_2/batchnorm_2/gamma/Initializer/onesConst*6
_class,
*(loc:@dnn/hiddenlayer_2/batchnorm_2/gamma*
_output_shapes
:d*
dtype0*
valueBd*  ??
?
#dnn/hiddenlayer_2/batchnorm_2/gammaVarHandleOp*6
_class,
*(loc:@dnn/hiddenlayer_2/batchnorm_2/gamma*
_output_shapes
: *
dtype0*
shape:d*4
shared_name%#dnn/hiddenlayer_2/batchnorm_2/gamma
?
Ddnn/hiddenlayer_2/batchnorm_2/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp#dnn/hiddenlayer_2/batchnorm_2/gamma*
_output_shapes
: 
?
*dnn/hiddenlayer_2/batchnorm_2/gamma/AssignAssignVariableOp#dnn/hiddenlayer_2/batchnorm_2/gamma4dnn/hiddenlayer_2/batchnorm_2/gamma/Initializer/ones*
dtype0
?
7dnn/hiddenlayer_2/batchnorm_2/gamma/Read/ReadVariableOpReadVariableOp#dnn/hiddenlayer_2/batchnorm_2/gamma*
_output_shapes
:d*
dtype0
?
4dnn/hiddenlayer_2/batchnorm_2/beta/Initializer/zerosConst*5
_class+
)'loc:@dnn/hiddenlayer_2/batchnorm_2/beta*
_output_shapes
:d*
dtype0*
valueBd*    
?
"dnn/hiddenlayer_2/batchnorm_2/betaVarHandleOp*5
_class+
)'loc:@dnn/hiddenlayer_2/batchnorm_2/beta*
_output_shapes
: *
dtype0*
shape:d*3
shared_name$"dnn/hiddenlayer_2/batchnorm_2/beta
?
Cdnn/hiddenlayer_2/batchnorm_2/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp"dnn/hiddenlayer_2/batchnorm_2/beta*
_output_shapes
: 
?
)dnn/hiddenlayer_2/batchnorm_2/beta/AssignAssignVariableOp"dnn/hiddenlayer_2/batchnorm_2/beta4dnn/hiddenlayer_2/batchnorm_2/beta/Initializer/zeros*
dtype0
?
6dnn/hiddenlayer_2/batchnorm_2/beta/Read/ReadVariableOpReadVariableOp"dnn/hiddenlayer_2/batchnorm_2/beta*
_output_shapes
:d*
dtype0
?
;dnn/hiddenlayer_2/batchnorm_2/moving_mean/Initializer/zerosConst*<
_class2
0.loc:@dnn/hiddenlayer_2/batchnorm_2/moving_mean*
_output_shapes
:d*
dtype0*
valueBd*    
?
)dnn/hiddenlayer_2/batchnorm_2/moving_meanVarHandleOp*<
_class2
0.loc:@dnn/hiddenlayer_2/batchnorm_2/moving_mean*
_output_shapes
: *
dtype0*
shape:d*:
shared_name+)dnn/hiddenlayer_2/batchnorm_2/moving_mean
?
Jdnn/hiddenlayer_2/batchnorm_2/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp)dnn/hiddenlayer_2/batchnorm_2/moving_mean*
_output_shapes
: 
?
0dnn/hiddenlayer_2/batchnorm_2/moving_mean/AssignAssignVariableOp)dnn/hiddenlayer_2/batchnorm_2/moving_mean;dnn/hiddenlayer_2/batchnorm_2/moving_mean/Initializer/zeros*
dtype0
?
=dnn/hiddenlayer_2/batchnorm_2/moving_mean/Read/ReadVariableOpReadVariableOp)dnn/hiddenlayer_2/batchnorm_2/moving_mean*
_output_shapes
:d*
dtype0
?
>dnn/hiddenlayer_2/batchnorm_2/moving_variance/Initializer/onesConst*@
_class6
42loc:@dnn/hiddenlayer_2/batchnorm_2/moving_variance*
_output_shapes
:d*
dtype0*
valueBd*  ??
?
-dnn/hiddenlayer_2/batchnorm_2/moving_varianceVarHandleOp*@
_class6
42loc:@dnn/hiddenlayer_2/batchnorm_2/moving_variance*
_output_shapes
: *
dtype0*
shape:d*>
shared_name/-dnn/hiddenlayer_2/batchnorm_2/moving_variance
?
Ndnn/hiddenlayer_2/batchnorm_2/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp-dnn/hiddenlayer_2/batchnorm_2/moving_variance*
_output_shapes
: 
?
4dnn/hiddenlayer_2/batchnorm_2/moving_variance/AssignAssignVariableOp-dnn/hiddenlayer_2/batchnorm_2/moving_variance>dnn/hiddenlayer_2/batchnorm_2/moving_variance/Initializer/ones*
dtype0
?
Adnn/hiddenlayer_2/batchnorm_2/moving_variance/Read/ReadVariableOpReadVariableOp-dnn/hiddenlayer_2/batchnorm_2/moving_variance*
_output_shapes
:d*
dtype0
?
6dnn/hiddenlayer_2/batchnorm_2/batchnorm/ReadVariableOpReadVariableOp-dnn/hiddenlayer_2/batchnorm_2/moving_variance*
_output_shapes
:d*
dtype0
r
-dnn/hiddenlayer_2/batchnorm_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:
?
+dnn/hiddenlayer_2/batchnorm_2/batchnorm/addAddV26dnn/hiddenlayer_2/batchnorm_2/batchnorm/ReadVariableOp-dnn/hiddenlayer_2/batchnorm_2/batchnorm/add/y*
T0*
_output_shapes
:d
?
-dnn/hiddenlayer_2/batchnorm_2/batchnorm/RsqrtRsqrt+dnn/hiddenlayer_2/batchnorm_2/batchnorm/add*
T0*
_output_shapes
:d
?
:dnn/hiddenlayer_2/batchnorm_2/batchnorm/mul/ReadVariableOpReadVariableOp#dnn/hiddenlayer_2/batchnorm_2/gamma*
_output_shapes
:d*
dtype0
?
+dnn/hiddenlayer_2/batchnorm_2/batchnorm/mulMul-dnn/hiddenlayer_2/batchnorm_2/batchnorm/Rsqrt:dnn/hiddenlayer_2/batchnorm_2/batchnorm/mul/ReadVariableOp*
T0*
_output_shapes
:d
?
-dnn/hiddenlayer_2/batchnorm_2/batchnorm/mul_1Muldnn/hiddenlayer_2/LeakyRelu+dnn/hiddenlayer_2/batchnorm_2/batchnorm/mul*
T0*'
_output_shapes
:?????????d
?
8dnn/hiddenlayer_2/batchnorm_2/batchnorm/ReadVariableOp_1ReadVariableOp)dnn/hiddenlayer_2/batchnorm_2/moving_mean*
_output_shapes
:d*
dtype0
?
-dnn/hiddenlayer_2/batchnorm_2/batchnorm/mul_2Mul8dnn/hiddenlayer_2/batchnorm_2/batchnorm/ReadVariableOp_1+dnn/hiddenlayer_2/batchnorm_2/batchnorm/mul*
T0*
_output_shapes
:d
?
8dnn/hiddenlayer_2/batchnorm_2/batchnorm/ReadVariableOp_2ReadVariableOp"dnn/hiddenlayer_2/batchnorm_2/beta*
_output_shapes
:d*
dtype0
?
+dnn/hiddenlayer_2/batchnorm_2/batchnorm/subSub8dnn/hiddenlayer_2/batchnorm_2/batchnorm/ReadVariableOp_2-dnn/hiddenlayer_2/batchnorm_2/batchnorm/mul_2*
T0*
_output_shapes
:d
?
-dnn/hiddenlayer_2/batchnorm_2/batchnorm/add_1AddV2-dnn/hiddenlayer_2/batchnorm_2/batchnorm/mul_1+dnn/hiddenlayer_2/batchnorm_2/batchnorm/sub*
T0*'
_output_shapes
:?????????d
?
dnn/zero_fraction_2/SizeSize-dnn/hiddenlayer_2/batchnorm_2/batchnorm/add_1*
T0*
_output_shapes
: *
out_type0	
e
dnn/zero_fraction_2/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R????
?
dnn/zero_fraction_2/LessEqual	LessEqualdnn/zero_fraction_2/Sizednn/zero_fraction_2/LessEqual/y*
T0	*
_output_shapes
: 
?
dnn/zero_fraction_2/condStatelessIfdnn/zero_fraction_2/LessEqual-dnn/hiddenlayer_2/batchnorm_2/batchnorm/add_1*
Tcond0
*
Tin
2*
Tout

2	*
_lower_using_switch_merge(* 
_output_shapes
: : : : : : * 
_read_only_resource_inputs
 *8
else_branch)R'
%dnn_zero_fraction_2_cond_false_511724*
output_shapes
: : : : : : *7
then_branch(R&
$dnn_zero_fraction_2_cond_true_511723
h
!dnn/zero_fraction_2/cond/IdentityIdentitydnn/zero_fraction_2/cond*
T0	*
_output_shapes
: 
l
#dnn/zero_fraction_2/cond/Identity_1Identitydnn/zero_fraction_2/cond:1*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_2/cond/Identity_2Identitydnn/zero_fraction_2/cond:2*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_2/cond/Identity_3Identitydnn/zero_fraction_2/cond:3*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_2/cond/Identity_4Identitydnn/zero_fraction_2/cond:4*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_2/cond/Identity_5Identitydnn/zero_fraction_2/cond:5*
T0*
_output_shapes
: 
?
*dnn/zero_fraction_2/counts_to_fraction/subSubdnn/zero_fraction_2/Size!dnn/zero_fraction_2/cond/Identity*
T0	*
_output_shapes
: 
?
+dnn/zero_fraction_2/counts_to_fraction/CastCast*dnn/zero_fraction_2/counts_to_fraction/sub*

DstT0*

SrcT0	*
_output_shapes
: 

-dnn/zero_fraction_2/counts_to_fraction/Cast_1Castdnn/zero_fraction_2/Size*

DstT0*

SrcT0	*
_output_shapes
: 
?
.dnn/zero_fraction_2/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_2/counts_to_fraction/Cast-dnn/zero_fraction_2/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
y
dnn/zero_fraction_2/fractionIdentity.dnn/zero_fraction_2/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
.dnn/hiddenlayer_2/fraction_of_zero_values/tagsConst*
_output_shapes
: *
dtype0*:
value1B/ B)dnn/hiddenlayer_2/fraction_of_zero_values
?
)dnn/hiddenlayer_2/fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_2/fraction_of_zero_values/tagsdnn/zero_fraction_2/fraction*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_2/activation/tagConst*
_output_shapes
: *
dtype0*-
value$B" Bdnn/hiddenlayer_2/activation
?
dnn/hiddenlayer_2/activationHistogramSummary dnn/hiddenlayer_2/activation/tag-dnn/hiddenlayer_2/batchnorm_2/batchnorm/add_1*
_output_shapes
: 
?
9dnn/hiddenlayer_3/kernel/Initializer/random_uniform/shapeConst*+
_class!
loc:@dnn/hiddenlayer_3/kernel*
_output_shapes
:*
dtype0*
valueB"d   2   
?
7dnn/hiddenlayer_3/kernel/Initializer/random_uniform/minConst*+
_class!
loc:@dnn/hiddenlayer_3/kernel*
_output_shapes
: *
dtype0*
valueB
 *??L?
?
7dnn/hiddenlayer_3/kernel/Initializer/random_uniform/maxConst*+
_class!
loc:@dnn/hiddenlayer_3/kernel*
_output_shapes
: *
dtype0*
valueB
 *??L>
?
Adnn/hiddenlayer_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform9dnn/hiddenlayer_3/kernel/Initializer/random_uniform/shape*
T0*+
_class!
loc:@dnn/hiddenlayer_3/kernel*
_output_shapes

:d2*
dtype0*
seed????*
seed2	
?
7dnn/hiddenlayer_3/kernel/Initializer/random_uniform/subSub7dnn/hiddenlayer_3/kernel/Initializer/random_uniform/max7dnn/hiddenlayer_3/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/hiddenlayer_3/kernel*
_output_shapes
: 
?
7dnn/hiddenlayer_3/kernel/Initializer/random_uniform/mulMulAdnn/hiddenlayer_3/kernel/Initializer/random_uniform/RandomUniform7dnn/hiddenlayer_3/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@dnn/hiddenlayer_3/kernel*
_output_shapes

:d2
?
3dnn/hiddenlayer_3/kernel/Initializer/random_uniformAdd7dnn/hiddenlayer_3/kernel/Initializer/random_uniform/mul7dnn/hiddenlayer_3/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/hiddenlayer_3/kernel*
_output_shapes

:d2
?
dnn/hiddenlayer_3/kernelVarHandleOp*+
_class!
loc:@dnn/hiddenlayer_3/kernel*
_output_shapes
: *
dtype0*
shape
:d2*)
shared_namednn/hiddenlayer_3/kernel
?
9dnn/hiddenlayer_3/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_3/kernel*
_output_shapes
: 
?
dnn/hiddenlayer_3/kernel/AssignAssignVariableOpdnn/hiddenlayer_3/kernel3dnn/hiddenlayer_3/kernel/Initializer/random_uniform*
dtype0
?
,dnn/hiddenlayer_3/kernel/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_3/kernel*
_output_shapes

:d2*
dtype0
?
(dnn/hiddenlayer_3/bias/Initializer/zerosConst*)
_class
loc:@dnn/hiddenlayer_3/bias*
_output_shapes
:2*
dtype0*
valueB2*    
?
dnn/hiddenlayer_3/biasVarHandleOp*)
_class
loc:@dnn/hiddenlayer_3/bias*
_output_shapes
: *
dtype0*
shape:2*'
shared_namednn/hiddenlayer_3/bias
}
7dnn/hiddenlayer_3/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_3/bias*
_output_shapes
: 
?
dnn/hiddenlayer_3/bias/AssignAssignVariableOpdnn/hiddenlayer_3/bias(dnn/hiddenlayer_3/bias/Initializer/zeros*
dtype0
}
*dnn/hiddenlayer_3/bias/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_3/bias*
_output_shapes
:2*
dtype0
?
'dnn/hiddenlayer_3/MatMul/ReadVariableOpReadVariableOpdnn/hiddenlayer_3/kernel*
_output_shapes

:d2*
dtype0
?
dnn/hiddenlayer_3/MatMulMatMul-dnn/hiddenlayer_2/batchnorm_2/batchnorm/add_1'dnn/hiddenlayer_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2
{
(dnn/hiddenlayer_3/BiasAdd/ReadVariableOpReadVariableOpdnn/hiddenlayer_3/bias*
_output_shapes
:2*
dtype0
?
dnn/hiddenlayer_3/BiasAddBiasAdddnn/hiddenlayer_3/MatMul(dnn/hiddenlayer_3/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:?????????2
l
dnn/hiddenlayer_3/LeakyRelu	LeakyReludnn/hiddenlayer_3/BiasAdd*'
_output_shapes
:?????????2
?
4dnn/hiddenlayer_3/batchnorm_3/gamma/Initializer/onesConst*6
_class,
*(loc:@dnn/hiddenlayer_3/batchnorm_3/gamma*
_output_shapes
:2*
dtype0*
valueB2*  ??
?
#dnn/hiddenlayer_3/batchnorm_3/gammaVarHandleOp*6
_class,
*(loc:@dnn/hiddenlayer_3/batchnorm_3/gamma*
_output_shapes
: *
dtype0*
shape:2*4
shared_name%#dnn/hiddenlayer_3/batchnorm_3/gamma
?
Ddnn/hiddenlayer_3/batchnorm_3/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp#dnn/hiddenlayer_3/batchnorm_3/gamma*
_output_shapes
: 
?
*dnn/hiddenlayer_3/batchnorm_3/gamma/AssignAssignVariableOp#dnn/hiddenlayer_3/batchnorm_3/gamma4dnn/hiddenlayer_3/batchnorm_3/gamma/Initializer/ones*
dtype0
?
7dnn/hiddenlayer_3/batchnorm_3/gamma/Read/ReadVariableOpReadVariableOp#dnn/hiddenlayer_3/batchnorm_3/gamma*
_output_shapes
:2*
dtype0
?
4dnn/hiddenlayer_3/batchnorm_3/beta/Initializer/zerosConst*5
_class+
)'loc:@dnn/hiddenlayer_3/batchnorm_3/beta*
_output_shapes
:2*
dtype0*
valueB2*    
?
"dnn/hiddenlayer_3/batchnorm_3/betaVarHandleOp*5
_class+
)'loc:@dnn/hiddenlayer_3/batchnorm_3/beta*
_output_shapes
: *
dtype0*
shape:2*3
shared_name$"dnn/hiddenlayer_3/batchnorm_3/beta
?
Cdnn/hiddenlayer_3/batchnorm_3/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp"dnn/hiddenlayer_3/batchnorm_3/beta*
_output_shapes
: 
?
)dnn/hiddenlayer_3/batchnorm_3/beta/AssignAssignVariableOp"dnn/hiddenlayer_3/batchnorm_3/beta4dnn/hiddenlayer_3/batchnorm_3/beta/Initializer/zeros*
dtype0
?
6dnn/hiddenlayer_3/batchnorm_3/beta/Read/ReadVariableOpReadVariableOp"dnn/hiddenlayer_3/batchnorm_3/beta*
_output_shapes
:2*
dtype0
?
;dnn/hiddenlayer_3/batchnorm_3/moving_mean/Initializer/zerosConst*<
_class2
0.loc:@dnn/hiddenlayer_3/batchnorm_3/moving_mean*
_output_shapes
:2*
dtype0*
valueB2*    
?
)dnn/hiddenlayer_3/batchnorm_3/moving_meanVarHandleOp*<
_class2
0.loc:@dnn/hiddenlayer_3/batchnorm_3/moving_mean*
_output_shapes
: *
dtype0*
shape:2*:
shared_name+)dnn/hiddenlayer_3/batchnorm_3/moving_mean
?
Jdnn/hiddenlayer_3/batchnorm_3/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp)dnn/hiddenlayer_3/batchnorm_3/moving_mean*
_output_shapes
: 
?
0dnn/hiddenlayer_3/batchnorm_3/moving_mean/AssignAssignVariableOp)dnn/hiddenlayer_3/batchnorm_3/moving_mean;dnn/hiddenlayer_3/batchnorm_3/moving_mean/Initializer/zeros*
dtype0
?
=dnn/hiddenlayer_3/batchnorm_3/moving_mean/Read/ReadVariableOpReadVariableOp)dnn/hiddenlayer_3/batchnorm_3/moving_mean*
_output_shapes
:2*
dtype0
?
>dnn/hiddenlayer_3/batchnorm_3/moving_variance/Initializer/onesConst*@
_class6
42loc:@dnn/hiddenlayer_3/batchnorm_3/moving_variance*
_output_shapes
:2*
dtype0*
valueB2*  ??
?
-dnn/hiddenlayer_3/batchnorm_3/moving_varianceVarHandleOp*@
_class6
42loc:@dnn/hiddenlayer_3/batchnorm_3/moving_variance*
_output_shapes
: *
dtype0*
shape:2*>
shared_name/-dnn/hiddenlayer_3/batchnorm_3/moving_variance
?
Ndnn/hiddenlayer_3/batchnorm_3/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp-dnn/hiddenlayer_3/batchnorm_3/moving_variance*
_output_shapes
: 
?
4dnn/hiddenlayer_3/batchnorm_3/moving_variance/AssignAssignVariableOp-dnn/hiddenlayer_3/batchnorm_3/moving_variance>dnn/hiddenlayer_3/batchnorm_3/moving_variance/Initializer/ones*
dtype0
?
Adnn/hiddenlayer_3/batchnorm_3/moving_variance/Read/ReadVariableOpReadVariableOp-dnn/hiddenlayer_3/batchnorm_3/moving_variance*
_output_shapes
:2*
dtype0
?
6dnn/hiddenlayer_3/batchnorm_3/batchnorm/ReadVariableOpReadVariableOp-dnn/hiddenlayer_3/batchnorm_3/moving_variance*
_output_shapes
:2*
dtype0
r
-dnn/hiddenlayer_3/batchnorm_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:
?
+dnn/hiddenlayer_3/batchnorm_3/batchnorm/addAddV26dnn/hiddenlayer_3/batchnorm_3/batchnorm/ReadVariableOp-dnn/hiddenlayer_3/batchnorm_3/batchnorm/add/y*
T0*
_output_shapes
:2
?
-dnn/hiddenlayer_3/batchnorm_3/batchnorm/RsqrtRsqrt+dnn/hiddenlayer_3/batchnorm_3/batchnorm/add*
T0*
_output_shapes
:2
?
:dnn/hiddenlayer_3/batchnorm_3/batchnorm/mul/ReadVariableOpReadVariableOp#dnn/hiddenlayer_3/batchnorm_3/gamma*
_output_shapes
:2*
dtype0
?
+dnn/hiddenlayer_3/batchnorm_3/batchnorm/mulMul-dnn/hiddenlayer_3/batchnorm_3/batchnorm/Rsqrt:dnn/hiddenlayer_3/batchnorm_3/batchnorm/mul/ReadVariableOp*
T0*
_output_shapes
:2
?
-dnn/hiddenlayer_3/batchnorm_3/batchnorm/mul_1Muldnn/hiddenlayer_3/LeakyRelu+dnn/hiddenlayer_3/batchnorm_3/batchnorm/mul*
T0*'
_output_shapes
:?????????2
?
8dnn/hiddenlayer_3/batchnorm_3/batchnorm/ReadVariableOp_1ReadVariableOp)dnn/hiddenlayer_3/batchnorm_3/moving_mean*
_output_shapes
:2*
dtype0
?
-dnn/hiddenlayer_3/batchnorm_3/batchnorm/mul_2Mul8dnn/hiddenlayer_3/batchnorm_3/batchnorm/ReadVariableOp_1+dnn/hiddenlayer_3/batchnorm_3/batchnorm/mul*
T0*
_output_shapes
:2
?
8dnn/hiddenlayer_3/batchnorm_3/batchnorm/ReadVariableOp_2ReadVariableOp"dnn/hiddenlayer_3/batchnorm_3/beta*
_output_shapes
:2*
dtype0
?
+dnn/hiddenlayer_3/batchnorm_3/batchnorm/subSub8dnn/hiddenlayer_3/batchnorm_3/batchnorm/ReadVariableOp_2-dnn/hiddenlayer_3/batchnorm_3/batchnorm/mul_2*
T0*
_output_shapes
:2
?
-dnn/hiddenlayer_3/batchnorm_3/batchnorm/add_1AddV2-dnn/hiddenlayer_3/batchnorm_3/batchnorm/mul_1+dnn/hiddenlayer_3/batchnorm_3/batchnorm/sub*
T0*'
_output_shapes
:?????????2
?
dnn/zero_fraction_3/SizeSize-dnn/hiddenlayer_3/batchnorm_3/batchnorm/add_1*
T0*
_output_shapes
: *
out_type0	
e
dnn/zero_fraction_3/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R????
?
dnn/zero_fraction_3/LessEqual	LessEqualdnn/zero_fraction_3/Sizednn/zero_fraction_3/LessEqual/y*
T0	*
_output_shapes
: 
?
dnn/zero_fraction_3/condStatelessIfdnn/zero_fraction_3/LessEqual-dnn/hiddenlayer_3/batchnorm_3/batchnorm/add_1*
Tcond0
*
Tin
2*
Tout

2	*
_lower_using_switch_merge(* 
_output_shapes
: : : : : : * 
_read_only_resource_inputs
 *8
else_branch)R'
%dnn_zero_fraction_3_cond_false_511822*
output_shapes
: : : : : : *7
then_branch(R&
$dnn_zero_fraction_3_cond_true_511821
h
!dnn/zero_fraction_3/cond/IdentityIdentitydnn/zero_fraction_3/cond*
T0	*
_output_shapes
: 
l
#dnn/zero_fraction_3/cond/Identity_1Identitydnn/zero_fraction_3/cond:1*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_3/cond/Identity_2Identitydnn/zero_fraction_3/cond:2*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_3/cond/Identity_3Identitydnn/zero_fraction_3/cond:3*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_3/cond/Identity_4Identitydnn/zero_fraction_3/cond:4*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_3/cond/Identity_5Identitydnn/zero_fraction_3/cond:5*
T0*
_output_shapes
: 
?
*dnn/zero_fraction_3/counts_to_fraction/subSubdnn/zero_fraction_3/Size!dnn/zero_fraction_3/cond/Identity*
T0	*
_output_shapes
: 
?
+dnn/zero_fraction_3/counts_to_fraction/CastCast*dnn/zero_fraction_3/counts_to_fraction/sub*

DstT0*

SrcT0	*
_output_shapes
: 

-dnn/zero_fraction_3/counts_to_fraction/Cast_1Castdnn/zero_fraction_3/Size*

DstT0*

SrcT0	*
_output_shapes
: 
?
.dnn/zero_fraction_3/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_3/counts_to_fraction/Cast-dnn/zero_fraction_3/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
y
dnn/zero_fraction_3/fractionIdentity.dnn/zero_fraction_3/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
.dnn/hiddenlayer_3/fraction_of_zero_values/tagsConst*
_output_shapes
: *
dtype0*:
value1B/ B)dnn/hiddenlayer_3/fraction_of_zero_values
?
)dnn/hiddenlayer_3/fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_3/fraction_of_zero_values/tagsdnn/zero_fraction_3/fraction*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_3/activation/tagConst*
_output_shapes
: *
dtype0*-
value$B" Bdnn/hiddenlayer_3/activation
?
dnn/hiddenlayer_3/activationHistogramSummary dnn/hiddenlayer_3/activation/tag-dnn/hiddenlayer_3/batchnorm_3/batchnorm/add_1*
_output_shapes
: 
?
9dnn/hiddenlayer_4/kernel/Initializer/random_uniform/shapeConst*+
_class!
loc:@dnn/hiddenlayer_4/kernel*
_output_shapes
:*
dtype0*
valueB"2   2   
?
7dnn/hiddenlayer_4/kernel/Initializer/random_uniform/minConst*+
_class!
loc:@dnn/hiddenlayer_4/kernel*
_output_shapes
: *
dtype0*
valueB
 *??z?
?
7dnn/hiddenlayer_4/kernel/Initializer/random_uniform/maxConst*+
_class!
loc:@dnn/hiddenlayer_4/kernel*
_output_shapes
: *
dtype0*
valueB
 *??z>
?
Adnn/hiddenlayer_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform9dnn/hiddenlayer_4/kernel/Initializer/random_uniform/shape*
T0*+
_class!
loc:@dnn/hiddenlayer_4/kernel*
_output_shapes

:22*
dtype0*
seed????*
seed2

?
7dnn/hiddenlayer_4/kernel/Initializer/random_uniform/subSub7dnn/hiddenlayer_4/kernel/Initializer/random_uniform/max7dnn/hiddenlayer_4/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/hiddenlayer_4/kernel*
_output_shapes
: 
?
7dnn/hiddenlayer_4/kernel/Initializer/random_uniform/mulMulAdnn/hiddenlayer_4/kernel/Initializer/random_uniform/RandomUniform7dnn/hiddenlayer_4/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@dnn/hiddenlayer_4/kernel*
_output_shapes

:22
?
3dnn/hiddenlayer_4/kernel/Initializer/random_uniformAdd7dnn/hiddenlayer_4/kernel/Initializer/random_uniform/mul7dnn/hiddenlayer_4/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/hiddenlayer_4/kernel*
_output_shapes

:22
?
dnn/hiddenlayer_4/kernelVarHandleOp*+
_class!
loc:@dnn/hiddenlayer_4/kernel*
_output_shapes
: *
dtype0*
shape
:22*)
shared_namednn/hiddenlayer_4/kernel
?
9dnn/hiddenlayer_4/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_4/kernel*
_output_shapes
: 
?
dnn/hiddenlayer_4/kernel/AssignAssignVariableOpdnn/hiddenlayer_4/kernel3dnn/hiddenlayer_4/kernel/Initializer/random_uniform*
dtype0
?
,dnn/hiddenlayer_4/kernel/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_4/kernel*
_output_shapes

:22*
dtype0
?
(dnn/hiddenlayer_4/bias/Initializer/zerosConst*)
_class
loc:@dnn/hiddenlayer_4/bias*
_output_shapes
:2*
dtype0*
valueB2*    
?
dnn/hiddenlayer_4/biasVarHandleOp*)
_class
loc:@dnn/hiddenlayer_4/bias*
_output_shapes
: *
dtype0*
shape:2*'
shared_namednn/hiddenlayer_4/bias
}
7dnn/hiddenlayer_4/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_4/bias*
_output_shapes
: 
?
dnn/hiddenlayer_4/bias/AssignAssignVariableOpdnn/hiddenlayer_4/bias(dnn/hiddenlayer_4/bias/Initializer/zeros*
dtype0
}
*dnn/hiddenlayer_4/bias/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_4/bias*
_output_shapes
:2*
dtype0
?
'dnn/hiddenlayer_4/MatMul/ReadVariableOpReadVariableOpdnn/hiddenlayer_4/kernel*
_output_shapes

:22*
dtype0
?
dnn/hiddenlayer_4/MatMulMatMul-dnn/hiddenlayer_3/batchnorm_3/batchnorm/add_1'dnn/hiddenlayer_4/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2
{
(dnn/hiddenlayer_4/BiasAdd/ReadVariableOpReadVariableOpdnn/hiddenlayer_4/bias*
_output_shapes
:2*
dtype0
?
dnn/hiddenlayer_4/BiasAddBiasAdddnn/hiddenlayer_4/MatMul(dnn/hiddenlayer_4/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:?????????2
l
dnn/hiddenlayer_4/LeakyRelu	LeakyReludnn/hiddenlayer_4/BiasAdd*'
_output_shapes
:?????????2
?
4dnn/hiddenlayer_4/batchnorm_4/gamma/Initializer/onesConst*6
_class,
*(loc:@dnn/hiddenlayer_4/batchnorm_4/gamma*
_output_shapes
:2*
dtype0*
valueB2*  ??
?
#dnn/hiddenlayer_4/batchnorm_4/gammaVarHandleOp*6
_class,
*(loc:@dnn/hiddenlayer_4/batchnorm_4/gamma*
_output_shapes
: *
dtype0*
shape:2*4
shared_name%#dnn/hiddenlayer_4/batchnorm_4/gamma
?
Ddnn/hiddenlayer_4/batchnorm_4/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp#dnn/hiddenlayer_4/batchnorm_4/gamma*
_output_shapes
: 
?
*dnn/hiddenlayer_4/batchnorm_4/gamma/AssignAssignVariableOp#dnn/hiddenlayer_4/batchnorm_4/gamma4dnn/hiddenlayer_4/batchnorm_4/gamma/Initializer/ones*
dtype0
?
7dnn/hiddenlayer_4/batchnorm_4/gamma/Read/ReadVariableOpReadVariableOp#dnn/hiddenlayer_4/batchnorm_4/gamma*
_output_shapes
:2*
dtype0
?
4dnn/hiddenlayer_4/batchnorm_4/beta/Initializer/zerosConst*5
_class+
)'loc:@dnn/hiddenlayer_4/batchnorm_4/beta*
_output_shapes
:2*
dtype0*
valueB2*    
?
"dnn/hiddenlayer_4/batchnorm_4/betaVarHandleOp*5
_class+
)'loc:@dnn/hiddenlayer_4/batchnorm_4/beta*
_output_shapes
: *
dtype0*
shape:2*3
shared_name$"dnn/hiddenlayer_4/batchnorm_4/beta
?
Cdnn/hiddenlayer_4/batchnorm_4/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp"dnn/hiddenlayer_4/batchnorm_4/beta*
_output_shapes
: 
?
)dnn/hiddenlayer_4/batchnorm_4/beta/AssignAssignVariableOp"dnn/hiddenlayer_4/batchnorm_4/beta4dnn/hiddenlayer_4/batchnorm_4/beta/Initializer/zeros*
dtype0
?
6dnn/hiddenlayer_4/batchnorm_4/beta/Read/ReadVariableOpReadVariableOp"dnn/hiddenlayer_4/batchnorm_4/beta*
_output_shapes
:2*
dtype0
?
;dnn/hiddenlayer_4/batchnorm_4/moving_mean/Initializer/zerosConst*<
_class2
0.loc:@dnn/hiddenlayer_4/batchnorm_4/moving_mean*
_output_shapes
:2*
dtype0*
valueB2*    
?
)dnn/hiddenlayer_4/batchnorm_4/moving_meanVarHandleOp*<
_class2
0.loc:@dnn/hiddenlayer_4/batchnorm_4/moving_mean*
_output_shapes
: *
dtype0*
shape:2*:
shared_name+)dnn/hiddenlayer_4/batchnorm_4/moving_mean
?
Jdnn/hiddenlayer_4/batchnorm_4/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp)dnn/hiddenlayer_4/batchnorm_4/moving_mean*
_output_shapes
: 
?
0dnn/hiddenlayer_4/batchnorm_4/moving_mean/AssignAssignVariableOp)dnn/hiddenlayer_4/batchnorm_4/moving_mean;dnn/hiddenlayer_4/batchnorm_4/moving_mean/Initializer/zeros*
dtype0
?
=dnn/hiddenlayer_4/batchnorm_4/moving_mean/Read/ReadVariableOpReadVariableOp)dnn/hiddenlayer_4/batchnorm_4/moving_mean*
_output_shapes
:2*
dtype0
?
>dnn/hiddenlayer_4/batchnorm_4/moving_variance/Initializer/onesConst*@
_class6
42loc:@dnn/hiddenlayer_4/batchnorm_4/moving_variance*
_output_shapes
:2*
dtype0*
valueB2*  ??
?
-dnn/hiddenlayer_4/batchnorm_4/moving_varianceVarHandleOp*@
_class6
42loc:@dnn/hiddenlayer_4/batchnorm_4/moving_variance*
_output_shapes
: *
dtype0*
shape:2*>
shared_name/-dnn/hiddenlayer_4/batchnorm_4/moving_variance
?
Ndnn/hiddenlayer_4/batchnorm_4/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp-dnn/hiddenlayer_4/batchnorm_4/moving_variance*
_output_shapes
: 
?
4dnn/hiddenlayer_4/batchnorm_4/moving_variance/AssignAssignVariableOp-dnn/hiddenlayer_4/batchnorm_4/moving_variance>dnn/hiddenlayer_4/batchnorm_4/moving_variance/Initializer/ones*
dtype0
?
Adnn/hiddenlayer_4/batchnorm_4/moving_variance/Read/ReadVariableOpReadVariableOp-dnn/hiddenlayer_4/batchnorm_4/moving_variance*
_output_shapes
:2*
dtype0
?
6dnn/hiddenlayer_4/batchnorm_4/batchnorm/ReadVariableOpReadVariableOp-dnn/hiddenlayer_4/batchnorm_4/moving_variance*
_output_shapes
:2*
dtype0
r
-dnn/hiddenlayer_4/batchnorm_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:
?
+dnn/hiddenlayer_4/batchnorm_4/batchnorm/addAddV26dnn/hiddenlayer_4/batchnorm_4/batchnorm/ReadVariableOp-dnn/hiddenlayer_4/batchnorm_4/batchnorm/add/y*
T0*
_output_shapes
:2
?
-dnn/hiddenlayer_4/batchnorm_4/batchnorm/RsqrtRsqrt+dnn/hiddenlayer_4/batchnorm_4/batchnorm/add*
T0*
_output_shapes
:2
?
:dnn/hiddenlayer_4/batchnorm_4/batchnorm/mul/ReadVariableOpReadVariableOp#dnn/hiddenlayer_4/batchnorm_4/gamma*
_output_shapes
:2*
dtype0
?
+dnn/hiddenlayer_4/batchnorm_4/batchnorm/mulMul-dnn/hiddenlayer_4/batchnorm_4/batchnorm/Rsqrt:dnn/hiddenlayer_4/batchnorm_4/batchnorm/mul/ReadVariableOp*
T0*
_output_shapes
:2
?
-dnn/hiddenlayer_4/batchnorm_4/batchnorm/mul_1Muldnn/hiddenlayer_4/LeakyRelu+dnn/hiddenlayer_4/batchnorm_4/batchnorm/mul*
T0*'
_output_shapes
:?????????2
?
8dnn/hiddenlayer_4/batchnorm_4/batchnorm/ReadVariableOp_1ReadVariableOp)dnn/hiddenlayer_4/batchnorm_4/moving_mean*
_output_shapes
:2*
dtype0
?
-dnn/hiddenlayer_4/batchnorm_4/batchnorm/mul_2Mul8dnn/hiddenlayer_4/batchnorm_4/batchnorm/ReadVariableOp_1+dnn/hiddenlayer_4/batchnorm_4/batchnorm/mul*
T0*
_output_shapes
:2
?
8dnn/hiddenlayer_4/batchnorm_4/batchnorm/ReadVariableOp_2ReadVariableOp"dnn/hiddenlayer_4/batchnorm_4/beta*
_output_shapes
:2*
dtype0
?
+dnn/hiddenlayer_4/batchnorm_4/batchnorm/subSub8dnn/hiddenlayer_4/batchnorm_4/batchnorm/ReadVariableOp_2-dnn/hiddenlayer_4/batchnorm_4/batchnorm/mul_2*
T0*
_output_shapes
:2
?
-dnn/hiddenlayer_4/batchnorm_4/batchnorm/add_1AddV2-dnn/hiddenlayer_4/batchnorm_4/batchnorm/mul_1+dnn/hiddenlayer_4/batchnorm_4/batchnorm/sub*
T0*'
_output_shapes
:?????????2
?
dnn/zero_fraction_4/SizeSize-dnn/hiddenlayer_4/batchnorm_4/batchnorm/add_1*
T0*
_output_shapes
: *
out_type0	
e
dnn/zero_fraction_4/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R????
?
dnn/zero_fraction_4/LessEqual	LessEqualdnn/zero_fraction_4/Sizednn/zero_fraction_4/LessEqual/y*
T0	*
_output_shapes
: 
?
dnn/zero_fraction_4/condStatelessIfdnn/zero_fraction_4/LessEqual-dnn/hiddenlayer_4/batchnorm_4/batchnorm/add_1*
Tcond0
*
Tin
2*
Tout

2	*
_lower_using_switch_merge(* 
_output_shapes
: : : : : : * 
_read_only_resource_inputs
 *8
else_branch)R'
%dnn_zero_fraction_4_cond_false_511920*
output_shapes
: : : : : : *7
then_branch(R&
$dnn_zero_fraction_4_cond_true_511919
h
!dnn/zero_fraction_4/cond/IdentityIdentitydnn/zero_fraction_4/cond*
T0	*
_output_shapes
: 
l
#dnn/zero_fraction_4/cond/Identity_1Identitydnn/zero_fraction_4/cond:1*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_4/cond/Identity_2Identitydnn/zero_fraction_4/cond:2*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_4/cond/Identity_3Identitydnn/zero_fraction_4/cond:3*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_4/cond/Identity_4Identitydnn/zero_fraction_4/cond:4*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_4/cond/Identity_5Identitydnn/zero_fraction_4/cond:5*
T0*
_output_shapes
: 
?
*dnn/zero_fraction_4/counts_to_fraction/subSubdnn/zero_fraction_4/Size!dnn/zero_fraction_4/cond/Identity*
T0	*
_output_shapes
: 
?
+dnn/zero_fraction_4/counts_to_fraction/CastCast*dnn/zero_fraction_4/counts_to_fraction/sub*

DstT0*

SrcT0	*
_output_shapes
: 

-dnn/zero_fraction_4/counts_to_fraction/Cast_1Castdnn/zero_fraction_4/Size*

DstT0*

SrcT0	*
_output_shapes
: 
?
.dnn/zero_fraction_4/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_4/counts_to_fraction/Cast-dnn/zero_fraction_4/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
y
dnn/zero_fraction_4/fractionIdentity.dnn/zero_fraction_4/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
.dnn/hiddenlayer_4/fraction_of_zero_values/tagsConst*
_output_shapes
: *
dtype0*:
value1B/ B)dnn/hiddenlayer_4/fraction_of_zero_values
?
)dnn/hiddenlayer_4/fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_4/fraction_of_zero_values/tagsdnn/zero_fraction_4/fraction*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_4/activation/tagConst*
_output_shapes
: *
dtype0*-
value$B" Bdnn/hiddenlayer_4/activation
?
dnn/hiddenlayer_4/activationHistogramSummary dnn/hiddenlayer_4/activation/tag-dnn/hiddenlayer_4/batchnorm_4/batchnorm/add_1*
_output_shapes
: 
?
9dnn/hiddenlayer_5/kernel/Initializer/random_uniform/shapeConst*+
_class!
loc:@dnn/hiddenlayer_5/kernel*
_output_shapes
:*
dtype0*
valueB"2   2   
?
7dnn/hiddenlayer_5/kernel/Initializer/random_uniform/minConst*+
_class!
loc:@dnn/hiddenlayer_5/kernel*
_output_shapes
: *
dtype0*
valueB
 *??z?
?
7dnn/hiddenlayer_5/kernel/Initializer/random_uniform/maxConst*+
_class!
loc:@dnn/hiddenlayer_5/kernel*
_output_shapes
: *
dtype0*
valueB
 *??z>
?
Adnn/hiddenlayer_5/kernel/Initializer/random_uniform/RandomUniformRandomUniform9dnn/hiddenlayer_5/kernel/Initializer/random_uniform/shape*
T0*+
_class!
loc:@dnn/hiddenlayer_5/kernel*
_output_shapes

:22*
dtype0*
seed????*
seed2
?
7dnn/hiddenlayer_5/kernel/Initializer/random_uniform/subSub7dnn/hiddenlayer_5/kernel/Initializer/random_uniform/max7dnn/hiddenlayer_5/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/hiddenlayer_5/kernel*
_output_shapes
: 
?
7dnn/hiddenlayer_5/kernel/Initializer/random_uniform/mulMulAdnn/hiddenlayer_5/kernel/Initializer/random_uniform/RandomUniform7dnn/hiddenlayer_5/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@dnn/hiddenlayer_5/kernel*
_output_shapes

:22
?
3dnn/hiddenlayer_5/kernel/Initializer/random_uniformAdd7dnn/hiddenlayer_5/kernel/Initializer/random_uniform/mul7dnn/hiddenlayer_5/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/hiddenlayer_5/kernel*
_output_shapes

:22
?
dnn/hiddenlayer_5/kernelVarHandleOp*+
_class!
loc:@dnn/hiddenlayer_5/kernel*
_output_shapes
: *
dtype0*
shape
:22*)
shared_namednn/hiddenlayer_5/kernel
?
9dnn/hiddenlayer_5/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_5/kernel*
_output_shapes
: 
?
dnn/hiddenlayer_5/kernel/AssignAssignVariableOpdnn/hiddenlayer_5/kernel3dnn/hiddenlayer_5/kernel/Initializer/random_uniform*
dtype0
?
,dnn/hiddenlayer_5/kernel/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_5/kernel*
_output_shapes

:22*
dtype0
?
(dnn/hiddenlayer_5/bias/Initializer/zerosConst*)
_class
loc:@dnn/hiddenlayer_5/bias*
_output_shapes
:2*
dtype0*
valueB2*    
?
dnn/hiddenlayer_5/biasVarHandleOp*)
_class
loc:@dnn/hiddenlayer_5/bias*
_output_shapes
: *
dtype0*
shape:2*'
shared_namednn/hiddenlayer_5/bias
}
7dnn/hiddenlayer_5/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_5/bias*
_output_shapes
: 
?
dnn/hiddenlayer_5/bias/AssignAssignVariableOpdnn/hiddenlayer_5/bias(dnn/hiddenlayer_5/bias/Initializer/zeros*
dtype0
}
*dnn/hiddenlayer_5/bias/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_5/bias*
_output_shapes
:2*
dtype0
?
'dnn/hiddenlayer_5/MatMul/ReadVariableOpReadVariableOpdnn/hiddenlayer_5/kernel*
_output_shapes

:22*
dtype0
?
dnn/hiddenlayer_5/MatMulMatMul-dnn/hiddenlayer_4/batchnorm_4/batchnorm/add_1'dnn/hiddenlayer_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2
{
(dnn/hiddenlayer_5/BiasAdd/ReadVariableOpReadVariableOpdnn/hiddenlayer_5/bias*
_output_shapes
:2*
dtype0
?
dnn/hiddenlayer_5/BiasAddBiasAdddnn/hiddenlayer_5/MatMul(dnn/hiddenlayer_5/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:?????????2
l
dnn/hiddenlayer_5/LeakyRelu	LeakyReludnn/hiddenlayer_5/BiasAdd*'
_output_shapes
:?????????2
?
4dnn/hiddenlayer_5/batchnorm_5/gamma/Initializer/onesConst*6
_class,
*(loc:@dnn/hiddenlayer_5/batchnorm_5/gamma*
_output_shapes
:2*
dtype0*
valueB2*  ??
?
#dnn/hiddenlayer_5/batchnorm_5/gammaVarHandleOp*6
_class,
*(loc:@dnn/hiddenlayer_5/batchnorm_5/gamma*
_output_shapes
: *
dtype0*
shape:2*4
shared_name%#dnn/hiddenlayer_5/batchnorm_5/gamma
?
Ddnn/hiddenlayer_5/batchnorm_5/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp#dnn/hiddenlayer_5/batchnorm_5/gamma*
_output_shapes
: 
?
*dnn/hiddenlayer_5/batchnorm_5/gamma/AssignAssignVariableOp#dnn/hiddenlayer_5/batchnorm_5/gamma4dnn/hiddenlayer_5/batchnorm_5/gamma/Initializer/ones*
dtype0
?
7dnn/hiddenlayer_5/batchnorm_5/gamma/Read/ReadVariableOpReadVariableOp#dnn/hiddenlayer_5/batchnorm_5/gamma*
_output_shapes
:2*
dtype0
?
4dnn/hiddenlayer_5/batchnorm_5/beta/Initializer/zerosConst*5
_class+
)'loc:@dnn/hiddenlayer_5/batchnorm_5/beta*
_output_shapes
:2*
dtype0*
valueB2*    
?
"dnn/hiddenlayer_5/batchnorm_5/betaVarHandleOp*5
_class+
)'loc:@dnn/hiddenlayer_5/batchnorm_5/beta*
_output_shapes
: *
dtype0*
shape:2*3
shared_name$"dnn/hiddenlayer_5/batchnorm_5/beta
?
Cdnn/hiddenlayer_5/batchnorm_5/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp"dnn/hiddenlayer_5/batchnorm_5/beta*
_output_shapes
: 
?
)dnn/hiddenlayer_5/batchnorm_5/beta/AssignAssignVariableOp"dnn/hiddenlayer_5/batchnorm_5/beta4dnn/hiddenlayer_5/batchnorm_5/beta/Initializer/zeros*
dtype0
?
6dnn/hiddenlayer_5/batchnorm_5/beta/Read/ReadVariableOpReadVariableOp"dnn/hiddenlayer_5/batchnorm_5/beta*
_output_shapes
:2*
dtype0
?
;dnn/hiddenlayer_5/batchnorm_5/moving_mean/Initializer/zerosConst*<
_class2
0.loc:@dnn/hiddenlayer_5/batchnorm_5/moving_mean*
_output_shapes
:2*
dtype0*
valueB2*    
?
)dnn/hiddenlayer_5/batchnorm_5/moving_meanVarHandleOp*<
_class2
0.loc:@dnn/hiddenlayer_5/batchnorm_5/moving_mean*
_output_shapes
: *
dtype0*
shape:2*:
shared_name+)dnn/hiddenlayer_5/batchnorm_5/moving_mean
?
Jdnn/hiddenlayer_5/batchnorm_5/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp)dnn/hiddenlayer_5/batchnorm_5/moving_mean*
_output_shapes
: 
?
0dnn/hiddenlayer_5/batchnorm_5/moving_mean/AssignAssignVariableOp)dnn/hiddenlayer_5/batchnorm_5/moving_mean;dnn/hiddenlayer_5/batchnorm_5/moving_mean/Initializer/zeros*
dtype0
?
=dnn/hiddenlayer_5/batchnorm_5/moving_mean/Read/ReadVariableOpReadVariableOp)dnn/hiddenlayer_5/batchnorm_5/moving_mean*
_output_shapes
:2*
dtype0
?
>dnn/hiddenlayer_5/batchnorm_5/moving_variance/Initializer/onesConst*@
_class6
42loc:@dnn/hiddenlayer_5/batchnorm_5/moving_variance*
_output_shapes
:2*
dtype0*
valueB2*  ??
?
-dnn/hiddenlayer_5/batchnorm_5/moving_varianceVarHandleOp*@
_class6
42loc:@dnn/hiddenlayer_5/batchnorm_5/moving_variance*
_output_shapes
: *
dtype0*
shape:2*>
shared_name/-dnn/hiddenlayer_5/batchnorm_5/moving_variance
?
Ndnn/hiddenlayer_5/batchnorm_5/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp-dnn/hiddenlayer_5/batchnorm_5/moving_variance*
_output_shapes
: 
?
4dnn/hiddenlayer_5/batchnorm_5/moving_variance/AssignAssignVariableOp-dnn/hiddenlayer_5/batchnorm_5/moving_variance>dnn/hiddenlayer_5/batchnorm_5/moving_variance/Initializer/ones*
dtype0
?
Adnn/hiddenlayer_5/batchnorm_5/moving_variance/Read/ReadVariableOpReadVariableOp-dnn/hiddenlayer_5/batchnorm_5/moving_variance*
_output_shapes
:2*
dtype0
?
6dnn/hiddenlayer_5/batchnorm_5/batchnorm/ReadVariableOpReadVariableOp-dnn/hiddenlayer_5/batchnorm_5/moving_variance*
_output_shapes
:2*
dtype0
r
-dnn/hiddenlayer_5/batchnorm_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:
?
+dnn/hiddenlayer_5/batchnorm_5/batchnorm/addAddV26dnn/hiddenlayer_5/batchnorm_5/batchnorm/ReadVariableOp-dnn/hiddenlayer_5/batchnorm_5/batchnorm/add/y*
T0*
_output_shapes
:2
?
-dnn/hiddenlayer_5/batchnorm_5/batchnorm/RsqrtRsqrt+dnn/hiddenlayer_5/batchnorm_5/batchnorm/add*
T0*
_output_shapes
:2
?
:dnn/hiddenlayer_5/batchnorm_5/batchnorm/mul/ReadVariableOpReadVariableOp#dnn/hiddenlayer_5/batchnorm_5/gamma*
_output_shapes
:2*
dtype0
?
+dnn/hiddenlayer_5/batchnorm_5/batchnorm/mulMul-dnn/hiddenlayer_5/batchnorm_5/batchnorm/Rsqrt:dnn/hiddenlayer_5/batchnorm_5/batchnorm/mul/ReadVariableOp*
T0*
_output_shapes
:2
?
-dnn/hiddenlayer_5/batchnorm_5/batchnorm/mul_1Muldnn/hiddenlayer_5/LeakyRelu+dnn/hiddenlayer_5/batchnorm_5/batchnorm/mul*
T0*'
_output_shapes
:?????????2
?
8dnn/hiddenlayer_5/batchnorm_5/batchnorm/ReadVariableOp_1ReadVariableOp)dnn/hiddenlayer_5/batchnorm_5/moving_mean*
_output_shapes
:2*
dtype0
?
-dnn/hiddenlayer_5/batchnorm_5/batchnorm/mul_2Mul8dnn/hiddenlayer_5/batchnorm_5/batchnorm/ReadVariableOp_1+dnn/hiddenlayer_5/batchnorm_5/batchnorm/mul*
T0*
_output_shapes
:2
?
8dnn/hiddenlayer_5/batchnorm_5/batchnorm/ReadVariableOp_2ReadVariableOp"dnn/hiddenlayer_5/batchnorm_5/beta*
_output_shapes
:2*
dtype0
?
+dnn/hiddenlayer_5/batchnorm_5/batchnorm/subSub8dnn/hiddenlayer_5/batchnorm_5/batchnorm/ReadVariableOp_2-dnn/hiddenlayer_5/batchnorm_5/batchnorm/mul_2*
T0*
_output_shapes
:2
?
-dnn/hiddenlayer_5/batchnorm_5/batchnorm/add_1AddV2-dnn/hiddenlayer_5/batchnorm_5/batchnorm/mul_1+dnn/hiddenlayer_5/batchnorm_5/batchnorm/sub*
T0*'
_output_shapes
:?????????2
?
dnn/zero_fraction_5/SizeSize-dnn/hiddenlayer_5/batchnorm_5/batchnorm/add_1*
T0*
_output_shapes
: *
out_type0	
e
dnn/zero_fraction_5/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R????
?
dnn/zero_fraction_5/LessEqual	LessEqualdnn/zero_fraction_5/Sizednn/zero_fraction_5/LessEqual/y*
T0	*
_output_shapes
: 
?
dnn/zero_fraction_5/condStatelessIfdnn/zero_fraction_5/LessEqual-dnn/hiddenlayer_5/batchnorm_5/batchnorm/add_1*
Tcond0
*
Tin
2*
Tout

2	*
_lower_using_switch_merge(* 
_output_shapes
: : : : : : * 
_read_only_resource_inputs
 *8
else_branch)R'
%dnn_zero_fraction_5_cond_false_512018*
output_shapes
: : : : : : *7
then_branch(R&
$dnn_zero_fraction_5_cond_true_512017
h
!dnn/zero_fraction_5/cond/IdentityIdentitydnn/zero_fraction_5/cond*
T0	*
_output_shapes
: 
l
#dnn/zero_fraction_5/cond/Identity_1Identitydnn/zero_fraction_5/cond:1*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_5/cond/Identity_2Identitydnn/zero_fraction_5/cond:2*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_5/cond/Identity_3Identitydnn/zero_fraction_5/cond:3*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_5/cond/Identity_4Identitydnn/zero_fraction_5/cond:4*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_5/cond/Identity_5Identitydnn/zero_fraction_5/cond:5*
T0*
_output_shapes
: 
?
*dnn/zero_fraction_5/counts_to_fraction/subSubdnn/zero_fraction_5/Size!dnn/zero_fraction_5/cond/Identity*
T0	*
_output_shapes
: 
?
+dnn/zero_fraction_5/counts_to_fraction/CastCast*dnn/zero_fraction_5/counts_to_fraction/sub*

DstT0*

SrcT0	*
_output_shapes
: 

-dnn/zero_fraction_5/counts_to_fraction/Cast_1Castdnn/zero_fraction_5/Size*

DstT0*

SrcT0	*
_output_shapes
: 
?
.dnn/zero_fraction_5/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_5/counts_to_fraction/Cast-dnn/zero_fraction_5/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
y
dnn/zero_fraction_5/fractionIdentity.dnn/zero_fraction_5/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
.dnn/hiddenlayer_5/fraction_of_zero_values/tagsConst*
_output_shapes
: *
dtype0*:
value1B/ B)dnn/hiddenlayer_5/fraction_of_zero_values
?
)dnn/hiddenlayer_5/fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_5/fraction_of_zero_values/tagsdnn/zero_fraction_5/fraction*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_5/activation/tagConst*
_output_shapes
: *
dtype0*-
value$B" Bdnn/hiddenlayer_5/activation
?
dnn/hiddenlayer_5/activationHistogramSummary dnn/hiddenlayer_5/activation/tag-dnn/hiddenlayer_5/batchnorm_5/batchnorm/add_1*
_output_shapes
: 
?
9dnn/hiddenlayer_6/kernel/Initializer/random_uniform/shapeConst*+
_class!
loc:@dnn/hiddenlayer_6/kernel*
_output_shapes
:*
dtype0*
valueB"2   
   
?
7dnn/hiddenlayer_6/kernel/Initializer/random_uniform/minConst*+
_class!
loc:@dnn/hiddenlayer_6/kernel*
_output_shapes
: *
dtype0*
valueB
 *?衾
?
7dnn/hiddenlayer_6/kernel/Initializer/random_uniform/maxConst*+
_class!
loc:@dnn/hiddenlayer_6/kernel*
_output_shapes
: *
dtype0*
valueB
 *???>
?
Adnn/hiddenlayer_6/kernel/Initializer/random_uniform/RandomUniformRandomUniform9dnn/hiddenlayer_6/kernel/Initializer/random_uniform/shape*
T0*+
_class!
loc:@dnn/hiddenlayer_6/kernel*
_output_shapes

:2
*
dtype0*
seed????*
seed2
?
7dnn/hiddenlayer_6/kernel/Initializer/random_uniform/subSub7dnn/hiddenlayer_6/kernel/Initializer/random_uniform/max7dnn/hiddenlayer_6/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/hiddenlayer_6/kernel*
_output_shapes
: 
?
7dnn/hiddenlayer_6/kernel/Initializer/random_uniform/mulMulAdnn/hiddenlayer_6/kernel/Initializer/random_uniform/RandomUniform7dnn/hiddenlayer_6/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@dnn/hiddenlayer_6/kernel*
_output_shapes

:2

?
3dnn/hiddenlayer_6/kernel/Initializer/random_uniformAdd7dnn/hiddenlayer_6/kernel/Initializer/random_uniform/mul7dnn/hiddenlayer_6/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/hiddenlayer_6/kernel*
_output_shapes

:2

?
dnn/hiddenlayer_6/kernelVarHandleOp*+
_class!
loc:@dnn/hiddenlayer_6/kernel*
_output_shapes
: *
dtype0*
shape
:2
*)
shared_namednn/hiddenlayer_6/kernel
?
9dnn/hiddenlayer_6/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_6/kernel*
_output_shapes
: 
?
dnn/hiddenlayer_6/kernel/AssignAssignVariableOpdnn/hiddenlayer_6/kernel3dnn/hiddenlayer_6/kernel/Initializer/random_uniform*
dtype0
?
,dnn/hiddenlayer_6/kernel/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_6/kernel*
_output_shapes

:2
*
dtype0
?
(dnn/hiddenlayer_6/bias/Initializer/zerosConst*)
_class
loc:@dnn/hiddenlayer_6/bias*
_output_shapes
:
*
dtype0*
valueB
*    
?
dnn/hiddenlayer_6/biasVarHandleOp*)
_class
loc:@dnn/hiddenlayer_6/bias*
_output_shapes
: *
dtype0*
shape:
*'
shared_namednn/hiddenlayer_6/bias
}
7dnn/hiddenlayer_6/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_6/bias*
_output_shapes
: 
?
dnn/hiddenlayer_6/bias/AssignAssignVariableOpdnn/hiddenlayer_6/bias(dnn/hiddenlayer_6/bias/Initializer/zeros*
dtype0
}
*dnn/hiddenlayer_6/bias/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_6/bias*
_output_shapes
:
*
dtype0
?
'dnn/hiddenlayer_6/MatMul/ReadVariableOpReadVariableOpdnn/hiddenlayer_6/kernel*
_output_shapes

:2
*
dtype0
?
dnn/hiddenlayer_6/MatMulMatMul-dnn/hiddenlayer_5/batchnorm_5/batchnorm/add_1'dnn/hiddenlayer_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????

{
(dnn/hiddenlayer_6/BiasAdd/ReadVariableOpReadVariableOpdnn/hiddenlayer_6/bias*
_output_shapes
:
*
dtype0
?
dnn/hiddenlayer_6/BiasAddBiasAdddnn/hiddenlayer_6/MatMul(dnn/hiddenlayer_6/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:?????????

l
dnn/hiddenlayer_6/LeakyRelu	LeakyReludnn/hiddenlayer_6/BiasAdd*'
_output_shapes
:?????????

?
4dnn/hiddenlayer_6/batchnorm_6/gamma/Initializer/onesConst*6
_class,
*(loc:@dnn/hiddenlayer_6/batchnorm_6/gamma*
_output_shapes
:
*
dtype0*
valueB
*  ??
?
#dnn/hiddenlayer_6/batchnorm_6/gammaVarHandleOp*6
_class,
*(loc:@dnn/hiddenlayer_6/batchnorm_6/gamma*
_output_shapes
: *
dtype0*
shape:
*4
shared_name%#dnn/hiddenlayer_6/batchnorm_6/gamma
?
Ddnn/hiddenlayer_6/batchnorm_6/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp#dnn/hiddenlayer_6/batchnorm_6/gamma*
_output_shapes
: 
?
*dnn/hiddenlayer_6/batchnorm_6/gamma/AssignAssignVariableOp#dnn/hiddenlayer_6/batchnorm_6/gamma4dnn/hiddenlayer_6/batchnorm_6/gamma/Initializer/ones*
dtype0
?
7dnn/hiddenlayer_6/batchnorm_6/gamma/Read/ReadVariableOpReadVariableOp#dnn/hiddenlayer_6/batchnorm_6/gamma*
_output_shapes
:
*
dtype0
?
4dnn/hiddenlayer_6/batchnorm_6/beta/Initializer/zerosConst*5
_class+
)'loc:@dnn/hiddenlayer_6/batchnorm_6/beta*
_output_shapes
:
*
dtype0*
valueB
*    
?
"dnn/hiddenlayer_6/batchnorm_6/betaVarHandleOp*5
_class+
)'loc:@dnn/hiddenlayer_6/batchnorm_6/beta*
_output_shapes
: *
dtype0*
shape:
*3
shared_name$"dnn/hiddenlayer_6/batchnorm_6/beta
?
Cdnn/hiddenlayer_6/batchnorm_6/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp"dnn/hiddenlayer_6/batchnorm_6/beta*
_output_shapes
: 
?
)dnn/hiddenlayer_6/batchnorm_6/beta/AssignAssignVariableOp"dnn/hiddenlayer_6/batchnorm_6/beta4dnn/hiddenlayer_6/batchnorm_6/beta/Initializer/zeros*
dtype0
?
6dnn/hiddenlayer_6/batchnorm_6/beta/Read/ReadVariableOpReadVariableOp"dnn/hiddenlayer_6/batchnorm_6/beta*
_output_shapes
:
*
dtype0
?
;dnn/hiddenlayer_6/batchnorm_6/moving_mean/Initializer/zerosConst*<
_class2
0.loc:@dnn/hiddenlayer_6/batchnorm_6/moving_mean*
_output_shapes
:
*
dtype0*
valueB
*    
?
)dnn/hiddenlayer_6/batchnorm_6/moving_meanVarHandleOp*<
_class2
0.loc:@dnn/hiddenlayer_6/batchnorm_6/moving_mean*
_output_shapes
: *
dtype0*
shape:
*:
shared_name+)dnn/hiddenlayer_6/batchnorm_6/moving_mean
?
Jdnn/hiddenlayer_6/batchnorm_6/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp)dnn/hiddenlayer_6/batchnorm_6/moving_mean*
_output_shapes
: 
?
0dnn/hiddenlayer_6/batchnorm_6/moving_mean/AssignAssignVariableOp)dnn/hiddenlayer_6/batchnorm_6/moving_mean;dnn/hiddenlayer_6/batchnorm_6/moving_mean/Initializer/zeros*
dtype0
?
=dnn/hiddenlayer_6/batchnorm_6/moving_mean/Read/ReadVariableOpReadVariableOp)dnn/hiddenlayer_6/batchnorm_6/moving_mean*
_output_shapes
:
*
dtype0
?
>dnn/hiddenlayer_6/batchnorm_6/moving_variance/Initializer/onesConst*@
_class6
42loc:@dnn/hiddenlayer_6/batchnorm_6/moving_variance*
_output_shapes
:
*
dtype0*
valueB
*  ??
?
-dnn/hiddenlayer_6/batchnorm_6/moving_varianceVarHandleOp*@
_class6
42loc:@dnn/hiddenlayer_6/batchnorm_6/moving_variance*
_output_shapes
: *
dtype0*
shape:
*>
shared_name/-dnn/hiddenlayer_6/batchnorm_6/moving_variance
?
Ndnn/hiddenlayer_6/batchnorm_6/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp-dnn/hiddenlayer_6/batchnorm_6/moving_variance*
_output_shapes
: 
?
4dnn/hiddenlayer_6/batchnorm_6/moving_variance/AssignAssignVariableOp-dnn/hiddenlayer_6/batchnorm_6/moving_variance>dnn/hiddenlayer_6/batchnorm_6/moving_variance/Initializer/ones*
dtype0
?
Adnn/hiddenlayer_6/batchnorm_6/moving_variance/Read/ReadVariableOpReadVariableOp-dnn/hiddenlayer_6/batchnorm_6/moving_variance*
_output_shapes
:
*
dtype0
?
6dnn/hiddenlayer_6/batchnorm_6/batchnorm/ReadVariableOpReadVariableOp-dnn/hiddenlayer_6/batchnorm_6/moving_variance*
_output_shapes
:
*
dtype0
r
-dnn/hiddenlayer_6/batchnorm_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:
?
+dnn/hiddenlayer_6/batchnorm_6/batchnorm/addAddV26dnn/hiddenlayer_6/batchnorm_6/batchnorm/ReadVariableOp-dnn/hiddenlayer_6/batchnorm_6/batchnorm/add/y*
T0*
_output_shapes
:

?
-dnn/hiddenlayer_6/batchnorm_6/batchnorm/RsqrtRsqrt+dnn/hiddenlayer_6/batchnorm_6/batchnorm/add*
T0*
_output_shapes
:

?
:dnn/hiddenlayer_6/batchnorm_6/batchnorm/mul/ReadVariableOpReadVariableOp#dnn/hiddenlayer_6/batchnorm_6/gamma*
_output_shapes
:
*
dtype0
?
+dnn/hiddenlayer_6/batchnorm_6/batchnorm/mulMul-dnn/hiddenlayer_6/batchnorm_6/batchnorm/Rsqrt:dnn/hiddenlayer_6/batchnorm_6/batchnorm/mul/ReadVariableOp*
T0*
_output_shapes
:

?
-dnn/hiddenlayer_6/batchnorm_6/batchnorm/mul_1Muldnn/hiddenlayer_6/LeakyRelu+dnn/hiddenlayer_6/batchnorm_6/batchnorm/mul*
T0*'
_output_shapes
:?????????

?
8dnn/hiddenlayer_6/batchnorm_6/batchnorm/ReadVariableOp_1ReadVariableOp)dnn/hiddenlayer_6/batchnorm_6/moving_mean*
_output_shapes
:
*
dtype0
?
-dnn/hiddenlayer_6/batchnorm_6/batchnorm/mul_2Mul8dnn/hiddenlayer_6/batchnorm_6/batchnorm/ReadVariableOp_1+dnn/hiddenlayer_6/batchnorm_6/batchnorm/mul*
T0*
_output_shapes
:

?
8dnn/hiddenlayer_6/batchnorm_6/batchnorm/ReadVariableOp_2ReadVariableOp"dnn/hiddenlayer_6/batchnorm_6/beta*
_output_shapes
:
*
dtype0
?
+dnn/hiddenlayer_6/batchnorm_6/batchnorm/subSub8dnn/hiddenlayer_6/batchnorm_6/batchnorm/ReadVariableOp_2-dnn/hiddenlayer_6/batchnorm_6/batchnorm/mul_2*
T0*
_output_shapes
:

?
-dnn/hiddenlayer_6/batchnorm_6/batchnorm/add_1AddV2-dnn/hiddenlayer_6/batchnorm_6/batchnorm/mul_1+dnn/hiddenlayer_6/batchnorm_6/batchnorm/sub*
T0*'
_output_shapes
:?????????

?
dnn/zero_fraction_6/SizeSize-dnn/hiddenlayer_6/batchnorm_6/batchnorm/add_1*
T0*
_output_shapes
: *
out_type0	
e
dnn/zero_fraction_6/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R????
?
dnn/zero_fraction_6/LessEqual	LessEqualdnn/zero_fraction_6/Sizednn/zero_fraction_6/LessEqual/y*
T0	*
_output_shapes
: 
?
dnn/zero_fraction_6/condStatelessIfdnn/zero_fraction_6/LessEqual-dnn/hiddenlayer_6/batchnorm_6/batchnorm/add_1*
Tcond0
*
Tin
2*
Tout

2	*
_lower_using_switch_merge(* 
_output_shapes
: : : : : : * 
_read_only_resource_inputs
 *8
else_branch)R'
%dnn_zero_fraction_6_cond_false_512116*
output_shapes
: : : : : : *7
then_branch(R&
$dnn_zero_fraction_6_cond_true_512115
h
!dnn/zero_fraction_6/cond/IdentityIdentitydnn/zero_fraction_6/cond*
T0	*
_output_shapes
: 
l
#dnn/zero_fraction_6/cond/Identity_1Identitydnn/zero_fraction_6/cond:1*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_6/cond/Identity_2Identitydnn/zero_fraction_6/cond:2*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_6/cond/Identity_3Identitydnn/zero_fraction_6/cond:3*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_6/cond/Identity_4Identitydnn/zero_fraction_6/cond:4*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_6/cond/Identity_5Identitydnn/zero_fraction_6/cond:5*
T0*
_output_shapes
: 
?
*dnn/zero_fraction_6/counts_to_fraction/subSubdnn/zero_fraction_6/Size!dnn/zero_fraction_6/cond/Identity*
T0	*
_output_shapes
: 
?
+dnn/zero_fraction_6/counts_to_fraction/CastCast*dnn/zero_fraction_6/counts_to_fraction/sub*

DstT0*

SrcT0	*
_output_shapes
: 

-dnn/zero_fraction_6/counts_to_fraction/Cast_1Castdnn/zero_fraction_6/Size*

DstT0*

SrcT0	*
_output_shapes
: 
?
.dnn/zero_fraction_6/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_6/counts_to_fraction/Cast-dnn/zero_fraction_6/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
y
dnn/zero_fraction_6/fractionIdentity.dnn/zero_fraction_6/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
.dnn/hiddenlayer_6/fraction_of_zero_values/tagsConst*
_output_shapes
: *
dtype0*:
value1B/ B)dnn/hiddenlayer_6/fraction_of_zero_values
?
)dnn/hiddenlayer_6/fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_6/fraction_of_zero_values/tagsdnn/zero_fraction_6/fraction*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_6/activation/tagConst*
_output_shapes
: *
dtype0*-
value$B" Bdnn/hiddenlayer_6/activation
?
dnn/hiddenlayer_6/activationHistogramSummary dnn/hiddenlayer_6/activation/tag-dnn/hiddenlayer_6/batchnorm_6/batchnorm/add_1*
_output_shapes
: 
?
9dnn/hiddenlayer_7/kernel/Initializer/random_uniform/shapeConst*+
_class!
loc:@dnn/hiddenlayer_7/kernel*
_output_shapes
:*
dtype0*
valueB"
   
   
?
7dnn/hiddenlayer_7/kernel/Initializer/random_uniform/minConst*+
_class!
loc:@dnn/hiddenlayer_7/kernel*
_output_shapes
: *
dtype0*
valueB
 *?7?
?
7dnn/hiddenlayer_7/kernel/Initializer/random_uniform/maxConst*+
_class!
loc:@dnn/hiddenlayer_7/kernel*
_output_shapes
: *
dtype0*
valueB
 *?7?
?
Adnn/hiddenlayer_7/kernel/Initializer/random_uniform/RandomUniformRandomUniform9dnn/hiddenlayer_7/kernel/Initializer/random_uniform/shape*
T0*+
_class!
loc:@dnn/hiddenlayer_7/kernel*
_output_shapes

:

*
dtype0*
seed????*
seed2
?
7dnn/hiddenlayer_7/kernel/Initializer/random_uniform/subSub7dnn/hiddenlayer_7/kernel/Initializer/random_uniform/max7dnn/hiddenlayer_7/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/hiddenlayer_7/kernel*
_output_shapes
: 
?
7dnn/hiddenlayer_7/kernel/Initializer/random_uniform/mulMulAdnn/hiddenlayer_7/kernel/Initializer/random_uniform/RandomUniform7dnn/hiddenlayer_7/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@dnn/hiddenlayer_7/kernel*
_output_shapes

:


?
3dnn/hiddenlayer_7/kernel/Initializer/random_uniformAdd7dnn/hiddenlayer_7/kernel/Initializer/random_uniform/mul7dnn/hiddenlayer_7/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/hiddenlayer_7/kernel*
_output_shapes

:


?
dnn/hiddenlayer_7/kernelVarHandleOp*+
_class!
loc:@dnn/hiddenlayer_7/kernel*
_output_shapes
: *
dtype0*
shape
:

*)
shared_namednn/hiddenlayer_7/kernel
?
9dnn/hiddenlayer_7/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_7/kernel*
_output_shapes
: 
?
dnn/hiddenlayer_7/kernel/AssignAssignVariableOpdnn/hiddenlayer_7/kernel3dnn/hiddenlayer_7/kernel/Initializer/random_uniform*
dtype0
?
,dnn/hiddenlayer_7/kernel/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_7/kernel*
_output_shapes

:

*
dtype0
?
(dnn/hiddenlayer_7/bias/Initializer/zerosConst*)
_class
loc:@dnn/hiddenlayer_7/bias*
_output_shapes
:
*
dtype0*
valueB
*    
?
dnn/hiddenlayer_7/biasVarHandleOp*)
_class
loc:@dnn/hiddenlayer_7/bias*
_output_shapes
: *
dtype0*
shape:
*'
shared_namednn/hiddenlayer_7/bias
}
7dnn/hiddenlayer_7/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_7/bias*
_output_shapes
: 
?
dnn/hiddenlayer_7/bias/AssignAssignVariableOpdnn/hiddenlayer_7/bias(dnn/hiddenlayer_7/bias/Initializer/zeros*
dtype0
}
*dnn/hiddenlayer_7/bias/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_7/bias*
_output_shapes
:
*
dtype0
?
'dnn/hiddenlayer_7/MatMul/ReadVariableOpReadVariableOpdnn/hiddenlayer_7/kernel*
_output_shapes

:

*
dtype0
?
dnn/hiddenlayer_7/MatMulMatMul-dnn/hiddenlayer_6/batchnorm_6/batchnorm/add_1'dnn/hiddenlayer_7/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????

{
(dnn/hiddenlayer_7/BiasAdd/ReadVariableOpReadVariableOpdnn/hiddenlayer_7/bias*
_output_shapes
:
*
dtype0
?
dnn/hiddenlayer_7/BiasAddBiasAdddnn/hiddenlayer_7/MatMul(dnn/hiddenlayer_7/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:?????????

l
dnn/hiddenlayer_7/LeakyRelu	LeakyReludnn/hiddenlayer_7/BiasAdd*'
_output_shapes
:?????????

?
4dnn/hiddenlayer_7/batchnorm_7/gamma/Initializer/onesConst*6
_class,
*(loc:@dnn/hiddenlayer_7/batchnorm_7/gamma*
_output_shapes
:
*
dtype0*
valueB
*  ??
?
#dnn/hiddenlayer_7/batchnorm_7/gammaVarHandleOp*6
_class,
*(loc:@dnn/hiddenlayer_7/batchnorm_7/gamma*
_output_shapes
: *
dtype0*
shape:
*4
shared_name%#dnn/hiddenlayer_7/batchnorm_7/gamma
?
Ddnn/hiddenlayer_7/batchnorm_7/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp#dnn/hiddenlayer_7/batchnorm_7/gamma*
_output_shapes
: 
?
*dnn/hiddenlayer_7/batchnorm_7/gamma/AssignAssignVariableOp#dnn/hiddenlayer_7/batchnorm_7/gamma4dnn/hiddenlayer_7/batchnorm_7/gamma/Initializer/ones*
dtype0
?
7dnn/hiddenlayer_7/batchnorm_7/gamma/Read/ReadVariableOpReadVariableOp#dnn/hiddenlayer_7/batchnorm_7/gamma*
_output_shapes
:
*
dtype0
?
4dnn/hiddenlayer_7/batchnorm_7/beta/Initializer/zerosConst*5
_class+
)'loc:@dnn/hiddenlayer_7/batchnorm_7/beta*
_output_shapes
:
*
dtype0*
valueB
*    
?
"dnn/hiddenlayer_7/batchnorm_7/betaVarHandleOp*5
_class+
)'loc:@dnn/hiddenlayer_7/batchnorm_7/beta*
_output_shapes
: *
dtype0*
shape:
*3
shared_name$"dnn/hiddenlayer_7/batchnorm_7/beta
?
Cdnn/hiddenlayer_7/batchnorm_7/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp"dnn/hiddenlayer_7/batchnorm_7/beta*
_output_shapes
: 
?
)dnn/hiddenlayer_7/batchnorm_7/beta/AssignAssignVariableOp"dnn/hiddenlayer_7/batchnorm_7/beta4dnn/hiddenlayer_7/batchnorm_7/beta/Initializer/zeros*
dtype0
?
6dnn/hiddenlayer_7/batchnorm_7/beta/Read/ReadVariableOpReadVariableOp"dnn/hiddenlayer_7/batchnorm_7/beta*
_output_shapes
:
*
dtype0
?
;dnn/hiddenlayer_7/batchnorm_7/moving_mean/Initializer/zerosConst*<
_class2
0.loc:@dnn/hiddenlayer_7/batchnorm_7/moving_mean*
_output_shapes
:
*
dtype0*
valueB
*    
?
)dnn/hiddenlayer_7/batchnorm_7/moving_meanVarHandleOp*<
_class2
0.loc:@dnn/hiddenlayer_7/batchnorm_7/moving_mean*
_output_shapes
: *
dtype0*
shape:
*:
shared_name+)dnn/hiddenlayer_7/batchnorm_7/moving_mean
?
Jdnn/hiddenlayer_7/batchnorm_7/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp)dnn/hiddenlayer_7/batchnorm_7/moving_mean*
_output_shapes
: 
?
0dnn/hiddenlayer_7/batchnorm_7/moving_mean/AssignAssignVariableOp)dnn/hiddenlayer_7/batchnorm_7/moving_mean;dnn/hiddenlayer_7/batchnorm_7/moving_mean/Initializer/zeros*
dtype0
?
=dnn/hiddenlayer_7/batchnorm_7/moving_mean/Read/ReadVariableOpReadVariableOp)dnn/hiddenlayer_7/batchnorm_7/moving_mean*
_output_shapes
:
*
dtype0
?
>dnn/hiddenlayer_7/batchnorm_7/moving_variance/Initializer/onesConst*@
_class6
42loc:@dnn/hiddenlayer_7/batchnorm_7/moving_variance*
_output_shapes
:
*
dtype0*
valueB
*  ??
?
-dnn/hiddenlayer_7/batchnorm_7/moving_varianceVarHandleOp*@
_class6
42loc:@dnn/hiddenlayer_7/batchnorm_7/moving_variance*
_output_shapes
: *
dtype0*
shape:
*>
shared_name/-dnn/hiddenlayer_7/batchnorm_7/moving_variance
?
Ndnn/hiddenlayer_7/batchnorm_7/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp-dnn/hiddenlayer_7/batchnorm_7/moving_variance*
_output_shapes
: 
?
4dnn/hiddenlayer_7/batchnorm_7/moving_variance/AssignAssignVariableOp-dnn/hiddenlayer_7/batchnorm_7/moving_variance>dnn/hiddenlayer_7/batchnorm_7/moving_variance/Initializer/ones*
dtype0
?
Adnn/hiddenlayer_7/batchnorm_7/moving_variance/Read/ReadVariableOpReadVariableOp-dnn/hiddenlayer_7/batchnorm_7/moving_variance*
_output_shapes
:
*
dtype0
?
6dnn/hiddenlayer_7/batchnorm_7/batchnorm/ReadVariableOpReadVariableOp-dnn/hiddenlayer_7/batchnorm_7/moving_variance*
_output_shapes
:
*
dtype0
r
-dnn/hiddenlayer_7/batchnorm_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:
?
+dnn/hiddenlayer_7/batchnorm_7/batchnorm/addAddV26dnn/hiddenlayer_7/batchnorm_7/batchnorm/ReadVariableOp-dnn/hiddenlayer_7/batchnorm_7/batchnorm/add/y*
T0*
_output_shapes
:

?
-dnn/hiddenlayer_7/batchnorm_7/batchnorm/RsqrtRsqrt+dnn/hiddenlayer_7/batchnorm_7/batchnorm/add*
T0*
_output_shapes
:

?
:dnn/hiddenlayer_7/batchnorm_7/batchnorm/mul/ReadVariableOpReadVariableOp#dnn/hiddenlayer_7/batchnorm_7/gamma*
_output_shapes
:
*
dtype0
?
+dnn/hiddenlayer_7/batchnorm_7/batchnorm/mulMul-dnn/hiddenlayer_7/batchnorm_7/batchnorm/Rsqrt:dnn/hiddenlayer_7/batchnorm_7/batchnorm/mul/ReadVariableOp*
T0*
_output_shapes
:

?
-dnn/hiddenlayer_7/batchnorm_7/batchnorm/mul_1Muldnn/hiddenlayer_7/LeakyRelu+dnn/hiddenlayer_7/batchnorm_7/batchnorm/mul*
T0*'
_output_shapes
:?????????

?
8dnn/hiddenlayer_7/batchnorm_7/batchnorm/ReadVariableOp_1ReadVariableOp)dnn/hiddenlayer_7/batchnorm_7/moving_mean*
_output_shapes
:
*
dtype0
?
-dnn/hiddenlayer_7/batchnorm_7/batchnorm/mul_2Mul8dnn/hiddenlayer_7/batchnorm_7/batchnorm/ReadVariableOp_1+dnn/hiddenlayer_7/batchnorm_7/batchnorm/mul*
T0*
_output_shapes
:

?
8dnn/hiddenlayer_7/batchnorm_7/batchnorm/ReadVariableOp_2ReadVariableOp"dnn/hiddenlayer_7/batchnorm_7/beta*
_output_shapes
:
*
dtype0
?
+dnn/hiddenlayer_7/batchnorm_7/batchnorm/subSub8dnn/hiddenlayer_7/batchnorm_7/batchnorm/ReadVariableOp_2-dnn/hiddenlayer_7/batchnorm_7/batchnorm/mul_2*
T0*
_output_shapes
:

?
-dnn/hiddenlayer_7/batchnorm_7/batchnorm/add_1AddV2-dnn/hiddenlayer_7/batchnorm_7/batchnorm/mul_1+dnn/hiddenlayer_7/batchnorm_7/batchnorm/sub*
T0*'
_output_shapes
:?????????

?
dnn/zero_fraction_7/SizeSize-dnn/hiddenlayer_7/batchnorm_7/batchnorm/add_1*
T0*
_output_shapes
: *
out_type0	
e
dnn/zero_fraction_7/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R????
?
dnn/zero_fraction_7/LessEqual	LessEqualdnn/zero_fraction_7/Sizednn/zero_fraction_7/LessEqual/y*
T0	*
_output_shapes
: 
?
dnn/zero_fraction_7/condStatelessIfdnn/zero_fraction_7/LessEqual-dnn/hiddenlayer_7/batchnorm_7/batchnorm/add_1*
Tcond0
*
Tin
2*
Tout

2	*
_lower_using_switch_merge(* 
_output_shapes
: : : : : : * 
_read_only_resource_inputs
 *8
else_branch)R'
%dnn_zero_fraction_7_cond_false_512214*
output_shapes
: : : : : : *7
then_branch(R&
$dnn_zero_fraction_7_cond_true_512213
h
!dnn/zero_fraction_7/cond/IdentityIdentitydnn/zero_fraction_7/cond*
T0	*
_output_shapes
: 
l
#dnn/zero_fraction_7/cond/Identity_1Identitydnn/zero_fraction_7/cond:1*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_7/cond/Identity_2Identitydnn/zero_fraction_7/cond:2*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_7/cond/Identity_3Identitydnn/zero_fraction_7/cond:3*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_7/cond/Identity_4Identitydnn/zero_fraction_7/cond:4*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_7/cond/Identity_5Identitydnn/zero_fraction_7/cond:5*
T0*
_output_shapes
: 
?
*dnn/zero_fraction_7/counts_to_fraction/subSubdnn/zero_fraction_7/Size!dnn/zero_fraction_7/cond/Identity*
T0	*
_output_shapes
: 
?
+dnn/zero_fraction_7/counts_to_fraction/CastCast*dnn/zero_fraction_7/counts_to_fraction/sub*

DstT0*

SrcT0	*
_output_shapes
: 

-dnn/zero_fraction_7/counts_to_fraction/Cast_1Castdnn/zero_fraction_7/Size*

DstT0*

SrcT0	*
_output_shapes
: 
?
.dnn/zero_fraction_7/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_7/counts_to_fraction/Cast-dnn/zero_fraction_7/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
y
dnn/zero_fraction_7/fractionIdentity.dnn/zero_fraction_7/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
.dnn/hiddenlayer_7/fraction_of_zero_values/tagsConst*
_output_shapes
: *
dtype0*:
value1B/ B)dnn/hiddenlayer_7/fraction_of_zero_values
?
)dnn/hiddenlayer_7/fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_7/fraction_of_zero_values/tagsdnn/zero_fraction_7/fraction*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_7/activation/tagConst*
_output_shapes
: *
dtype0*-
value$B" Bdnn/hiddenlayer_7/activation
?
dnn/hiddenlayer_7/activationHistogramSummary dnn/hiddenlayer_7/activation/tag-dnn/hiddenlayer_7/batchnorm_7/batchnorm/add_1*
_output_shapes
: 
?
9dnn/hiddenlayer_8/kernel/Initializer/random_uniform/shapeConst*+
_class!
loc:@dnn/hiddenlayer_8/kernel*
_output_shapes
:*
dtype0*
valueB"
   
   
?
7dnn/hiddenlayer_8/kernel/Initializer/random_uniform/minConst*+
_class!
loc:@dnn/hiddenlayer_8/kernel*
_output_shapes
: *
dtype0*
valueB
 *?7?
?
7dnn/hiddenlayer_8/kernel/Initializer/random_uniform/maxConst*+
_class!
loc:@dnn/hiddenlayer_8/kernel*
_output_shapes
: *
dtype0*
valueB
 *?7?
?
Adnn/hiddenlayer_8/kernel/Initializer/random_uniform/RandomUniformRandomUniform9dnn/hiddenlayer_8/kernel/Initializer/random_uniform/shape*
T0*+
_class!
loc:@dnn/hiddenlayer_8/kernel*
_output_shapes

:

*
dtype0*
seed????*
seed2
?
7dnn/hiddenlayer_8/kernel/Initializer/random_uniform/subSub7dnn/hiddenlayer_8/kernel/Initializer/random_uniform/max7dnn/hiddenlayer_8/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/hiddenlayer_8/kernel*
_output_shapes
: 
?
7dnn/hiddenlayer_8/kernel/Initializer/random_uniform/mulMulAdnn/hiddenlayer_8/kernel/Initializer/random_uniform/RandomUniform7dnn/hiddenlayer_8/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@dnn/hiddenlayer_8/kernel*
_output_shapes

:


?
3dnn/hiddenlayer_8/kernel/Initializer/random_uniformAdd7dnn/hiddenlayer_8/kernel/Initializer/random_uniform/mul7dnn/hiddenlayer_8/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/hiddenlayer_8/kernel*
_output_shapes

:


?
dnn/hiddenlayer_8/kernelVarHandleOp*+
_class!
loc:@dnn/hiddenlayer_8/kernel*
_output_shapes
: *
dtype0*
shape
:

*)
shared_namednn/hiddenlayer_8/kernel
?
9dnn/hiddenlayer_8/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_8/kernel*
_output_shapes
: 
?
dnn/hiddenlayer_8/kernel/AssignAssignVariableOpdnn/hiddenlayer_8/kernel3dnn/hiddenlayer_8/kernel/Initializer/random_uniform*
dtype0
?
,dnn/hiddenlayer_8/kernel/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_8/kernel*
_output_shapes

:

*
dtype0
?
(dnn/hiddenlayer_8/bias/Initializer/zerosConst*)
_class
loc:@dnn/hiddenlayer_8/bias*
_output_shapes
:
*
dtype0*
valueB
*    
?
dnn/hiddenlayer_8/biasVarHandleOp*)
_class
loc:@dnn/hiddenlayer_8/bias*
_output_shapes
: *
dtype0*
shape:
*'
shared_namednn/hiddenlayer_8/bias
}
7dnn/hiddenlayer_8/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_8/bias*
_output_shapes
: 
?
dnn/hiddenlayer_8/bias/AssignAssignVariableOpdnn/hiddenlayer_8/bias(dnn/hiddenlayer_8/bias/Initializer/zeros*
dtype0
}
*dnn/hiddenlayer_8/bias/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_8/bias*
_output_shapes
:
*
dtype0
?
'dnn/hiddenlayer_8/MatMul/ReadVariableOpReadVariableOpdnn/hiddenlayer_8/kernel*
_output_shapes

:

*
dtype0
?
dnn/hiddenlayer_8/MatMulMatMul-dnn/hiddenlayer_7/batchnorm_7/batchnorm/add_1'dnn/hiddenlayer_8/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????

{
(dnn/hiddenlayer_8/BiasAdd/ReadVariableOpReadVariableOpdnn/hiddenlayer_8/bias*
_output_shapes
:
*
dtype0
?
dnn/hiddenlayer_8/BiasAddBiasAdddnn/hiddenlayer_8/MatMul(dnn/hiddenlayer_8/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:?????????

l
dnn/hiddenlayer_8/LeakyRelu	LeakyReludnn/hiddenlayer_8/BiasAdd*'
_output_shapes
:?????????

?
4dnn/hiddenlayer_8/batchnorm_8/gamma/Initializer/onesConst*6
_class,
*(loc:@dnn/hiddenlayer_8/batchnorm_8/gamma*
_output_shapes
:
*
dtype0*
valueB
*  ??
?
#dnn/hiddenlayer_8/batchnorm_8/gammaVarHandleOp*6
_class,
*(loc:@dnn/hiddenlayer_8/batchnorm_8/gamma*
_output_shapes
: *
dtype0*
shape:
*4
shared_name%#dnn/hiddenlayer_8/batchnorm_8/gamma
?
Ddnn/hiddenlayer_8/batchnorm_8/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp#dnn/hiddenlayer_8/batchnorm_8/gamma*
_output_shapes
: 
?
*dnn/hiddenlayer_8/batchnorm_8/gamma/AssignAssignVariableOp#dnn/hiddenlayer_8/batchnorm_8/gamma4dnn/hiddenlayer_8/batchnorm_8/gamma/Initializer/ones*
dtype0
?
7dnn/hiddenlayer_8/batchnorm_8/gamma/Read/ReadVariableOpReadVariableOp#dnn/hiddenlayer_8/batchnorm_8/gamma*
_output_shapes
:
*
dtype0
?
4dnn/hiddenlayer_8/batchnorm_8/beta/Initializer/zerosConst*5
_class+
)'loc:@dnn/hiddenlayer_8/batchnorm_8/beta*
_output_shapes
:
*
dtype0*
valueB
*    
?
"dnn/hiddenlayer_8/batchnorm_8/betaVarHandleOp*5
_class+
)'loc:@dnn/hiddenlayer_8/batchnorm_8/beta*
_output_shapes
: *
dtype0*
shape:
*3
shared_name$"dnn/hiddenlayer_8/batchnorm_8/beta
?
Cdnn/hiddenlayer_8/batchnorm_8/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp"dnn/hiddenlayer_8/batchnorm_8/beta*
_output_shapes
: 
?
)dnn/hiddenlayer_8/batchnorm_8/beta/AssignAssignVariableOp"dnn/hiddenlayer_8/batchnorm_8/beta4dnn/hiddenlayer_8/batchnorm_8/beta/Initializer/zeros*
dtype0
?
6dnn/hiddenlayer_8/batchnorm_8/beta/Read/ReadVariableOpReadVariableOp"dnn/hiddenlayer_8/batchnorm_8/beta*
_output_shapes
:
*
dtype0
?
;dnn/hiddenlayer_8/batchnorm_8/moving_mean/Initializer/zerosConst*<
_class2
0.loc:@dnn/hiddenlayer_8/batchnorm_8/moving_mean*
_output_shapes
:
*
dtype0*
valueB
*    
?
)dnn/hiddenlayer_8/batchnorm_8/moving_meanVarHandleOp*<
_class2
0.loc:@dnn/hiddenlayer_8/batchnorm_8/moving_mean*
_output_shapes
: *
dtype0*
shape:
*:
shared_name+)dnn/hiddenlayer_8/batchnorm_8/moving_mean
?
Jdnn/hiddenlayer_8/batchnorm_8/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp)dnn/hiddenlayer_8/batchnorm_8/moving_mean*
_output_shapes
: 
?
0dnn/hiddenlayer_8/batchnorm_8/moving_mean/AssignAssignVariableOp)dnn/hiddenlayer_8/batchnorm_8/moving_mean;dnn/hiddenlayer_8/batchnorm_8/moving_mean/Initializer/zeros*
dtype0
?
=dnn/hiddenlayer_8/batchnorm_8/moving_mean/Read/ReadVariableOpReadVariableOp)dnn/hiddenlayer_8/batchnorm_8/moving_mean*
_output_shapes
:
*
dtype0
?
>dnn/hiddenlayer_8/batchnorm_8/moving_variance/Initializer/onesConst*@
_class6
42loc:@dnn/hiddenlayer_8/batchnorm_8/moving_variance*
_output_shapes
:
*
dtype0*
valueB
*  ??
?
-dnn/hiddenlayer_8/batchnorm_8/moving_varianceVarHandleOp*@
_class6
42loc:@dnn/hiddenlayer_8/batchnorm_8/moving_variance*
_output_shapes
: *
dtype0*
shape:
*>
shared_name/-dnn/hiddenlayer_8/batchnorm_8/moving_variance
?
Ndnn/hiddenlayer_8/batchnorm_8/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp-dnn/hiddenlayer_8/batchnorm_8/moving_variance*
_output_shapes
: 
?
4dnn/hiddenlayer_8/batchnorm_8/moving_variance/AssignAssignVariableOp-dnn/hiddenlayer_8/batchnorm_8/moving_variance>dnn/hiddenlayer_8/batchnorm_8/moving_variance/Initializer/ones*
dtype0
?
Adnn/hiddenlayer_8/batchnorm_8/moving_variance/Read/ReadVariableOpReadVariableOp-dnn/hiddenlayer_8/batchnorm_8/moving_variance*
_output_shapes
:
*
dtype0
?
6dnn/hiddenlayer_8/batchnorm_8/batchnorm/ReadVariableOpReadVariableOp-dnn/hiddenlayer_8/batchnorm_8/moving_variance*
_output_shapes
:
*
dtype0
r
-dnn/hiddenlayer_8/batchnorm_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:
?
+dnn/hiddenlayer_8/batchnorm_8/batchnorm/addAddV26dnn/hiddenlayer_8/batchnorm_8/batchnorm/ReadVariableOp-dnn/hiddenlayer_8/batchnorm_8/batchnorm/add/y*
T0*
_output_shapes
:

?
-dnn/hiddenlayer_8/batchnorm_8/batchnorm/RsqrtRsqrt+dnn/hiddenlayer_8/batchnorm_8/batchnorm/add*
T0*
_output_shapes
:

?
:dnn/hiddenlayer_8/batchnorm_8/batchnorm/mul/ReadVariableOpReadVariableOp#dnn/hiddenlayer_8/batchnorm_8/gamma*
_output_shapes
:
*
dtype0
?
+dnn/hiddenlayer_8/batchnorm_8/batchnorm/mulMul-dnn/hiddenlayer_8/batchnorm_8/batchnorm/Rsqrt:dnn/hiddenlayer_8/batchnorm_8/batchnorm/mul/ReadVariableOp*
T0*
_output_shapes
:

?
-dnn/hiddenlayer_8/batchnorm_8/batchnorm/mul_1Muldnn/hiddenlayer_8/LeakyRelu+dnn/hiddenlayer_8/batchnorm_8/batchnorm/mul*
T0*'
_output_shapes
:?????????

?
8dnn/hiddenlayer_8/batchnorm_8/batchnorm/ReadVariableOp_1ReadVariableOp)dnn/hiddenlayer_8/batchnorm_8/moving_mean*
_output_shapes
:
*
dtype0
?
-dnn/hiddenlayer_8/batchnorm_8/batchnorm/mul_2Mul8dnn/hiddenlayer_8/batchnorm_8/batchnorm/ReadVariableOp_1+dnn/hiddenlayer_8/batchnorm_8/batchnorm/mul*
T0*
_output_shapes
:

?
8dnn/hiddenlayer_8/batchnorm_8/batchnorm/ReadVariableOp_2ReadVariableOp"dnn/hiddenlayer_8/batchnorm_8/beta*
_output_shapes
:
*
dtype0
?
+dnn/hiddenlayer_8/batchnorm_8/batchnorm/subSub8dnn/hiddenlayer_8/batchnorm_8/batchnorm/ReadVariableOp_2-dnn/hiddenlayer_8/batchnorm_8/batchnorm/mul_2*
T0*
_output_shapes
:

?
-dnn/hiddenlayer_8/batchnorm_8/batchnorm/add_1AddV2-dnn/hiddenlayer_8/batchnorm_8/batchnorm/mul_1+dnn/hiddenlayer_8/batchnorm_8/batchnorm/sub*
T0*'
_output_shapes
:?????????

?
dnn/zero_fraction_8/SizeSize-dnn/hiddenlayer_8/batchnorm_8/batchnorm/add_1*
T0*
_output_shapes
: *
out_type0	
e
dnn/zero_fraction_8/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R????
?
dnn/zero_fraction_8/LessEqual	LessEqualdnn/zero_fraction_8/Sizednn/zero_fraction_8/LessEqual/y*
T0	*
_output_shapes
: 
?
dnn/zero_fraction_8/condStatelessIfdnn/zero_fraction_8/LessEqual-dnn/hiddenlayer_8/batchnorm_8/batchnorm/add_1*
Tcond0
*
Tin
2*
Tout

2	*
_lower_using_switch_merge(* 
_output_shapes
: : : : : : * 
_read_only_resource_inputs
 *8
else_branch)R'
%dnn_zero_fraction_8_cond_false_512312*
output_shapes
: : : : : : *7
then_branch(R&
$dnn_zero_fraction_8_cond_true_512311
h
!dnn/zero_fraction_8/cond/IdentityIdentitydnn/zero_fraction_8/cond*
T0	*
_output_shapes
: 
l
#dnn/zero_fraction_8/cond/Identity_1Identitydnn/zero_fraction_8/cond:1*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_8/cond/Identity_2Identitydnn/zero_fraction_8/cond:2*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_8/cond/Identity_3Identitydnn/zero_fraction_8/cond:3*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_8/cond/Identity_4Identitydnn/zero_fraction_8/cond:4*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_8/cond/Identity_5Identitydnn/zero_fraction_8/cond:5*
T0*
_output_shapes
: 
?
*dnn/zero_fraction_8/counts_to_fraction/subSubdnn/zero_fraction_8/Size!dnn/zero_fraction_8/cond/Identity*
T0	*
_output_shapes
: 
?
+dnn/zero_fraction_8/counts_to_fraction/CastCast*dnn/zero_fraction_8/counts_to_fraction/sub*

DstT0*

SrcT0	*
_output_shapes
: 

-dnn/zero_fraction_8/counts_to_fraction/Cast_1Castdnn/zero_fraction_8/Size*

DstT0*

SrcT0	*
_output_shapes
: 
?
.dnn/zero_fraction_8/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_8/counts_to_fraction/Cast-dnn/zero_fraction_8/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
y
dnn/zero_fraction_8/fractionIdentity.dnn/zero_fraction_8/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
.dnn/hiddenlayer_8/fraction_of_zero_values/tagsConst*
_output_shapes
: *
dtype0*:
value1B/ B)dnn/hiddenlayer_8/fraction_of_zero_values
?
)dnn/hiddenlayer_8/fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_8/fraction_of_zero_values/tagsdnn/zero_fraction_8/fraction*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_8/activation/tagConst*
_output_shapes
: *
dtype0*-
value$B" Bdnn/hiddenlayer_8/activation
?
dnn/hiddenlayer_8/activationHistogramSummary dnn/hiddenlayer_8/activation/tag-dnn/hiddenlayer_8/batchnorm_8/batchnorm/add_1*
_output_shapes
: 
?
2dnn/logits/kernel/Initializer/random_uniform/shapeConst*$
_class
loc:@dnn/logits/kernel*
_output_shapes
:*
dtype0*
valueB"
      
?
0dnn/logits/kernel/Initializer/random_uniform/minConst*$
_class
loc:@dnn/logits/kernel*
_output_shapes
: *
dtype0*
valueB
 *?=?
?
0dnn/logits/kernel/Initializer/random_uniform/maxConst*$
_class
loc:@dnn/logits/kernel*
_output_shapes
: *
dtype0*
valueB
 *?=?
?
:dnn/logits/kernel/Initializer/random_uniform/RandomUniformRandomUniform2dnn/logits/kernel/Initializer/random_uniform/shape*
T0*$
_class
loc:@dnn/logits/kernel*
_output_shapes

:
*
dtype0*
seed????*
seed2
?
0dnn/logits/kernel/Initializer/random_uniform/subSub0dnn/logits/kernel/Initializer/random_uniform/max0dnn/logits/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@dnn/logits/kernel*
_output_shapes
: 
?
0dnn/logits/kernel/Initializer/random_uniform/mulMul:dnn/logits/kernel/Initializer/random_uniform/RandomUniform0dnn/logits/kernel/Initializer/random_uniform/sub*
T0*$
_class
loc:@dnn/logits/kernel*
_output_shapes

:

?
,dnn/logits/kernel/Initializer/random_uniformAdd0dnn/logits/kernel/Initializer/random_uniform/mul0dnn/logits/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@dnn/logits/kernel*
_output_shapes

:

?
dnn/logits/kernelVarHandleOp*$
_class
loc:@dnn/logits/kernel*
_output_shapes
: *
dtype0*
shape
:
*"
shared_namednn/logits/kernel
s
2dnn/logits/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/logits/kernel*
_output_shapes
: 
z
dnn/logits/kernel/AssignAssignVariableOpdnn/logits/kernel,dnn/logits/kernel/Initializer/random_uniform*
dtype0
w
%dnn/logits/kernel/Read/ReadVariableOpReadVariableOpdnn/logits/kernel*
_output_shapes

:
*
dtype0
?
!dnn/logits/bias/Initializer/zerosConst*"
_class
loc:@dnn/logits/bias*
_output_shapes
:*
dtype0*
valueB*    
?
dnn/logits/biasVarHandleOp*"
_class
loc:@dnn/logits/bias*
_output_shapes
: *
dtype0*
shape:* 
shared_namednn/logits/bias
o
0dnn/logits/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/logits/bias*
_output_shapes
: 
k
dnn/logits/bias/AssignAssignVariableOpdnn/logits/bias!dnn/logits/bias/Initializer/zeros*
dtype0
o
#dnn/logits/bias/Read/ReadVariableOpReadVariableOpdnn/logits/bias*
_output_shapes
:*
dtype0
r
 dnn/logits/MatMul/ReadVariableOpReadVariableOpdnn/logits/kernel*
_output_shapes

:
*
dtype0
?
dnn/logits/MatMulMatMul-dnn/hiddenlayer_8/batchnorm_8/batchnorm/add_1 dnn/logits/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
m
!dnn/logits/BiasAdd/ReadVariableOpReadVariableOpdnn/logits/bias*
_output_shapes
:*
dtype0
?
dnn/logits/BiasAddBiasAdddnn/logits/MatMul!dnn/logits/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:?????????
e
dnn/zero_fraction_9/SizeSizednn/logits/BiasAdd*
T0*
_output_shapes
: *
out_type0	
e
dnn/zero_fraction_9/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R????
?
dnn/zero_fraction_9/LessEqual	LessEqualdnn/zero_fraction_9/Sizednn/zero_fraction_9/LessEqual/y*
T0	*
_output_shapes
: 
?
dnn/zero_fraction_9/condStatelessIfdnn/zero_fraction_9/LessEqualdnn/logits/BiasAdd*
Tcond0
*
Tin
2*
Tout

2	*
_lower_using_switch_merge(* 
_output_shapes
: : : : : : * 
_read_only_resource_inputs
 *8
else_branch)R'
%dnn_zero_fraction_9_cond_false_512381*
output_shapes
: : : : : : *7
then_branch(R&
$dnn_zero_fraction_9_cond_true_512380
h
!dnn/zero_fraction_9/cond/IdentityIdentitydnn/zero_fraction_9/cond*
T0	*
_output_shapes
: 
l
#dnn/zero_fraction_9/cond/Identity_1Identitydnn/zero_fraction_9/cond:1*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_9/cond/Identity_2Identitydnn/zero_fraction_9/cond:2*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_9/cond/Identity_3Identitydnn/zero_fraction_9/cond:3*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_9/cond/Identity_4Identitydnn/zero_fraction_9/cond:4*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_9/cond/Identity_5Identitydnn/zero_fraction_9/cond:5*
T0*
_output_shapes
: 
?
*dnn/zero_fraction_9/counts_to_fraction/subSubdnn/zero_fraction_9/Size!dnn/zero_fraction_9/cond/Identity*
T0	*
_output_shapes
: 
?
+dnn/zero_fraction_9/counts_to_fraction/CastCast*dnn/zero_fraction_9/counts_to_fraction/sub*

DstT0*

SrcT0	*
_output_shapes
: 

-dnn/zero_fraction_9/counts_to_fraction/Cast_1Castdnn/zero_fraction_9/Size*

DstT0*

SrcT0	*
_output_shapes
: 
?
.dnn/zero_fraction_9/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_9/counts_to_fraction/Cast-dnn/zero_fraction_9/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
y
dnn/zero_fraction_9/fractionIdentity.dnn/zero_fraction_9/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
'dnn/logits/fraction_of_zero_values/tagsConst*
_output_shapes
: *
dtype0*3
value*B( B"dnn/logits/fraction_of_zero_values
?
"dnn/logits/fraction_of_zero_valuesScalarSummary'dnn/logits/fraction_of_zero_values/tagsdnn/zero_fraction_9/fraction*
T0*
_output_shapes
: 
o
dnn/logits/activation/tagConst*
_output_shapes
: *
dtype0*&
valueB Bdnn/logits/activation
p
dnn/logits/activationHistogramSummarydnn/logits/activation/tagdnn/logits/BiasAdd*
_output_shapes
: 
S
head/logits/ShapeShapednn/logits/BiasAdd*
T0*
_output_shapes
:
g
%head/logits/assert_rank_at_least/rankConst*
_output_shapes
: *
dtype0*
value	B :
W
Ohead/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
H
@head/logits/assert_rank_at_least/static_checks_determined_all_okNoOp
j
head/predictions/logisticSigmoiddnn/logits/BiasAdd*
T0*'
_output_shapes
:?????????
n
head/predictions/zeros_like	ZerosLikednn/logits/BiasAdd*
T0*'
_output_shapes
:?????????
q
&head/predictions/two_class_logits/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????
?
!head/predictions/two_class_logitsConcatV2head/predictions/zeros_likednn/logits/BiasAdd&head/predictions/two_class_logits/axis*
N*
T0*'
_output_shapes
:?????????
~
head/predictions/probabilitiesSoftmax!head/predictions/two_class_logits*
T0*'
_output_shapes
:?????????
o
$head/predictions/class_ids/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
?????????
?
head/predictions/class_idsArgMax!head/predictions/two_class_logits$head/predictions/class_ids/dimension*
T0*#
_output_shapes
:?????????
j
head/predictions/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????
?
head/predictions/ExpandDims
ExpandDimshead/predictions/class_idshead/predictions/ExpandDims/dim*
T0	*'
_output_shapes
:?????????
w
head/predictions/str_classesAsStringhead/predictions/ExpandDims*
T0	*'
_output_shapes
:?????????
X
head/predictions/ShapeShapednn/logits/BiasAdd*
T0*
_output_shapes
:
n
$head/predictions/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
p
&head/predictions/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
p
&head/predictions/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
head/predictions/strided_sliceStridedSlicehead/predictions/Shape$head/predictions/strided_slice/stack&head/predictions/strided_slice/stack_1&head/predictions/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
^
head/predictions/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
^
head/predictions/range/limitConst*
_output_shapes
: *
dtype0*
value	B :
^
head/predictions/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
?
head/predictions/rangeRangehead/predictions/range/starthead/predictions/range/limithead/predictions/range/delta*
_output_shapes
:
c
!head/predictions/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
?
head/predictions/ExpandDims_1
ExpandDimshead/predictions/range!head/predictions/ExpandDims_1/dim*
T0*
_output_shapes

:
c
!head/predictions/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :
?
head/predictions/Tile/multiplesPackhead/predictions/strided_slice!head/predictions/Tile/multiples/1*
N*
T0*
_output_shapes
:
?
head/predictions/TileTilehead/predictions/ExpandDims_1head/predictions/Tile/multiples*
T0*'
_output_shapes
:?????????
Z
head/predictions/Shape_1Shapednn/logits/BiasAdd*
T0*
_output_shapes
:
p
&head/predictions/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
r
(head/predictions/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
r
(head/predictions/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
 head/predictions/strided_slice_1StridedSlicehead/predictions/Shape_1&head/predictions/strided_slice_1/stack(head/predictions/strided_slice_1/stack_1(head/predictions/strided_slice_1/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
`
head/predictions/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 
`
head/predictions/range_1/limitConst*
_output_shapes
: *
dtype0*
value	B :
`
head/predictions/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :
?
head/predictions/range_1Rangehead/predictions/range_1/starthead/predictions/range_1/limithead/predictions/range_1/delta*
_output_shapes
:
d
head/predictions/AsStringAsStringhead/predictions/range_1*
T0*
_output_shapes
:
c
!head/predictions/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : 
?
head/predictions/ExpandDims_2
ExpandDimshead/predictions/AsString!head/predictions/ExpandDims_2/dim*
T0*
_output_shapes

:
e
#head/predictions/Tile_1/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :
?
!head/predictions/Tile_1/multiplesPack head/predictions/strided_slice_1#head/predictions/Tile_1/multiples/1*
N*
T0*
_output_shapes
:
?
head/predictions/Tile_1Tilehead/predictions/ExpandDims_2!head/predictions/Tile_1/multiples*
T0*'
_output_shapes
:?????????
X

head/ShapeShapehead/predictions/probabilities*
T0*
_output_shapes
:
b
head/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
d
head/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
d
head/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
head/strided_sliceStridedSlice
head/Shapehead/strided_slice/stackhead/strided_slice/stack_1head/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
R
head/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
R
head/range/limitConst*
_output_shapes
: *
dtype0*
value	B :
R
head/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
e

head/rangeRangehead/range/starthead/range/limithead/range/delta*
_output_shapes
:
J
head/AsStringAsString
head/range*
T0*
_output_shapes
:
U
head/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
j
head/ExpandDims
ExpandDimshead/AsStringhead/ExpandDims/dim*
T0*
_output_shapes

:
W
head/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :
t
head/Tile/multiplesPackhead/strided_slicehead/Tile/multiples/1*
N*
T0*
_output_shapes
:
i
	head/TileTilehead/ExpandDimshead/Tile/multiples*
T0*'
_output_shapes
:?????????

initNoOp
?
init_all_tablesNoOp?^dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_lookup/hash_table/table_init/LookupTableImportV2?^dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_lookup/hash_table/table_init/LookupTableImportV2?^dnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_lookup/hash_table/table_init/LookupTableImportV2z^dnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_lookup/hash_table/table_init/LookupTableImportV2~^dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_lookup/hash_table/table_init/LookupTableImportV2

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
Y
save/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
dtype0*
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
dtype0*
shape: 
{
save/StaticRegexFullMatchStaticRegexFullMatch
save/Const"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
a
save/Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part
f
save/Const_2Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
|
save/SelectSelectsave/StaticRegexFullMatchsave/Const_1save/Const_2"/device:CPU:**
T0*
_output_shapes
: 
f
save/StringJoin
StringJoin
save/Constsave/Select"/device:CPU:**
N*
_output_shapes
: 
Q
save/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
k
save/ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
?
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
?
save/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*?
value?B??B"dnn/hiddenlayer_0/batchnorm_0/betaB#dnn/hiddenlayer_0/batchnorm_0/gammaB)dnn/hiddenlayer_0/batchnorm_0/moving_meanB-dnn/hiddenlayer_0/batchnorm_0/moving_varianceBdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelB"dnn/hiddenlayer_1/batchnorm_1/betaB#dnn/hiddenlayer_1/batchnorm_1/gammaB)dnn/hiddenlayer_1/batchnorm_1/moving_meanB-dnn/hiddenlayer_1/batchnorm_1/moving_varianceBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelB"dnn/hiddenlayer_2/batchnorm_2/betaB#dnn/hiddenlayer_2/batchnorm_2/gammaB)dnn/hiddenlayer_2/batchnorm_2/moving_meanB-dnn/hiddenlayer_2/batchnorm_2/moving_varianceBdnn/hiddenlayer_2/biasBdnn/hiddenlayer_2/kernelB"dnn/hiddenlayer_3/batchnorm_3/betaB#dnn/hiddenlayer_3/batchnorm_3/gammaB)dnn/hiddenlayer_3/batchnorm_3/moving_meanB-dnn/hiddenlayer_3/batchnorm_3/moving_varianceBdnn/hiddenlayer_3/biasBdnn/hiddenlayer_3/kernelB"dnn/hiddenlayer_4/batchnorm_4/betaB#dnn/hiddenlayer_4/batchnorm_4/gammaB)dnn/hiddenlayer_4/batchnorm_4/moving_meanB-dnn/hiddenlayer_4/batchnorm_4/moving_varianceBdnn/hiddenlayer_4/biasBdnn/hiddenlayer_4/kernelB"dnn/hiddenlayer_5/batchnorm_5/betaB#dnn/hiddenlayer_5/batchnorm_5/gammaB)dnn/hiddenlayer_5/batchnorm_5/moving_meanB-dnn/hiddenlayer_5/batchnorm_5/moving_varianceBdnn/hiddenlayer_5/biasBdnn/hiddenlayer_5/kernelB"dnn/hiddenlayer_6/batchnorm_6/betaB#dnn/hiddenlayer_6/batchnorm_6/gammaB)dnn/hiddenlayer_6/batchnorm_6/moving_meanB-dnn/hiddenlayer_6/batchnorm_6/moving_varianceBdnn/hiddenlayer_6/biasBdnn/hiddenlayer_6/kernelB"dnn/hiddenlayer_7/batchnorm_7/betaB#dnn/hiddenlayer_7/batchnorm_7/gammaB)dnn/hiddenlayer_7/batchnorm_7/moving_meanB-dnn/hiddenlayer_7/batchnorm_7/moving_varianceBdnn/hiddenlayer_7/biasBdnn/hiddenlayer_7/kernelB"dnn/hiddenlayer_8/batchnorm_8/betaB#dnn/hiddenlayer_8/batchnorm_8/gammaB)dnn/hiddenlayer_8/batchnorm_8/moving_meanB-dnn/hiddenlayer_8/batchnorm_8/moving_varianceBdnn/hiddenlayer_8/biasBdnn/hiddenlayer_8/kernelBQdnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weightsB_dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weightsBSdnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weightsBSdnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weightsBOdnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weightsBQdnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weightsBdnn/logits/biasBdnn/logits/kernelBglobal_step
?
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices6dnn/hiddenlayer_0/batchnorm_0/beta/Read/ReadVariableOp7dnn/hiddenlayer_0/batchnorm_0/gamma/Read/ReadVariableOp=dnn/hiddenlayer_0/batchnorm_0/moving_mean/Read/ReadVariableOpAdnn/hiddenlayer_0/batchnorm_0/moving_variance/Read/ReadVariableOp*dnn/hiddenlayer_0/bias/Read/ReadVariableOp,dnn/hiddenlayer_0/kernel/Read/ReadVariableOp6dnn/hiddenlayer_1/batchnorm_1/beta/Read/ReadVariableOp7dnn/hiddenlayer_1/batchnorm_1/gamma/Read/ReadVariableOp=dnn/hiddenlayer_1/batchnorm_1/moving_mean/Read/ReadVariableOpAdnn/hiddenlayer_1/batchnorm_1/moving_variance/Read/ReadVariableOp*dnn/hiddenlayer_1/bias/Read/ReadVariableOp,dnn/hiddenlayer_1/kernel/Read/ReadVariableOp6dnn/hiddenlayer_2/batchnorm_2/beta/Read/ReadVariableOp7dnn/hiddenlayer_2/batchnorm_2/gamma/Read/ReadVariableOp=dnn/hiddenlayer_2/batchnorm_2/moving_mean/Read/ReadVariableOpAdnn/hiddenlayer_2/batchnorm_2/moving_variance/Read/ReadVariableOp*dnn/hiddenlayer_2/bias/Read/ReadVariableOp,dnn/hiddenlayer_2/kernel/Read/ReadVariableOp6dnn/hiddenlayer_3/batchnorm_3/beta/Read/ReadVariableOp7dnn/hiddenlayer_3/batchnorm_3/gamma/Read/ReadVariableOp=dnn/hiddenlayer_3/batchnorm_3/moving_mean/Read/ReadVariableOpAdnn/hiddenlayer_3/batchnorm_3/moving_variance/Read/ReadVariableOp*dnn/hiddenlayer_3/bias/Read/ReadVariableOp,dnn/hiddenlayer_3/kernel/Read/ReadVariableOp6dnn/hiddenlayer_4/batchnorm_4/beta/Read/ReadVariableOp7dnn/hiddenlayer_4/batchnorm_4/gamma/Read/ReadVariableOp=dnn/hiddenlayer_4/batchnorm_4/moving_mean/Read/ReadVariableOpAdnn/hiddenlayer_4/batchnorm_4/moving_variance/Read/ReadVariableOp*dnn/hiddenlayer_4/bias/Read/ReadVariableOp,dnn/hiddenlayer_4/kernel/Read/ReadVariableOp6dnn/hiddenlayer_5/batchnorm_5/beta/Read/ReadVariableOp7dnn/hiddenlayer_5/batchnorm_5/gamma/Read/ReadVariableOp=dnn/hiddenlayer_5/batchnorm_5/moving_mean/Read/ReadVariableOpAdnn/hiddenlayer_5/batchnorm_5/moving_variance/Read/ReadVariableOp*dnn/hiddenlayer_5/bias/Read/ReadVariableOp,dnn/hiddenlayer_5/kernel/Read/ReadVariableOp6dnn/hiddenlayer_6/batchnorm_6/beta/Read/ReadVariableOp7dnn/hiddenlayer_6/batchnorm_6/gamma/Read/ReadVariableOp=dnn/hiddenlayer_6/batchnorm_6/moving_mean/Read/ReadVariableOpAdnn/hiddenlayer_6/batchnorm_6/moving_variance/Read/ReadVariableOp*dnn/hiddenlayer_6/bias/Read/ReadVariableOp,dnn/hiddenlayer_6/kernel/Read/ReadVariableOp6dnn/hiddenlayer_7/batchnorm_7/beta/Read/ReadVariableOp7dnn/hiddenlayer_7/batchnorm_7/gamma/Read/ReadVariableOp=dnn/hiddenlayer_7/batchnorm_7/moving_mean/Read/ReadVariableOpAdnn/hiddenlayer_7/batchnorm_7/moving_variance/Read/ReadVariableOp*dnn/hiddenlayer_7/bias/Read/ReadVariableOp,dnn/hiddenlayer_7/kernel/Read/ReadVariableOp6dnn/hiddenlayer_8/batchnorm_8/beta/Read/ReadVariableOp7dnn/hiddenlayer_8/batchnorm_8/gamma/Read/ReadVariableOp=dnn/hiddenlayer_8/batchnorm_8/moving_mean/Read/ReadVariableOpAdnn/hiddenlayer_8/batchnorm_8/moving_variance/Read/ReadVariableOp*dnn/hiddenlayer_8/bias/Read/ReadVariableOp,dnn/hiddenlayer_8/kernel/Read/ReadVariableOpednn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights/Read/ReadVariableOpsdnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights/Read/ReadVariableOpgdnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights/Read/ReadVariableOpgdnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights/Read/ReadVariableOpcdnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights/Read/ReadVariableOpednn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights/Read/ReadVariableOp#dnn/logits/bias/Read/ReadVariableOp%dnn/logits/kernel/Read/ReadVariableOpglobal_step/Read/ReadVariableOp"/device:CPU:0*M
dtypesC
A2?	
?
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
?
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
N*
T0*
_output_shapes
:
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0
?
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
?
save/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*?
value?B??B"dnn/hiddenlayer_0/batchnorm_0/betaB#dnn/hiddenlayer_0/batchnorm_0/gammaB)dnn/hiddenlayer_0/batchnorm_0/moving_meanB-dnn/hiddenlayer_0/batchnorm_0/moving_varianceBdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelB"dnn/hiddenlayer_1/batchnorm_1/betaB#dnn/hiddenlayer_1/batchnorm_1/gammaB)dnn/hiddenlayer_1/batchnorm_1/moving_meanB-dnn/hiddenlayer_1/batchnorm_1/moving_varianceBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelB"dnn/hiddenlayer_2/batchnorm_2/betaB#dnn/hiddenlayer_2/batchnorm_2/gammaB)dnn/hiddenlayer_2/batchnorm_2/moving_meanB-dnn/hiddenlayer_2/batchnorm_2/moving_varianceBdnn/hiddenlayer_2/biasBdnn/hiddenlayer_2/kernelB"dnn/hiddenlayer_3/batchnorm_3/betaB#dnn/hiddenlayer_3/batchnorm_3/gammaB)dnn/hiddenlayer_3/batchnorm_3/moving_meanB-dnn/hiddenlayer_3/batchnorm_3/moving_varianceBdnn/hiddenlayer_3/biasBdnn/hiddenlayer_3/kernelB"dnn/hiddenlayer_4/batchnorm_4/betaB#dnn/hiddenlayer_4/batchnorm_4/gammaB)dnn/hiddenlayer_4/batchnorm_4/moving_meanB-dnn/hiddenlayer_4/batchnorm_4/moving_varianceBdnn/hiddenlayer_4/biasBdnn/hiddenlayer_4/kernelB"dnn/hiddenlayer_5/batchnorm_5/betaB#dnn/hiddenlayer_5/batchnorm_5/gammaB)dnn/hiddenlayer_5/batchnorm_5/moving_meanB-dnn/hiddenlayer_5/batchnorm_5/moving_varianceBdnn/hiddenlayer_5/biasBdnn/hiddenlayer_5/kernelB"dnn/hiddenlayer_6/batchnorm_6/betaB#dnn/hiddenlayer_6/batchnorm_6/gammaB)dnn/hiddenlayer_6/batchnorm_6/moving_meanB-dnn/hiddenlayer_6/batchnorm_6/moving_varianceBdnn/hiddenlayer_6/biasBdnn/hiddenlayer_6/kernelB"dnn/hiddenlayer_7/batchnorm_7/betaB#dnn/hiddenlayer_7/batchnorm_7/gammaB)dnn/hiddenlayer_7/batchnorm_7/moving_meanB-dnn/hiddenlayer_7/batchnorm_7/moving_varianceBdnn/hiddenlayer_7/biasBdnn/hiddenlayer_7/kernelB"dnn/hiddenlayer_8/batchnorm_8/betaB#dnn/hiddenlayer_8/batchnorm_8/gammaB)dnn/hiddenlayer_8/batchnorm_8/moving_meanB-dnn/hiddenlayer_8/batchnorm_8/moving_varianceBdnn/hiddenlayer_8/biasBdnn/hiddenlayer_8/kernelBQdnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weightsB_dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weightsBSdnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weightsBSdnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weightsBOdnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weightsBQdnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weightsBdnn/logits/biasBdnn/logits/kernelBglobal_step
?
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*M
dtypesC
A2?	
N
save/Identity_1Identitysave/RestoreV2*
T0*
_output_shapes
:
k
save/AssignVariableOpAssignVariableOp"dnn/hiddenlayer_0/batchnorm_0/betasave/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:1*
T0*
_output_shapes
:
n
save/AssignVariableOp_1AssignVariableOp#dnn/hiddenlayer_0/batchnorm_0/gammasave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:2*
T0*
_output_shapes
:
t
save/AssignVariableOp_2AssignVariableOp)dnn/hiddenlayer_0/batchnorm_0/moving_meansave/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:3*
T0*
_output_shapes
:
x
save/AssignVariableOp_3AssignVariableOp-dnn/hiddenlayer_0/batchnorm_0/moving_variancesave/Identity_4*
dtype0
P
save/Identity_5Identitysave/RestoreV2:4*
T0*
_output_shapes
:
a
save/AssignVariableOp_4AssignVariableOpdnn/hiddenlayer_0/biassave/Identity_5*
dtype0
P
save/Identity_6Identitysave/RestoreV2:5*
T0*
_output_shapes
:
c
save/AssignVariableOp_5AssignVariableOpdnn/hiddenlayer_0/kernelsave/Identity_6*
dtype0
P
save/Identity_7Identitysave/RestoreV2:6*
T0*
_output_shapes
:
m
save/AssignVariableOp_6AssignVariableOp"dnn/hiddenlayer_1/batchnorm_1/betasave/Identity_7*
dtype0
P
save/Identity_8Identitysave/RestoreV2:7*
T0*
_output_shapes
:
n
save/AssignVariableOp_7AssignVariableOp#dnn/hiddenlayer_1/batchnorm_1/gammasave/Identity_8*
dtype0
P
save/Identity_9Identitysave/RestoreV2:8*
T0*
_output_shapes
:
t
save/AssignVariableOp_8AssignVariableOp)dnn/hiddenlayer_1/batchnorm_1/moving_meansave/Identity_9*
dtype0
Q
save/Identity_10Identitysave/RestoreV2:9*
T0*
_output_shapes
:
y
save/AssignVariableOp_9AssignVariableOp-dnn/hiddenlayer_1/batchnorm_1/moving_variancesave/Identity_10*
dtype0
R
save/Identity_11Identitysave/RestoreV2:10*
T0*
_output_shapes
:
c
save/AssignVariableOp_10AssignVariableOpdnn/hiddenlayer_1/biassave/Identity_11*
dtype0
R
save/Identity_12Identitysave/RestoreV2:11*
T0*
_output_shapes
:
e
save/AssignVariableOp_11AssignVariableOpdnn/hiddenlayer_1/kernelsave/Identity_12*
dtype0
R
save/Identity_13Identitysave/RestoreV2:12*
T0*
_output_shapes
:
o
save/AssignVariableOp_12AssignVariableOp"dnn/hiddenlayer_2/batchnorm_2/betasave/Identity_13*
dtype0
R
save/Identity_14Identitysave/RestoreV2:13*
T0*
_output_shapes
:
p
save/AssignVariableOp_13AssignVariableOp#dnn/hiddenlayer_2/batchnorm_2/gammasave/Identity_14*
dtype0
R
save/Identity_15Identitysave/RestoreV2:14*
T0*
_output_shapes
:
v
save/AssignVariableOp_14AssignVariableOp)dnn/hiddenlayer_2/batchnorm_2/moving_meansave/Identity_15*
dtype0
R
save/Identity_16Identitysave/RestoreV2:15*
T0*
_output_shapes
:
z
save/AssignVariableOp_15AssignVariableOp-dnn/hiddenlayer_2/batchnorm_2/moving_variancesave/Identity_16*
dtype0
R
save/Identity_17Identitysave/RestoreV2:16*
T0*
_output_shapes
:
c
save/AssignVariableOp_16AssignVariableOpdnn/hiddenlayer_2/biassave/Identity_17*
dtype0
R
save/Identity_18Identitysave/RestoreV2:17*
T0*
_output_shapes
:
e
save/AssignVariableOp_17AssignVariableOpdnn/hiddenlayer_2/kernelsave/Identity_18*
dtype0
R
save/Identity_19Identitysave/RestoreV2:18*
T0*
_output_shapes
:
o
save/AssignVariableOp_18AssignVariableOp"dnn/hiddenlayer_3/batchnorm_3/betasave/Identity_19*
dtype0
R
save/Identity_20Identitysave/RestoreV2:19*
T0*
_output_shapes
:
p
save/AssignVariableOp_19AssignVariableOp#dnn/hiddenlayer_3/batchnorm_3/gammasave/Identity_20*
dtype0
R
save/Identity_21Identitysave/RestoreV2:20*
T0*
_output_shapes
:
v
save/AssignVariableOp_20AssignVariableOp)dnn/hiddenlayer_3/batchnorm_3/moving_meansave/Identity_21*
dtype0
R
save/Identity_22Identitysave/RestoreV2:21*
T0*
_output_shapes
:
z
save/AssignVariableOp_21AssignVariableOp-dnn/hiddenlayer_3/batchnorm_3/moving_variancesave/Identity_22*
dtype0
R
save/Identity_23Identitysave/RestoreV2:22*
T0*
_output_shapes
:
c
save/AssignVariableOp_22AssignVariableOpdnn/hiddenlayer_3/biassave/Identity_23*
dtype0
R
save/Identity_24Identitysave/RestoreV2:23*
T0*
_output_shapes
:
e
save/AssignVariableOp_23AssignVariableOpdnn/hiddenlayer_3/kernelsave/Identity_24*
dtype0
R
save/Identity_25Identitysave/RestoreV2:24*
T0*
_output_shapes
:
o
save/AssignVariableOp_24AssignVariableOp"dnn/hiddenlayer_4/batchnorm_4/betasave/Identity_25*
dtype0
R
save/Identity_26Identitysave/RestoreV2:25*
T0*
_output_shapes
:
p
save/AssignVariableOp_25AssignVariableOp#dnn/hiddenlayer_4/batchnorm_4/gammasave/Identity_26*
dtype0
R
save/Identity_27Identitysave/RestoreV2:26*
T0*
_output_shapes
:
v
save/AssignVariableOp_26AssignVariableOp)dnn/hiddenlayer_4/batchnorm_4/moving_meansave/Identity_27*
dtype0
R
save/Identity_28Identitysave/RestoreV2:27*
T0*
_output_shapes
:
z
save/AssignVariableOp_27AssignVariableOp-dnn/hiddenlayer_4/batchnorm_4/moving_variancesave/Identity_28*
dtype0
R
save/Identity_29Identitysave/RestoreV2:28*
T0*
_output_shapes
:
c
save/AssignVariableOp_28AssignVariableOpdnn/hiddenlayer_4/biassave/Identity_29*
dtype0
R
save/Identity_30Identitysave/RestoreV2:29*
T0*
_output_shapes
:
e
save/AssignVariableOp_29AssignVariableOpdnn/hiddenlayer_4/kernelsave/Identity_30*
dtype0
R
save/Identity_31Identitysave/RestoreV2:30*
T0*
_output_shapes
:
o
save/AssignVariableOp_30AssignVariableOp"dnn/hiddenlayer_5/batchnorm_5/betasave/Identity_31*
dtype0
R
save/Identity_32Identitysave/RestoreV2:31*
T0*
_output_shapes
:
p
save/AssignVariableOp_31AssignVariableOp#dnn/hiddenlayer_5/batchnorm_5/gammasave/Identity_32*
dtype0
R
save/Identity_33Identitysave/RestoreV2:32*
T0*
_output_shapes
:
v
save/AssignVariableOp_32AssignVariableOp)dnn/hiddenlayer_5/batchnorm_5/moving_meansave/Identity_33*
dtype0
R
save/Identity_34Identitysave/RestoreV2:33*
T0*
_output_shapes
:
z
save/AssignVariableOp_33AssignVariableOp-dnn/hiddenlayer_5/batchnorm_5/moving_variancesave/Identity_34*
dtype0
R
save/Identity_35Identitysave/RestoreV2:34*
T0*
_output_shapes
:
c
save/AssignVariableOp_34AssignVariableOpdnn/hiddenlayer_5/biassave/Identity_35*
dtype0
R
save/Identity_36Identitysave/RestoreV2:35*
T0*
_output_shapes
:
e
save/AssignVariableOp_35AssignVariableOpdnn/hiddenlayer_5/kernelsave/Identity_36*
dtype0
R
save/Identity_37Identitysave/RestoreV2:36*
T0*
_output_shapes
:
o
save/AssignVariableOp_36AssignVariableOp"dnn/hiddenlayer_6/batchnorm_6/betasave/Identity_37*
dtype0
R
save/Identity_38Identitysave/RestoreV2:37*
T0*
_output_shapes
:
p
save/AssignVariableOp_37AssignVariableOp#dnn/hiddenlayer_6/batchnorm_6/gammasave/Identity_38*
dtype0
R
save/Identity_39Identitysave/RestoreV2:38*
T0*
_output_shapes
:
v
save/AssignVariableOp_38AssignVariableOp)dnn/hiddenlayer_6/batchnorm_6/moving_meansave/Identity_39*
dtype0
R
save/Identity_40Identitysave/RestoreV2:39*
T0*
_output_shapes
:
z
save/AssignVariableOp_39AssignVariableOp-dnn/hiddenlayer_6/batchnorm_6/moving_variancesave/Identity_40*
dtype0
R
save/Identity_41Identitysave/RestoreV2:40*
T0*
_output_shapes
:
c
save/AssignVariableOp_40AssignVariableOpdnn/hiddenlayer_6/biassave/Identity_41*
dtype0
R
save/Identity_42Identitysave/RestoreV2:41*
T0*
_output_shapes
:
e
save/AssignVariableOp_41AssignVariableOpdnn/hiddenlayer_6/kernelsave/Identity_42*
dtype0
R
save/Identity_43Identitysave/RestoreV2:42*
T0*
_output_shapes
:
o
save/AssignVariableOp_42AssignVariableOp"dnn/hiddenlayer_7/batchnorm_7/betasave/Identity_43*
dtype0
R
save/Identity_44Identitysave/RestoreV2:43*
T0*
_output_shapes
:
p
save/AssignVariableOp_43AssignVariableOp#dnn/hiddenlayer_7/batchnorm_7/gammasave/Identity_44*
dtype0
R
save/Identity_45Identitysave/RestoreV2:44*
T0*
_output_shapes
:
v
save/AssignVariableOp_44AssignVariableOp)dnn/hiddenlayer_7/batchnorm_7/moving_meansave/Identity_45*
dtype0
R
save/Identity_46Identitysave/RestoreV2:45*
T0*
_output_shapes
:
z
save/AssignVariableOp_45AssignVariableOp-dnn/hiddenlayer_7/batchnorm_7/moving_variancesave/Identity_46*
dtype0
R
save/Identity_47Identitysave/RestoreV2:46*
T0*
_output_shapes
:
c
save/AssignVariableOp_46AssignVariableOpdnn/hiddenlayer_7/biassave/Identity_47*
dtype0
R
save/Identity_48Identitysave/RestoreV2:47*
T0*
_output_shapes
:
e
save/AssignVariableOp_47AssignVariableOpdnn/hiddenlayer_7/kernelsave/Identity_48*
dtype0
R
save/Identity_49Identitysave/RestoreV2:48*
T0*
_output_shapes
:
o
save/AssignVariableOp_48AssignVariableOp"dnn/hiddenlayer_8/batchnorm_8/betasave/Identity_49*
dtype0
R
save/Identity_50Identitysave/RestoreV2:49*
T0*
_output_shapes
:
p
save/AssignVariableOp_49AssignVariableOp#dnn/hiddenlayer_8/batchnorm_8/gammasave/Identity_50*
dtype0
R
save/Identity_51Identitysave/RestoreV2:50*
T0*
_output_shapes
:
v
save/AssignVariableOp_50AssignVariableOp)dnn/hiddenlayer_8/batchnorm_8/moving_meansave/Identity_51*
dtype0
R
save/Identity_52Identitysave/RestoreV2:51*
T0*
_output_shapes
:
z
save/AssignVariableOp_51AssignVariableOp-dnn/hiddenlayer_8/batchnorm_8/moving_variancesave/Identity_52*
dtype0
R
save/Identity_53Identitysave/RestoreV2:52*
T0*
_output_shapes
:
c
save/AssignVariableOp_52AssignVariableOpdnn/hiddenlayer_8/biassave/Identity_53*
dtype0
R
save/Identity_54Identitysave/RestoreV2:53*
T0*
_output_shapes
:
e
save/AssignVariableOp_53AssignVariableOpdnn/hiddenlayer_8/kernelsave/Identity_54*
dtype0
R
save/Identity_55Identitysave/RestoreV2:54*
T0*
_output_shapes
:
?
save/AssignVariableOp_54AssignVariableOpQdnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weightssave/Identity_55*
dtype0
R
save/Identity_56Identitysave/RestoreV2:55*
T0*
_output_shapes
:
?
save/AssignVariableOp_55AssignVariableOp_dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weightssave/Identity_56*
dtype0
R
save/Identity_57Identitysave/RestoreV2:56*
T0*
_output_shapes
:
?
save/AssignVariableOp_56AssignVariableOpSdnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weightssave/Identity_57*
dtype0
R
save/Identity_58Identitysave/RestoreV2:57*
T0*
_output_shapes
:
?
save/AssignVariableOp_57AssignVariableOpSdnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weightssave/Identity_58*
dtype0
R
save/Identity_59Identitysave/RestoreV2:58*
T0*
_output_shapes
:
?
save/AssignVariableOp_58AssignVariableOpOdnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weightssave/Identity_59*
dtype0
R
save/Identity_60Identitysave/RestoreV2:59*
T0*
_output_shapes
:
?
save/AssignVariableOp_59AssignVariableOpQdnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weightssave/Identity_60*
dtype0
R
save/Identity_61Identitysave/RestoreV2:60*
T0*
_output_shapes
:
\
save/AssignVariableOp_60AssignVariableOpdnn/logits/biassave/Identity_61*
dtype0
R
save/Identity_62Identitysave/RestoreV2:61*
T0*
_output_shapes
:
^
save/AssignVariableOp_61AssignVariableOpdnn/logits/kernelsave/Identity_62*
dtype0
R
save/Identity_63Identitysave/RestoreV2:62*
T0	*
_output_shapes
:
X
save/AssignVariableOp_62AssignVariableOpglobal_stepsave/Identity_63*
dtype0	
?
save/restore_shardNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_12^save/AssignVariableOp_13^save/AssignVariableOp_14^save/AssignVariableOp_15^save/AssignVariableOp_16^save/AssignVariableOp_17^save/AssignVariableOp_18^save/AssignVariableOp_19^save/AssignVariableOp_2^save/AssignVariableOp_20^save/AssignVariableOp_21^save/AssignVariableOp_22^save/AssignVariableOp_23^save/AssignVariableOp_24^save/AssignVariableOp_25^save/AssignVariableOp_26^save/AssignVariableOp_27^save/AssignVariableOp_28^save/AssignVariableOp_29^save/AssignVariableOp_3^save/AssignVariableOp_30^save/AssignVariableOp_31^save/AssignVariableOp_32^save/AssignVariableOp_33^save/AssignVariableOp_34^save/AssignVariableOp_35^save/AssignVariableOp_36^save/AssignVariableOp_37^save/AssignVariableOp_38^save/AssignVariableOp_39^save/AssignVariableOp_4^save/AssignVariableOp_40^save/AssignVariableOp_41^save/AssignVariableOp_42^save/AssignVariableOp_43^save/AssignVariableOp_44^save/AssignVariableOp_45^save/AssignVariableOp_46^save/AssignVariableOp_47^save/AssignVariableOp_48^save/AssignVariableOp_49^save/AssignVariableOp_5^save/AssignVariableOp_50^save/AssignVariableOp_51^save/AssignVariableOp_52^save/AssignVariableOp_53^save/AssignVariableOp_54^save/AssignVariableOp_55^save/AssignVariableOp_56^save/AssignVariableOp_57^save/AssignVariableOp_58^save/AssignVariableOp_59^save/AssignVariableOp_6^save/AssignVariableOp_60^save/AssignVariableOp_61^save/AssignVariableOp_62^save/AssignVariableOp_7^save/AssignVariableOp_8^save/AssignVariableOp_9
-
save/restore_allNoOp^save/restore_shard??
?
?
%dnn_zero_fraction_1_cond_false_511626H
Dcount_nonzero_notequal_dnn_hiddenlayer_1_batchnorm_1_batchnorm_add_1
count_nonzero_nonzero_count	
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalnoneo
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros?
count_nonzero/NotEqualNotEqualDcount_nonzero_notequal_dnn_hiddenlayer_1_batchnorm_1_batchnorm_add_1count_nonzero/zeros:output:0*
T0*'
_output_shapes
:?????????d2
count_nonzero/NotEqual?
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*'
_output_shapes
:?????????d2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const?
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count?
OptionalFromValueOptionalFromValuecount_nonzero/zeros:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue?
OptionalFromValue_1OptionalFromValuecount_nonzero/NotEqual:z:0*
Toutput_types
2
*
_output_shapes
: 2
OptionalFromValue_1?
OptionalFromValue_2OptionalFromValuecount_nonzero/Cast:y:0*
Toutput_types
2	*
_output_shapes
: 2
OptionalFromValue_2?
OptionalFromValue_3OptionalFromValuecount_nonzero/Const:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_3D
OptionalNoneOptionalNone*
_output_shapes
: 2
OptionalNone"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"'
optionalnoneOptionalNone:optional:0*&
_input_shapes
:?????????d:- )
'
_output_shapes
:?????????d
?
?
%dnn_zero_fraction_2_cond_false_511724H
Dcount_nonzero_notequal_dnn_hiddenlayer_2_batchnorm_2_batchnorm_add_1
count_nonzero_nonzero_count	
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalnoneo
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros?
count_nonzero/NotEqualNotEqualDcount_nonzero_notequal_dnn_hiddenlayer_2_batchnorm_2_batchnorm_add_1count_nonzero/zeros:output:0*
T0*'
_output_shapes
:?????????d2
count_nonzero/NotEqual?
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*'
_output_shapes
:?????????d2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const?
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count?
OptionalFromValueOptionalFromValuecount_nonzero/zeros:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue?
OptionalFromValue_1OptionalFromValuecount_nonzero/NotEqual:z:0*
Toutput_types
2
*
_output_shapes
: 2
OptionalFromValue_1?
OptionalFromValue_2OptionalFromValuecount_nonzero/Cast:y:0*
Toutput_types
2	*
_output_shapes
: 2
OptionalFromValue_2?
OptionalFromValue_3OptionalFromValuecount_nonzero/Const:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_3D
OptionalNoneOptionalNone*
_output_shapes
: 2
OptionalNone"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"'
optionalnoneOptionalNone:optional:0*&
_input_shapes
:?????????d:- )
'
_output_shapes
:?????????d
?
?
$dnn_zero_fraction_5_cond_true_512017H
Dcount_nonzero_notequal_dnn_hiddenlayer_5_batchnorm_5_batchnorm_add_1
cast	
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros?
count_nonzero/NotEqualNotEqualDcount_nonzero_notequal_dnn_hiddenlayer_5_batchnorm_5_batchnorm_add_1count_nonzero/zeros:output:0*
T0*'
_output_shapes
:?????????22
count_nonzero/NotEqual?
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const?
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast?
OptionalFromValueOptionalFromValuecount_nonzero/zeros:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue?
OptionalFromValue_1OptionalFromValuecount_nonzero/NotEqual:z:0*
Toutput_types
2
*
_output_shapes
: 2
OptionalFromValue_1?
OptionalFromValue_2OptionalFromValuecount_nonzero/Cast:y:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_2?
OptionalFromValue_3OptionalFromValuecount_nonzero/Const:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_3?
OptionalFromValue_4OptionalFromValue$count_nonzero/nonzero_count:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_4"
castCast:y:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0*&
_input_shapes
:?????????2:- )
'
_output_shapes
:?????????2
?
?
%dnn_zero_fraction_4_cond_false_511920H
Dcount_nonzero_notequal_dnn_hiddenlayer_4_batchnorm_4_batchnorm_add_1
count_nonzero_nonzero_count	
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalnoneo
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros?
count_nonzero/NotEqualNotEqualDcount_nonzero_notequal_dnn_hiddenlayer_4_batchnorm_4_batchnorm_add_1count_nonzero/zeros:output:0*
T0*'
_output_shapes
:?????????22
count_nonzero/NotEqual?
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*'
_output_shapes
:?????????22
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const?
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count?
OptionalFromValueOptionalFromValuecount_nonzero/zeros:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue?
OptionalFromValue_1OptionalFromValuecount_nonzero/NotEqual:z:0*
Toutput_types
2
*
_output_shapes
: 2
OptionalFromValue_1?
OptionalFromValue_2OptionalFromValuecount_nonzero/Cast:y:0*
Toutput_types
2	*
_output_shapes
: 2
OptionalFromValue_2?
OptionalFromValue_3OptionalFromValuecount_nonzero/Const:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_3D
OptionalNoneOptionalNone*
_output_shapes
: 2
OptionalNone"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"'
optionalnoneOptionalNone:optional:0*&
_input_shapes
:?????????2:- )
'
_output_shapes
:?????????2
?
?
$dnn_zero_fraction_2_cond_true_511723H
Dcount_nonzero_notequal_dnn_hiddenlayer_2_batchnorm_2_batchnorm_add_1
cast	
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros?
count_nonzero/NotEqualNotEqualDcount_nonzero_notequal_dnn_hiddenlayer_2_batchnorm_2_batchnorm_add_1count_nonzero/zeros:output:0*
T0*'
_output_shapes
:?????????d2
count_nonzero/NotEqual?
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const?
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast?
OptionalFromValueOptionalFromValuecount_nonzero/zeros:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue?
OptionalFromValue_1OptionalFromValuecount_nonzero/NotEqual:z:0*
Toutput_types
2
*
_output_shapes
: 2
OptionalFromValue_1?
OptionalFromValue_2OptionalFromValuecount_nonzero/Cast:y:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_2?
OptionalFromValue_3OptionalFromValuecount_nonzero/Const:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_3?
OptionalFromValue_4OptionalFromValue$count_nonzero/nonzero_count:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_4"
castCast:y:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0*&
_input_shapes
:?????????d:- )
'
_output_shapes
:?????????d
?
?
%dnn_zero_fraction_7_cond_false_512214H
Dcount_nonzero_notequal_dnn_hiddenlayer_7_batchnorm_7_batchnorm_add_1
count_nonzero_nonzero_count	
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalnoneo
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros?
count_nonzero/NotEqualNotEqualDcount_nonzero_notequal_dnn_hiddenlayer_7_batchnorm_7_batchnorm_add_1count_nonzero/zeros:output:0*
T0*'
_output_shapes
:?????????
2
count_nonzero/NotEqual?
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*'
_output_shapes
:?????????
2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const?
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count?
OptionalFromValueOptionalFromValuecount_nonzero/zeros:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue?
OptionalFromValue_1OptionalFromValuecount_nonzero/NotEqual:z:0*
Toutput_types
2
*
_output_shapes
: 2
OptionalFromValue_1?
OptionalFromValue_2OptionalFromValuecount_nonzero/Cast:y:0*
Toutput_types
2	*
_output_shapes
: 2
OptionalFromValue_2?
OptionalFromValue_3OptionalFromValuecount_nonzero/Const:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_3D
OptionalNoneOptionalNone*
_output_shapes
: 2
OptionalNone"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"'
optionalnoneOptionalNone:optional:0*&
_input_shapes
:?????????
:- )
'
_output_shapes
:?????????

?
?
%dnn_zero_fraction_9_cond_false_512381-
)count_nonzero_notequal_dnn_logits_biasadd
count_nonzero_nonzero_count	
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalnoneo
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros?
count_nonzero/NotEqualNotEqual)count_nonzero_notequal_dnn_logits_biasaddcount_nonzero/zeros:output:0*
T0*'
_output_shapes
:?????????2
count_nonzero/NotEqual?
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*'
_output_shapes
:?????????2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const?
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count?
OptionalFromValueOptionalFromValuecount_nonzero/zeros:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue?
OptionalFromValue_1OptionalFromValuecount_nonzero/NotEqual:z:0*
Toutput_types
2
*
_output_shapes
: 2
OptionalFromValue_1?
OptionalFromValue_2OptionalFromValuecount_nonzero/Cast:y:0*
Toutput_types
2	*
_output_shapes
: 2
OptionalFromValue_2?
OptionalFromValue_3OptionalFromValuecount_nonzero/Const:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_3D
OptionalNoneOptionalNone*
_output_shapes
: 2
OptionalNone"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"'
optionalnoneOptionalNone:optional:0*&
_input_shapes
:?????????:- )
'
_output_shapes
:?????????
?
?
$dnn_zero_fraction_9_cond_true_512380-
)count_nonzero_notequal_dnn_logits_biasadd
cast	
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros?
count_nonzero/NotEqualNotEqual)count_nonzero_notequal_dnn_logits_biasaddcount_nonzero/zeros:output:0*
T0*'
_output_shapes
:?????????2
count_nonzero/NotEqual?
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const?
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast?
OptionalFromValueOptionalFromValuecount_nonzero/zeros:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue?
OptionalFromValue_1OptionalFromValuecount_nonzero/NotEqual:z:0*
Toutput_types
2
*
_output_shapes
: 2
OptionalFromValue_1?
OptionalFromValue_2OptionalFromValuecount_nonzero/Cast:y:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_2?
OptionalFromValue_3OptionalFromValuecount_nonzero/Const:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_3?
OptionalFromValue_4OptionalFromValue$count_nonzero/nonzero_count:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_4"
castCast:y:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0*&
_input_shapes
:?????????:- )
'
_output_shapes
:?????????
?
?
"dnn_zero_fraction_cond_true_511527H
Dcount_nonzero_notequal_dnn_hiddenlayer_0_batchnorm_0_batchnorm_add_1
cast	
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros?
count_nonzero/NotEqualNotEqualDcount_nonzero_notequal_dnn_hiddenlayer_0_batchnorm_0_batchnorm_add_1count_nonzero/zeros:output:0*
T0*(
_output_shapes
:??????????2
count_nonzero/NotEqual?
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const?
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast?
OptionalFromValueOptionalFromValuecount_nonzero/zeros:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue?
OptionalFromValue_1OptionalFromValuecount_nonzero/NotEqual:z:0*
Toutput_types
2
*
_output_shapes
: 2
OptionalFromValue_1?
OptionalFromValue_2OptionalFromValuecount_nonzero/Cast:y:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_2?
OptionalFromValue_3OptionalFromValuecount_nonzero/Const:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_3?
OptionalFromValue_4OptionalFromValue$count_nonzero/nonzero_count:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_4"
castCast:y:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0*'
_input_shapes
:??????????:. *
(
_output_shapes
:??????????
?
?
$dnn_zero_fraction_8_cond_true_512311H
Dcount_nonzero_notequal_dnn_hiddenlayer_8_batchnorm_8_batchnorm_add_1
cast	
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros?
count_nonzero/NotEqualNotEqualDcount_nonzero_notequal_dnn_hiddenlayer_8_batchnorm_8_batchnorm_add_1count_nonzero/zeros:output:0*
T0*'
_output_shapes
:?????????
2
count_nonzero/NotEqual?
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const?
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast?
OptionalFromValueOptionalFromValuecount_nonzero/zeros:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue?
OptionalFromValue_1OptionalFromValuecount_nonzero/NotEqual:z:0*
Toutput_types
2
*
_output_shapes
: 2
OptionalFromValue_1?
OptionalFromValue_2OptionalFromValuecount_nonzero/Cast:y:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_2?
OptionalFromValue_3OptionalFromValuecount_nonzero/Const:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_3?
OptionalFromValue_4OptionalFromValue$count_nonzero/nonzero_count:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_4"
castCast:y:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0*&
_input_shapes
:?????????
:- )
'
_output_shapes
:?????????

?
?
$dnn_zero_fraction_6_cond_true_512115H
Dcount_nonzero_notequal_dnn_hiddenlayer_6_batchnorm_6_batchnorm_add_1
cast	
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros?
count_nonzero/NotEqualNotEqualDcount_nonzero_notequal_dnn_hiddenlayer_6_batchnorm_6_batchnorm_add_1count_nonzero/zeros:output:0*
T0*'
_output_shapes
:?????????
2
count_nonzero/NotEqual?
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const?
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast?
OptionalFromValueOptionalFromValuecount_nonzero/zeros:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue?
OptionalFromValue_1OptionalFromValuecount_nonzero/NotEqual:z:0*
Toutput_types
2
*
_output_shapes
: 2
OptionalFromValue_1?
OptionalFromValue_2OptionalFromValuecount_nonzero/Cast:y:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_2?
OptionalFromValue_3OptionalFromValuecount_nonzero/Const:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_3?
OptionalFromValue_4OptionalFromValue$count_nonzero/nonzero_count:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_4"
castCast:y:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0*&
_input_shapes
:?????????
:- )
'
_output_shapes
:?????????

?
?
%dnn_zero_fraction_6_cond_false_512116H
Dcount_nonzero_notequal_dnn_hiddenlayer_6_batchnorm_6_batchnorm_add_1
count_nonzero_nonzero_count	
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalnoneo
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros?
count_nonzero/NotEqualNotEqualDcount_nonzero_notequal_dnn_hiddenlayer_6_batchnorm_6_batchnorm_add_1count_nonzero/zeros:output:0*
T0*'
_output_shapes
:?????????
2
count_nonzero/NotEqual?
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*'
_output_shapes
:?????????
2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const?
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count?
OptionalFromValueOptionalFromValuecount_nonzero/zeros:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue?
OptionalFromValue_1OptionalFromValuecount_nonzero/NotEqual:z:0*
Toutput_types
2
*
_output_shapes
: 2
OptionalFromValue_1?
OptionalFromValue_2OptionalFromValuecount_nonzero/Cast:y:0*
Toutput_types
2	*
_output_shapes
: 2
OptionalFromValue_2?
OptionalFromValue_3OptionalFromValuecount_nonzero/Const:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_3D
OptionalNoneOptionalNone*
_output_shapes
: 2
OptionalNone"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"'
optionalnoneOptionalNone:optional:0*&
_input_shapes
:?????????
:- )
'
_output_shapes
:?????????

?
?
%dnn_zero_fraction_8_cond_false_512312H
Dcount_nonzero_notequal_dnn_hiddenlayer_8_batchnorm_8_batchnorm_add_1
count_nonzero_nonzero_count	
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalnoneo
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros?
count_nonzero/NotEqualNotEqualDcount_nonzero_notequal_dnn_hiddenlayer_8_batchnorm_8_batchnorm_add_1count_nonzero/zeros:output:0*
T0*'
_output_shapes
:?????????
2
count_nonzero/NotEqual?
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*'
_output_shapes
:?????????
2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const?
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count?
OptionalFromValueOptionalFromValuecount_nonzero/zeros:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue?
OptionalFromValue_1OptionalFromValuecount_nonzero/NotEqual:z:0*
Toutput_types
2
*
_output_shapes
: 2
OptionalFromValue_1?
OptionalFromValue_2OptionalFromValuecount_nonzero/Cast:y:0*
Toutput_types
2	*
_output_shapes
: 2
OptionalFromValue_2?
OptionalFromValue_3OptionalFromValuecount_nonzero/Const:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_3D
OptionalNoneOptionalNone*
_output_shapes
: 2
OptionalNone"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"'
optionalnoneOptionalNone:optional:0*&
_input_shapes
:?????????
:- )
'
_output_shapes
:?????????

?
?
$dnn_zero_fraction_1_cond_true_511625H
Dcount_nonzero_notequal_dnn_hiddenlayer_1_batchnorm_1_batchnorm_add_1
cast	
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros?
count_nonzero/NotEqualNotEqualDcount_nonzero_notequal_dnn_hiddenlayer_1_batchnorm_1_batchnorm_add_1count_nonzero/zeros:output:0*
T0*'
_output_shapes
:?????????d2
count_nonzero/NotEqual?
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const?
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast?
OptionalFromValueOptionalFromValuecount_nonzero/zeros:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue?
OptionalFromValue_1OptionalFromValuecount_nonzero/NotEqual:z:0*
Toutput_types
2
*
_output_shapes
: 2
OptionalFromValue_1?
OptionalFromValue_2OptionalFromValuecount_nonzero/Cast:y:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_2?
OptionalFromValue_3OptionalFromValuecount_nonzero/Const:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_3?
OptionalFromValue_4OptionalFromValue$count_nonzero/nonzero_count:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_4"
castCast:y:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0*&
_input_shapes
:?????????d:- )
'
_output_shapes
:?????????d
?
?
%dnn_zero_fraction_3_cond_false_511822H
Dcount_nonzero_notequal_dnn_hiddenlayer_3_batchnorm_3_batchnorm_add_1
count_nonzero_nonzero_count	
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalnoneo
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros?
count_nonzero/NotEqualNotEqualDcount_nonzero_notequal_dnn_hiddenlayer_3_batchnorm_3_batchnorm_add_1count_nonzero/zeros:output:0*
T0*'
_output_shapes
:?????????22
count_nonzero/NotEqual?
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*'
_output_shapes
:?????????22
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const?
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count?
OptionalFromValueOptionalFromValuecount_nonzero/zeros:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue?
OptionalFromValue_1OptionalFromValuecount_nonzero/NotEqual:z:0*
Toutput_types
2
*
_output_shapes
: 2
OptionalFromValue_1?
OptionalFromValue_2OptionalFromValuecount_nonzero/Cast:y:0*
Toutput_types
2	*
_output_shapes
: 2
OptionalFromValue_2?
OptionalFromValue_3OptionalFromValuecount_nonzero/Const:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_3D
OptionalNoneOptionalNone*
_output_shapes
: 2
OptionalNone"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"'
optionalnoneOptionalNone:optional:0*&
_input_shapes
:?????????2:- )
'
_output_shapes
:?????????2
?
?
%dnn_zero_fraction_5_cond_false_512018H
Dcount_nonzero_notequal_dnn_hiddenlayer_5_batchnorm_5_batchnorm_add_1
count_nonzero_nonzero_count	
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalnoneo
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros?
count_nonzero/NotEqualNotEqualDcount_nonzero_notequal_dnn_hiddenlayer_5_batchnorm_5_batchnorm_add_1count_nonzero/zeros:output:0*
T0*'
_output_shapes
:?????????22
count_nonzero/NotEqual?
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*'
_output_shapes
:?????????22
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const?
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count?
OptionalFromValueOptionalFromValuecount_nonzero/zeros:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue?
OptionalFromValue_1OptionalFromValuecount_nonzero/NotEqual:z:0*
Toutput_types
2
*
_output_shapes
: 2
OptionalFromValue_1?
OptionalFromValue_2OptionalFromValuecount_nonzero/Cast:y:0*
Toutput_types
2	*
_output_shapes
: 2
OptionalFromValue_2?
OptionalFromValue_3OptionalFromValuecount_nonzero/Const:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_3D
OptionalNoneOptionalNone*
_output_shapes
: 2
OptionalNone"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"'
optionalnoneOptionalNone:optional:0*&
_input_shapes
:?????????2:- )
'
_output_shapes
:?????????2
?
?
$dnn_zero_fraction_7_cond_true_512213H
Dcount_nonzero_notequal_dnn_hiddenlayer_7_batchnorm_7_batchnorm_add_1
cast	
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros?
count_nonzero/NotEqualNotEqualDcount_nonzero_notequal_dnn_hiddenlayer_7_batchnorm_7_batchnorm_add_1count_nonzero/zeros:output:0*
T0*'
_output_shapes
:?????????
2
count_nonzero/NotEqual?
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const?
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast?
OptionalFromValueOptionalFromValuecount_nonzero/zeros:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue?
OptionalFromValue_1OptionalFromValuecount_nonzero/NotEqual:z:0*
Toutput_types
2
*
_output_shapes
: 2
OptionalFromValue_1?
OptionalFromValue_2OptionalFromValuecount_nonzero/Cast:y:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_2?
OptionalFromValue_3OptionalFromValuecount_nonzero/Const:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_3?
OptionalFromValue_4OptionalFromValue$count_nonzero/nonzero_count:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_4"
castCast:y:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0*&
_input_shapes
:?????????
:- )
'
_output_shapes
:?????????

?
?
#dnn_zero_fraction_cond_false_511528H
Dcount_nonzero_notequal_dnn_hiddenlayer_0_batchnorm_0_batchnorm_add_1
count_nonzero_nonzero_count	
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalnoneo
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros?
count_nonzero/NotEqualNotEqualDcount_nonzero_notequal_dnn_hiddenlayer_0_batchnorm_0_batchnorm_add_1count_nonzero/zeros:output:0*
T0*(
_output_shapes
:??????????2
count_nonzero/NotEqual?
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*(
_output_shapes
:??????????2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const?
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count?
OptionalFromValueOptionalFromValuecount_nonzero/zeros:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue?
OptionalFromValue_1OptionalFromValuecount_nonzero/NotEqual:z:0*
Toutput_types
2
*
_output_shapes
: 2
OptionalFromValue_1?
OptionalFromValue_2OptionalFromValuecount_nonzero/Cast:y:0*
Toutput_types
2	*
_output_shapes
: 2
OptionalFromValue_2?
OptionalFromValue_3OptionalFromValuecount_nonzero/Const:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_3D
OptionalNoneOptionalNone*
_output_shapes
: 2
OptionalNone"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"'
optionalnoneOptionalNone:optional:0*'
_input_shapes
:??????????:. *
(
_output_shapes
:??????????
?
?
$dnn_zero_fraction_3_cond_true_511821H
Dcount_nonzero_notequal_dnn_hiddenlayer_3_batchnorm_3_batchnorm_add_1
cast	
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros?
count_nonzero/NotEqualNotEqualDcount_nonzero_notequal_dnn_hiddenlayer_3_batchnorm_3_batchnorm_add_1count_nonzero/zeros:output:0*
T0*'
_output_shapes
:?????????22
count_nonzero/NotEqual?
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const?
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast?
OptionalFromValueOptionalFromValuecount_nonzero/zeros:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue?
OptionalFromValue_1OptionalFromValuecount_nonzero/NotEqual:z:0*
Toutput_types
2
*
_output_shapes
: 2
OptionalFromValue_1?
OptionalFromValue_2OptionalFromValuecount_nonzero/Cast:y:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_2?
OptionalFromValue_3OptionalFromValuecount_nonzero/Const:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_3?
OptionalFromValue_4OptionalFromValue$count_nonzero/nonzero_count:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_4"
castCast:y:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0*&
_input_shapes
:?????????2:- )
'
_output_shapes
:?????????2
?
?
$dnn_zero_fraction_4_cond_true_511919H
Dcount_nonzero_notequal_dnn_hiddenlayer_4_batchnorm_4_batchnorm_add_1
cast	
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros?
count_nonzero/NotEqualNotEqualDcount_nonzero_notequal_dnn_hiddenlayer_4_batchnorm_4_batchnorm_add_1count_nonzero/zeros:output:0*
T0*'
_output_shapes
:?????????22
count_nonzero/NotEqual?
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const?
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast?
OptionalFromValueOptionalFromValuecount_nonzero/zeros:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue?
OptionalFromValue_1OptionalFromValuecount_nonzero/NotEqual:z:0*
Toutput_types
2
*
_output_shapes
: 2
OptionalFromValue_1?
OptionalFromValue_2OptionalFromValuecount_nonzero/Cast:y:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_2?
OptionalFromValue_3OptionalFromValuecount_nonzero/Const:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_3?
OptionalFromValue_4OptionalFromValue$count_nonzero/nonzero_count:output:0*
Toutput_types
2*
_output_shapes
: 2
OptionalFromValue_4"
castCast:y:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0*&
_input_shapes
:?????????2:- )
'
_output_shapes
:?????????2"?<
save/Const:0save/Identity:0save/restore_all (5 @F8"~
global_stepom
k
global_step:0global_step/Assign!global_step/Read/ReadVariableOp:0(2global_step/Initializer/zeros:0H"%
saved_model_main_op


group_deps"?
	summaries?
?
+dnn/hiddenlayer_0/fraction_of_zero_values:0
dnn/hiddenlayer_0/activation:0
+dnn/hiddenlayer_1/fraction_of_zero_values:0
dnn/hiddenlayer_1/activation:0
+dnn/hiddenlayer_2/fraction_of_zero_values:0
dnn/hiddenlayer_2/activation:0
+dnn/hiddenlayer_3/fraction_of_zero_values:0
dnn/hiddenlayer_3/activation:0
+dnn/hiddenlayer_4/fraction_of_zero_values:0
dnn/hiddenlayer_4/activation:0
+dnn/hiddenlayer_5/fraction_of_zero_values:0
dnn/hiddenlayer_5/activation:0
+dnn/hiddenlayer_6/fraction_of_zero_values:0
dnn/hiddenlayer_6/activation:0
+dnn/hiddenlayer_7/fraction_of_zero_values:0
dnn/hiddenlayer_7/activation:0
+dnn/hiddenlayer_8/fraction_of_zero_values:0
dnn/hiddenlayer_8/activation:0
$dnn/logits/fraction_of_zero_values:0
dnn/logits/activation:0"?
table_initializer?
?
?dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding_1/browser_major_version_na_lookup/hash_table/table_init/LookupTableImportV2
?dnn/input_from_feature_columns/input_layer/browser_name_embedding_1/browser_name_lookup/hash_table/table_init/LookupTableImportV2
?dnn/input_from_feature_columns/input_layer/country_code_embedding_1/country_code_lookup/hash_table/table_init/LookupTableImportV2
ydnn/input_from_feature_columns/input_layer/osfamily_embedding_1/osfamily_lookup/hash_table/table_init/LookupTableImportV2
}dnn/input_from_feature_columns/input_layer/osmajor_na_embedding_1/osmajor_na_lookup/hash_table/table_init/LookupTableImportV2"?I
trainable_variables?H?H
?
Sdnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights:0Xdnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights/Assigngdnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights/Read/ReadVariableOp:0(2pdnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights/Initializer/truncated_normal:08
?
adnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights:0fdnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights/Assignudnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights/Read/ReadVariableOp:0(2~dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights/Initializer/truncated_normal:08
?
Udnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights:0Zdnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights/Assignidnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights/Read/ReadVariableOp:0(2rdnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights/Initializer/truncated_normal:08
?
Udnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights:0Zdnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights/Assignidnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights/Read/ReadVariableOp:0(2rdnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights/Initializer/truncated_normal:08
?
Qdnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights:0Vdnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights/Assignednn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights/Read/ReadVariableOp:0(2ndnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights/Initializer/truncated_normal:08
?
Sdnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights:0Xdnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights/Assigngdnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights/Read/ReadVariableOp:0(2pdnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights/Initializer/truncated_normal:08
?
dnn/hiddenlayer_0/kernel:0dnn/hiddenlayer_0/kernel/Assign.dnn/hiddenlayer_0/kernel/Read/ReadVariableOp:0(25dnn/hiddenlayer_0/kernel/Initializer/random_uniform:08
?
dnn/hiddenlayer_0/bias:0dnn/hiddenlayer_0/bias/Assign,dnn/hiddenlayer_0/bias/Read/ReadVariableOp:0(2*dnn/hiddenlayer_0/bias/Initializer/zeros:08
?
%dnn/hiddenlayer_0/batchnorm_0/gamma:0*dnn/hiddenlayer_0/batchnorm_0/gamma/Assign9dnn/hiddenlayer_0/batchnorm_0/gamma/Read/ReadVariableOp:0(26dnn/hiddenlayer_0/batchnorm_0/gamma/Initializer/ones:08
?
$dnn/hiddenlayer_0/batchnorm_0/beta:0)dnn/hiddenlayer_0/batchnorm_0/beta/Assign8dnn/hiddenlayer_0/batchnorm_0/beta/Read/ReadVariableOp:0(26dnn/hiddenlayer_0/batchnorm_0/beta/Initializer/zeros:08
?
dnn/hiddenlayer_1/kernel:0dnn/hiddenlayer_1/kernel/Assign.dnn/hiddenlayer_1/kernel/Read/ReadVariableOp:0(25dnn/hiddenlayer_1/kernel/Initializer/random_uniform:08
?
dnn/hiddenlayer_1/bias:0dnn/hiddenlayer_1/bias/Assign,dnn/hiddenlayer_1/bias/Read/ReadVariableOp:0(2*dnn/hiddenlayer_1/bias/Initializer/zeros:08
?
%dnn/hiddenlayer_1/batchnorm_1/gamma:0*dnn/hiddenlayer_1/batchnorm_1/gamma/Assign9dnn/hiddenlayer_1/batchnorm_1/gamma/Read/ReadVariableOp:0(26dnn/hiddenlayer_1/batchnorm_1/gamma/Initializer/ones:08
?
$dnn/hiddenlayer_1/batchnorm_1/beta:0)dnn/hiddenlayer_1/batchnorm_1/beta/Assign8dnn/hiddenlayer_1/batchnorm_1/beta/Read/ReadVariableOp:0(26dnn/hiddenlayer_1/batchnorm_1/beta/Initializer/zeros:08
?
dnn/hiddenlayer_2/kernel:0dnn/hiddenlayer_2/kernel/Assign.dnn/hiddenlayer_2/kernel/Read/ReadVariableOp:0(25dnn/hiddenlayer_2/kernel/Initializer/random_uniform:08
?
dnn/hiddenlayer_2/bias:0dnn/hiddenlayer_2/bias/Assign,dnn/hiddenlayer_2/bias/Read/ReadVariableOp:0(2*dnn/hiddenlayer_2/bias/Initializer/zeros:08
?
%dnn/hiddenlayer_2/batchnorm_2/gamma:0*dnn/hiddenlayer_2/batchnorm_2/gamma/Assign9dnn/hiddenlayer_2/batchnorm_2/gamma/Read/ReadVariableOp:0(26dnn/hiddenlayer_2/batchnorm_2/gamma/Initializer/ones:08
?
$dnn/hiddenlayer_2/batchnorm_2/beta:0)dnn/hiddenlayer_2/batchnorm_2/beta/Assign8dnn/hiddenlayer_2/batchnorm_2/beta/Read/ReadVariableOp:0(26dnn/hiddenlayer_2/batchnorm_2/beta/Initializer/zeros:08
?
dnn/hiddenlayer_3/kernel:0dnn/hiddenlayer_3/kernel/Assign.dnn/hiddenlayer_3/kernel/Read/ReadVariableOp:0(25dnn/hiddenlayer_3/kernel/Initializer/random_uniform:08
?
dnn/hiddenlayer_3/bias:0dnn/hiddenlayer_3/bias/Assign,dnn/hiddenlayer_3/bias/Read/ReadVariableOp:0(2*dnn/hiddenlayer_3/bias/Initializer/zeros:08
?
%dnn/hiddenlayer_3/batchnorm_3/gamma:0*dnn/hiddenlayer_3/batchnorm_3/gamma/Assign9dnn/hiddenlayer_3/batchnorm_3/gamma/Read/ReadVariableOp:0(26dnn/hiddenlayer_3/batchnorm_3/gamma/Initializer/ones:08
?
$dnn/hiddenlayer_3/batchnorm_3/beta:0)dnn/hiddenlayer_3/batchnorm_3/beta/Assign8dnn/hiddenlayer_3/batchnorm_3/beta/Read/ReadVariableOp:0(26dnn/hiddenlayer_3/batchnorm_3/beta/Initializer/zeros:08
?
dnn/hiddenlayer_4/kernel:0dnn/hiddenlayer_4/kernel/Assign.dnn/hiddenlayer_4/kernel/Read/ReadVariableOp:0(25dnn/hiddenlayer_4/kernel/Initializer/random_uniform:08
?
dnn/hiddenlayer_4/bias:0dnn/hiddenlayer_4/bias/Assign,dnn/hiddenlayer_4/bias/Read/ReadVariableOp:0(2*dnn/hiddenlayer_4/bias/Initializer/zeros:08
?
%dnn/hiddenlayer_4/batchnorm_4/gamma:0*dnn/hiddenlayer_4/batchnorm_4/gamma/Assign9dnn/hiddenlayer_4/batchnorm_4/gamma/Read/ReadVariableOp:0(26dnn/hiddenlayer_4/batchnorm_4/gamma/Initializer/ones:08
?
$dnn/hiddenlayer_4/batchnorm_4/beta:0)dnn/hiddenlayer_4/batchnorm_4/beta/Assign8dnn/hiddenlayer_4/batchnorm_4/beta/Read/ReadVariableOp:0(26dnn/hiddenlayer_4/batchnorm_4/beta/Initializer/zeros:08
?
dnn/hiddenlayer_5/kernel:0dnn/hiddenlayer_5/kernel/Assign.dnn/hiddenlayer_5/kernel/Read/ReadVariableOp:0(25dnn/hiddenlayer_5/kernel/Initializer/random_uniform:08
?
dnn/hiddenlayer_5/bias:0dnn/hiddenlayer_5/bias/Assign,dnn/hiddenlayer_5/bias/Read/ReadVariableOp:0(2*dnn/hiddenlayer_5/bias/Initializer/zeros:08
?
%dnn/hiddenlayer_5/batchnorm_5/gamma:0*dnn/hiddenlayer_5/batchnorm_5/gamma/Assign9dnn/hiddenlayer_5/batchnorm_5/gamma/Read/ReadVariableOp:0(26dnn/hiddenlayer_5/batchnorm_5/gamma/Initializer/ones:08
?
$dnn/hiddenlayer_5/batchnorm_5/beta:0)dnn/hiddenlayer_5/batchnorm_5/beta/Assign8dnn/hiddenlayer_5/batchnorm_5/beta/Read/ReadVariableOp:0(26dnn/hiddenlayer_5/batchnorm_5/beta/Initializer/zeros:08
?
dnn/hiddenlayer_6/kernel:0dnn/hiddenlayer_6/kernel/Assign.dnn/hiddenlayer_6/kernel/Read/ReadVariableOp:0(25dnn/hiddenlayer_6/kernel/Initializer/random_uniform:08
?
dnn/hiddenlayer_6/bias:0dnn/hiddenlayer_6/bias/Assign,dnn/hiddenlayer_6/bias/Read/ReadVariableOp:0(2*dnn/hiddenlayer_6/bias/Initializer/zeros:08
?
%dnn/hiddenlayer_6/batchnorm_6/gamma:0*dnn/hiddenlayer_6/batchnorm_6/gamma/Assign9dnn/hiddenlayer_6/batchnorm_6/gamma/Read/ReadVariableOp:0(26dnn/hiddenlayer_6/batchnorm_6/gamma/Initializer/ones:08
?
$dnn/hiddenlayer_6/batchnorm_6/beta:0)dnn/hiddenlayer_6/batchnorm_6/beta/Assign8dnn/hiddenlayer_6/batchnorm_6/beta/Read/ReadVariableOp:0(26dnn/hiddenlayer_6/batchnorm_6/beta/Initializer/zeros:08
?
dnn/hiddenlayer_7/kernel:0dnn/hiddenlayer_7/kernel/Assign.dnn/hiddenlayer_7/kernel/Read/ReadVariableOp:0(25dnn/hiddenlayer_7/kernel/Initializer/random_uniform:08
?
dnn/hiddenlayer_7/bias:0dnn/hiddenlayer_7/bias/Assign,dnn/hiddenlayer_7/bias/Read/ReadVariableOp:0(2*dnn/hiddenlayer_7/bias/Initializer/zeros:08
?
%dnn/hiddenlayer_7/batchnorm_7/gamma:0*dnn/hiddenlayer_7/batchnorm_7/gamma/Assign9dnn/hiddenlayer_7/batchnorm_7/gamma/Read/ReadVariableOp:0(26dnn/hiddenlayer_7/batchnorm_7/gamma/Initializer/ones:08
?
$dnn/hiddenlayer_7/batchnorm_7/beta:0)dnn/hiddenlayer_7/batchnorm_7/beta/Assign8dnn/hiddenlayer_7/batchnorm_7/beta/Read/ReadVariableOp:0(26dnn/hiddenlayer_7/batchnorm_7/beta/Initializer/zeros:08
?
dnn/hiddenlayer_8/kernel:0dnn/hiddenlayer_8/kernel/Assign.dnn/hiddenlayer_8/kernel/Read/ReadVariableOp:0(25dnn/hiddenlayer_8/kernel/Initializer/random_uniform:08
?
dnn/hiddenlayer_8/bias:0dnn/hiddenlayer_8/bias/Assign,dnn/hiddenlayer_8/bias/Read/ReadVariableOp:0(2*dnn/hiddenlayer_8/bias/Initializer/zeros:08
?
%dnn/hiddenlayer_8/batchnorm_8/gamma:0*dnn/hiddenlayer_8/batchnorm_8/gamma/Assign9dnn/hiddenlayer_8/batchnorm_8/gamma/Read/ReadVariableOp:0(26dnn/hiddenlayer_8/batchnorm_8/gamma/Initializer/ones:08
?
$dnn/hiddenlayer_8/batchnorm_8/beta:0)dnn/hiddenlayer_8/batchnorm_8/beta/Assign8dnn/hiddenlayer_8/batchnorm_8/beta/Read/ReadVariableOp:0(26dnn/hiddenlayer_8/batchnorm_8/beta/Initializer/zeros:08
?
dnn/logits/kernel:0dnn/logits/kernel/Assign'dnn/logits/kernel/Read/ReadVariableOp:0(2.dnn/logits/kernel/Initializer/random_uniform:08
{
dnn/logits/bias:0dnn/logits/bias/Assign%dnn/logits/bias/Read/ReadVariableOp:0(2#dnn/logits/bias/Initializer/zeros:08"?k
	variables?k?k
k
global_step:0global_step/Assign!global_step/Read/ReadVariableOp:0(2global_step/Initializer/zeros:0H
?
Sdnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights:0Xdnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights/Assigngdnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights/Read/ReadVariableOp:0(2pdnn/input_from_feature_columns/input_layer/asn_number_embedding/embedding_weights/Initializer/truncated_normal:08
?
adnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights:0fdnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights/Assignudnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights/Read/ReadVariableOp:0(2~dnn/input_from_feature_columns/input_layer/browser_major_version_na_embedding/embedding_weights/Initializer/truncated_normal:08
?
Udnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights:0Zdnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights/Assignidnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights/Read/ReadVariableOp:0(2rdnn/input_from_feature_columns/input_layer/browser_name_embedding/embedding_weights/Initializer/truncated_normal:08
?
Udnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights:0Zdnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights/Assignidnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights/Read/ReadVariableOp:0(2rdnn/input_from_feature_columns/input_layer/country_code_embedding/embedding_weights/Initializer/truncated_normal:08
?
Qdnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights:0Vdnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights/Assignednn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights/Read/ReadVariableOp:0(2ndnn/input_from_feature_columns/input_layer/osfamily_embedding/embedding_weights/Initializer/truncated_normal:08
?
Sdnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights:0Xdnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights/Assigngdnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights/Read/ReadVariableOp:0(2pdnn/input_from_feature_columns/input_layer/osmajor_na_embedding/embedding_weights/Initializer/truncated_normal:08
?
dnn/hiddenlayer_0/kernel:0dnn/hiddenlayer_0/kernel/Assign.dnn/hiddenlayer_0/kernel/Read/ReadVariableOp:0(25dnn/hiddenlayer_0/kernel/Initializer/random_uniform:08
?
dnn/hiddenlayer_0/bias:0dnn/hiddenlayer_0/bias/Assign,dnn/hiddenlayer_0/bias/Read/ReadVariableOp:0(2*dnn/hiddenlayer_0/bias/Initializer/zeros:08
?
%dnn/hiddenlayer_0/batchnorm_0/gamma:0*dnn/hiddenlayer_0/batchnorm_0/gamma/Assign9dnn/hiddenlayer_0/batchnorm_0/gamma/Read/ReadVariableOp:0(26dnn/hiddenlayer_0/batchnorm_0/gamma/Initializer/ones:08
?
$dnn/hiddenlayer_0/batchnorm_0/beta:0)dnn/hiddenlayer_0/batchnorm_0/beta/Assign8dnn/hiddenlayer_0/batchnorm_0/beta/Read/ReadVariableOp:0(26dnn/hiddenlayer_0/batchnorm_0/beta/Initializer/zeros:08
?
+dnn/hiddenlayer_0/batchnorm_0/moving_mean:00dnn/hiddenlayer_0/batchnorm_0/moving_mean/Assign?dnn/hiddenlayer_0/batchnorm_0/moving_mean/Read/ReadVariableOp:0(2=dnn/hiddenlayer_0/batchnorm_0/moving_mean/Initializer/zeros:0@H
?
/dnn/hiddenlayer_0/batchnorm_0/moving_variance:04dnn/hiddenlayer_0/batchnorm_0/moving_variance/AssignCdnn/hiddenlayer_0/batchnorm_0/moving_variance/Read/ReadVariableOp:0(2@dnn/hiddenlayer_0/batchnorm_0/moving_variance/Initializer/ones:0@H
?
dnn/hiddenlayer_1/kernel:0dnn/hiddenlayer_1/kernel/Assign.dnn/hiddenlayer_1/kernel/Read/ReadVariableOp:0(25dnn/hiddenlayer_1/kernel/Initializer/random_uniform:08
?
dnn/hiddenlayer_1/bias:0dnn/hiddenlayer_1/bias/Assign,dnn/hiddenlayer_1/bias/Read/ReadVariableOp:0(2*dnn/hiddenlayer_1/bias/Initializer/zeros:08
?
%dnn/hiddenlayer_1/batchnorm_1/gamma:0*dnn/hiddenlayer_1/batchnorm_1/gamma/Assign9dnn/hiddenlayer_1/batchnorm_1/gamma/Read/ReadVariableOp:0(26dnn/hiddenlayer_1/batchnorm_1/gamma/Initializer/ones:08
?
$dnn/hiddenlayer_1/batchnorm_1/beta:0)dnn/hiddenlayer_1/batchnorm_1/beta/Assign8dnn/hiddenlayer_1/batchnorm_1/beta/Read/ReadVariableOp:0(26dnn/hiddenlayer_1/batchnorm_1/beta/Initializer/zeros:08
?
+dnn/hiddenlayer_1/batchnorm_1/moving_mean:00dnn/hiddenlayer_1/batchnorm_1/moving_mean/Assign?dnn/hiddenlayer_1/batchnorm_1/moving_mean/Read/ReadVariableOp:0(2=dnn/hiddenlayer_1/batchnorm_1/moving_mean/Initializer/zeros:0@H
?
/dnn/hiddenlayer_1/batchnorm_1/moving_variance:04dnn/hiddenlayer_1/batchnorm_1/moving_variance/AssignCdnn/hiddenlayer_1/batchnorm_1/moving_variance/Read/ReadVariableOp:0(2@dnn/hiddenlayer_1/batchnorm_1/moving_variance/Initializer/ones:0@H
?
dnn/hiddenlayer_2/kernel:0dnn/hiddenlayer_2/kernel/Assign.dnn/hiddenlayer_2/kernel/Read/ReadVariableOp:0(25dnn/hiddenlayer_2/kernel/Initializer/random_uniform:08
?
dnn/hiddenlayer_2/bias:0dnn/hiddenlayer_2/bias/Assign,dnn/hiddenlayer_2/bias/Read/ReadVariableOp:0(2*dnn/hiddenlayer_2/bias/Initializer/zeros:08
?
%dnn/hiddenlayer_2/batchnorm_2/gamma:0*dnn/hiddenlayer_2/batchnorm_2/gamma/Assign9dnn/hiddenlayer_2/batchnorm_2/gamma/Read/ReadVariableOp:0(26dnn/hiddenlayer_2/batchnorm_2/gamma/Initializer/ones:08
?
$dnn/hiddenlayer_2/batchnorm_2/beta:0)dnn/hiddenlayer_2/batchnorm_2/beta/Assign8dnn/hiddenlayer_2/batchnorm_2/beta/Read/ReadVariableOp:0(26dnn/hiddenlayer_2/batchnorm_2/beta/Initializer/zeros:08
?
+dnn/hiddenlayer_2/batchnorm_2/moving_mean:00dnn/hiddenlayer_2/batchnorm_2/moving_mean/Assign?dnn/hiddenlayer_2/batchnorm_2/moving_mean/Read/ReadVariableOp:0(2=dnn/hiddenlayer_2/batchnorm_2/moving_mean/Initializer/zeros:0@H
?
/dnn/hiddenlayer_2/batchnorm_2/moving_variance:04dnn/hiddenlayer_2/batchnorm_2/moving_variance/AssignCdnn/hiddenlayer_2/batchnorm_2/moving_variance/Read/ReadVariableOp:0(2@dnn/hiddenlayer_2/batchnorm_2/moving_variance/Initializer/ones:0@H
?
dnn/hiddenlayer_3/kernel:0dnn/hiddenlayer_3/kernel/Assign.dnn/hiddenlayer_3/kernel/Read/ReadVariableOp:0(25dnn/hiddenlayer_3/kernel/Initializer/random_uniform:08
?
dnn/hiddenlayer_3/bias:0dnn/hiddenlayer_3/bias/Assign,dnn/hiddenlayer_3/bias/Read/ReadVariableOp:0(2*dnn/hiddenlayer_3/bias/Initializer/zeros:08
?
%dnn/hiddenlayer_3/batchnorm_3/gamma:0*dnn/hiddenlayer_3/batchnorm_3/gamma/Assign9dnn/hiddenlayer_3/batchnorm_3/gamma/Read/ReadVariableOp:0(26dnn/hiddenlayer_3/batchnorm_3/gamma/Initializer/ones:08
?
$dnn/hiddenlayer_3/batchnorm_3/beta:0)dnn/hiddenlayer_3/batchnorm_3/beta/Assign8dnn/hiddenlayer_3/batchnorm_3/beta/Read/ReadVariableOp:0(26dnn/hiddenlayer_3/batchnorm_3/beta/Initializer/zeros:08
?
+dnn/hiddenlayer_3/batchnorm_3/moving_mean:00dnn/hiddenlayer_3/batchnorm_3/moving_mean/Assign?dnn/hiddenlayer_3/batchnorm_3/moving_mean/Read/ReadVariableOp:0(2=dnn/hiddenlayer_3/batchnorm_3/moving_mean/Initializer/zeros:0@H
?
/dnn/hiddenlayer_3/batchnorm_3/moving_variance:04dnn/hiddenlayer_3/batchnorm_3/moving_variance/AssignCdnn/hiddenlayer_3/batchnorm_3/moving_variance/Read/ReadVariableOp:0(2@dnn/hiddenlayer_3/batchnorm_3/moving_variance/Initializer/ones:0@H
?
dnn/hiddenlayer_4/kernel:0dnn/hiddenlayer_4/kernel/Assign.dnn/hiddenlayer_4/kernel/Read/ReadVariableOp:0(25dnn/hiddenlayer_4/kernel/Initializer/random_uniform:08
?
dnn/hiddenlayer_4/bias:0dnn/hiddenlayer_4/bias/Assign,dnn/hiddenlayer_4/bias/Read/ReadVariableOp:0(2*dnn/hiddenlayer_4/bias/Initializer/zeros:08
?
%dnn/hiddenlayer_4/batchnorm_4/gamma:0*dnn/hiddenlayer_4/batchnorm_4/gamma/Assign9dnn/hiddenlayer_4/batchnorm_4/gamma/Read/ReadVariableOp:0(26dnn/hiddenlayer_4/batchnorm_4/gamma/Initializer/ones:08
?
$dnn/hiddenlayer_4/batchnorm_4/beta:0)dnn/hiddenlayer_4/batchnorm_4/beta/Assign8dnn/hiddenlayer_4/batchnorm_4/beta/Read/ReadVariableOp:0(26dnn/hiddenlayer_4/batchnorm_4/beta/Initializer/zeros:08
?
+dnn/hiddenlayer_4/batchnorm_4/moving_mean:00dnn/hiddenlayer_4/batchnorm_4/moving_mean/Assign?dnn/hiddenlayer_4/batchnorm_4/moving_mean/Read/ReadVariableOp:0(2=dnn/hiddenlayer_4/batchnorm_4/moving_mean/Initializer/zeros:0@H
?
/dnn/hiddenlayer_4/batchnorm_4/moving_variance:04dnn/hiddenlayer_4/batchnorm_4/moving_variance/AssignCdnn/hiddenlayer_4/batchnorm_4/moving_variance/Read/ReadVariableOp:0(2@dnn/hiddenlayer_4/batchnorm_4/moving_variance/Initializer/ones:0@H
?
dnn/hiddenlayer_5/kernel:0dnn/hiddenlayer_5/kernel/Assign.dnn/hiddenlayer_5/kernel/Read/ReadVariableOp:0(25dnn/hiddenlayer_5/kernel/Initializer/random_uniform:08
?
dnn/hiddenlayer_5/bias:0dnn/hiddenlayer_5/bias/Assign,dnn/hiddenlayer_5/bias/Read/ReadVariableOp:0(2*dnn/hiddenlayer_5/bias/Initializer/zeros:08
?
%dnn/hiddenlayer_5/batchnorm_5/gamma:0*dnn/hiddenlayer_5/batchnorm_5/gamma/Assign9dnn/hiddenlayer_5/batchnorm_5/gamma/Read/ReadVariableOp:0(26dnn/hiddenlayer_5/batchnorm_5/gamma/Initializer/ones:08
?
$dnn/hiddenlayer_5/batchnorm_5/beta:0)dnn/hiddenlayer_5/batchnorm_5/beta/Assign8dnn/hiddenlayer_5/batchnorm_5/beta/Read/ReadVariableOp:0(26dnn/hiddenlayer_5/batchnorm_5/beta/Initializer/zeros:08
?
+dnn/hiddenlayer_5/batchnorm_5/moving_mean:00dnn/hiddenlayer_5/batchnorm_5/moving_mean/Assign?dnn/hiddenlayer_5/batchnorm_5/moving_mean/Read/ReadVariableOp:0(2=dnn/hiddenlayer_5/batchnorm_5/moving_mean/Initializer/zeros:0@H
?
/dnn/hiddenlayer_5/batchnorm_5/moving_variance:04dnn/hiddenlayer_5/batchnorm_5/moving_variance/AssignCdnn/hiddenlayer_5/batchnorm_5/moving_variance/Read/ReadVariableOp:0(2@dnn/hiddenlayer_5/batchnorm_5/moving_variance/Initializer/ones:0@H
?
dnn/hiddenlayer_6/kernel:0dnn/hiddenlayer_6/kernel/Assign.dnn/hiddenlayer_6/kernel/Read/ReadVariableOp:0(25dnn/hiddenlayer_6/kernel/Initializer/random_uniform:08
?
dnn/hiddenlayer_6/bias:0dnn/hiddenlayer_6/bias/Assign,dnn/hiddenlayer_6/bias/Read/ReadVariableOp:0(2*dnn/hiddenlayer_6/bias/Initializer/zeros:08
?
%dnn/hiddenlayer_6/batchnorm_6/gamma:0*dnn/hiddenlayer_6/batchnorm_6/gamma/Assign9dnn/hiddenlayer_6/batchnorm_6/gamma/Read/ReadVariableOp:0(26dnn/hiddenlayer_6/batchnorm_6/gamma/Initializer/ones:08
?
$dnn/hiddenlayer_6/batchnorm_6/beta:0)dnn/hiddenlayer_6/batchnorm_6/beta/Assign8dnn/hiddenlayer_6/batchnorm_6/beta/Read/ReadVariableOp:0(26dnn/hiddenlayer_6/batchnorm_6/beta/Initializer/zeros:08
?
+dnn/hiddenlayer_6/batchnorm_6/moving_mean:00dnn/hiddenlayer_6/batchnorm_6/moving_mean/Assign?dnn/hiddenlayer_6/batchnorm_6/moving_mean/Read/ReadVariableOp:0(2=dnn/hiddenlayer_6/batchnorm_6/moving_mean/Initializer/zeros:0@H
?
/dnn/hiddenlayer_6/batchnorm_6/moving_variance:04dnn/hiddenlayer_6/batchnorm_6/moving_variance/AssignCdnn/hiddenlayer_6/batchnorm_6/moving_variance/Read/ReadVariableOp:0(2@dnn/hiddenlayer_6/batchnorm_6/moving_variance/Initializer/ones:0@H
?
dnn/hiddenlayer_7/kernel:0dnn/hiddenlayer_7/kernel/Assign.dnn/hiddenlayer_7/kernel/Read/ReadVariableOp:0(25dnn/hiddenlayer_7/kernel/Initializer/random_uniform:08
?
dnn/hiddenlayer_7/bias:0dnn/hiddenlayer_7/bias/Assign,dnn/hiddenlayer_7/bias/Read/ReadVariableOp:0(2*dnn/hiddenlayer_7/bias/Initializer/zeros:08
?
%dnn/hiddenlayer_7/batchnorm_7/gamma:0*dnn/hiddenlayer_7/batchnorm_7/gamma/Assign9dnn/hiddenlayer_7/batchnorm_7/gamma/Read/ReadVariableOp:0(26dnn/hiddenlayer_7/batchnorm_7/gamma/Initializer/ones:08
?
$dnn/hiddenlayer_7/batchnorm_7/beta:0)dnn/hiddenlayer_7/batchnorm_7/beta/Assign8dnn/hiddenlayer_7/batchnorm_7/beta/Read/ReadVariableOp:0(26dnn/hiddenlayer_7/batchnorm_7/beta/Initializer/zeros:08
?
+dnn/hiddenlayer_7/batchnorm_7/moving_mean:00dnn/hiddenlayer_7/batchnorm_7/moving_mean/Assign?dnn/hiddenlayer_7/batchnorm_7/moving_mean/Read/ReadVariableOp:0(2=dnn/hiddenlayer_7/batchnorm_7/moving_mean/Initializer/zeros:0@H
?
/dnn/hiddenlayer_7/batchnorm_7/moving_variance:04dnn/hiddenlayer_7/batchnorm_7/moving_variance/AssignCdnn/hiddenlayer_7/batchnorm_7/moving_variance/Read/ReadVariableOp:0(2@dnn/hiddenlayer_7/batchnorm_7/moving_variance/Initializer/ones:0@H
?
dnn/hiddenlayer_8/kernel:0dnn/hiddenlayer_8/kernel/Assign.dnn/hiddenlayer_8/kernel/Read/ReadVariableOp:0(25dnn/hiddenlayer_8/kernel/Initializer/random_uniform:08
?
dnn/hiddenlayer_8/bias:0dnn/hiddenlayer_8/bias/Assign,dnn/hiddenlayer_8/bias/Read/ReadVariableOp:0(2*dnn/hiddenlayer_8/bias/Initializer/zeros:08
?
%dnn/hiddenlayer_8/batchnorm_8/gamma:0*dnn/hiddenlayer_8/batchnorm_8/gamma/Assign9dnn/hiddenlayer_8/batchnorm_8/gamma/Read/ReadVariableOp:0(26dnn/hiddenlayer_8/batchnorm_8/gamma/Initializer/ones:08
?
$dnn/hiddenlayer_8/batchnorm_8/beta:0)dnn/hiddenlayer_8/batchnorm_8/beta/Assign8dnn/hiddenlayer_8/batchnorm_8/beta/Read/ReadVariableOp:0(26dnn/hiddenlayer_8/batchnorm_8/beta/Initializer/zeros:08
?
+dnn/hiddenlayer_8/batchnorm_8/moving_mean:00dnn/hiddenlayer_8/batchnorm_8/moving_mean/Assign?dnn/hiddenlayer_8/batchnorm_8/moving_mean/Read/ReadVariableOp:0(2=dnn/hiddenlayer_8/batchnorm_8/moving_mean/Initializer/zeros:0@H
?
/dnn/hiddenlayer_8/batchnorm_8/moving_variance:04dnn/hiddenlayer_8/batchnorm_8/moving_variance/AssignCdnn/hiddenlayer_8/batchnorm_8/moving_variance/Read/ReadVariableOp:0(2@dnn/hiddenlayer_8/batchnorm_8/moving_variance/Initializer/ones:0@H
?
dnn/logits/kernel:0dnn/logits/kernel/Assign'dnn/logits/kernel/Read/ReadVariableOp:0(2.dnn/logits/kernel/Initializer/random_uniform:08
{
dnn/logits/bias:0dnn/logits/bias/Assign%dnn/logits/bias/Read/ReadVariableOp:0(2#dnn/logits/bias/Initializer/zeros:08*?
classification?
3
inputs)
input_example_tensor:0?????????-
classes"
head/Tile:0?????????A
scores7
 head/predictions/probabilities:0?????????tensorflow/serving/classify*?
predict?
5
examples)
input_example_tensor:0??????????
all_class_ids.
head/predictions/Tile:0??????????
all_classes0
head/predictions/Tile_1:0?????????A
	class_ids4
head/predictions/ExpandDims:0	?????????@
classes5
head/predictions/str_classes:0?????????>
logistic2
head/predictions/logistic:0?????????5
logits+
dnn/logits/BiasAdd:0?????????H
probabilities7
 head/predictions/probabilities:0?????????tensorflow/serving/predict*?

regression?
3
inputs)
input_example_tensor:0?????????=
outputs2
head/predictions/logistic:0?????????tensorflow/serving/regress*?
serving_default?
3
inputs)
input_example_tensor:0?????????-
classes"
head/Tile:0?????????A
scores7
 head/predictions/probabilities:0?????????tensorflow/serving/classify