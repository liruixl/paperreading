`[TOC]`

###  tf.Variable() shortcoming

运行图时，会有两组变量被创建。。看图：

![1532006018740](assets/1532006018740.png)
![1532006049582](assets/1532006049582.png)

### Global step

```python
global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
```

Very common in TensorFlow program.

### tf.train.Saver()

#### 保存特定参数

![1532006410140](assets/1532006410140.png)

![1532006424558](assets/1532006424558.png)

#### 检查checkpoint

![1532006463863](assets/1532006463863.png)

### tf.summary

#### 步骤

Visualize our summary statistics during our training

+ tf.summary.scalar
+ tf.summary.histogram
+ tf.summary.image

三步走：1、creat summarie and op 2、run it 3、write summaries to file

![1532006628378](assets/1532006628378.png)

### Tensorboard

#### 曲线对比

If you save your summaries into different sub-folder in your graph folder, you can compare your progresses. For example, the first time we run our model with learning rate 1.0, we save it in ‘improved_graph/lr1.0’ and the second time we run our model, we save it in ‘improved_graph/lr0.5：

![1532006814902](assets/1532006814902.png)

### TensorFlow Padding Options？

#### Valid：

![1532006941352](assets/1532006941352.png)可以想象，先不考虑第一个F×F的卷积（+1），（也可以倒着考虑，但由于可能要删除最后及列，故还是正方向考虑）或者说以卷积核右边为对象，它会走直到N×N的右边，走过的长度为N-F，步长是S，所以卷积出来的尺寸为(N - F) / stride，取下整。可能会droped右面，下面的像素。

或者new_height = new_width = (N– F + 1) / S，结果向上取整，从几何意义上理解不了这个公式。

#### Same(with zero padding)

![1532007128380](assets/1532007128380.png)

Same的意思可以理解为，~~卷积出来的尺寸与原图像尺寸大小相等~~。原来不是，上面一句话仅仅适用于步长为1的情况。实际情况看上图。“SAME”尝试向左和向右均匀填充，但如果要添加的列数为奇数，则会向右添加额外的列，如本示例中的情况（只展示水平方向）。

+ stride=1，核大小F是奇数：![1532007247735](assets/1532007247735.png)

  积核尺寸是奇数，那么只需让卷积核的中间与图像左上角对齐，卷积直到中间与图像的右上角对齐，那么卷积核相对于图像多出来的行数即为padding的像素值。卷积后大小：（F-1）/2。

+ stride=S：

  stride=1卷积出来的尺寸与原图大小一致。

  当stride=S时，卷积（无论卷积核多大）出来的尺寸为**N / S （结果向上取整）**，这个可以从卷积核的**左边**考虑，卷积直到无法移动。这里Padding的数量要从结果往回推导。

  参考上图，现在已知卷积核大小F×F（为了简化问题，这里选长宽一样的），步长为S，卷积后的尺寸大小为N / S （结果向上取整）。补0后的问题就是Valid卷积方式而且是能整除的方式哦。

  考虑宽度：Valid的公式为(N - F) / stride + 1= N/S=new_weight，所以补0后的宽度为**(new_weight – 1) × S + F**。这里可以沿着valid公式的几何思路反过来考虑。

  左边：Pddding_left = padding_need/2  右边：Pddding = padding_need-padding_left。

### Visualizing convnet features

不知道。

![1532007599501](assets/1532007599501.png)

### tf.transpose()

```python
 net = tf.transpose(inputs, perm=(0, 2, 3, 1))  # 转换维度，(0,1,2,3)=>(0,2,1,3))
```

### list相加

```python
import numpy as np
a = [5,10,10,3]   # shape(4,)
b =[6,4]  # shape(2,)

print(a+b)  # [5, 10, 10, 3, 6, 4]
print(np.add(a, b)) # ValueError: operands could not be broadcast together with shapes (4,) (2,) 
```

### tf.concat()

```python
t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
tf.concat([t1, t2], axis=0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
tf.concat([t1, t2], axis=1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

# tensor t3 with shape [2, 3]
# tensor t4 with shape [2, 3]
tf.shape(tf.concat([t3, t4], 0))  # [4, 3]
tf.shape(tf.concat([t3, t4], 1))  # [2, 6]
```

就是在某一维度上连接起来。。

怎么想象两个矩阵在某一维度上的连接呢？

多维度矩阵，可以想象成：从0维展开，又会得到len(Numpy[0])个多维矩阵。

```python
import numpy as np
t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
print(t1[0])  # [1, 2, 3]
print(t1[1])  # [4, 5, 6]
```

axis=0的链接即：t1[0]，t1[1]...，+  t2[0]，t2[1]...也就是：t1+t2

axis=1的连接即：a\[0]+b\[0]，a[1]+b[1]

下面看一下：

```python
t1 = [[[1, 2], [2, 3]], [[4, 4], [5, 3]]]  # shape:(2,2,2)
t2 = [[[7, 4], [8, 4]], [[2, 10], [15, 11]]] # shape:(2,2,2)
tf.concat([t1, t2], -1)  # 倒数第一维，-1+rank = -1+3 = 2
>>>
[[[ 1,  2,  7,  4],
  [ 2,  3,  8,  4]],

 [[ 4,  4,  2, 10],
  [ 5,  3, 15, 11]]]
```

axis=2的连接即：a\[0]\[0]+b\[0]\[0]，a\[0]\[1]+b\[0]\[1]，a\[1]\[0]+b\[1]\[0]，a\[1]\[1]+b\[1]\[1]

尝试想象上面，令axis=1的连接。a\[0]+b\[0]，a[1]+b[1]

连接操作，形状维度不变，只是在某一维度膨胀了。

### tf.stack() and tf.unstack()

> **Note:** If you are concatenating along a new axis consider using stack. E.g. 

与concat不同，这是堆叠。

```python
import numpy as np
a = [[[1, 2], [2, 3]], [[4, 4], [5, 3]]]  # shape:(2,2,2)
b = [[[7, 4], [8, 4]], [[2, 10], [15, 11]]]
c = np.stack([a,b],-1)
print(c)
print(c.shape)
>>>
[[[[ 1  7]
   [ 2  4]]

  [[ 2  8]
   [ 3  4]]]


 [[[ 4  2]
   [ 4 10]]

  [[ 5 15]
   [ 3 11]]]]

(2, 2, 2, 2)
```

### tf.gfile.FastGFile

```
image_data = tf.gfile.FastGFile('img/a.jpg', 'rb').read()
>>>
b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00O\x00\x00\x00\x17\x08\x06\x00\x00\x00J\xe9\x12\x1c\x00\x00\x00\x01sRGB\x00\xae\xce\x1c\xe9\x00……
```

返回什么呢？image_data: string, JPEG encoding of RGB image

### TensorFlow-Slim Data

[TensorFlow-Slim Data](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim/python/slim/data)

# 常用函数

## 创建变量

参考：[TensorFlow变量管理](https://ilewseu.github.io/2018/03/11/Tensorflow%E5%8F%98%E9%87%8F%E7%AE%A1%E7%90%86/) 。

创建变量是最最常用的，权重与偏差，都需要创建。tensorflow中常用的两个函数是：`tf.Variable `和`tf.get_variable` ，那么他们有什么不同呢啊？

### Variable类

创建变量，想象一下，是要有初始值的（那就包括形状shape和类型dtype，以及以何种方式初始化这些值），说白了就是要用 tf中提供的一些 initializer functions来初始化。

> Just like any Tensor, variables created with Variable() can be used as inputs for other Ops in the graph.
>
> 在run Ops之前，变量要被确切的初始化，由于tensorflow是先构建图后执行操作，变量不会自己初始化哦，所以：
>
>  You can initialize a variable by 
>
> + running  its initializer op：sess.run(w.initializer)
> + restoring the variable from a save file, 
> + simply running an assign Op that assigns a value to the variable. 
>
> 事实上，variable initializer op 就是一个 assign Op，通过他将初始值赋给变量。
>
> ============================
>
> 最常用的手段是使用常规函数`global_variables_initializer() `添加 an Op 给这个 graph来初始化所有变量，在启动graph后，使用变量前运行这个Op。
>
> 两个与变量有关的 graph collection：
>
> + GraphKeys.GLOBAL_VARIABLES：新变量默认添加到此集合，可通过`global_variables()` returns the contents of that collection.
> +  GraphKeys.TRAINABLE_VARIABLES：可训练变量集合，用于区别**可训练的模型参数**和其他变量，例如我们常常用` global_step`记录 training steps，由参数 trainable 控制。可通过`trainable_variables() `返回内容，各种各样的`Optimizer` classes使用此集合作为默认变量list去优化

You add a variable to the graph by constructing an instance of the `class Variable`：

```python
# This constructor creates both a variable Op and an assign Op 
# to set the variable to its initial value
def __init__(self,
             initial_value=None,
             trainable=True,
             collections=None,   # 一般用不到
             validate_shape=True,
             caching_device=None,  # ？？
             name=None,
             variable_def=None,    # ？？？
             dtype=None,
             expected_shape=None,# If set, initial_value is expected to have this shape.
             import_scope=None,  # ？？
             constraint=None):   # 优化器更新后的可选投影函数
```

+ **initial_value**--A Tensor，这个Tensor常常用  initializer functions 来构造， (Note that initializer functions from `init_ops.py` must first be bound(绑定) to a shape before being used here.) 

  ```python
  def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.01) # 形状必须指定
      return tf.Variable(initial)
  ```

+ **trainable**--上面说了

+ **collections**--List of graph collections keys. The new variable is added to these collections. Defaults to [GraphKeys.GLOBAL_VARIABLES].

+ **validate_shape**--False，允许以未知的shape初始化；True，不可以，必须指定**initial_value**的shape。

+ **name**--可选参数字符串，默认为'Variable' ，并且自动区分。

+ **dtype**--If set, initial_value will be converted to the given type. If None, either the datatype will be kept (if initial_value is a Tensor), or convert_to_tensor will decide.

###  get_variable函数

创建或者获取变量

```python
def get_variable(name,       # 这不同于Variable类，是必须给出的哦
                 shape=None,
                 dtype=None,
                 initializer=None, # 前三个变量相当于Variable的initial_value一个参数
                 regularizer=None,
                 trainable=True,
                 collections=None,
                 # 后面不知道干啥的..
                 caching_device=None,
                 partitioner=None,
                 validate_shape=True,
                 use_resource=None,
                 custom_getter=None,
                 constraint=None)
```

对于tf.get_variable函数，变量名称是一个必填的参数。 tf.get_variable会根据这个名字去创建或者获取变量。 

1. 变量共享机制

**如果需要通过tf.get_variable获取一个已经创建的变量，需要通过tf.variable_scope函数来生成一个上下文管理器，并明确指定在这个上下文管理器中，tf.get_variable将直接获取已经生成的变量。** 

```python
# 在名字为foo的命名空间内创建名字为v的变量
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1], initializer=tf.constant_initializer(1.0))
# 因为在命名空间foo中已经存在名为v的变量，所以下面的代码将会报错
with tf.variable_scope("foo"):
    v = tf.get_variable("v",[1])
# 在生成上下文管理器时，将参数reuse设置为True。这样tf.get_vaiable函数将直接获取
# 已经声明的变量。
with tf.variable_scope("foo",reuse=True):
    v1 = tf.get_variable("v",[1])
    print v==v1 #输出为True
# 将参数reuse设置为True时，tf.variable_scope将只能获取已经创建过的变量。因为在命名
# 空间bar中还没有创建变量v，所以下面的代码将会报错
with tf.variable_scope("bar",reuse=True):
    v = tf.get_variable("v",[1])
```

2. 命名空间管理

**tf.variable_scope函数生成的上下文管理器也会创建一个TensorFlow中的命名空间：**

```python
v1 = tf.get_variable("v",[1])
print v1.name    # 输出v:0,"v"为变量名称，“：0”表示这个变量时生成变量这个运算的第一个结果
with tf.variable_scope("foo"):
    v2 = tf.get_variable("v",[1])
    print v2.name  # 输出为foo/v:0。在tf.variable_scope中创建的变量，名称前面会
                   # 加入命名空间的名称，通过/来分隔命名空间的名称和变量的名称。
with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v3 = tf.get_variable("v",[1])
        print v3.name # 输出为foo/bar/v:0。命名空间可以嵌套，同时变量的名称也会
                      # 加入所有命名空间的名称作为前缀。
    v4 = tf.get_variable("v1",[1])
    print v4.name  # 输出foo/v1:0。当命名空间退出之后，变量名称也就不会再被加入其前缀了。
    
# 创建一个名称为空的命名空间，并设置为reuse=True
with tf.variable_scope("", reuse=True):
    v5 = tf.get_variable("foo/bar/v",[1]) # 可以直接通过带命名空间名称的变量名来
                                          # 获取其他命名空间下的变量
    print v5 == v3 # 输出为True
    v6 = tf.get_variable("foo/v1",[1])
    print v6 == v4 # 输出为True
```



### 初始化函数(initializer)

| 初始化函数                          | 功能                                                         | 主要参数                        |
| ----------------------------------- | :----------------------------------------------------------- | ------------------------------- |
| tf.constant_initializer             | 将变量初始化为给定常量                                       | 常量的取值                      |
| tf.random_normal_initializer        | 将变量初始化为满足正态分布的随机值                           | 正态分布的均值和标准差          |
| tf.truncated_normal_initializer     | 将变量初始化为满足正态分布的随机值，但如果随机出来的值 偏离平均值超过2个标准差，那么这个数将会被重新随机 | 正态分布的均值和标准差          |
| tf.random_uniform_initializer       | 将变量初始化为满足均匀分布的随机值                           | 最大值、最小值                  |
| tf.uniform_unit_scaling_initializer | 将变量初始化为满足均匀分布但不影响输出数量级的随机值         | factor(产生随机数时 乘以的系数) |
| tf.zeros_initializer                | 将被变量设置为0                                              | 变量维度                        |
| tf.ones_initializer                 | 将变量设置为1                                                | 变量的维度                      |

### 两者区别：

参考：

https://zhuanlan.zhihu.com/p/26996077

[TensorFlow 变量共享](https://www.cnblogs.com/Charles-Wan/p/6200446.html)

tensorflow中用tf.Variable()，tf.get_variable()，tf.Variable_scope()，tf.name_scope()几个函数来维护变量。

+ tf.Variable()会自动检测命名冲突并自行处理，
+ tf.get_variable()则遇到重名的变量创建且变量名没有设置为共享变量时，则会报错。  

```python
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name = 'weights')

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name = 'biases')

def conv2d(input, in_features, out_features, kernel_size):
    W = weight_variable([ kernel_size, kernel_size, in_features, out_features ])
    conv = tf.nn.conv2d(input, W, [ 1, 1, 1, 1 ], padding='SAME')
    return conv + bias_variable([ out_features ])
```

然后有：

```python
# 下面会生成两套变量
result1 = conv2d(image1, 3, 16, 3)
result2 = conv2d(image2, 3, 16, 3)
```

如果是用的是 tf.get_variable ，那么就会报错，

```python
# Raises ValueError(... weights already exists ...)
```

为了解决这个问题，TensorFlow 又提出了 tf.variable_scope 函数。

+ tf.Variable()用于创建一个新变量，在同一个name_scope下面，可以创建相同名字的变量，底层实现会自动引入别名机制，两次调用产生了其实是两个不同的变量。 
+ tf.get_variable(<variable_name>)用于获取或创建一个变量，并且不受name_scope的约束。当这个变量已经存在时，则自动获取；如果不存在，则自动创建一个变量。 

```python
with tf.variable_scope("foo"):
    with tf.name_scope("bar"):
        v = tf.get_variable("v", [1])  # 不受约束
        x = 1.0 + v
        a = tf.Variable(tf.constant(1.0,shape=[2,2]))  #  受约束
        
print(v.name)     # "foo/v:0"
print(x.op.name)  # "foo/bar/add"
print(a.name)     # "foo/bar/Variable:0"
```

+ tf.name_scope(<scope_name>):主要用于管理一个图里面的**各种op的名字**，返回的是一个以scope_name命名的context manager。一个graph会维护一个name_space的堆，每一个namespace下面可以定义各种op或者子namespace，实现一种层次化有条理的管理，避免各个op之间命名冲突。 
+ tf.variable_scope(<scope_name>)：一般与tf.name_scope()配合使用，用于管理一个**graph中变量的名字**，避免变量之间的命名冲突，tf.variable_scope(<scope_name>)允许在一个variable_scope下面共享变量。 



## 卷积

TensorFlow中，卷积都有哪些函数？

### tf.nn.conv2d

```python
# tf.nn.conv2d
def conv2d(input,   # 4D Tensor,shape [batch, height, width, in_channels]
           filter,  # A Tensor, [height, width, in_channels, out_channels],类型一致和输入
           strides, # 一般来说，[1, stride, stride, 1].在高和宽之间以步长stride滑动
           padding, #SAME" or "VALID".
           use_cudnn_on_gpu=True, 
           data_format="NHWC", dilations=[1, 1, 1, 1], name=None):
```

 对于2D卷积来说，卷积核 filter的可以理解为 [height，width，卷积核通道数，卷积核个数]

那么有，1）输入的通道数 in_channels = = 卷积核通道数 in_channels。2）大多数情况，卷积核的第0 和第3 维度大小为 1 。但是这个函数要重复指定。

而且，卷积操作前，要定义权重参数W；卷积操作后，1）还要加bias项，2）后面可能还要接激活函数，3）还要考虑其是否正则化参数。

所以使用这个函数使有一点点麻烦。看下一个函数：

### tf.layers.conv2d

函数中用到了Class `Conv2D`：tf.layers.Conv2D

```python
def conv2d(inputs,
           filters,         # 卷积核个数，输出通道数
           kernel_size,     # An integer or tuple/list of 2 integers，用于指定卷积核高宽
           strides=(1, 1),  # An integer or tuple/list of 2 integers，指定步长
           padding='valid',
           data_format='channels_last',
           dilation_rate=(1, 1),
           
           activation=None,
           use_bias=True,
           
           kernel_initializer=None,
           bias_initializer=init_ops.zeros_initializer(),
           
           kernel_regularizer=None,
           bias_regularizer=None,
           activity_regularizer=None, # Optional regularizer function for the output？
           
           kernel_constraint=None,
           bias_constraint=None,
           trainable=True,
           name=None,
           reuse=None) # Boolean, 是否使用通过相同的名字使用先前的wights
```

参数：

+ data_format：string

  channels_last ：(batch, height, width, channels)

  channels_first ： (batch, channels, height, width)

+ dilation_rate：空洞卷积？不等于1的时候与stride不等于1不兼容

这个函数，就结合各种操作为一体了。

## 激活函数

## Pooling

## 损失函数

## 保存模型

## 图形



