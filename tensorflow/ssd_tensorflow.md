### 1.数据集TFrecoeds制作

数据集：PASCALVOC 2007，用于物体检测的注释在Annotations文件夹中，每张图片对应信息保存为xml文件。

```xml
<annotation>
	<folder>VOC2007</folder>
	<filename>000001.jpg</filename>
	<source>  # 没啥用
		<database>The VOC2007 Database</database>
		<annotation>PASCAL VOC2007</annotation>
		<image>flickr</image>
		<flickrid>341012865</flickrid>
	</source>
	<owner>  # 没啥用
		<flickrid>Fried Camels</flickrid>
		<name>Jinky the Fruit Bat</name>
	</owner>
	<size>  # 图像尺寸及通道数
		<width>353</width>
		<height>500</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>dog</name>
		<pose>Left</pose>  # 好像是从哪个角度拍的？
		<truncated>1</truncated>  # 这个目标是否因为各种原因没有被框完整（被截断了）
		<difficult>0</difficult>  # 待检测目标是否很难识别，为1的目标在测评估中一般会被忽略
		<bndbox>  # 轴对其矩形，框住的是目标在照片中的可见部分
			<xmin>48</xmin>
			<ymin>240</ymin>
			<xmax>195</xmax>
			<ymax>371</ymax>
		</bndbox>
	</object>
	<object>
		……
	</object>
</annotation>

```

20类别+1背景信息如下，以字典类型存储

```python
VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}
```

#### run

```python
DIRECTORY_ANNOTATIONS = 'Annotations/'
DIRECTORY_IMAGES = 'JPEGImages/'
# TFRecords convertion parameters.
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 200  # 每个tfrecor存储的sample数量

def run(dataset_dir, output_dir, name='voc_train', shuffling=False):
    """Runs the conversion operation.
    Args:
      dataset_dir: The dataset directory. mine:F:/data/VOCdevkit/VOC2007
      output_dir: Output directory.比如：./tfrecords
      name：默认为voc_train,对应VOC2007为voc_2007_train
    """
    # True if the path exists, whether its a file or a directory
    if not tf.gfile.Exists(dataset_dir): 
        tf.gfile.MakeDirs(dataset_dir)
    # Dataset filenames, and shuffling.对于检测任务，洗不洗牌感觉没那么重要
    path = os.path.join(dataset_dir, DIRECTORY_ANNOTATIONS)
    filenames = sorted(os.listdir(path))  # 必须排序，因为系统都出来的可能不按照顺序
    if shuffling:
        random.seed(RANDOM_SEED)  # 保证下一句（洗牌）每次输出一样的结果
        random.shuffle(filenames)
    # Process dataset files.
    i = 0
    fidx = 0  # file_index,tfrecords的名字中的 %03d，9663=200*48+63，会生成49个tfrecords
    while i < len(filenames):  # 9963个图片
        # Open new TFRecord file.
        tf_filename = _get_output_filename(output_dir, name, fidx)  # 就获得个路径名字。。。
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(filenames)))
                sys.stdout.flush()
                
                filename = filenames[i]
                img_name = filename[:-4]  # 去掉.jpg
                _add_to_tfrecord(dataset_dir, img_name, tfrecord_writer)
                i += 1
                j += 1
            fidx += 1
    print('\nFinished converting the Pascal VOC dataset!')
```

其中：

```python
def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)
```

#####  _add_to_tfrecord()

1. 处理一张图片及其xml文件，读取信息
2. 将信息转换为example
3. 运用*tf.python_io.TFRecordWriter*写入一张图片信息（example）到文件中：

> 真不明白为什么单独写一个函数。。

```python
def _add_to_tfrecord(dataset_dir, name, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    image_data, shape, bboxes, labels, labels_text, difficult, truncated = \
        _process_image(dataset_dir, name)
    example = _convert_to_example(image_data, labels, labels_text,
                                  bboxes, shape, difficult, truncated)
    tfrecord_writer.write(example.SerializeToString())
```

###### _process_image

```python
def _process_image(directory, name):
    """Process a image and annotation file.
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
    """
    # 原来是：filename = directory + DIRECTORY_IMAGES + name + '.jpg'
    # 很奇怪，directory后面没有'/'啊，怎么不用join
    # 为什么不直接传入带后缀.jpg的文件名
    # 卧槽，懂了还有xml文件呢
    filename = directory +'/'+ DIRECTORY_IMAGES + name + '.jpg'
    # Read the image file.
    # 原来是：image_data = tf.gfile.FastGFile(filename, 'r').read()，有bug
    # rb表示按字节读，r默认是rt模式：字节文本。不太懂
    image_data = tf.gfile.FastGFile(filename, 'rb').read()

    # Read the XML annotation file.这又用join。。。。。。
    filename = os.path.join(directory, DIRECTORY_ANNOTATIONS, name + '.xml')
    tree = ET.parse(filename)  # 解析
    root = tree.getroot()

    # Image shape.
    size = root.find('size')
    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]  # 比如1.jpg, [500,353,3]
    # Find annotations.每个object有以下信息
    bboxes = []  # 归一化哦，元素：tuple(ymin,xmin,ymax,xmax)
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(int(VOC_LABELS[label][0]))
        labels_text.append(label.encode('ascii'))

        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)
        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)

        bbox = obj.find('bndbox')
        bboxes.append((float(bbox.find('ymin').text) / shape[0],  # 归一化了啊
                       float(bbox.find('xmin').text) / shape[1],
                       float(bbox.find('ymax').text) / shape[0],
                       float(bbox.find('xmax').text) / shape[1]
                       ))
    return image_data, shape, bboxes, labels, labels_text, difficult, truncated
```

得到的主要的有用的图片信息：

```python
img_data:image_data = tf.gfile.FastGFile(filename, 'rb').read(),一大串\x进制的
shape:[H,W,3]，形状为(3,)
下面对应多个object：
bboxes:[(ymin,xmin,ymax,xmax),(),()],形状为(num_object,)，记住这里已经归一化，并且将坐标放在了一起。
labels:[c1,c2,c3……],形状为(num_object,)
labels_text, difficult, truncated好像没啥用。
```

######  _convert_to_example

返回tf.train.Example(features=tf.train.Features(feature={……}))实例。

```python
def _convert_to_example(image_data, labels, labels_text, bboxes, shape,
                        difficult, truncated):
    """Build an Example proto for an image example.

    Args:
      image_data: string, JPEG encoding of RGB image;
      labels: list of integers, identifier for the ground truth;
      labels_text: list of strings, human-readable labels;
      bboxes: list of bounding boxes; each box is a list of integers; # tuple吧？？
          specifying [xmin, ymin, xmax, ymax]. [ymin,xmin,ymax,xmax]吧？？？？
          All boxes are assumed to belong to the same label as the image label.
      shape: 3 integers, image shapes in pixels.
    Returns:
      Example proto
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        
    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/shape': int64_feature(shape),
            'image/object/bbox/xmin': float_feature(xmin),
            'image/object/bbox/xmax': float_feature(xmax),
            'image/object/bbox/ymin': float_feature(ymin),
            'image/object/bbox/ymax': float_feature(ymax),
            'image/object/bbox/label': int64_feature(labels),
            'image/object/bbox/label_text': bytes_feature(labels_text),
            'image/object/bbox/difficult': int64_feature(difficult),
            'image/object/bbox/truncated': int64_feature(truncated),
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data)}))
    return example

def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def float_feature(value):
    """Wrapper for inserting float features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
```

其中：

```python
"""zip([iterable, ...])
参数说明：
	iterabl -- 一个或多个迭代器;返回zip对象，需要用list转换展示，打包为tuple
"""
z = zip([ymin, xmin, ymax, xmax], (1,2,3,4))
print (list(z))  # [([], 1), ([], 2), ([], 3), ([], 4)]
```

#### 关键步骤

```python
while i < len(filenames):
    tf_filename = ''
	with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
    	image_data, shape, bboxes, labels = process(info(i)) # 自己去处理数据第i个数据，最好list
         example = tf.train.Example(features=tf.train.Features(feature={
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data),
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/shape': int64_feature(shape),
            'image/object/bbox/xmin': float_feature(xmin),
            'image/object/bbox/xmax': float_feature(xmax),
            'image/object/bbox/ymin': float_feature(ymin),
            'image/object/bbox/ymax': float_feature(ymax),
            'image/object/bbox/label': int64_feature(labels)}))
    	tfrecord_writer.write(example.SerializeToString())
        i+=1
```

1. 生成TFRecord Writer：

   ```python
   writer = tf.python_io.TFRecordWriter(path, options=None)
   ```

2. tf.train.Features生成协议信息，内层feature是字典，字典key用于读取时索引。列表类型一般有BytesList, FloatList, Int64List， 例如：

   ```python
   feature = {
   "width":tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
   "weights":tf.train.Feature(float_list=tf.train.FloatList(value=[weights])),
   "image_raw":tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
   }
   ```
   其中

   ```python
   tf.train.BytesList(value=[value]) # value转化为字符串（二进制）列表
   tf.train.FloatList(value=[value]) # value转化为浮点型列表
   tf.train.Int64List(value=[value]) # value转化为整型列表
   ```

   外层features再将内层字典编码： 

   ```python
   features = tf.train.Features(feature)
   ```

3. 使用tf.train.Example将features编码数据封装成特定的PB协议格式

   ```python
   example = tf.train.Example(features)
   ```

4. 将example数据系列化为字符串

   ```python
   example_str = example.SerializeToString()
   ```

5. 将系列化为字符串的example数据写入协议缓冲区

   ```python
   writer.write(example_str)
   writer.close()
   ```


### 1.数据集读取

#### tf.TFRecordReader

```python
    fliename = r'./voc_2007_train_000.tfrecord'
    # 一、根据文件名生成文件名队列
    filename_queue = tf.train.string_input_producer([filename])  # tfrecord文件名list
    reader = tf.TFRecordReader()  # 二、生成TFRecordReader
    tf_name, serialized_example = reader.read(filename_queue)  # 返回文件名和读取的内容
    # 三、解析器解析序列化example
    features = tf.parse_single_example(serialized_example,
                features={
                # 对于单个元素的变量，我们使用FixlenFeature来读取，需要指明变量存储的数据类型
                'image/encoded': tf.FixedLenFeature([], tf.string,default_value=''),
                'image/format': tf.FixedLenFeature([], tf.string,default_value='jpeg'),
                'image/height': tf.FixedLenFeature([1], tf.int64),
                'image/width': tf.FixedLenFeature([1], tf.int64),
                'image/channels': tf.FixedLenFeature([1], tf.int64),
                'image/shape': tf.FixedLenFeature([3], tf.int64),
                # 对于list类型的变量，我们使用VarLenFeature来读取，同样需要指明读取变量的类型
                # bbox
                'image/object/bbox/xmin':tf.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/ymin':tf.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/xmax':tf.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/ymax':tf.VarLenFeature(dtype=tf.float32),
                # label
                'image/object/bbox/label':tf.VarLenFeature(dtype=tf.int64),
                })
    
    # features字典，接下来用key索引得到数据，为我所用。
```

例如，显示000001.jpg图片：

```python
print(features) # 1标注
import matplotlib.pyplot as plt
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    shape,ymin,img = \
sess.run([features['image/shape'],features['image/object/bbox/ymin'],features['image/encoded']])
    print(shape)  # [500 353   3]
    print(ymin)  # 2标注

    image = tf.image.decode_jpeg(img, channels=3) #Tensor("DecodeJpeg:0", shape=(?, ?, 3), dtype=uint8) 最好不要在session里面写Op的操作。。。。。。。。。
    image = sess.run(image)  # type：numpy.ndarray
    print(image.dtype)  # uint8

    plt.imshow(image)
    plt.show()

    coord.request_stop()
    coord.join(threads)
```

注意：#1标注，#2标注的打印结果：

```python
#1 features:是字典
{'image/object/bbox/label': 
<tensorflow.python.framework.sparse_tensor.SparseTensor object at 0x000001DBA49CDD68>,
'image/object/bbox/ymax':
<tensorflow.python.framework.sparse_tensor.SparseTensor object at 0x000001DBA4A06D68>,
 'image/encoded': 
 <tf.Tensor 'ParseSingleExample/ParseSingleExample:16' shape=() dtype=string>,
 'image/height': 
 <tf.Tensor 'ParseSingleExample/ParseSingleExample:18' shape=(1,) dtype=int64>,
 }
```

+ 以list存储的label和ymax等坐标值取出来的类型是**SparseTensor**；
+ 而image是**string**类型，应该是字节型字符串，像这样：

```
b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00………………………………………………………………'
```

+ 高、宽、通道数、和形状都是int64类型的，情理之中。

```python
#2 而ymin的类型是：tf.VarLenFeature(dtype=tf.float32),取出来是稀疏Tensor..........
SparseTensorValue(indices=array([[0],[1]], dtype=int64), 
                  values=array([0.48 , 0.024], dtype=float32), 
                  dense_shape=array([2], dtype=int64))
# 怎么就是稀疏矩阵了呢？？？如何转换成可以正常的array([0.48 , 0.024], dtype=float32)呢？？靠
```

#### 读取多张

在构建好图之后，run，image节点多次就可以依次取出图像数据。

```python
filename_queue = tf.train.string_input_producer([tfrecord_path])
reader = tf.TFRecordReader()
# 用reader去read数据集队列
tf_name, serialized_example = reader.read(filename_queue,name='reading')
features = tf.parse_single_example(serialized_example,
    features={
              'image/encoded': tf.FixedLenFeature([], tf.string, default_value=''),
    })
img = features['image/encoded']
image = tf.image.decode_jpeg(img, channels=3)
=======================构建图完成=========================
with tf.Session() as sess:
writer = tf.summary.FileWriter('./graphs',sess.graph)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

print(image)
img1 = sess.run(image)
img2 = sess.run(image)  # run了两次image结点就可以依次取出图像
plt.imshow(img1)
plt.show()
plt.imshow(img2)
plt.show()
coord.request_stop()
coord.join(threads)
```

![1533044907809](assets/1533044907809.png)
![1533044917245](assets/1533044917245.png)


#### tf.python_io.tf_record_iterator 

```python
with tf.Session() as sess:
    for serialized_example in tf.python_io.tf_record_iterator(tfrecord_path):
        # An iterator that read the records from a TFRecords file
        features = tf.parse_single_example(serialized_example,
            features={
            'image/encoded': tf.FixedLenFeature([], tf.string, default_value=''),
            'image/height': tf.FixedLenFeature([1], tf.int64),
            'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
            })
        height = features['image/height']
        print(sess.run(height))
```


### 2.Anchors的生成

为所有特征图生成anchors：layers_anchors: [(y,x,h,w),(y,x,h,w),(y,x,h,w)……]。元素为tuple类型。

```python
def ssd_anchors_all_layers(img_shape,
                           layers_shape,
                           anchor_sizes,
                           anchor_ratios,
                           anchor_steps,
                           offset=0.5,
                           dtype=np.float32):
    """Compute anchor boxes for all feature layers.
    """
    layers_anchors = []
    for i, s in enumerate(layers_shape):  # 例如第一层
        anchor_bboxes = ssd_anchor_one_layer(img_shape, s,  # (38,38)
                                             anchor_sizes[i],  # (21,45)
                                             anchor_ratios[i],  # [2,.5]
                                             anchor_steps[i],  # 8
                                             offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors
```

下面函数返回：

y，x：形状：(H, W, 1)

h，w : 形状：(num_anchors, ) 比如：4个或6个，也要归一化


```python
def ssd_anchor_one_layer(img_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         step,
                         offset=0.5,
                         dtype=np.float32):
    """Computer SSD default anchor boxes for one feature layer.
    Determine the relative position grid of the centers, and the relative
    width and height.
    以sdd300的第一个特征图为例:
    Arguments:
      feat_shape: Feature shape, used for computing relative position grids;  (38,38)
      size: Absolute reference sizes;  (21.,45.)
      ratios: Ratios to use on these features;  [2,.5]
      img_shape: Image shape, used for computing height, width relatively to the
        former;  (300,300)
      offset: Grid offset.  0.5
    Return:
      y, x, h, w: Relative x and y grids of the centers, and relative height and width.
    """
    # Weird SSD-Caffe computation using steps values...
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) * step / img_shape[0] 
    # y.astype(dtype):Copy of the array, cast to a specified type.
    x = (x.astype(dtype) + offset) * step / img_shape[1] # 除以300/8≈38，即归一化
    
    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = len(sizes) + len(ratios)
    h = np.zeros((num_anchors, ), dtype=dtype)  
    w = np.zeros((num_anchors, ), dtype=dtype) 
    # Add first anchor boxes with ratio=1.
    h[0] = sizes[0] / img_shape[0] # 21/300 默认ratio=1的一个框
    w[0] = sizes[0] / img_shape[1] # 21/300
    di = 1
    if len(sizes) > 1:
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]  # 默认ratio=更号下s0*s1的框
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        di += 1
    for i, r in enumerate(ratios):  #  其他2或者4个框
        h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
    return y, x, h, w
```

其中：

#### np.mgrid

相当于生成了一个坐标(x,y)网格，分别存储横坐标和纵坐标，而且正好与图像的对应，原点在左上角。

```python
import numpy as np
a = np.mgrid[0:5, 0:5]  # np.mgrid[0:a,0:b] 生成(2,a,b)形状的数组，数字排列规则见例子。先|后——
y, x = a
print(a)
print(a.shape) # (2, 5, 5)
print(y)
>>>
[
y [[0 0 0 0 0]
  [1 1 1 1 1]
  [2 2 2 2 2]
  [3 3 3 3 3]
  [4 4 4 4 4]]

x [[0 1 2 3 4]
  [0 1 2 3 4]
  [0 1 2 3 4]
  [0 1 2 3 4]
  [0 1 2 3 4]]
]
```

#### np.expand_dims

正如作者所说：Expand dims to support easy broadcasting.

```python
y = np.expand_dims(y, axis=-1) # 增加一个axis维度 (5,5)->(5,5,1)
print(y)
>>>
[[[0][0][0][0][0]]
[[1][1][1][1][1]]
[[2][2][2][2][2]]
[[3][3][3][3][3]]
[[4][4][4][4][4]]]
```

### print_configuration

#### pprint

####  parallel_reader

```python
from tensorflow.contrib.slim.python.slim.data import parallel_reader
data_files = parallel_reader.get_data_files(data_sources)
```



### Preprocess

```python
def preprocess_for_train(image, labels, bboxes,
                         out_shape, data_format='NHWC',
                         scope='ssd_preprocessing_train'):
    """Preprocesses the given image for training.

    Note that the actual resizing scale is sampled from
        [`resize_size_min`, `resize_size_max`].

    Args:
        image: A `Tensor` representing an image of arbitrary size.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        resize_side_min: The lower bound for the smallest side of the image for
            aspect-preserving resizing.
        resize_side_max: The upper bound for the smallest side of the image for
            aspect-preserving resizing.

    Returns:
        A preprocessed image.
    """
    fast_mode = False
    with tf.name_scope(scope, 'ssd_preprocessing_train', [image, labels, bboxes]):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')
        # Convert to float scaled [0, 1].
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        tf_summary_image(image, bboxes, 'image_with_bboxes')

        # Distort image and bounding boxes.
        dst_image = image
        dst_image, labels, bboxes, distort_bbox = \
            distorted_bounding_box_crop(image, labels, bboxes,
                                        min_object_covered=MIN_OBJECT_COVERED,
                                        aspect_ratio_range=CROP_RATIO_RANGE)
        # Resize image to output size.
        dst_image = tf_image.resize_image(dst_image, out_shape,
                                          method=tf.image.ResizeMethod.BILINEAR,
                                          align_corners=False)
        tf_summary_image(dst_image, bboxes, 'image_shape_distorted')

        # Randomly flip the image horizontally.
        dst_image, bboxes = tf_image.random_flip_left_right(dst_image, bboxes)

        # Randomly distort the colors. There are 4 ways to do it.
        dst_image = apply_with_random_selector(
                dst_image,
                lambda x, ordering: distort_color(x, ordering, fast_mode),
                num_cases=4)
        tf_summary_image(dst_image, bboxes, 'image_color_distorted')

        # Rescale to VGG input scale.
        image = dst_image * 255.
        image = tf_image_whitened(image, [_R_MEAN, _G_MEAN, _B_MEAN])
        # Image data format.
        if data_format == 'NCHW':
            image = tf.transpose(image, perm=(2, 0, 1))
        return image, labels, bboxes
```

#### tf.boolean_mask



