# SPPnet——引入金字塔池化层改进RCNN



## Problem&Achievement

Abstract

+ can generate a fixed-length representation（每一个window生成一个1维向量） regardless of image size/scale. we compute the feature maps from the entire image only once（RCNN一张图片分割为2000张）, and then pool features in arbitrary regions (sub-images) to generate fixed-length representations for training the detectors.
+ SPP-net should in general improve all CNN-based image classification methods.
+ SPP-net achieves state-of-the art classification results using a single full-image representation and **no fine-tuning**.
+ In processing test images, our method is 24-102× faster than the R-CNN method, while achieving better or comparable accuracy on Pascal VOC 2007    

## Structure

### SPP-net

下面是文中给出与传统CNN的对比图：

![SPP_compare](img\SPP_compare.png)

可以看到，在图片输入之前不用在对图片进行crop/warp的操作，将CNN的最后一层pooling改为金字塔pooling，最好接上分类层。

### spatial pyramid pooling

> we replace the last pooling layer (e.g., pool5, after the last convolutional layer) with a spatial pyramid pooling layer.

![SPP_spatial pyramid pooling](E:\paperreading\img\SPP_spatial pyramid pooling.png)

文献中把图中pooling的小方块叫做local spatial bins，尺寸与输入图像大小无关。**注意**，每个bins在pooling时可能重叠，由核尺寸和步长决定。

> In each spatial bin, we pool the responses of each filter (throughout this paper we use max pooling).
>
> 这里pooling整个特征图（each filter），是只为了分类任务吧。。。。。。。。

得到固定维度的 *kM*-dimensional vectors，k是最后一层卷积核数量，也是输出特征图的通道数。M是local spatial bins的数量。This not only allows arbitrary aspect ratios, but also allows arbitrary scales. We can resize the input image to any scale。值得一提的是，最右边1×1的叫做“global pooling”，是受到前人工作的启发。

## Train Solution 

> Theoretically, the above network structure can be trained with standard back-propagation [1], regardless of the input image size. But in practice the GPU implementations (such as cuda-convnet [3] and Caffe [35]) are preferably run on fixed input images. Next we describe our training solution that takes advantage of these GPU implementations while still preserving the spatial pyramid pooling behaviors.    
>
> 所以是：单尺寸训练+另外的单尺寸训练+另外另外的单尺寸训练…………??

### Single-size training

+ 输入：224×224，按照前人的工作，首先考虑的尺寸。
+ **SPP-pooling**：例如：3-level pyramid pooling (3×3, 2×2, 1×1) ，**win=a/n取上整，stride=a/n取下整**， AlexNet的conv5的输出是13×13，3×3的输出对应win=13/3=5，str=13/3=4 。另两个（7，6），（13，13），但在训练中使用The pyramid is {6×6, 3×3, 2×2, 1×1} (totally 50 bins)    
+ 损失Loss：没提。
+ 训练超参数：没提。

### Multi-size training

+ 输入：考虑了180×180，不是裁剪180×180的区域，而是resize the aforementioned 224×224 region to 180×180。这样：分辨率不同，但内容和布局相同。当然也考虑了[180,224]之间的尺寸
+ SPP-pooling，conv5的输出10×10，pooling策略与之前相同，可生成与224×224相同的输出。共享参数哦！
+ LOSS：没提。
+ 训练超参数：没提。

### SPP-NET FOR OBJECT DETECTION

> our method (SPP) enables feature extraction in arbitrary windows from the deep convolutional feature maps.   

流程如下：

![SPP_detection](img\SPP_detection.png)

对比RCNN，整个过程是：

1. RCNN：首先通过选择性搜索，对待检测的图片进行搜索出2000个候选窗口。这一步和R-CNN一样。

2. 特征提取阶段。这一步就是和R-CNN最大的区别了，这一步骤的具体操作如下：we resize the image such that min(w，h) = s，然输入到CNN中，进行一次性特征提取，得到feature maps，然后在feature maps中找到各个候选框的区域，再对各个候选框采用金字塔空间池化，提取出固定长度的特征向量。而R-CNN输入的是每个候选框，然后在进入CNN，因为SPP-Net只需要一次对整张图片进行特征提取，速度会大大提升。~~SPPnet：This generates a 12,800- d (256×50) representation **for each window** ，而RCNN中是4096。~~

   这理解错了，**原文**：This generates a 12,800- d (256×50) representation for each window. These representations are provided to the fully-connected layers of the network.  后面还接着两个全连接层。最后输出是多少不知道了。。。。。也是2000×4096？？？   

3. RCNN：最后一步也是和R-CNN一样，采用SVM算法进行特征向量分类识别，We apply the standard hard negative mining [23] to train the SVM. This step is iterated once. It takes less than 1 hour to train SVMs for all 20 categories.     

步骤2中涉及的**难点**：**原始图像的~~ROI~~建议框如何映射到特征图上呢（一系列卷积层的最后输出）** 。文中也提到了这点，有时间补上，其实是没看懂。。。。。。。。。。。。。。。。。

#### 模型：

+ SPP-net model of ZF-5 (single-size trained)   
+ we use a 4-level spatial pyramid (1×1, 2×2, 3×3, 6×6, totally 50 bins) to pool the features.
+ 后面fc6，fc7，还有21-way fc8。      

#### 标签：

与RCNN策略一样。positive samples [0.5,1]，negative samples [0.1,0.5]。数字代表IoU的值。

#### 训练：

SVM训练与RCNN策略一样。不多说。

> Our implementation of the SVM training follows [20], [7]. We use the ground-truth windows to generate the positive samples. The negative samples are those overlapping a positive window by at most 30% (measured by the intersection-over-union (IoU) ratio). Any negative sample is removed if it overlaps another negative sample by more than 70%. We apply the standard hard negative mining [23] to train the SVM. This step is iterated once.    

1. 微调预训练模型：We also fine-tune our pre-trained network, following RCNN. Since our features are pooled from the conv5 feature maps from windows of any sizes, for simplicity we **only fine-tune the fully-connected layers**.

   文中说之训练fc层是为了简单起见，而Fast R-CNN中打脸地提到：

   > Training all network weights with back-propagation is an important capability of Fast R-CNN. First, let’s elucidate why SPPnet is unable to update weights below the spatial pyramid pooling layer    

2. SVM训练。输出2000×20，每一类对应的2000个框的得分。

3. 边框回归训练：Also following RCNN, we use bounding box regression to post-process the prediction windows. The features used for regression are the pooled features from conv5 (as a counterpart of the pool5 features used in [7]RCNN). The windows used for the regression training are those overlapping with a ground-truth window by at least 50% .   

#### 超参数：

+ The fc8 weights are initialized with a Gaussian distribution of σ=0.01   
+ 学习速率： We train 250k mini-batches using the learning rate 1e-4, and then 50k mini-batches using 1e-5
+ Because we only fine-tune the fc layers, the training is very fast and takes about 2 hours on the GPU (excluding pre-caching feature maps which takes about 1 hour)  

## Test

可以输入任意尺寸啦，下图比较了与RCNN的区别：

![SPP_vs_RCNN](img\SPP_vs_RCNN.png)

## Experiment

为了说明 Multi-level Pooling Improves Accuracy ，要做三组实验：不加SPP的、单尺度的和多尺度的。

为了说明不是参数变多导致效果好，做了{4×4, 3×3, 2×2, 1×1} (totally 30 bins)的实验与no SPP的对比 ，参数：30×256<36×256。

不懂：Multi-view Testing on Feature Maps

For the standard 10-view, we use s = 256 and the views are 224×224 windows on the corners or center.    

## 遗留问题

Multi-view Testing on Feature Maps：在feature map上取multi-view（multi-window），然后平均化softmax的预测分数，进一步地再加入多尺寸，同样有助于提升准确率 。

Mapping a Window to Feature Maps （文末）：

> In the detection algorithm (and multi-view testing on feature maps), a window is given in the image domain, and we use it to crop the convolutional feature maps (e.g., conv5) which have been sub-sampled several times. So we need to align the window on the feature maps.    

参考：[原始图片中的ROI如何映射到到feature map?](https://zhuanlan.zhihu.com/p/24780433)

