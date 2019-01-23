# PIL

对于彩色图像，不管其图像格式是PNG，还是BMP，或者JPG，在PIL中，使用Image模块的open()函数打开后，返回的图像对象的模式都是“RGB”。

而对于灰度图像，不管其图像格式是PNG，还是BMP，或者JPG，打开后，其模式为“L”。 

概念：通道（bands）、模式（mode）、尺寸（size）、坐标系统（coordinate system）、调色板（palette）、信息（info）和滤波器（filters） 

## 1 通道

### getbands()

对于一张图片的通道数量和名称，可以通过方法getbands()来获取。

返回一个字符串元组（tuple），包括每一个通道的名称。例如：('L',)、('R', 'G', 'B') 。

## 2 模式

0是黑，255是白

### mode属性

1：1位像素，表示黑和白，但是存储的时候每个像素存储为8bit。

L：8位像素，表示黑和白。

P：8位像素，使用调色板映射到其他模式。

RGB：3x8位像素，为真彩色。

RGBA：4x8位像素，有透明通道的真彩色。

CMYK：4x8位像素，颜色分离。

> 模式“CMYK”就是印刷四分色模式，它是彩色印刷时采用的一种套色模式，利用色料的三原色混色原理，加上黑色油墨，共计四种颜色混合叠加，形成所谓“全彩印刷”。 
>
> 四种标准颜色是：C：Cyan = 青色，又称为‘天蓝色’或是‘湛蓝’M：Magenta = 品红色，又称为‘洋红色’；Y：Yellow = 黄色；K：Key Plate(blacK) = 定位套版色（黑色）。 

YCbCr：3x8位像素，彩色视频格式。

> YCbCr其中Y是指亮度分量，Cb指蓝色色度分量，而Cr指红色色度分量。 人的肉眼对视频的Y分量更敏感，因此在通过对色度分量进行子采样来减少色度分量后，肉眼将察觉不到的图像质量的变化。 

I：32位整型像素。

F：32位浮点型像素

### 格式/模式转化convert()

格式转化：

对于PNG、BMP和JPG彩色图像格式之间的互相转换都可以通过Image模块的open()和save()函数来完成。具体说就是，在打开这些图像时，PIL会将它们解码为三通道的“RGB”图像。用户可以基于这个“RGB”图像，对其进行处理。处理完毕，使用函数save()，可以将处理结果保存成PNG、BMP和JPG中任何格式。

当然，对于不同格式的灰度图像，也可通过类似途径完成，只是PIL解码后是模式为“L”的图像。 

**但是，对于后缀为 .jpg 而实际格式为 TIFF 或 BMP 的图像，在 save 之后并没有改变格式。。。。。**

```python
img_list = os.listdir(r'F:\data\Steel_VOC\SteelVOC\JPEGImages')
for img_name in img_list:
    img = Image.open(os.path.join(r'F:\data\Steel_VOC\SteelVOC\JPEGImages',img_name))
    if img.format is not 'JPEG':
        print(img_name,img.format,img.getbands())
        img.save(os.path.join(r'F:\data\Steel_VOC\SteelVOC\JPEGImages',img_name))
        print(img_name, img.format, img.getbands())
```

结果：转化前和保存后还是一样一样的。

```
000902.jpg TIFF ('L',)
000902.jpg TIFF ('L',)
```

哈哈哈哈哈哈，可不一样吗，打印的都是img变量的，应该重新打开：

```python
 img2 = Image.open(os.path.join(r'F:\data\Steel_VOC\SteelVOC\JPEGImages', img_name))
```

仍然报错，BMP的不能保存为 JPEG格式的？？？

```
000916.jpg BMP ('L',)
Traceback (most recent call last):
  File "D:/Users/lirui/PycharmProjects/SSD.TensorFlow/dataset/test.py", line 13, in <module>
    img.save(os.path.join(r'F:\data\Steel_VOC\SteelVOC\JPEGImages',img_name))
  File "D:\Users\lirui\Anaconda3\lib\site-packages\PIL\Image.py", line 1932, in save
    fp = builtins.open(filename, "w+b")
PermissionError: [Errno 13] Permission denied: 'F:\\data\\Steel_VOC\\SteelVOC\\JPEGImages\\000916.jpg'
```

**不知道这是为什么？难道是因为在open BMP文件之后，被加锁了？？但是处理 TIFF的时候没发生这种情况。**





模式转化：

convert()有三种形式的定义：

```python
im.convert(mode) # ⇒ image
im.convert("P", **options) # ⇒ image
im.convert(mode, matrix) # ⇒ image
```

模式“RGB”转换为“L”模式是按照下面的公式转换的：

L = R * 299/1000 + G * 587/1000+ B * 114/1000



## 3 尺寸

### size属性

是一个二元组，包含水平和垂直方向上的像素数。

## 4 坐标系统

PIL使用笛卡尔像素坐标系统，坐标(0，0)位于左上角。注意：坐标值表示像素的角；位于坐标（0，0）处的像素的中心实际上位于（0.5，0.5）。

长方形则表示为四元组，前面是左上角坐标。例如，一个覆盖800x600的像素图像的长方形表示为（0，0，800，600）。

## 5 滤波器

Image模块中的方法resize()和thumbnail()用到了滤波器。 

### resize()

```python
resize(size, filter=None)  # =>Image，默认使用NEAREST滤波器。
```

四种滤波器：

+ NEAREST：最近滤波。从输入图像中选取最近的像素作为输出像素。它忽略了所有其他的像素。 
+ BILINEAR：双线性滤波。在输入图像的2x2矩阵上进行线性插值。 
+ BICUBIC：双立方滤波。在输入图像的4x4矩阵上进行立方插值。 
+ ANTIALIAS：平滑滤波。这是PIL 1.1.3版本中新的滤波器。对所有可以影响输出像素的输入像素进行高质量的重采样滤波，以计算输出像素值。在当前的PIL版本中，这个滤波器只用于改变尺寸和缩略图方法。 

