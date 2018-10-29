 0 (black), 255 (white) 

### depth

```c++
#define CV_8U   0
#define CV_8S   1 
#define CV_16U  2
#define CV_16S  3
#define CV_32S  4
#define CV_32F  5
#define CV_64F  6
//多通道
#define CV_8UC4 CV_MAKETYPE(CV_8U,4)   //0+((4-1) << 3) == 24
```

[What are the differences between CV_8U and CV_32F](https://stackoverflow.com/questions/8377091/what-are-the-differences-between-cv-8u-and-cv-32f-and-what-should-i-worry-about)

需要知道：

+ `CV_8U`: 1-byte unsigned integer (`unsigned char`).
+ `CV_32S`: 4-byte signed integer (`int`).
+ `CV_32F`: 4-byte floating point (`float`).
+ `CV_64F`: 8-byte doublepoint (`double`).



### addWeighted

函数原型为：`dst = cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]]) `

其中alpha是第一幅图片中元素的权重，beta是第二个的权重， gamma是加到最后结果上的一个值。

**功能：**实现以不同的权重将两幅图片叠加，对于不同的权重，叠加后的图像会有不同的透明度

### convertScaleAbs

一般将经过Sobel处理后的深度为CV_16S（梯度是有符号的）的图像转化为CV_8U，否则将无法显示图像，而只是一副灰色的窗口。

```c++
#include <opencv2/core/core.hpp>
//! scales array elements, 
//computes absolute values and converts the results to 8-bit unsigned integers:
//dst(i)=saturate_cast<uchar>abs(src(i)*alpha+beta)
CV_EXPORTS_W void convertScaleAbs(InputArray src, OutputArray dst,
                                  double alpha=1,//乘数因子
                                  double beta=0);//偏移量
```

saturate_cast防止数据溢出 [0,255]。



### dilate（膨胀）



### findContours

例子：

```c++
Mat m_fold = grad.clone();
vector<vector<Point>> contours;
cv::findContours(m_fold, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
Mat sal_edge = Mat::zeros(m_fold.size(), CV_8UC1);
cout << contours.size() << endl;

for (auto it = contours.begin(); it != contours.end(); it++)
{
    int point_sum = (*it).size();  //point是-14了
    Rect rect = boundingRect(*it);  //坐标变成负数了。。
    if (point_sum > 40)
        m_fold(rect).copyTo(sal_edge(rect));
}

namedWindow("去小点");
imshow("去小点", sal_edge);
```



### flip

```c++
void cv::flip(
	cv::InputArray src, // 输入图像
	cv::OutputArray dst, // 输出
	int flipCode = 0 // 0: 沿y-轴翻转, 0: 沿x-轴翻转, <0: x、y轴同时翻转
);
```

+ **flipCode** – a flag to specify how to flip the array; **0 means flipping around the x-axis** and **positive value (for example, 1) means flipping around y-axis**. Negative value (for example, -1) means flipping around both axes (see the discussion below for the formulas).

其实搞不清的点是opencv里的坐标系，原点是哪个？哪个是x轴，哪个是y轴？

沿着x轴翻转是垂直翻转（Vertical flipping ），沿着y轴是水平翻转（Horizontal flipping ）。

应该是左上角（ top-left ）是原点。

### GaussianBlur

去噪以平滑图像（smooth images），当然也有其他方法平滑图像。 [原理](https://docs.opencv.org/2.4/doc/tutorials/imgproc/gausian_median_blur_bilateral_filter/gausian_median_blur_bilateral_filter.html) 以及 [API](https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#gaussianblur)。

```c++
#include "opencv2/imgproc/imgproc.hpp
void GaussianBlur( InputArray src,//可以有多个channels，独立处理
                  OutputArray dst, Size ksize,
                  double sigmaX, double sigmaY=0,
                  int borderType=BORDER_DEFAULT );
```

+ src - depth 应该是以下类型`CV_8U`, `CV_16U`, `CV_16S`, `CV_32F` or `CV_64F`. 

+ ksize - 高斯核函数，`ksize.width` and `ksize.height` can differ，但必须是正数和奇数。也可以为0，然后根据 sigmaX 计算。In-place filtering is supported.

+ **sigmaX** – Gaussian kernel standard deviation in X direction. 高斯函数中有此参数。

+ **sigmaY** – Gaussian kernel standard deviation in Y direction;

   if `sigmaY` is zero, it is set to be equal to `sigmaX`。

  假如两个sigmas都是0，他们各自从核函数的宽、高来计算。

  为了完全控制结果而不考虑未来对语义的修改，建议指定所有ksize，sigmaX和sigmaY。

+ **borderType** – pixel extrapolation method。不了解

平滑公式：

![smooth](assets/smooth.png)

h(k,l) 是 kernel，只不过是过滤器的系数。

A 1D Gaussian kernel，可以想象2D的核函数：

![Smoothing_Tutorial_theory_gaussian_0.jpg](assets/Smoothing_Tutorial_theory_gaussian_0.jpg)

 A 2D Gaussian can be represented as : 

![G_{0}(x,y) = A  e^{ \dfrac{ -(x - \mu_{x})^{2} }{ 2\sigma^{2}_{x} } +  \dfrac{ -(y - \mu_{y})^{2} }{ 2\sigma^{2}_{y} } }](assets/gaussian.png)

μ 是平均值（峰值），σ 是标准差。(per each of the variables x and y)。A是？？？？



### minMaxLoc

```c++
double gray_min, gray_max;
minMaxLoc(orgin_cut(r_max),&gray_min,&gray_max);
```



### rectangle

```c++
cv::rectangle(caibian,cvPoint(beginX,0),cvPoint(endX,height),cvScalar(255,255,255),1); 
cv::rectangle(m_orginal, cv::Rect(beginX,beginY,width,height), cv::Scalar(255));
```



### reduce



### Sobel

`#include "opencv2/imgproc/imgproc.hpp"`，[api](https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#sobel)以及[例子](https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html)

```c++
void Sobel( InputArray src, OutputArray dst, int ddepth,
                         int dx, int dy, int ksize=3,
                         double scale=1, double delta=0,
                         int borderType=BORDER_DEFAULT );
```

+ **ddepth**：dst输出图像的深度。支持对应关系：

  + `src.depth()` = `CV_8U`, `ddepth` = -1/`CV_16S`/`CV_32F`/`CV_64F`
  + `src.depth()` = `CV_16U`/`CV_16S`, `ddepth` = -1/`CV_32F`/`CV_64F`
  + `src.depth()` = `CV_32F`, `ddepth` = -1/`CV_32F`/`CV_64F`
  + `src.depth()` = `CV_64F`, `ddepth` = -1/`CV_64F`

+ **dx** – 对x轴方向求导的阶数。

+ **dy** – order of the derivative y.

+ **ksize** – size of the extended Sobel kernel; it must be 1, 3, 5, or 7.

+ **scale** – optional scale factor for the computed derivative values; by default, no scaling is applied (see [`getDerivKernels()`](https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#void%20getDerivKernels(OutputArray%20kx,%20OutputArray%20ky,%20int%20dx,%20int%20dy,%20int%20ksize,%20bool%20normalize,%20int%20ktype)) for details).  

  比例常数，默认情况下没有伸缩系数 。

+ **delta** – optional delta value that is added to the results prior to storing them in `dst`. 

  可选增量， 将会加到最终的dst中。

+ **borderType** – pixel extrapolation(外推) method (see [`borderInterpolate()`](https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#int%20borderInterpolate(int%20p,%20int%20len,%20int%20borderType)) for details).

示例代码：

```c++
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
using namespace cv;
int main( int argc, char** argv )
{
  Mat src, src_gray;
  Mat grad;
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;

  int c;

  /// Load an image
  src = imread( argv[1] );
    
  if( !src.data )
  { return -1; }

  GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );
  /// 去噪后将其转化为灰度图
  cvtColor( src, src_gray, CV_BGR2GRAY );
  namedWindow( "sobel", CV_WINDOW_AUTOSIZE );
  /// Generate grad_x and grad_y
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;

  /// Gradient X
  //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
  Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_x, abs_grad_x );//We convert our partial results back to CV_8U
  /// Gradient Y
  //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
  Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_y, abs_grad_y );

  //computing (half of) L1 gradient: "abs(x) + abs(y)", instead of L2: "sqrt(x^2 + y^2)" 
  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

  imshow( "sobel", grad );
  waitKey(0);
  return 0;
  }
```

Most often, the function is called with ( `dx` = 1, `dy` = 0, `ksize` = 3) or ( dx= 0, dy= 1, `ksize` = 3) to calculate the first x- or y- image derivative. The first case corresponds to a kernel of: 

![\vecthreethree{-1}{0}{1}{-2}{0}{2}{-1}{0}{1}](assets/f2531c53069c2dabcab2bcb391518bd65dc535eb.png)

The second case corresponds to a kernel of: 

![\vecthreethree{-1}{-2}{-1}{0}{0}{0}{1}{2}{1}](assets/03e50d0ac972c69085ccbff5cadd0b53f791fce8.png)

There is also the special value `ksize = CV_SCHARR` (-1) that corresponds to the  3×3   Scharr filter that may give more accurate results ：

![\vecthreethree{-3}{0}{3}{-10}{0}{10}{-3}{0}{3}](assets/3ab98ff1a5283f63057e5f3ff52c25e49ef01318.png)

也可以在示例代码中去掉 Scharr 的注释。

### threshold

下面的代码应该解释的很清楚了

```c++
#include "opencv2/imgproc/imgproc.hpp"
double threshold( InputArray src, OutputArray dst,
                  double thresh, double maxval, int type );
```

```c++
/* Threshold types */
enum
{
    CV_THRESH_BINARY      =0,  /* value = value > threshold ? max_value : 0       */
    CV_THRESH_BINARY_INV  =1,  /* value = value > threshold ? 0 : max_value       */
    CV_THRESH_TRUNC       =2,  /* value = value > threshold ? threshold : value   */
    CV_THRESH_TOZERO      =3,  /* value = value > threshold ? value : 0           */
    CV_THRESH_TOZERO_INV  =4,  /* value = value > threshold ? 0 : value           */
    CV_THRESH_MASK        =7,  
    CV_THRESH_OTSU        =8  /* use Otsu algorithm to choose the optimal threshold value;
                                 combine the flag with one of the above CV_THRESH_* values */
};
```

![threshold.png](assets/threshold.png)

遗留问题：[How Otsu's Binarization Works?](https://docs.opencv.org/3.4.0/d7/d4d/tutorial_py_thresholding.html)。在页面的最下方。



