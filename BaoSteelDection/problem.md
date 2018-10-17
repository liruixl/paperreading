0是黑，255是白

# 变量

```c++
Mat m_cut; //在原图上裁剪之后的图
Mat m_oricla = m_cut.clone();
Mat m_black；//threshold(m_cut,m_black,over_mean,255,CV_THRESH_BINARY_INV);x小于mean的都转白色255
Mat orgin_cut = m_cut.clone();
//m_cut被光照均衡
Mat m_ori = m_cut.clone();
sal_img = m_cut.clone();//用于计算边缘图
Mat m_img = m_cut.clone();//求图像平均梯度

Mat testgau;//存储合并图像
```



# 裁边

1. 二值化的阈值怎么定？
2. 

```c++
height = endY-beginY+1;
	if (height & 1)
	{
		height += 1; 
	}
```

3. 

```c++
if((beginX+1) / 20 *20 == (beginX+1))
```

4. 64

```c++
long maxcondtion = maxpixel/64;//daijijing{0,64}
	if (maxcondtion < 100)
	{
		return false;
	}
```

5. 规格*7？  实际长度转化为像素数。水平方向：实际/0.14=像素数，垂直：实际高度/0.5=像素数

```c++
float guige = 73;//img->sDou;像素数
float length = guige * 7;//用钢管规格*7，用来判断裁边
```

6. 2怎么取得

```c++
for (up_beginX = 1; up_beginX < up_pixels.size(); up_beginX++)
{
        if (up_pixels[up_beginX] - up_pixels[up_beginX - 1]> 2 /*&& 20 * (pos - up_beginX) < length*/)
        {
            break;
        }
}
```

7. 让width变为偶数

```c++
if (width & 1)
	{
		width += 1; 
	}
```

8. cut_ratio  经验值

```c++
for (beginX = 1; beginX < pixels.size(); beginX++)
	{
		if (pixels[beginX] - pixels[beginX - 1]> cut_ratio * maxpixel && 20 * (pos - beginX) < length*1.1 /*&& gray_pixels[beginX] > 0.1 * maxgray*/)
		{
			break;
		}
	}
```

# 预处理

1. 投影，光照均衡？相当于128是阈值，高于128减一点，低于128的加一点

```c++
project_Img(int width,int height,unsigned char* pData){
    for (int i = 0; i < width; i++)
	{
		float sum_t = 0;
		int	count_t = 0;
		for (int k = i - 9; k < i + 9; k++)
		{
			if (k >= 0 && k < width)
			{
				sum_t += projection[k];
				count_t++;
			}
		}
		float res = (float)sum_t/count_t;
		scale[i] = 128.0f / res;
	}
}
```

2. 干净度怎么定义

   ```c++
   if(imageAvG < 2.5 && sobel1_Y < 2.3)//1.35 ->2.5 
   { 
       paramIndex=0;
   }
   //imageAvG>3.26 ||sobel1_Y >4.19       imageAvG >4 && sobel1_Y >3.85
   else if((imageAvG >3 && sobel1_Y >4.19 &&sobel1_X >3.5)||(imageAvG >4.5 && sobel1_Y >3.85  &&sobel1_X >3.4))
   {
       paramIndex=2;
   }
   else
   {
       paramIndex=1;
   }
   ```

   

# 干净度三算法

干净度是通过，求图像的平均梯度imageAvGdouble sobel1_Y = mean(grad_y)[0];double sobel1_X = mean(grad_x)[0];  确定的

1410-2410=1000行

```c++
void operator_clean3(Mat& m_ori,     //m_ori(m_cut):光照均衡图
                     Mat& orgin_cut, //orgin_cut(m_oricla):裁边图
                     Mat& m_oricla,  //裁边图Mat m_oricla = m_cut.clone();
                     Mat& m_black,   //m_black:裁边后的二值图
                     Mat& m_cut,     //最后光照均衡修改project_Img(width,height,m_cut.data)
                     Mat& testgau,   //边缘图
                     Mat& sal_edge,  //sal_edge:去小点图
                     Mat& cannyarea, //未定义 Mat cannyarea;
                     double imageAvG,//原裁边图平均梯度
                     DPUParams params, 
                     int height, 
                     int width, 
                     GlobalEntity::DefectObjectList& tmpObjectList){...}

Mat& m_ori,     //m_ori(m_cut):光照均衡图
Mat& orgin_cut, //orgin_cut(m_oricla):裁边图
Mat& m_oricla,  //裁边图Mat m_oricla = m_cut.clone();
Mat& m_black,   //m_black:裁边后的二值图
Mat& m_cut,     //最后光照均衡修改project_Img(width,height,m_cut.data)
Mat& testgau,   //边缘图
Mat& sal_edge,  //sal_edge:去小点图
Mat& cannyarea,


mean_gobel;//裁剪图平均灰度
```

```c++
typedef struct defectRecord
{
	int lu_x;    //left up   1000 000
	int lu_y;    
	int rd_x;    //right down  0
	int rd_y;
	int pixelNum;
}defectRecord;
```



1. ？？？

```c++
int conutSum = 0,countB = 0,countH = 0;
```

1684行

```c++
double mean_roi = mean(orgin_cut(r_max))[0];
		if(mean_gobel > 58)
			minMaxLoc(orgin_cut(r_max),&gray_min,&gray_max);
		else
			minMaxLoc(m_ori(r_max),&gray_min,&gray_max);

```

这都表示什么

```c++
int black_mean = 0,cut_mean = 0;
int writeSum = 0, blackSum = 0;//缺陷框中大于某值或小于某值的像素点个数
int orgin_sum = 0, orgin_num = 0;//边缘图中边缘区域 对应在原图中的像素点个数
int unorgin_sum = 0, unorgin_num = 0;//边缘图中非边缘区域 对应在原图中的像素点个数
```



2. 1738

```c++
vsun = vsun + tmp_edge.data[ii * width + jj];//其中tmp_edge为未经过膨胀操作的原边缘图
```

3. 1478

```c++
int edge_ld = 10000, edge_rd = 0;
edge_ld = edge_ld < jj ? edge_ld : jj;
edge_rd = edge_rd > jj ? edge_rd : jj;
```

4. 1798朝后

```c++
int vsun = 0;//vsun代表边缘图中每一行缺陷的像素点个数
vsun = vsun + tmp_edge.data[ii * width + jj];//其中tmp_edge为未经过膨胀操作的原边缘图
vsun = vsun /255 ;   //  0/255，统计结果应该是像素值为255像素点的个数
proX.push_back(vsun);//每一行像素点个数
```

5. 1772？更新坐标？？   m_black到底是什么

```c++
if (m_black.data[ii * width + jj] != 0 )//b_rs b_re b_cs b_ce为黑斑的左上角右下角坐标
{
    if (ii < b_rs)
    {
        b_re = ii;
    }
    if (ii > b_re)
    {
        b_rs = ii;
    }
    if (jj < b_cs)
    {
        b_ce = jj;
    }
    if (jj > b_ce)
    {
        b_cs = jj;
    }
    countB++;//代表二值化之后图像中白点个数，认为count表示黑斑像素点数
    black_mean += m_cut.data[ii * width + jj];
}
```

6. 2177

```c++
//去除边缘线 2018/7/8 是什么
```

7. 如何合并? 矩形框重叠的算合并

```c++
////新增 用作合并
unsigned char* cluster = new unsigned char[width*height]();
```

