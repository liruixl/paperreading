# 20190429

## 1、重写调用运算符，实现隐式转化：

`EDefect::Defect defect = (*defectPair.second)();`

需求：当从本地实体类转化到服务器实体类的时候，调用。

```c++
class DefectEntity : private EDefect::Defect //私有继承后不能直接隐式转化。
{
	EDefect::Defect operator()();
}

EDefect::Defect DefectEntity::operator()()
{
	EDefect::Defect defect;

	defect.id =				this->id;
	defect.coilId =			this->coilId;
	defect.defectId =		this->defectId;
	
    return defect;
}

```



# 20190507

## 1、QT中，单击也触发了mouseMoveEvent函数？？？

有时候正常：

鼠标操作：移动移动单击单击：

```c++
2.moveEvent
=====mouseEvent and not down
2.moveEvent
=====mouseEvent and not down
1.pressEvent
3.release
release after:
181 132 230 145
1.pressEvent
3.release
release after:
181 132 230 145
```

有时候不正常，单击也触发move事件，导致画出宽高为0的矩形，打印如下：

```c++
1.pressEvent
2.moveEvent
move event and isLbtnDown
movePoint:
106 330
3.release
release after:
106 330 0 0
```

#### 解决办法：

1、`if(event->buttons()&Qt::LeftButton) `，鼠标左键按下的同时移动鼠标。不起作用

2、如果宽高都是0，就是无效操作。采用

```c++
pathfinderRect.setRect(pressPoint.x(), pressPoint.y(), moveW, moveH);
if (!pathfinderRect.isNull())
{
    //操作
}
```

3、判断movingPoint以及pressPoint是不是一个点。是就没移动。

```c++
int moveW = movingPoint.x() - pressPoint.x();
		int moveH = movingPoint.y() - pressPoint.y();

		if (moveW == 0 && moveH == 0)
		{
			qDebug() << "00000000000";
			return;
		}
```

输出：

```
1.pressEvent
2.moveEvent================================
move event and ===============isLbtnDown
00000000000
3.release
```

经过尝试：确实是没移动就触发了moveEvent。

怎么发生的这个问题：

+ 快速移动鼠标然后单击。
+ 点击其他应用程序后，回来再单击。

## 2、矩形右侧靠边界时不能显示

判断矩形是否在图像label内部：

```c++
bool ImageLabel::rectInLabel(const QRect & rect)
{
	QRect labelRect(0, 0, size().width(), size().height());
	return labelRect.contains(rect, false);
}
```

由于渲染原因，当矩形靠近图像右侧和底部，判断为ture，但是右侧和底部的渲染不能显示。

![1557280605135](assets/1557280605135.png)

解决方法：

1、绘制为2个像素，应该可以

![1557280701781](assets/1557280701781.png)

2、`QRect labelRect(0, 0, size().width()-1, size().height()-1);` 这样的话，实际上矩形的右边界不能与Label边界重合（相差一像素）。上、左边界可以重合：都为0。

# 20190508

## 1、阅读了QRect文档，一些点值得注意。

![1557282252114](assets/1557282252114.png)

+ 渲染问题。

+ set函数改变一些东西  move函数不改变宽高。

+ The `isEmpty()` function returns true if left() > right() or top() > bottom(). Note that an empty rectangle is not valid: 

  The `isValid()` function returns true if left() <= right() and top() <= bottom(). 

  A null rectangle (`isNull() == true`) on the other hand, has both width and height set to 0. 就是right() == left() - 1 and bottom() == top() - 1). A null rectangle is also empty, and hence is not valid.

+ translate()、translated()、adjust、adjusted、normalized函数功能。

  平移，调整，纠正宽高为负的矩形。

## 2、绘制矩形注意，已知起点、终点

例如方格表示像素，起点和终点已给出，注意如何计算宽高，以及图像如何渲染。

![1557283393315](assets/1557283393315.png)

# 20190509

1、双缓冲需要了解（没有用到）

2、删除矩形框（删除按钮）

实际删除的是缺陷

# 20190510

1、Qt的坐标系统

## 2、放大功能一些想法

+ QPainter的

+ 放大QPainter坐标系统QPainter::scale() 。重新用QPainter绘制时就按照新的坐标系统绘制。

  就本功能而言，当Label放大时，放大QPainter的坐标系统来绘制矩形框。

  当新建矩形框时，单击得到的坐标是放大后Label的，而不是正常尺寸下的，所以新建的矩形框的时候，坐标也要转化。

+ resize控件本身，绘制矩形框时按照正常尺寸和伸缩因子重新计算坐标。

# 20190520

1、label.setPixmap() 仍然不显示图像！！！

+ 本想作为调试程序的setText()会导致设置的Pixmap为空。导致label大小异常，本应该与图像一样大，adjustSize()来调整尺寸去适应内容。
+ 删除setText()后还是没有显示图像，此时pixmap()不为空，但是没有显示图像。

## 2、**卧槽 我忘了调用父类QLabel的paint事件**。

```c++
void ImageLabel::paintEvent(QPaintEvent * event)
{
	QLabel::paintEvent(event);//!!!!!!!!!!!!!!!
    
	QPainter painter(this);
	painter.setPen(QPen(Qt::red, 2));
	for (auto &defectRect : defectRectList)
	{
		painter.drawRect(defectRect.rect);
	}
}
```

3、增加编辑、修改、新建开关。为什么要增加开关：防止用户错误操作。

# 20190524

## 1、返回局部变量指针

```c++
std::vector<DefectPtr> ActionManager::searchDefectbyTime(long long startTime, long long endTime)
{
	auto items = defectMock.searchDefectbyTime(startTime, startTime); //server interface

	std::vector<DefectPtr> datas;
	for (auto & item : items) 
		datas.emplace_back(std::make_shared<DefectEntity>(item));

	return datas;
}
```

上述例子，先从服务器查询缺陷列表，然后将每个缺陷的智能指针，保存到vector中。返回这个vector。

我们知道，不能返回局部变量的指针。**为什么这里返回了？？？？？**

# 20190526

## 1、多缺陷框模拟

缺陷与图像：多对一。一个缺陷对应一个图像，一个图像对应多个缺陷。

主要是数据。

## 2、样本空间列表UI

未完成，怎样设计？可按照类别查看。

# 20190527

## 1、DefectEntity与DefectSample如何协调

由于之前都使用的是DefectEntity，有许多样本并不关心的信息。对于新建缺陷样本，以及要提交到样本空间的缺陷样本信息，需要的是DefectSample的类。

还是在服务器一查询下来就变为DefectSample类。

# 20190530

## 1、模型Index创建

在手动编辑缺陷框后，希望利用proxy模型的setData()函数来修改底层数据，这样也可以像修改类别一样将用户编辑的item展示为红色。

20190601 发现，protected成员函数不能在其他类MainGui中访问！！！！！额

setData中咋创建这个index呢。

```c++
QModelIndex topLeft = this->index(0, getColumnOfHeader("classType"));//QAbstractTableModel
```



## 2、QTabelView切换行

默认响应单击：

```c++
ui.classifyDefectTableView->scrollTo(proxyIndex);
ui.classifyDefectTableView->selectRow(proxyIndex.row());//这个是
//emit ui.classifyDefectTableView->clicked(proxyIndex);
```

# 20190601

## 1、this不能在常量表达式中使用？

```c++
switch (index.column())
{
    case getNum(): //error
        
        break;
}
```

## 2、protected成员函数不可访问？

内部类提供接口调用保护方法。

## 3、QLabel没有单击事件clicked

## 4、将新建Rect改为只能新建一个

用布尔变量变量`isNewStatus`控制，新建状态下（newEnable==true），当绘制新矩形时，通过isNewStatus来控制在真正拖动鼠标新建时添加 新的Rect。而不是点击新建按钮时就创建新的Rect，因为用户可能不去新建 。。

```c++
if (!isNewStatus)
{
    //添加时id怎么计算,置为负数？？
    DefectSample newD = DefectSample(-defectRectList.size(), 15,
                                     selectDefectRect->getImageLabel(), newrect);
    defectRectList.emplace_back(newD);
    selectDefectRect = &defectRectList.at(defectRectList.size() - 1);
    isNewStatus = true;
    qDebug() << "===============start draw";
}
else
{
    //selectDefectRect->rect = newrect;
    selectDefectRect->setRect(newrect);
}
```

当用户点击提交时，再将isNewStatus置为false。

# 20190602

# 1、新建缺陷的问题

Id及类别Id在哪里初始化？？

暂时考虑id为负数，类别由ImageWidget提供。

# 21090603

## 1、线程池，模板实现

模板：

+ 非类型模板参数的模板实参必须是常量表达式
+ 模板直到实例化才会生成代码
+ ？与非模板函数不同，**为了生成一个实例化版本**，编译器需要掌握函数模板或类模板成员函数的定义。因此，与非模板代码相同，模板的头文件通常既包括声明也包括定义。



+ 默认情况下，对于一个已经实例化的类模板，其成员只有在使用时才被实例化。这一特性是的即使某种类型不能完全符合模板操作的要求，我们仍能使用该模板。
+ 类模板不是类型！！



+ static 成员，T::A
+ 类型成员，T::A



+ 声明与定义
+ 实例化声明与实例化定义

线程池中用到的：

+ 函数模板
+ 右值引用参数
+ 参数转发
+ 函数参数包
+ 后置返回类型
+ future-promise特性
+ thread取任务
+ queue任务队列

# 20190304

## 1、map的插入操作

+ insert，有则什么也不做
+ 下标操作，返回一个左值，可读写。没有则新建

