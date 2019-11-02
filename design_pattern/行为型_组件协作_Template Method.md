## 模板方法

20191031

非常基础的设计模式（虚函数）。看下面的一段代码就懂了。

简而言之，模板方法定义了一个算法的步骤，允许子类为一个或多个步骤提供实现。

lib.cpp

```c++
//程序库开发人员
class Library{
public:
	//稳定 template method
    void Run(){
        Step1();

        if (Step2()) { //支持变化 ==> 虚函数的多态调用
            Step3(); 
        }
        
        for (int i = 0; i < 4; i++){
            Step4(); //支持变化 ==> 虚函数的多态调用
        }
        
        Step5();
    }
	virtual ~Library(){ }
protected:
	void Step1() { //稳定
        //.....
    }
	void Step3() {//稳定
        //.....
    }
	void Step5() { //稳定
		//.....
	}

	virtual bool Step2() = 0;//变化
    virtual void Step4() =0; //变化
};
```

app.cpp

```c++
//应用程序开发人员
class Application : public Library {
protected:
	virtual bool Step2(){
		//... 子类重写实现
    }

    virtual void Step4() {
		//... 子类重写实现
    }
};

int main()
	{
	    Library* pLib=new Application();
	    lib->Run();

		delete pLib;
	}
}
```



## 我用到过吗？

最近在写一个QT的GUI程序。

对Model/View的使用只是停留在模仿例子以及重写其方法。例如，QAbstractItemModel类为项目模型类提供抽象接口，有五个纯虚函数（`rowCount()`, `columnCount()`,  `data()`， `index()` and `parent()）`，子类必须要重写这几个函数。应用时知其然不知其所以然。今天看到模板方法模式，感觉这不就是模板方法吗？Model/View的运行的一套流程已经由QT定义好，用户只需要填充虚函数即可。程序在运行时调用用户的代码（晚绑定）。

我常用的二维表格Table类 QAbstractTableModel 文档中有一段话：

> When subclassing QAbstractTableModel, you must implement rowCount(), columnCount(), and data(). Default implementations of the index() and parent() functions are provided by QAbstractTableModel. Well behaved models will also implement headerData().
>
> Editable models need to implement setData(), and implement flags() to return a value containing Qt::ItemIsEditable.

与模板方法一起理解，大致意思就是“我QT库已经把该写的步骤写好了，但是有些步骤需要你们用户自己编写，你们必须写哦”。

至于这些用户自己实现的虚函数用到了哪里，用在什么地方，什么时候被调用，不看源码是无从知晓了。不过可以猜测：比如`rowCount(), columnCount(), and data()`函数可以给View来调用，`setData()`可以给delegates来调用。在文档中Model/View Programming一章Model Subclassing Reference小节里的内容给出了分组，将这些函数分成三组。更具体的分组请查看文档或者网络上搜索。

>The functions that need to be implemented in a model subclass can be divided into three groups:
>
>+ Item data handling: All models need to implement functions to enable views and delegates to query the dimensions of the model, examine items, and retrieve data.
>+ Navigation and index creation: Hierarchical models need to provide functions that views can call to navigate the tree-like structures they expose, and obtain model indexes for items.
>+ Drag and drop support and MIME type handling: Models inherit functions that control the way that internal and external drag and drop operations are performed. These functions allow items of data to be described in terms of MIME types that other components and applications can understand.



## 钩子方法



钩子方法可以作为条件控制，影响抽象类中的算法流程。基类中有默认的实现，子类也可以覆盖它。

感觉Model/View中的`flags()`方法就是一个钩子方法，控制着一些开关。

## 其他的实际应用

排序算法：你需要提供一个比较算法。





