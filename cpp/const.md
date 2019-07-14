# const

+ 值不能改变，任何赋值操作将引发错误。
+ const对象必须初始化
+ 经常用的是const引用：对常量的引用。
+ 指向常量的指针：“自以为是”指向了常量
+ 常量指针

`const double *const px = &x`，在这条语句中，靠右的是顶层，靠左的是底层：

+ 顶层const表示指针本身是个常量，也可以表示任意对象是常量：

  ```c++
  int i = 0;
  int *const p1 = &i;
  const int ci = 42;
  ```

+ 底层const表示指针所指向的对象是个常量

+ 执行对象的拷贝操作时，是否时顶层const无所谓，要看底层const。

常量指针：int* const p = &a; 指的是指针是常量，也就是指针的值不能变了。

指向常量的指针：const int* p = &a；a是常量，通过指针不能修改a的值。



## 智能指针const

我惊呆了：

起因：

```c++
using EpochInfoPtr = std::shared_ptr<EpochInfoEntity>;
using EpochInfoConstPtr = std::shared_ptr<const EpochInfoEntity>;

EpochInfoConstPtr TrainConsoleWidget::updateEpochInfo(const EpochInfoEntity& epochInfo)
{
	EpochInfoPtr epoch = std::make_shared<EpochInfoEntity>(epochInfo);
    auto pairRes = epochList.insert(std::pair<int,EpochInfoPtr>(epoch->getEpochId(),epoch));
}
```

形参中的`epochInfo`是常量引用。然后可以用指向非常量的智能指针指向它？？？？

我们知道一般的指针是不可以的！

```c++
const int a = 5;
int *p = &a;    //error
const int* p = &a; //ok
```

但是智能指针可以唉：

```c++
class Test {
public:
	int getA() const { return a; }
	void setA(int a) { this->a = a; }
private:
	int a;
};

int main()
{
    Test t;
	t.setA(666);
	const Test & tt = t;
    
    std::shared_ptr<Test> pa = std::make_shared<Test>(tt);
    //std::shared_ptr<Test> pa = std::make_shared<Test>(t);
	pa->setA(999);
    
	std::cout << t.getA() << std::endl; //666
	std::cout << tt.getA() << std::endl;//666
	std::cout << pa->getA() << std::endl;//999
	
}
```

根据结果可以推断：智能指针根本没有指向原来的对象。。。。只是赋值了一份？？？？？

所以：

```c++
//创建智能指针并不是像：
int *p = &a;
//而是
int *p = new int;
```

所以，最开始的创建智能指针：

```c++
EpochInfoPtr epoch = std::make_shared<EpochInfoEntity>(epochInfo);
//类似
EpochInfoEntity * epoch = new EpochInfoEntity(epochInfo);//调用构造函数，创建了一个新对象
```



蠢B。

