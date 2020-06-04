制造麻烦很容易，

**在智能指针出现以前，我们通常使用 new 和 delete 来管理动态分配的内存，但这种方式存在几个常见的问题：**

+ **忘记 delete 内存：**会导致内存泄漏问题，且除非是内存耗尽否则很难检测到这种错误。
+ **使用已经释放掉的对象：**如果能够记得在释放掉内存后将指针置空并在下次使用前判空，尚可避免这种错误。
+ **同一块内存释放两次：**如果有两个指针指向相同的动态分配对象，则很容易发生这种错误。
+ **发生异常时的内存泄漏：**若在 new 和 delete 之间发生异常，则会导致内存泄漏。

# 背景

资源，有借有还。

在C++中，定义在栈空间上的局部对象称为自动存储对象，创建和销毁工作是由系统自动完成的。 

把资源放进对象内，用对象来管理资源，便是 C++ 编程中最重要的编程技法之一，即 RAII ，它是 "Resource Acquisition Is Initialization" 的首字母缩写。智能指针便是利用 RAII 的技术对普通的指针进行封装，这使得智能指针实质是一个对象，行为表现的却像一个指针。

RAII的本质内容是用对象代表资源 。

## 



# 成员函数

```c++
#include <memory>

// http://www.cplusplus.com/reference/memory/shared_ptr/
// std::shared_ptr模板类
template <class T> class shared_ptr;


// shared_ptr::shared_ptr
// shared_ptr::~shared_ptr
std::shared_ptr<Person> p1; // The object is empty (owns no pointer, use count of zero).
std::shared_ptr<Person> p1(nullptr); // 上同

auto p2 = std::make_shared<Person>(2);
std::shared_ptr<Person> p3(new Person(3)); // p2 和 p3 指向了两个不同的Person对象

// shared_ptr::operator=
p1 = p2; //Person2 引用计数为2
p2 = p3; //Person2 引用计数变为1，Person3 的变为 2

//此时，p1 指向Person2，p2 p3 指向 Person3

//shared_ptr::operator*
std::cout << (*p1).value;
//shared_ptr::operator->
 std::cout << p2->value;
//shared_ptr::swap
p1.swap(p2); //p2->Person2，p1,p3 ->Person3

//shared_ptr::reset
//void reset() noexcept;
//template <class U> void reset (U* p);
// Person 3的引用计数为 1，p3 指向它
p1.reset();
p3.reset();// Person 3的引用计数为 0，被销毁

// shared_ptr::operator bool
if (!p2){}

//shared_ptr::get
//shared_ptr::owner_before
//shared_ptr::unique
//shared_ptr::use_count
```

# 手撕✍智能指针：

[谈谈智能指针：原理及其实现](https://liam.page/2018/01/13/smart-pointer/)

## 该有哪些操作

暂且不考虑模板类，也不考虑继承。针对Point类去构造智能指针对象。

```c++
class Point {
 private:
  int x_ = 0;                               // 1.
  int y_ = 0;

 public:
  Point() = default;                        // 2.
  Point(int x, int y) : x_{x}, y_{y} {}     // 3.

 public:
  int x() const { return x_; }              // 4.
  int y() const { return y_; }

  Point& x(int x) { x_ = x; return *this; } // 5.
  Point& y(int y) { y_ = y; return *this; }
};
//对于移动和拷贝构造函数及赋值运算符，则可由编译器自动合成。

SmartPointer {
public:
    SmartPointer(Point* pp);   //传入new 得到的指针
    SmartPointer(Point p);     //拷贝一份
    SmartPointer(int x, int y);//Point构造函数的参数
public:
    SmartPointer();
    SmartPointer(const SmartPointer& other);
    SmartPointer(SmartPointer&& other);
    SmartPointer& operator=(const SmartPointer& other);
    SmartPointer& operator=(SmartPointer&& other) noexcept;
    ~SmartPointer();
public:
    //模拟指针
 	Point* operator->() const noexcept;
 	Point& operator*() const noexcept;
    
  // ...
};
```

## 引用计数问题

问题：这个计数器不应该是 `SmartPointer` 对象的一部分。这是因为，如果计数器是 `SmartPointer` 的一部分，那么每次增减计数器的值，都必须广播到每一个管理着目标对象的智能指针。这样做的代价太大了。 

+ 简单的int指针
+ 计数辅助类： `PointCounter`， 来做计数器使用。`PointCounter` 类完全是为了实现 `SmartPointer` 而设计的，它不应被 `SmartPointer` 类以外的代码修改。因此，`PointCounter` 类的所有成员都应是 `private` 的，并声明 `SmartPointer` 为其友元类。 

```c++
class PointCounter {
private:
    friend class SmartPointer;

    Point p_;  //这里应该是对象成员吗? 还是指针?
    size_t count_;

    PointCounter() : count_{1} {}                         // 1.
    PointCounter(const Point& p) : p_{p}, count_{1} {}    // 2.
    PointCounter(Point&& p) : p_{p}, count_{1} {}         // 3.
    PointCounter(int x, int y) : p_{x, y}, count_{1} {}   // 4.
};
```

由于 `PointCounter` 是为了计数而设计的，因此 (1) 处的所有构造函数，在构造时都将计数器设置为 `1`。(2), (3) 两处则分别是调用 `Point` 类的拷贝和移动构造函数。(4) 则调用 `Point(int x, int y)`。 

## 第一版实现

```c++
class SmartPointer {
 public:
  SmartPointer(Point* pp) : ptctr_{new PointCounter(*pp)} {}    // 1 pp资源岂不是泄露了。。
  SmartPointer(const Point& p) : ptctr_{new PointCounter(p)} {}
  SmartPointer(Point&& p) : ptctr_{new PointCounter(p)} {}
  SmartPointer(int x, int y) : ptctr_{new PointCounter(x, y)} {}

 private:
  void try_decrease() {
    if (nullptr != ptctr_) {
      if (1 == ptctr_->count_) {
        delete ptctr_;
      } else {
        --(ptctr_->count_);
      }
    } else {}
  }

 public:
  SmartPointer() : ptctr_{new PointCounter} {}  // 2.其计数值也为 1?
    
  SmartPointer(const SmartPointer& other) : ptctr_{other.ptctr_} {      // 3.
    ++(ptctr_->count_);
  }
  SmartPointer(SmartPointer&& other) noexcept : ptctr_{other.ptctr_} {
    other.ptctr_ = nullptr;
  }
    
  SmartPointer& operator=(const SmartPointer& other) {                  // 4.
    try_decrease();
    ptctr_ = other.ptctr_;
    ++(ptctr_->count_);
    return *this;
  }
  SmartPointer& operator=(SmartPointer&& other) noexcept {
    try_decrease();
    ptctr_ = other.ptctr_;
    other.ptctr_ = nullptr;
    return *this;
  }
    
  ~SmartPointer() {
    try_decrease();
    ptctr_ = nullptr;
  }

 public:
  Point* operator->() const noexcept {                                  // 5.
    return &(ptctr_->p_);
  }
  Point& operator*() const noexcept {
    return ptctr_->p_;
  }

 private:
  PointCounter* ptctr_ = nullptr;
};
```

## 第二次尝试

第一版缺点：

+ 首先，`PointCounter` 类内包含了一个类型为 `Point` 的成员，从而禁止了动态绑定。 
+ 其次，从抽象的角度来说，`SmartPointer` 需要的计数器，与 `SmartPointer` 内绑定的对象的类型没有关系，因此不应该针对 `Point` 类构建一个 `PointCounter` 辅助类。 

## 引用计数器

内含size_t指针，目的也是能让多个智能指针可以共享计数

```c++
class RefCount {
private:
    size_t* count_ = nullptr;

private:
    void try_decrease() {
        if (nullptr != count_) {
            if (1 == *count_) {
                delete count_;
            } else {
                --(*count_);
            }
        } else {}
    }

 public:
    RefCount() : count_{new size_t(1)} {}
    RefCount(const RefCount& other) : count_{other.count_} { ++(*count_); }
    RefCount(RefCount&& other) : count_{other.count_} { other.count_ = nullptr; }
    RefCount& operator=(const RefCount& other) {
        try_decrease();
        count_ = other.count_;
        ++(*count_);
        return *this;
    }
    RefCount& operator=(RefCount&& other) {
        try_decrease();
        count_ = other.count_;
        other.count_ = nullptr;
        return *this;
    }
    ~RefCount() {
        try_decrease();
        count_ = nullptr;
    }

    // ...
};
```

回顾 `SmartPointer` 的析构函数，我们不难发现：为了在合适的实际销毁 `Point` 对象，我们必须有办法知道当前析构的 `SmartPointer` 是否为最后一个绑定在目标 `Point` 对象上的智能指针。因此，我们的 `RefCount` 类必须提供这样的接口。 

```c++
class RefCounter {
  // ...

public:
  	bool only() const { return (1 == *count_); }
};
```

## 引用计数器的智能指针

```c++
class SmartPointer {
public:
    SmartPointer(Point* pp) : point_{pp} {}                   // 1.
    SmartPointer(const Point& p) : point_{new Point(p)} {}
    SmartPointer(Point&& p) : point_{new Point{p}} {}
    SmartPointer(int x, int y) : point_{new Point{x, y}} {}

public:
    SmartPointer() : point_{new Point} {}
    SmartPointer(const SmartPointer& other) = default;        // 2.
    SmartPointer(SmartPointer&& other) noexcept : point_{other.point_}, refc_{std::move(other.refc_)} {
        other.point_ = nullptr;
    }
    SmartPointer& operator=(const SmartPointer& other) {      // 3.
        if (refc_.only()) {
            delete point_;
        }
        refc_  = other.refc_;
        point_ = other.point_;
        return *this;
    }
    SmartPointer& operator=(SmartPointer&& other) noexcept {
        if (refc_.only()) {
            delete point_;
        }
        refc_  = std::move(other.refc_);
        point_ = other.point_;
        other.point_ = nullptr;
        return *this;
    }
    ~SmartPointer() {                                         // 4.
        if (point_ && refc_.only()) {
            delete point_;
            point_ = nullptr;
        }
    }

public:
    Point* operator->() const noexcept {
        return point_;
    }
    Point& operator*() const noexcept {
        return *point_;
    }

private:
    Point* point_ = nullptr; //这回是指针了。第一版应该是写错了吧
    RefCount refc_;
};
```

## 模板

相比第一版，第二版的 `SmartPointer` 有了不少改进。但是，它有点名不副实——虽然类的名字是 `SmartPointer`，但是却只能和 `Point` 这一个类联用。为此，我们需要 C++ 中的模板技术。 

我们需要思考：

+ 构造函数 `SmartPointer(int x, int y)` 是专为 `Point` 类设计的。 作为参数的转发，如果作为模板类， `T` 的构造函数可以千奇百怪，于是若要将 `T` 的构造函数都在 `SmartPointer` 中做转发，就不得不变成 `template <typename T, typename... Args> class SmartPointer;`。这很不好； 

+ 另一方面，我们又希望能够保留从 `T` 的构造函数出发，直接构造 `SmartPointer` 的能力。为此，我们需要引入一个函数模板 `make_smart`。 

  ```c++
  template<typename T, typename... Args>
  SmartPointer<T> make_smart(Args&&... args);
  ```



# 最终实现

```c++
template<typename T>
class smart_ptr {
 public:
  using value_type = T;

 public:
  smart_ptr(value_type* pp) : value_{pp} {}
  smart_ptr(const value_type& p) : value_{new value_type(p)} {}
  smart_ptr(value_type&& p) : value_{new value_type{p}} {}

 public:
  smart_ptr() : value_{new value_type} {}
  smart_ptr(const smart_ptr& other) = default;
  smart_ptr(smart_ptr&& other) = default;
  smart_ptr& operator=(const smart_ptr& other) {
    if (refc_.only()) {
      delete value_;
    }
    refc_  = other.refc_;
    value_ = other.value_;
    return *this;
  }
  smart_ptr& operator=(smart_ptr&& other) noexcept {
    if (refc_.only()) {
      delete value_;
    }
    refc_  = std::move(other.refc_);
    value_ = other.value_;
    other.value_ = nullptr;
    return *this;
  }
  ~smart_ptr() {
    if (value_ and refc_.only()) {
      delete value_;
      value_ = nullptr;
    }
  }

 public:
  value_type* operator->() const noexcept {
    return value_;
  }
  value_type& operator*() const noexcept {
    return *value_;
  }

 private:
  value_type* value_ = nullptr;
  RefCount refc_;
};
```

# 实现make_smart函数模板

```c++
template<typename T, typename... Args>                     // 1.
smart_ptr<T> make_smart(Args&&... args) {                  // 2.
  return smart_ptr<T>(new T(std::forward<Args>(args)...)); // 3.
}
```

这里，(1) 处使用了 C++11 中名为「参数包」的技术，使得函数模板可以接收任意多个任意类型的参数；(2) 处对参数包进行解包，使用右值引用模式接受参数，借助「引用折叠」技术接收任意类型的参数；(3) 处使用了 `std::forward`，将接收到的参数原封不动地完美地转发给 `T` 的构造函数。 