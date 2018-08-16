# Class

语法：

```c++
class class_name {
  access_specifier_1:
    member1;
  access_specifier_2:
    member2;
  ...
} object_names;
```

> Classes have the same format as plain *data structures*, except that they can also include **functions** and have these new things called ***access specifiers***. An *access specifier* is one of the following three keywords: `private`, `public` or `protected`. These specifiers modify the access rights for the members that follow them: 

+ `private` members of a class are accessible only from within other members of the same class (or from their *"friends"*).
+ `protected` members are accessible from other members of the same class (or from their *"friends"*), but also from members of their derived(派生) classes.
+ Finally, `public` members are accessible from anywhere where the object is visible.

举个栗子：

```c++
// example: one class, two objects
#include <iostream>
using namespace std;

class Rectangle {
    int width, height;
  public:
    void set_values (int,int);
    int area () {return width*height;}
};

void Rectangle::set_values (int x, int y) {
  width = x;
  height = y;
}

int main () {
  Rectangle rect, rectb;
  rect.set_values (3,4);
  rectb.set_values (5,6);
  cout << "rect area: " << rect.area() << endl;
  cout << "rectb area: " << rectb.area() << endl;
  return 0;
}
```

## Constructors(构造函数)

+ 名字与类名相同
+ 无返回，甚至是void
+ they simply initialize the object

```c++
class Rectangle {
    int width, height;
  public:
    Rectangle (int,int);  //构造函数
    int area () {return (width*height);}
};

Rectangle::Rectangle (int a, int b) {
  width = a;
  height = b;
}
```

## Overloading constructors

+ 不同类型的参数
+ 不同个数的参数
+ 前两者混合

```c++
class Rectangle {
    int width, height;
  public:
    Rectangle ();  // default constructor 
    //The compiler will automatically call the one whose parameters match the arguments
    Rectangle (int,int);
    int area (void) {return (width*height);}
};

Rectangle::Rectangle () {
  width = 5;
  height = 5;
}

Rectangle::Rectangle (int a, int b) {
  width = a;
  height = b;
}

//调用默认构造函数
Rectangle rectb;   // ok, default constructor called
Rectangle rectc(); // oops, default constructor NOT called 
// It would be a function that takes no arguments and returns a value of type Rectangle.
```

## Uniform initialization

 four ways to construct objects of a class whose constructor takes a single parameter: 

```c++
#include <iostream>
using namespace std;

class Circle {
    double radius;
  public:
    Circle(double r) { radius = r; }
    double circum() {return 2*radius*3.14159265;}
};

int main () {
  Circle foo (10.0);   // functional form
  Circle bar = 20.0;   // assignment init.
  Circle baz {30.0};   // uniform init.........
  Circle qux = {40.0}; // POD-like

  cout << "foo's circumference: " << foo.circum() << '\n';
  return 0;
}
```

uniform声明不像大括号，无参数时是函数声明：

```c++
Rectangle rectc(); // function declaration (default constructor NOT called)
Rectangle rectd{}; // default constructor called 
```



## Pointers to classes

```c++
#include <iostream>
using namespace std;

class Rectangle {
  int width, height;
public:
  Rectangle(int x, int y) : width(x), height(y) {}
  int area(void) { return width * height; }
};


int main() {
  Rectangle obj (3, 4);
  Rectangle * foo, * bar, * baz;  //定义指针
  foo = &obj;
  bar = new Rectangle (5, 6);
  baz = new Rectangle[2] { {2,5}, {3,6} };  //定义类数组。。。。
  cout << "obj's area: " << obj.area() << '\n';
  cout << "*foo's area: " << foo->area() << '\n';  //指针访问方式。
  cout << "*bar's area: " << bar->area() << '\n';
  cout << "baz[0]'s area:" << baz[0].area() << '\n';
  cout << "baz[1]'s area:" << baz[1].area() << '\n';       
  delete bar;
  delete[] baz;
  return 0;
}	
```

指针访问的一些表达：

| expression | can be read as                                               |
| ---------- | ------------------------------------------------------------ |
| `*x`       | pointed to by `x`                                            |
| `&x`       | address of `x`                                               |
| `x.y`      | member `y` of object `x`                                     |
| `x->y`     | member `y` of object pointed to by `x`                       |
| `(*x).y`   | member `y` of object pointed to by `x` (equivalent to the previous one) |
| `x[0]`     | first object pointed to by `x`                               |
| `x[1]`     | second object pointed to by `x`                              |
| `x[n]`     | (`n+1`)th object pointed to by `x`                           |

## Classes defined with struct and union

+ struct

  The only difference between both is that members of classes declared with the keyword `struct` have `public` access by default, while members of classes declared with the keyword `class` have `private` access by default. For all other purposes both keywords are equivalent in this context. 

+ union

  Conversely, the concept of *unions* is different from that of classes declared with `struct` and `class`, since unions only store one data member at a time, but nevertheless they are also classes and can thus also hold member functions. The default access in union classes is `public`. 

## Overloading operators(重载运算符)

```c++
class CVector {
  public:
    int x,y;
    CVector () {};
    CVector (int a,int b) : x(a), y(b) {}
    CVector operator + (const CVector&);
};

//作为成员函数加载
CVector CVector::operator+ (const CVector& param) {  //int& a,引用
  CVector temp;
  temp.x = x + param.x;
  temp.y = y + param.y;
  return temp;
}

//调用时
c = a + b;
c = a.operator+ (b);//直接带哦用函数
```



指针和引用的声明方式： 声明指针： char\* pc; 声明引用： char c = 'A' ;char& rc = c;  它们的区别：

①从现象上看，指针在运行时可以改变其所指向的值，而引用一旦和某个对象绑定后就不再改变。这句话可以理解为：指针可以被重新赋值以指向另一个不同的对象。但是**引用则总是指向在初始化时被指定的对象，以后不能改变，但是指定的对象其内容可以改变。** 

② 从内存分配上看，程序为指针变量分配内存区域，而不为引用分配内存区域，因为引用声明时必须初始化，从而指向一个已经存在的对象。引用不能指向空值。 

③ 从编译上看，程序在编译时分别将指针和引用添加到符号表上，符号表上记录的是变量名及变量所对应地址。指针变量在符号表上对应的地址值为指针变量的地址值，而引用在符号表上对应的地址值为引用对象的地址值。符号表生成后就不会再改，因此指针可以改变指向的对象（指针变量中的值可以改），而引用对象不能改。这是使用指针不安全而使用引用安全的主要原因。从某种意义上来说引用可以被认为是不能改变的指针。 

④不存在指向空值的引用这个事实意味着使用引用的代码效率比使用指针的要高。因为在使用引用之前不需要测试它的合法性。相反，指针则应该总是被测试，防止其为空。 

⑤理论上，对于指针的级数没有限制，但是引用只能是一级。 

## The keyword this

> The keyword `this` represents **a pointer** to the object whose member function is being executed. It is used within a class's member function to refer to the object itself. 

+ 可以检查传成员函数的参数是不是object自己。例如：

```c++
class Dummy {
  public:
    bool isitme (Dummy& param);
};

bool Dummy::isitme (Dummy& param)
{
  if (&param == this) return true;
  else return false;
}
```

+ It is also frequently used in `operator=` member functions that return objects by reference. 

  ```c++
  CVector& CVector::operator= (const CVector& param)
  {
    x=param.x;
    y=param.y;
    return *this;
  }
  ```

## Static members

>  A static data member of a class is also known as a "class variable" 

所有此类对象之间共享。

IDEA：对象计数器，每场建一个类就可以在构造函数中使其+1。

```c++
class Dummy {
  public:
    static int n;
    Dummy () { n++; };
};
```

 访问：

```c++
cout << a.n;
cout << Dummy::n;
```

## Const member functions (orz)

const可以让变量只读：一个const对象`const MyClass myobject;`，就好似对与来自外部的访问它的每个数据成员都是const。Note though, that the constructor is still called and is allowed to initialize and modify these data members.

const对象只能调用被明确声明为`const` 的member functions。

const member functions的声明：the `const` keyword shall follow the function prototype, after the closing parenthesis for its parameters ，像`int get() const {return x;}`

const也可以限定(qualify )成员函数返回类型：

```c++
int get() const {return x;}        // const member function
const int& get() {return x;}       // member function returning a const&=const int&
const int& get() const {return x;} // const member function returning a const& 
```

搞不清楚const加上指针再加上引用，Types in C++ are read right to left ：

+ `int* const` is a `const` pointer to a [non-`const`] `int`
+ `int const*` is a [`non-const`] pointer to a `const int`
+ `int const* const` is a `const` pointer to a `const int`

const&是什么东西？？？？

+ const member function性质：
  + cannot modify non-static data members 
  +  cnnnot call other non-`const` member functions 
  + 实质是：`const` members shall not modify the state of an object. 注意是对象(object)的状态哦！
+ const object性质：
  + limited to access only member functions marked as `const`
  +  but non-`const` objects are not restricted and thus can access both `const` and non-`const` member functions alike. 

作用：

> Most functions taking classes as parameters actually **take them by `const` reference**, and thus, these functions can only access their `const`members: 

```c++
#include <iostream>
using namespace std;

class MyClass {
    int x;
  public:
    MyClass(int val) : x(val) {}       //构造函数
    const int& get() const {return x;}  //返回x的const reference
};

void print (const MyClass& arg) {     //传类的const reference
  cout << arg.get() << '\n';  
}

int main() {
  MyClass foo (10);
  print(foo);

  return 0;
}
```

## Class templates

例1：

```c++
template <class T>
class mypair {
    T values [2];
  public:
    mypair (T first, T second)    //构造函数作为成员函数
    {
      values[0]=first; values[1]=second;
    }
};
//初始化对象
mypair<int> myobject (115, 36);
```

例2：

```c++
#include <iostream>
using namespace std;

template <class T>
class mypair {
    T a, b;
  public:
    mypair (T first, T second)
      {a=first; b=second;}
    T getmax ();
};

template <class T>
T mypair<T>::getmax ()   //成员函数定义在类外！要有template <...> prefix
{
  T retval;
  retval = a>b? a : b;
  return retval;
}

int main () {
  mypair <int> myobject (100, 75);
  cout << myobject.getmax();
  return 0;
}
```

## Template specialization

```c++
// template specialization
#include <iostream>
using namespace std;

// class template:
template <class T>
class mycontainer {
    T element;
  public:
    mycontainer (T arg) {element=arg;}
    T increase () {return ++element;}
};

// class template specialization:
template <>     
class mycontainer <char> {    //the syntax used for the class template specialization
    char element;
  public:
    mycontainer (char arg) {element=arg;}
    char uppercase ()
    {
      if ((element>='a')&&(element<='z'))
      element+='A'-'a';
      return element;
    }
};

int main () {
  mycontainer<int> myint (7);
  mycontainer<char> mychar ('j');
  cout << myint.increase() << endl;
  cout << mychar.uppercase() << endl;
  return 0;
}
```

# Special members

> Special member functions are member functions that are implicitly defined as member of classes under certain circumstances. There are six: 

| Member function                                              | typical form for class `C`: |
| ------------------------------------------------------------ | --------------------------- |
| [Default constructor](http://www.cplusplus.com/doc/tutorial/classes2/#default_constructor) | `C::C();`                   |
| [Destructor](http://www.cplusplus.com/doc/tutorial/classes2/#destructor) | `C::~C();`                  |
| [Copy constructor](http://www.cplusplus.com/doc/tutorial/classes2/#copy_constructor) | `C::C (const C&);`          |
| [Copy assignment](http://www.cplusplus.com/doc/tutorial/classes2/#copy_assignment) | `C& operator= (const C&);`  |
| [Move constructor](http://www.cplusplus.com/doc/tutorial/classes2/#move) | `C::C (C&&);`               |
| [Move assignment](http://www.cplusplus.com/doc/tutorial/classes2/#move) | `C& operator= (C&&);`       |

以后再看！

# Friendship and inheritance

## Friend functions

> A **non-member function** can access the private and protected members of a class if it is declared a *friend* of that class.  

```c++
// friend functions
#include <iostream>
using namespace std;

class Rectangle {
    int width, height;
  public:
    Rectangle() {}
    Rectangle (int x, int y) : width(x), height(y) {}
    int area() {return width * height;}
    friend Rectangle duplicate (const Rectangle&);  //声明哪一锅是友元函数,而不是成员函数
};

Rectangle duplicate (const Rectangle& param)  //友元函数实现
{
  Rectangle res;
  res.width = param.width*2;
  res.height = param.height*2;
  return res;
}

int main () {
  Rectangle foo;
  Rectangle bar (2,3);
  foo = duplicate (bar);  //可以访问bar的私有成员width、height
  cout << foo.area() << '\n';
  return 0;
}
```

## Friend classes

> A friend class is a class whose members have access to the private or protected members of another class.
>
> Another property of friendships is that they are not transitive: The friend of a friend is not considered a friend unless explicitly specified. 

```c++
#include <iostream>
using namespace std;

class Square;   //Rectangle类用到，必须提前申明声明。两个类互相用到

class Rectangle {
    int width, height;
  public:
    int area ()
      {return (width * height);}
    void convert (Square a);
};

class Square {
  friend class Rectangle;   //声明矩形类是我的友元类
  private:
    int side;
  public:
    Square (int a) : side(a) {}
};

void Rectangle::convert (Square a) {
  width = a.side;
  height = a.side;
}
  
int main () {
  Rectangle rect;
  Square sqr (4);
  rect.convert(sqr);  //那么矩形类就可以访问Square类里的私有成员
  cout << rect.area();
  return 0;
}
```

## Inheritance between classes

```c++
class Polygon {
  protected:
    int width, height;
  public:
    void set_values (int a, int b)
      { width=a; height=b;}
 };

class Rectangle: public Polygon {    //注意一下语法，在基类前有public
  public:
    int area ()
      { return width * height; }
 };

class Triangle: public Polygon {
  public:
    int area ()
      { return width * height / 2; }
  };
```

继承了哪些：

| Access                    | `public` | `protected` | `private` |
| ------------------------- | -------- | ----------- | --------- |
| members of the same class | yes      | yes         | yes       |
| members of derived class  | yes      | yes         | no        |
| not members               | yes      | no          | no        |

基类前的权限可改变子类继承之后成员的权限：

```c++
class Rectangle: public Polygon { /* ... */ }
```

+ public： inherit all the members with the same levels they had in the base class. 
+ protected： all public members of the base class are inherited as `protected` in the derived class. 
+ private： all the base class members are inherited as `private`. 

## 继承了什么

In principle, a publicly derived class inherits access to every member of a base class **except**：

+ its constructors and its destructor. 毕竟名字都不一样
+ its assignment operator members (operator=)
+ its friends. 长辈的朋友并不是你朋友
+ its private members. 私有成员

虽然没有被继承，父类的构造函数和析构函数依然会被子类的构造函数、析构函数调用（默认或者：

```c++
Son (int a) : Mother (a)  // constructor specified: call this specific constructor 
```

## Multiple inheritance

```c++
#include <iostream>
using namespace std;

class Polygon {
  protected:
    int width, height;
  public:
    Polygon (int a, int b) : width(a), height(b) {}
};

class Output {
  public:
    static void print (int i);
};

void Output::print (int i) {
  cout << i << '\n';
}

class Rectangle: public Polygon, public Output {
  public:
    Rectangle (int a, int b) : Polygon(a,b) {}
    int area ()
      { return width*height; }
};

class Triangle: public Polygon, public Output {
  public:
    Triangle (int a, int b) : Polygon(a,b) {}
    int area ()
      { return width*height/2; }
};
  
int main () {
  Rectangle rect (4,5);
  Triangle trgl (4,5);
  rect.print (rect.area());
  Triangle::print (trgl.area());
  return 0;
}
```

# Polymorphism(多态)

## Pointers to base class

与java中的接口类似吧。

```c++
//基类Polygon，子类Rectangle Triangle
Rectangle rect;
Triangle trgl;
Polygon * ppoly1 = &rect;
Polygon * ppoly2 = &trgl;
ppoly1->set_values (4,5); //rect.set_values (4,5);一样
```

But because the type of `ppoly1` and `ppoly2` is pointer to `Polygon` (and not pointer to `Rectangle` nor pointer to `Triangle`), only the members inherited from `Polygon` can be accessed.

## Virtual members

```c++
#include <iostream>
using namespace std;

class Polygon {
  protected:
    int width, height;
  public:
    void set_values (int a, int b)
      { width=a; height=b; }
    virtual int area ()   //虚函数，就好像给子类留了个接口
      { return 0; }
};

class Rectangle: public Polygon {
  public:
    int area ()
      { return width * height; }
};

class Triangle: public Polygon {
  public:
    int area ()
      { return (width * height / 2); }
};

int main () {
  Rectangle rect;
  Triangle trgl;
  Polygon poly;
  Polygon * ppoly1 = &rect;
  Polygon * ppoly2 = &trgl;
  Polygon * ppoly3 = &poly;
  ppoly1->set_values (4,5);
  ppoly2->set_values (4,5);
  ppoly3->set_values (4,5);
  cout << ppoly1->area() << '\n';  //20
  cout << ppoly2->area() << '\n';  //10
  cout << ppoly3->area() << '\n';  //0
  return 0;
}
```

## Abstract base classes

```c++
// abstract class CPolygon
class Polygon {
  protected:
    int width, height;
  public:
    void set_values (int a, int b)
      { width=a; height=b; }
    virtual int area () =0;// to replace their definition by =0(an equal sign and a zero)
};
```

Notice that `area()` has no definition; this has been replaced by `=0`, which makes it a *pure virtual function*. Classes that contain at least one *pure virtual function* are known as *abstract base classes*. 

Abstract base classes cannot be used to instantiate objects.  不能直接创建对象。

But an *abstract base class* is not totally useless. It can be used to create pointers to it, and take advantage of all its polymorphic abilities.  但可以创建指针。

```c++
// pure virtual members can be called
// from the abstract base class
#include <iostream>
using namespace std;

class Polygon {
  protected:
    int width, height;
  public:
    void set_values (int a, int b)
      { width=a; height=b; }
    virtual int area() =0;
    // it is even possible for a member of the abstract base class Polygon 
    //to use the special pointer this to access the proper virtual members, 
    //even though Polygon itself has no implementation for this function
    void printarea()
      { cout << this->area() << '\n'; }
};

class Rectangle: public Polygon {
  public:
    int area (void)
      { return (width * height); }
};

class Triangle: public Polygon {
  public:
    int area (void)
      { return (width * height / 2); }
};

int main () {
  Rectangle rect;
  Triangle trgl;
  Polygon * ppoly1 = &rect;
  Polygon * ppoly2 = &trgl;
  ppoly1->set_values (4,5);
  ppoly2->set_values (4,5);
  ppoly1->printarea();
  ppoly2->printarea();
  return 0;
}
```