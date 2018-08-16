## Statements and flow control

### Selection statements: if and else

if-else

```c++
if (x > 0)
  cout << "x is positive";
else if (x < 0)
  cout << "x is negative";
else
  cout << "x is 0";
```

### Iteration statements (loops)

#### while

#### do-while

#### the for loop

`for (initialization; condition; increase) statement; `

1. `initialization` is executed. This is executed a single time, at the beginning of the loop.
2. `condition` is checked. If it is true, the loop continues; otherwise, the loop ends, and `statement` is skipped, going directly to step 5.
3. `statement` is executed. As usual, it can be either a single statement or a block enclosed in curly braces `{ }`.
4. `increase` is executed, and the loop gets back to step 2.
5. the loop ends: execution continues by the next statement after it.

```c++
for (int n=10; n>0; n--) {
    cout << n << ", ";
  }
```

#### Range-based for loop

`for ( declaration : range ) statement; `

```c++
string str {"Hello!"};
  for (char c : str)
  {
    cout << "[" << c << "]";
  }
```

### Jump statements

+ break
+ continue
+ goto

### Another selection statement: switch

> Notice that `switch` is limited to compare its evaluated expression against labels that are constant expressions. It **is not possible to use variables as labels or ranges**, because they are not valid C++ **constant expressions**. 

## Functions

type name ( parameter1, parameter2, ...) { statements } 

### main函数的返回

| value          | description                                                  |
| -------------- | ------------------------------------------------------------ |
| `0`            | The program was successful                                   |
| `EXIT_SUCCESS` | The program was successful (same as above). This value is defined in header `<cstdlib>`. |
| `EXIT_FAILURE` | The program failed. This value is defined in header `<cstdlib>`. |

### Arguments passed by value and by reference

  传值什么的就不谈了

  > In certain cases, though, it may be useful to **access an external variable** from within a function. To do that, arguments can be passed *by reference* 
  >
  > 有时，从函数中访问外部变量可能很有用，传引用。

  ```c++
  // passing parameters by reference
  #include <iostream>
  using namespace std;
  
  void duplicate (int& a, int& b, int& c)
  {
    a*=2;
    b*=2;
    c*=2;
  }
  
  int main ()
  {
    int x=1, y=3, z=7;
    duplicate (x, y, z);
    cout << "x=" << x << ", y=" << y << ", z=" << z;
    return 0;
  }
  ```
### Efficiency considerations and const references

比较下面三个示例：

1. 传copy会使效率降低，万一字符串超级长呢？

```c++
string concatenate (string a, string b)
{
  return a+b;
}
```

2. 传引用就不存在1的问题，但是传引用一般被视为要对传递的参数进行修改，但本例只是返回a+b。

```c++
string concatenate (string& a, string& b)
{
  return a+b;
}
```

3. 所以const references可以派上用场。我只是为了提高效率用用你而已。

```c++
string concatenate (const string& a, const string& b)
{
  return a+b;
}
```

### Inline functions 内联函数

> Calling a function generally causes a certain overhead (stacking arguments, jumps, etc...), and thus for very short functions, it may be more efficient to simply insert the code of the function where it is called, instead of performing the process of formally calling a function. 

```c++
inline string concatenate (const string& a, const string& b)
{
  return a+b;
}
```

### Declaring functions 声明函数

在main函数之前声明哪个，并不定义。

```c++
int protofunction (int first, int second);
int protofunction (int, int);
```

有什么用呢？下面的例子两个函数相互调用，如果不适用声明函数，there is no way to structure the code so that `odd` is defined before `even`, and `even` before `odd`. 所以至少要提前声明一个函数。。

```c++
// declaring functions prototypes
#include <iostream>
using namespace std;

void odd (int x);
void even (int x);

int main()
{
  int i;
  do {
    cout << "Please, enter number (0 to exit): ";
    cin >> i;
    odd (i);
  } while (i!=0);
  return 0;
}

void odd (int x)
{
  if ((x%2)!=0) cout << "It is odd.\n";
  else even (x);
}

void even (int x)
{
  if ((x%2)==0) cout << "It is even.\n";
  else odd (x);
}
```

### Recursivity 递归

## Overloads and templates

重载和模板

### Overloaded functions

同样的参数名，不同的作用：

+ 参数个数？?
+ 参数类型

> Note that a function cannot be overloaded only by its return type. At least one of its parameters must have a different type. 

### Function templates

> Defining a function template follows the same syntax as a regular function, except that it is preceded by the template keyword and a series of template parameters enclosed in angle-brackets <>:
>
> 定义函数模板遵循与常规函数一样的语法，除了在它之前是……

`template <template-parameters> function-declaration`  

The template parameters are a series of parameters separated by commas. These parameters can be generic template types by specifying either the `class` or `typename` keyword followed by an identifier. **This identifier can then be used in the function declaration as if it was a regular type**. For example, a generic `sum` function could be defined as: 

```c++
template <class SomeType>
SomeType sum (SomeType a, SomeType b)
{
  return a+b;
}
```

It makes no difference whether the generic type is specified with keyword `class` or keyword `typename` in the template argument list (they are 100% synonyms(同义词) in template declarations). 

使用：

```c++
k=sum<int>(i,j);
h=sum<double>(f,g);
```

### Non-type template arguments

```c++
// template arguments
#include <iostream>
using namespace std;

template <class T, int N>
T fixed_multiply (T val)
{
  return val * N;
}

int main() {
  std::cout << fixed_multiply<int,2>(10) << '\n';
  std::cout << fixed_multiply<int,3>(10) << '\n';
}
```

模板的第二个函数之哦嗯那个使常量表达式，绝对不能是个变量，因为：

>  the value of template parameters is determined on **compile-time** to generate a different instantiation of the function `fixed_multiply`, and thus the value of that argument is never passed during runtime 

## Name visibility

### Scopes

这个好理解吧

+ global scope  // global variable
+ local scopw  // local variable

### Namespaces

有什么用呢？

> Non-local names bring more possibilities for name collision, especially considering that libraries may declare many functions, types, and variables, neither of them local in nature, and some of them very generic. 

举个栗子：

```c++
// namespaces
#include <iostream>
using namespace std;

namespace foo
{
  int value() { return 5; }
}

namespace bar
{
  const double pi = 3.1416;
  double value() { return 2*pi; }
}

int main () {
  cout << foo::value() << '\n';
  cout << bar::value() << '\n';
  cout << bar::pi << '\n';
  return 0;
}
```

#### using

加入在first和second的namespace里都定义了x和y，那么要使用其中一个的话，可以使用using关键字：

```c++
using first::x;
using second::y;
```

The variables `first::y` and `second::x` can still be accessed, but require fully qualified names. 

或者直接`using namespace first;`，就像之前程序里看到的`using namespace std;`

#### Namespace aliasing

`namespace new_name = current_name; `

#### The std namespace

```c++
using namespace std;
cout << "Hello world!";
std::cout << "Hello world!";
```

 It is mostly a matter of style preference, although for projects mixing libraries, explicit qualification tends to be preferred. 

### Storage classes

+ Variables with *static storage* (such as global variables) that are not explicitly initialized are automatically initialized to zeroes.
+ Variables with *automatic storage* (such as local variables) that are not explicitly initialized are left uninitialized, and thus have an undetermined value. 

```c++
#include <iostream>
using namespace std;

int x;  //0

int main ()
{
  int y;  //undetermined value.
  cout << x << '\n';
  cout << y << '\n';
  return 0;
 }
```

