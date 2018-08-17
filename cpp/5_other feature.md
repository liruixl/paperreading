# Type conversions

## Implicit conversion

。。。

# Preprocessor directives

无分号（；）仅一行，多行用backslash（\）

## macro definitions

 #define   #undef

```c++
#define TABLE_SIZE 100
int table1[TABLE_SIZE];  //100
#define getmax(a,b) ((a)>(b)?(a):(b))
#undef TABLE_SIZE  //...
#define TABLE_SIZE 200
int table2[TABLE_SIZE];  //200
```



被字符串x取代

```c++
#define str(x) #x
cout << str(test);
//等于
cout << "test";
```

##

```c++
#define glue(a,b) a ## b   //## concatenates two arguments leaving no blank spaces 
glue(c,out) << "test";
//等于
cout << "test";
```

## Conditional inclusions 

(#ifdef, #ifndef, #if, #endif, #else and #elif)

```c++
#ifdef TABLE_SIZE    //假如TABLE_SZIE定义了，编译代码
int table[TABLE_SIZE];
#endif  
```

```c++
#ifndef TABLE_SIZE   //假如没定义
#define TABLE_SIZE 100
#endif
int table[TABLE_SIZE];
```



```c++
#if TABLE_SIZE>200     //if
#undef TABLE_SIZE
#define TABLE_SIZE 200
 
#elif TABLE_SIZE<50    //else if
#undef TABLE_SIZE
#define TABLE_SIZE 50
 
#else                 //else
#undef TABLE_SIZE
#define TABLE_SIZE 100
#endif               //endif
 
int table[TABLE_SIZE]; 
```

## Line control (#line)

```c++
#line 20 "assigning variable"
int a?;
```

This code will generate an error that will be shown as error in file `"assigning variable"`, line 20. 

## Error directive (#error)

```c++
#ifndef __cplusplus
#error A C++ compiler is required!
#endif 
```

This example aborts the compilation process if the macro name `__cplusplus` is not defined (this macro name is defined by default in all C++ compilers).

## Source file inclusion (#include)

> When the preprocessor finds an `#include`directive it replaces it by the entire content of the specified header or file.  

```c++
#include <header>
#include "file" 
```

## Predefined macro names

The following macro names are always defined (they all begin and end with two underscore characters, `_`):  

| macro             | value                                                        |
| ----------------- | ------------------------------------------------------------ |
| `__LINE__`        | Integer value representing the current line in the source code file being compiled. |
| `__FILE__`        | A string literal containing the presumed name of the source file being compiled. |
| `__DATE__`        | A string literal in the form "Mmm dd yyyy" containing the date in which the compilation process began. |
| `__TIME__`        | A string literal in the form "hh:mm:ss" containing the time at which the compilation process began. |
| `__cplusplus`     | An integer value. All C++ compilers have this constant defined to some value. Its value depends on the version of the standard supported by the compiler: <br />+99711L: ISO C++ 1998/2003<br />+201103L: ISO C++ 2011<br />Non conforming compilers define this constant as some value at most five digits long. Note that many compilers are not fully conforming and thus will have this constant defined as neither of the values above. |
| `__STDC_HOSTED__` | `1` if the implementation is a *hosted implementation* (with all standard headers available) `0` otherwise. |

The following macros are optionally defined, generally depending on whether a feature is available: 

| macro                              | value                                                        |
| ---------------------------------- | ------------------------------------------------------------ |
| `__STDC__`                         | In C: if defined to `1`, the implementation conforms to the C standard. In C++: Implementation defined. |
| `__STDC_VERSION__`                 | In C: **199401L**: ISO C 1990, Ammendment 1<br />**199901L**: ISO C 1999<br />**201112L**: ISO C 2011<br />In C++: Implementation defined. |
| `__STDC_MB_MIGHT_NEQ_WC__`         | `1` if multibyte encoding might give a character a different value in character literals |
| `__STDC_ISO_10646__`               | A value in the form `yyyymmL`, specifying the date of the Unicode standard followed by the encoding of `wchar_t` characters |
| `__STDCPP_STRICT_POINTER_SAFETY__` | `1` if the implementation has *strict pointer safety* (see `get_pointer_safety`) |
| `__STDCPP_THREADS__`               | `1` if the program can have more than one thread             |