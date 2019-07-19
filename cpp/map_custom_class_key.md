## map_custom_class_key

```c++
using EvalTaskKey = typename std::pair<UserTaskEntity, int>;

std::map<EvalTaskKey, EvalConsoleWidget*> evalConsoles;
```

​	

1.当自定义的结构或类作为map的key值时，需要提供比较函数，重载小于操作符。为什么不重载==操作符呢？那map的find函数怎么办呢？

实际上，在map的实现里面就是靠 对调operator<两边的操作数实现的。简单的来说，当用map的find或者是set的find函数时，二叉树查找对应值，要涉及到比较操作，如果需要查找的key < element不成立，而且 element < key 也不成立，那么 如果element不是end()时，element即查找的key。



所以这里我们需要提供std::pair<UserTaskEntity, int>的比较操作符。

而标准库std::pair有比较操作符：<

```c++
template <class T1, class T2>
  bool operator<  (const pair<T1,T2>& lhs, const pair<T1,T2>& rhs)
{ return lhs.first<rhs.first || (!(rhs.first<lhs.first) && lhs.second<rhs.second); }
```

这return语句写的巧妙啊，只用小于号（哦，对，只能用小于号）。

`std::pair<UserTaskEntity, int>`，内置类型int不用考虑，所以我们只需要给我们自定义`UserTaskEntity`类定义`<`操作符。

如果不提供“<”比较操作符，我么可以自定义仿函数替代第三个参数Compare，该仿函数实现“()”操作符，提供比较功能。插入时各节点顺序以该仿函数为纲。 

><https://blog.csdn.net/seanyxie/article/details/6329408> 
>
>以std::pair为关键字掺入map
>下面我们先写一个有错误的函数，在分析错误原因之后，逐步进行修正。
>
>```c++
>#include <map>
>
>int main()
>
>{
>
>       std::map<std::pair<int, int>, int> res;
>
>       res.insert(std::make_pair(12,33), 33);
>
>}
>
>```
>
> 这个程序一定失败，如果非要如此使用，上述a方法显然不适合，std::pair是已定义好的结构体不可修改。只能使用b方法了，定义一个比较类改造如下： 
>
>```c++
>struct comp
>
>{
>       typedef std::pair<int, int> value_type;
>       bool operator () (const value_type & ls, const value_type &rs)
>       {
>
>              return ls.first < rs.first || 
>                  (ls.first == rs.first && ls.second < rs.second);
>       }
>
>};
>```

上面这一大段引用，虽然说明了意思。但有两点错误：

1. `res.insert(std::make_pair(std::make_pair(12,33), 33));` //插入语句应该是这样吧。
2. pair提供<运算符。基于piar模板参数类型的<运算符。



注意：在类的里面定义<只有一个参数，在外面有两个参数（左、右）。