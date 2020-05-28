> python语言最常见的括号有三种，分别是：小括号( )、中括号[ ]和大括号也叫做花括号{ }，分别用来代表不同的python基本内置数据类型。
>
> 1、python中的小括号( )：代表tuple元组数据类型，元组是一种不可变序列。
>
>  2、python中的中括号[ ]，代表list列表数据类型：
>
> 3、python大括号{ }花括号：代表dict字典数据类型，字典是由键对值组组成。冒号':'分开键和值，逗号','隔开组。

## 继承

```python
def get_dataset(dataset, task):
    class Dataset(dataset_factory[dataset], _sample_factory[task]):
        pass
    return Dataset
```

## List

参照[python基础数据类型--list列表](https://www.cnblogs.com/Kingfan1993/p/9435769.html)

```python
list1=['2','3','4']

s=''.join(list1)
print(s)
'234'
 
# 把元素都变为字符串 
list2=[3,4,5]
list2=[str(i) for i in list2]
```

要使`str.join()`起作用，迭代中包含的元素(即此处的列表)必须是字符串本身

```python
', '.join(map(str, l)) + ','
```

## 队列

[简析Python中的四种队列](https://blog.csdn.net/sinat_38682860/article/details/80392493）

```python
>>> from collections import deque
>>> d = deque('ghi')                 # make a new deque with three items
>>> for elem in d:                   # iterate over the deque's elements
...     print(elem.upper())
G
H
I

>>> d.append('j')                    # add a new entry to the right side
>>> d.appendleft('f')                # add a new entry to the left side
>>> d                                # show the representation of the deque
deque(['f', 'g', 'h', 'i', 'j'])

>>> d.pop()                          # return and remove the rightmost item
'j'
>>> d.popleft()                      # return and remove the leftmost item
'f'
>>> list(d)                          # list the contents of the deque
['g', 'h', 'i']
>>> d[0]                             # peek at leftmost item
'g'
>>> d[-1]                            # peek at rightmost item
'i'

>>> list(reversed(d))                # list the contents of a deque in reverse
['i', 'h', 'g']
>>> 'h' in d                         # search the deque
True
>>> d.extend('jkl')                  # add multiple elements at once
>>> d
deque(['g', 'h', 'i', 'j', 'k', 'l'])
>>> d.rotate(1)                      # right rotation
>>> d
deque(['l', 'g', 'h', 'i', 'j', 'k'])
>>> d.rotate(-1)                     # left rotation
>>> d
deque(['g', 'h', 'i', 'j', 'k', 'l'])

>>> deque(reversed(d))               # make a new deque in reverse order
deque(['l', 'k', 'j', 'i', 'h', 'g'])
>>> d.clear()                        # empty the deque
>>> d.pop()                          # cannot pop from an empty deque
Traceback (most recent call last):
    File "<pyshell#6>", line 1, in -toplevel-
        d.pop()
IndexError: pop from an empty deque

>>> d.extendleft('abc')              # extendleft() reverses the input order
>>> d
deque(['c', 'b', 'a'])
```



## 字符串

split() 方法语法：

`str.split(str="", num=string.count(str))`
参数
str -- 分隔符，默认为所有的空字符，包括空格、换行(\n)、制表符(\t)等。
num -- 分割次数。默认为 -1, 即分隔所有。



## 遍历map

https://www.jianshu.com/p/36a7c85b7243

如何判断 key 已经存在

Python 3.X 里不包含 has_key() 函数，被` __contains__(key) `替代

`key in m.keys()`