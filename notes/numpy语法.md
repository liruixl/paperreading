# numpy

np.inf无穷大

array.shape返回维度



# 初始化

https://blog.csdn.net/jasonzhoujx/article/details/81608555



numpy中logical_and和all函数

logical_and函数对表达式进行真值判断，相应输出true,false，

all函数对列表中false和true进行逻辑和运算，输出结果。



np.newaxis放在第几个位置，就会在shape里面看到相应的位置增加了一个维数。 `x[:, np.newaxis]`



计算乘积

`np.prod(N, axis=2)`



np.meshgrid

矩阵X的行向量是向量x的简单复制，而矩阵Y的列向量是向量y的简单复制。

假设x是长度为m的向量，y是长度为n的向量，则最终生成的矩阵X和Y的维度都是 n*m （注意不是m*n）

```python
"""
    # H = 16 W = 32
    >>> x,y = np.meshgrid([0,16,32],[0,16])
    >>> x
    array([[ 0, 16, 32],
        [ 0, 16, 32]])
    >>> y
    array([[ 0,  0,  0],
        [16, 16, 16]])
 """
```





numpy.stack(arrays, axis=0)

新的一个维度根据`axis`来确定的。

```python
>>> a = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
>>> b = np.array([[4, 5, 6], [4, 5, 6], [4, 5, 6]])
>>> a.shape
(3, 3)
>>> b.shape
(3, 3)
>>> np.stack((a, b), axis=0).shape
(2, 3, 3)
>>> np.stack((a, b), axis=1).shape
(3, 2, 3)
>>> np.stack((a, b), axis=2).shape
(3, 3, 2)

```



np.where(condition)

只有条件 (condition)，没有x和y，则输出满足条件 (即非0) 元素的坐标 (等价于[numpy.nonzero](https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html#numpy.nonzero))。这里的坐标以tuple的形式给出，通常原数组有多少维，输出的tuple中就包含几个数组，分别对应符合条件元素的各维坐标。

```python
>>> a = np.array([2,4,6,8,10])
>>> np.where(a > 5)             # 返回索引
(array([2, 3, 4]),)   
>>> a[np.where(a > 5)]              # 等价于 a[a>5]
array([ 6,  8, 10])

>>> np.where([[0, 1], [1, 0]])
(array([0, 1]), array([1, 0])) # 非0的坐标
```

