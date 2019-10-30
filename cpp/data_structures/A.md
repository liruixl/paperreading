## 线性表、栈、队列

+ 数组
+ 链表（带有表头结点）
+ 栈
  + 顺序栈（基于数组）
  + 链式栈（基于链表）
+ 队列
  + 顺序队列（基于数组，假定数组是循环的，数组的大小比实际队列允许的最大长度大1，区分满队列和空队列）
  + 链式队列（基于链表）

## 二叉树

+ 结点，根结点的深度为0，层数也为0
  + 深度：从节点到根路径的长度，从0开始
  + 高度：最深节点的深度加1，从1开始
  + 层数：=深度
+ 满二叉树：每个结点，或者是分支结点，并恰有两个非空子结点；或者叶结点。
  + 定理一：非空满二叉树，叶节点数 = 分支节点数 + 1。
  + 定理二：非空二叉树，空子树数目 = 结点数 + 1。why？试想：将所有空子树换成叶节点，则得到一颗满二叉树。此时，分支结点数目是原来的结点数目，叶结点数目是原来空子树数目。由定理一可得结论。
+ 完全二叉树：除最后一层外，每一层都是满的。底层叶节点集中在左边。
  + 可用于堆数据结构。堆经常用来实现优先队列和外排序算法。
  + 可用数组实现，由于完全二叉树结点位置完全由其序号决定。不存在结构性开销。公式如下，r 表示下标，n表示结点的总数。
    + Parent（r）=（r-1）/ 2，当r<0时
    + Left（r）= 2r+1，当2r+1<n时
    + Right（r）=2r+2，当2r+2<n时
+ 二叉查找树（Binary Search Tree，BST，也叫二叉检索树）
  + 对任意一个结点，设其值为K，左子树中任意一个结点的值都小于K，右子树中任意一个结点的值都大于等于K。
  + 中序周游（左子树、结点、右子树顺序），可以按顺序打印二叉树。
+ 最大值堆（完全二叉树）

## 图

实现方式

+ 相邻矩阵
+ 邻接表（以链表为元素的数组）

算法

+ 深度优先搜索（DFS）
+ 广度优先搜索（BFS）
+ 拓扑排序
+ 图的最短路径
+ 最小支撑树（网络路由）



分类：

+ 有向图
+ 无向图
+ 标号图
+ 带权图
