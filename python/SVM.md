参考：https://blog.csdn.net/c406495762/article/details/78072313

[深入理解拉格朗日乘子法（Lagrange Multiplier) 和KKT条件](https://www.cnblogs.com/sddai/p/5728195.html)

KKT条件是说**最优值**必须满足以下条件：

　　　　1）L(a, b, x)对x求导为零；

　　　　2）h(x) =0;

　　　　**3）a\*g(x) = 0;**

 求取这些等式之后就能得到候选最优值。其中第三个式子非常有趣，因为g(x)<=0，如果要满足这个等式，必须a=0或者g(x)=0. 这是SVM的很多重要性质的来源，如支持向量的概念。



SMO算法？？？？



[支持向量机(SVM)是什么意思？](https://www.zhihu.com/question/21094489)