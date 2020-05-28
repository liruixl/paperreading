# 下载Anaconda3

参考：[Anaconda3安装——Linux下](http://www.lwqgj.cn/586.html)

问题一：想选python3.6版本的

官方地址在下面，3.6的对应的是 Anaconda 5.2，5.3以后的都是python 3.7的不要看错了

反正这个是，这里下载对应 python3.6 版本的 anaconda3:

![Anaconda3安装——Linux](C:\ZZZ\notes\assert\2019-02-13_105530.png)

问题二：下载太慢

到清华镜像下载：https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/

用wget命令下载

```
wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh 
```

bash 安装，注意更改目录。

安装完后使生效：

```c++
source ~/.bashrc
```



注意`conda update conda`命令，会把包都更新，python会更新到3.7。。。注意注意。

# 管理虚拟环境

想把pytorch单独放入一个环境。

```python
# 创建一个名为python34的环境，指定Python版本是3.4（不用管是3.4.x，conda会为我们自动寻找3.4.x中的最新版本）
 
conda create --name python34 python=3.4
 
# 安装好后，使用activate激活某个环境
 
activate python34 # for Windows
 
source activate python34 # for Linux & Mac
 
# 激活后，会发现terminal输入的地方多了python34的字样，实际上，此时系统做的事情就是把默认2.7环境从PATH中去除，再把3.4对应的命令加入PATH
 
# 此时，再次输入
 
python --version
 
# 可以得到`Python 3.4.5 :: Anaconda 4.1.1 (64-bit)`，即系统已经切换到了3.4的环境
 
# 如果想返回默认的python 2.7环境，运行
 
deactivate python34 # for Windows
source deactivate python34 # for Linux & Mac
# 删除一个已有的环境
 
conda remove --name python34 --all
```



新环境不指定python版本就是最新版本，我\*你哥。

```python
# To activate this environment, use:
# > source activate pytorch12
#
# To deactivate an active environment, use:
# > source deactivate
#

查看环境
conda info --envs
```



安装pycharm

参考：https://blog.csdn.net/xiaoxiaofengsun/article/details/82257391

下载安装包linux下的：pycharm-professional-2019.3.tar.gz

解压即可` tar -zxvf pycharm-professional-2018.1.3.tar.gz -C /usr/local/pycharm`

最好把安装包放在当前目录，并解压到当前目录，以防操作失误。。

# 装pytorch

pytorch官网找命令去，对应cuda的版本要看好。

https://pytorch.org/get-started/previous-versions/

# Simple-Faster-Rcnn

记录下遇到的问题。

## Pillow版本问题

pillow7.0.0已经没有`PILLOW_VERSION`这个东西了，而pillow6.1还保留着
所以只需要`conda install pillow=6.1`即可



## Without the incoming socket 问题

you cannot receive events from the server or register event handlers to your Visdom client.

https://github.com/facebookresearch/visdom/issues/354

https://github.com/facebookresearch/visdom/commit/520f9acfee4320a2fb726759b2988338109c2cca



## object __array__ method not producing an array

https://github.com/chenyuntc/simple-faster-rcnn-pytorch/issues/72#issuecomment-554652274