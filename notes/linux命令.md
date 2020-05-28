nvidia-smi：GPU使用情况。

~ 表示代码主目录，也就是当前登录用户的用户目录。



复制

**1.cp命令**

命令：**cp dir1/a.doc dir2** 表示将dir1下的a.doc文件复制到dir2目录下

**cp -r dir1 dir2** 表示将dir1及其dir1下所包含的文件复制到dir2下

**cp -r dir1/. dir2** 表示将dir1下的文件复制到dir2,不包括dir1目录

说明：cp参数 -i：询问，如果目标文件已经存在，则会询问是否覆盖；

移动

**文件移动（mv）**

文件移动不同于文件拷贝，文件移动相当于我们word中的术语剪切和粘贴。

命令：mv AAA BBB 表示将AAA改名成BBB

说明：**目标目录与原目录一致，指定了新文件名，效果就是仅仅重命名。目标目录与原目录不一致，没有指定新文件名，效果就是仅仅移动。目标目录与原目录不一致，指定了新文件名，效果就是：移动 + 重命名。**





Wheels (precompiled binary packages)



查询进程 关闭进程

ps -ef |grep redis

> ps:将某个进程显示出来
> -A 显示所有程序
> -f 显示UID PPIP C与STIME栏位
> grep命令是查找
> 中间的|是管道命令 是指ps命令与grep同时执行
> 这条命令的意思是显示有关redis有关的进程

kill [参数] [进程号]

> kill -9 4394
> kill就是给某个进程id发送了一个信号。默认发送的信号是SIGTERM，而kill -9发送的信号是SIGKILL，即exit。
> exit信号不会被系统阻塞，所以kill -9能顺利杀掉进程。当然你也可以使用kill发送其他信号给进程