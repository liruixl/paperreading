QT 5.12 Threading Basics

## 什么是线程

单个进程内的并发——concurrency within one single process

如何实现并发？

单核CPU的并发是一种错觉，对于进程，是在进程之间切换。

单核CPU上多线程也是如此， 每个程序在开始时有一个线程，不过，程序可以滞后开启新的线程。通过重复保存程序计数器和寄存器，然后加载下一个线程的程序计数器和寄存器，可以在单核CPU上实现并发。 在活动线程之间循环不需要程序的任何协作。 切换到下一个线程时，线程可能处于任何状态。

多核CPU上可以实现真正的并发。

## GUI Thread and Worker Thread

界面应用程序，启动时运行的线程叫做主线程“main thread”也叫做“GUI thread”，QT GUI必须运行在这个线程上。所有Widget和几个相关的类，比如：QPixmap不会在辅助线程工作（A secondary thread）。辅助线程通常被称为“Worder Thread”，因为它用于从主线程分担处理工作。

所以，千万不能阻塞主线程，或者做一些长时间的工作。

## Simultaneous Access to Data

同时访问数据。静态成员、单例、或全局数据。需要熟悉线程安全和可重入的类和函数的概念。

线程共享同一地址空间，每个线程有自己的栈。



## 什么时候使用线程

+ 利用多处理器使程序更快
+ 通过减轻长时间的一些处理或一些阻塞调用，使GUI线程或对其他对时间要求严格的线程保持响应。

## QT中代替线程的其他方法

线程要小心使用！启动很容易，但是很难确保所有共享数据保持一致。

在创建线程之前，考虑一些可能替代它的方法。



+ QEventLoop::processEvents()

  在耗时的计算中重复调用processEvents，防止GUI阻塞。此解决方案无法很好地扩展，因为根据硬件的不同，对processEvents（）的调用可能发生的次数过多或不够频繁

+ QTimer

  有时可以使用计时器方便地完成后台处理，以安排将来某个时间执行插槽。 一旦没有更多事件要处理，间隔为0的计时器将超时。

+ QSocketNotifier QNetworkAccessManager QIODevice::readyRead()

  ？？？？？？？响应式涉及

+ QtConcurrent命名空间

  线程代码隐藏在框架后，用户无需关心。 但是，当需要与正在运行的线程进行通信时，不能使用QtConcurrent，并且不应将其用于处理阻塞操作。

## QT中使用什么线程技术

有关对Qt进行多线程处理的不同方法的介绍，以及有关如何选择这些方法的指南，请参见Qt中的多线程技术页面[Multithreading Technologies in Qt](.\Multithreading Technologise in Qt.md )。

## Qt线程基础

### 与QObject的关系

A QObject instance is said to have a thread affinity, 参考文档QObject类。

### 如何安全地从多个线程访问数据

保护数据完整性，必须格外小心避免数据损坏。请参见 [Synchronizing Threads](./Synchronizing Threads)

### 异步执行如何在不阻塞的情况下获得结果

处理异步执行。

为了获得“工作线程”的结果，我们可以等待直到线程结束。但是，长时间的阻塞等待可能致命，比如阻塞了“GUI线程”。

代替方法是posted events 或者 queued signals and slots。这会产生一定的开销，因为操作的结果不会出现在下一行代码中。Qt开发人员习惯使用这种异步行为，因为它与GUI程序中的**事件驱动编程**非常相似。

## Examples

简单例子， 参见文档类ref：QThread and QThreadPool 

其他例子：参见文档：[Threading and Concurrent Programming Examples](examples-threadandconcurrent.html)

## Digging Deeper

参见 [Thread Support in Qt]() 文档















