Qt文档Multithreading Technologise in Qt的翻译总结。

## 四种方法

Qt程序可以使用以下四种方法实现多线程应用。

### QThread

QThread可以直接实例化也可以子类化。 实例化QThread提供了一个并行事件循环，从而允许在辅助线程中调用QObject插槽。 子类化QThread允许应用程序在开始其事件循环之前初始化新线程，或者在没有事件循环的情况下运行并行代码。（没看明白这段20191120）

### QThreadPool and QRunnable: Reusing Threads

线程可以重新使用来运行新tasks，QThreadPool是线程池。

要在QThreadPool的某个线程中运行代码，需要重新实现QRunnable :: run()并实例化子类QRunnable。 使用QThreadPool :: start() 将QRunnable放入QThreadPool的运行队列中。 当线程可用时，QRunnable :: run中的代码将在该线程中执行。

每个Qt应用程序都有一个全局线程池，可通过QThreadPool :: globalInstance（）访问该线程池。 该全局线程池会根据CPU中的内核数自动维护最佳线程数。 但是，可以显式创建和管理单独的QThreadPool。

### Qt Concurrent: Using a High-level API

**The QtConcurrent namespace** provides high-level APIs that make it possible to write multi-threaded programs without using low-level threading primitives such as mutexes, read-write locks, wait conditions, or semaphores.

相关QFuture、QFutureWatcher、map, filter and reduce algorithms、run()

更多请参见：Qt Concurrent 模块

### WorkerScript: Threading in QML

没用过QML

## 用例指导

线程生命周期，和方案指导。

+ 一次调用，在另一个线程中运行新的线性函数，可以选择在运行期间更新进度。

  + 将函数放置在QThread::run() 并且 start 这个线程。发出信号更新进度。

  + 将函数放置在QRubable::run()中，并且添加QRunable到线程池QThreadPool。写一个线程安全的变量去更新进度。

  + 使用QtConcurrent::run()去运行函数。写一个线程安全的变量去更新进度。

    函数签名：QFuture\<T\> QtConcurrent::run(Function function, ...)

+ 一次调用，在另一个线程运行一个已经存在的函数，并且的得到它的返回值。

  使用QtConcurrent::run() 运行此函数， 让函数返回时让QFutureWatcher发出 finish() 信号，然后调用QFutureWatcher :: result() 以获取函数的返回值。

  Qt中让QFutureWatcher发出信号，C++ 中该如何知道future对象已经获得函数的结果了呢。

+ 一次调用，使用所有可用的CPU核对一个容器中所有的items执行操作，**例如，从图像列表生成缩略图**。

  使用Qt Concurrent的QtConcurrent :: filter（）函数选择容器元素，然后使用QtConcurrent :: map（）函数将操作应用于每个元素。 要将输出折叠为单个结果（？？？），请改用QtConcurrent :: filteredReduced（）和QtConcurrent :: mappedReduced（）。

+ 永久，使对象驻留在另一个线程中，该对象可以根据请求执行不同的任务或者可以接收要使用的新数据。

  子类化一个QObjet以创建一个worker，实例化这个worker对象和一个QThread。将此worker移动到（move to）新线程， Send commands or data to the worker object over **queued** signal-slot connections.

+ 永久，在另一个线程重复执行一些昂贵的操作，这个线程不需要接收任何的信号和事件。

  重新实现 QThread :: run() ，在其中直接编写无限循环。 启动线程而没有事件循环。 让线程发出信号以将数据发送回GUI线程。

  