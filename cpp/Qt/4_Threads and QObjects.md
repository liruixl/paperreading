文档Threads and QObjects 



QThread 继承 QObject，有信号指示线程的开始或完成，也提供了一些槽。

QObject可以被使用在多线程中，发出信号，或调用在其他线程中的槽函数，并且post events向其他线程中的对象。这些成为可能的原因是：线程允许有它自己的事件循环。



## QObject Reentrancy

QObject是可重入的，大多数它的非GUI子类也是可重入的。但要注意：**这些类旨在从单个线程中创建和使用。 不能保证在一个线程中创建对象并从另一个线程调用其功能**。 有三个约束要注意：

+ QObject的 child 必须总是在parent被创建的线程中被创建。这意味着，除其他事项外，你永远不要传递QThread object（this）作为在这个线程中创建的对象的parent（因为QThread object本身是在另一个线程中创建的）。
+ 事件驱动的对象只能在单个线程中使用。 具体地说，这适用于计时器机制和网络模块（the timer mechanism and the network module）。 例如，您不能启动计时器或在不是对象线程的线程中连接套接字。
+ 在删除QThread之前，必须确保删除在线程中创建的所有对象。 在run()实现中通过在栈上创建对象，可以轻松完成此操作。

### 惊了：

尽管QObject是可重入的，但GUI类（尤其是QWidget及其所有子类）不是可重入的。 它们只能在主线程中使用。 如前所述，还必须从该线程调用QCoreApplication :: exec()。

这段是最好理解的：

实际上，可以通过将耗时的操作放在单独的工作线程中并在工作线程完成后将结果显示在主线程的屏幕上来轻松解决在主线程以外的其他线程中使用GUI类的可能性。 这是用于实现Mandelbrot示例和Blocking Fortune Client示例的方法（参见文档<u>Threading and Concurrent Programming Examples</u>）。

### 卧槽：

通常，不支持在QApplication之前创建QObject，并且可能导致退出时发生奇怪的崩溃，具体取决于平台。 这意味着也不支持QObject的静态实例。 正确构造的单线程或多线程应用程序应使QApplication成为第一个创建，最后一个销毁的QObject。

## Per-Thread Event Loop

每个线程都有它自己的事件循环（event loop），初始线程使用QCoreApplication::exec()开始它的事件循环，或者对于单对话框GUI应用程序，有时使用QDialog::exec()开启事件循环。其他线程可以使用QThread::exec()开始事件循环，像，一样，QThread提供了exit（int）函数和quit（）槽函数。

### 为什么需要事件循环

1、可以使用一些需要事件循环支持的非GUI Qt类，比如 QTimer, QTcpSocket, and QProcess

2、可以将来自任何线程的信号连接到特定线程的插槽。 这在下面的“Signals and Slots Across Threads”部分中进行了详细说明。

### 几点说明

1. QObject 实例属于（live in）创建它的线程，该对象的事件由该线程的事件循环调度。QObject::thread()
2. The `QObject::moveToThread()` function changes the thread affinity for an object and its children (the object cannot be moved if it has a parent).
3. 拥有对象对线程负责delete这个对象，除非你保证这个对象不再处理任何事件。 使用 QObject::deleteLater() 代替。
4. 假如没有事件循环，事件不会传递到object，例如，假如你在一个线程中创建了QTimer对象，但是没有调用exec()，这个QTimr永远不会发出`timeout()`信号。调用deleteLater()也不会工作。（这些限制对于主线程也是如此）
5. 你可以手动post events到任何线程中的任何对象在任意事件，使用线程安全函数：QCoreApplication::postEvent()，事件将会通过对象被创建的线程的事件循环自动分发。
6. 所有线程均支持事件过滤器（Event filters），但限制是监视对象必须与被监视对象位于同一线程中。 同样，QCoreApplication :: sendEvent（）（与postEvent（）不同）只能用于将事件分发（调度）到属于调用该函数线程中的对象。

## Accessing QObject Subclasses from Other Threads

### 注意几点

1. QObject和所有他的子类不是线程安全的，包括整个的事件分发系统。所以时刻注意事件循环正在分发事件到你的QObject子类上，然而你正在从其他线程访问这个对象！！
2. 假如你在不属于这个对象的线程上调用它的函数，而这个对象可能在接收事件。这个时候，你必须用互斥量来保护你的QObject子类的内部数据。
3. QThread属于创建它的线程，而不是run中创建的那个线程。
4. 在QThread中提供槽函数是不安全的，除非用mutex保护成员变量。
5. 你可以在QThread : : run() 重实现中发出信号，因为**信号发出是线程安全的**。

## Signals and Slots Across Threads

主要讲解信号槽的连接方式。`enum Qt::ConnectionType`

+ Auto Connection (default) If the signal is emitted in the thread which the receiving object has affinity then the behavior is the same as the Direct Connection. Otherwise, the behavior is the same as the Queued Connection."

+ Direct Connection The slot is invoked immediately, when the signal is emitted. The slot is executed in the emitter's thread, which is not necessarily the receiver's thread.

+ Queued Connection The slot is invoked when control returns to the event loop of the receiver's thread. The slot is executed in the receiver's thread.

+ Blocking Queued Connection The slot is invoked as for the Queued Connection, except the current thread blocks until the slot returns.

  Note: Using this type to connect objects in the same thread will cause deadlock.

+ Unique Connection The behavior is the same as the Auto Connection, but the connection is made only if it does not duplicate an existing connection. i.e., if the same signal is already connected to the same slot for the same pair of objects, then the connection is not made and connect() returns false.

### 注意几点

1. Direct Connection，当发送者和接收者属于不用的线程时，是不安全的假如在接收者的线程中正在运行一个事件循环。这与调用其他线程的对象函数是一样的，这个对象可能在接收事件。
2. QObject::connect() 是线程安全的。