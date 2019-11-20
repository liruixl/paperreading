Qt文档 Synchronizing Threads 的翻译。



多个线程同时写入相同的变量，结果是未定义的。强制线程互相等待的原理称为互斥（mutual exclusion）。 这是保护共享资源（如数据）的常用技术。

Qt提供了低级原语以及用于同步线程的高级机制。

## Low-Level Synchronization Primitives

+ QMutex

  一个线程lock这个mutex以获取共享资源的访问权限，另一个线程则必须等待直到第一个线程unlock这个mutex。

+ QReadWriteLock

  与QMutex相似，但有“read”和“write”的区别，当数据没有被写入时，多个线程可以同时读取它。而QMutex则强迫多个readers轮流去读数据。

+ QSemaphore

  信号量是QMutex的泛化，可以保护一定数量的相同资源。 相反，QMutex仅保护一种资源。 信号量典型应用：在生产者和使用者之间同步访问循环缓冲区。

+ QWaitCondition

  QWaitCondition不是通过强制互斥（mutual exclusion）而是通过提供条件变量（a condition variable）来同步线程。其他原语使线程等待直到资源被解锁，QWaitCondition使线程等待直到满足特定条件。 要允许等待的线程继续进行，请调用 wakeOne() 唤醒一个随机选择的线程，或者调用 wakeAll() 同时唤醒它们。

### 风险

程序将会freeze，当：

没有释放锁，比如发生异常时。

死锁。

RAII：QMutexLocker, QReadLocker and QWriteLocker are convenience classes

## High-Level Event Queues

> Qt's event system（参见文档The Event System） is very useful for inter-thread communication. Every thread may have its own event loop. To call a slot (or any invokable method) in another thread, place that call in the target thread's event loop. This lets the target thread finish its current task before the slot starts running, while the original thread continues running in parallel.
>
> 怎么place that call in the target thread's event loop？
>
> To place an invocation in an event loop, make a queued signal-slot connection. Whenever the signal is emitted, its arguments will be recorded by the event system. The thread that the signal receiver lives in will then run the slot. Alternatively, call QMetaObject::invokeMethod() to achieve the same effect without signals. In both cases, a **queued connection**（参见Qt Namespace， Qt::ConnectionType） must be used because a direct connection bypasses the event system and runs the method immediately in the current thread.
>
> There is no risk of deadlocks when using the event system for thread synchronization, unlike using low-level primitives. However, the event system does not enforce mutual exclusion. If invokable methods access shared data, they must still be protected with low-level primitives.
>
> Having said that, Qt's event system, along with **implicitly shared** （参见文档Implicit Sharing ）data structures, offers an alternative to traditional thread locking. If signals and slots are used exclusively and no variables are shared between threads, a multithreaded program can do without low-level primitives altogether.

## 一些疑惑

Qt线程的事件循环？参见QThread::exec()、文档Threads and QObjects 

怎么place that call in the target thread's event loop？

Thread Affinity？参见QObject类。

隐式共享数据结构？

