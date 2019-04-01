# _T

```c++
CString strTemp=_T("MFC Tutorial");
```

_T是什么意思？

当工程是Unicode编码时，_T将括号内的字符串以Unicode方式保存；

当工程是多字节编码时，_T将括号内的字符串以ANSI方式保存。

关键字“L”，则是不管编码环境是什么，都是将其后面的字符串以Unicode方式保存。

Unicode字符是LPCWSTR

ASCII字符是LPCSTR

通过强制转换是无法完成的，需要用_T()和L()来完成转换

# 智能指针

[一文说尽C++智能指针的点点滴滴](https://www.itcodemonkey.com/article/10733.html)

# =delete

[C++11 标准新特性：Defaulted 和 Deleted函数](https://www.ibm.com/developerworks/cn/aix/library/1212_lufang_c11new/index.html)



# 信号量

Semaphore是旗语的意思，在Windows中，Semaphore对象用来控制对资源的并发访问数。Semaphore对象具有一个计数值，当值大于0时，**Semaphore被置信号**，当计数值等于0时，Semaphore被清除信号。每次针对Semaphore的wait functions返回时，计数值被减1，调用ReleaseSemaphore可以将计数值增加 lReleaseCount 参数值指定的值。 

1）创建信号量：

```c++
HANDLE CreateSemaphore(
  LPSECURITY_ATTRIBUTES lpSemaphoreAttributes,
  LONG lInitialCount,
  LONG lMaximumCount,
  LPCTSTR lpName
);
```

函数说明：

第一个参数表示安全控制，一般直接传入NULL。

第二个参数表示初始资源数量。

第三个参数表示最大并发数量。

第四个参数表示信号量的名称，传入NULL表示匿名信号量。

2）打开信号量：

```c++
HANDLE OpenSemaphore(
  DWORD dwDesiredAccess,
  BOOL bInheritHandle,
  LPCTSTR lpName
);
```

函数说明：

第一个参数表示访问权限，对一般传入SEMAPHORE_ALL_ACCESS。详细解释可以查看MSDN文档。

第二个参数表示信号量句柄继承性，一般传入TRUE即可。

第三个参数表示名称，不同进程中的各线程可以通过名称来确保它们访问同一个信号量。

3）ReleaseSemaphore :

```c++
BOOL ReleaseSemaphore(
  HANDLE hSemaphore,
  LONG lReleaseCount, 
  LPLONG lpPreviousCount
);
```

函数说明：

第一个参数是信号量的句柄。

第二个参数表示增加个数，必须大于0且不超过最大资源数量。

第三个参数可以用来传出先前的资源计数，设为NULL表示不需要传出。

4）信号量的清理与销毁 :

由于信号量是内核对象，因此使用CloseHandle()就可以完成清理与销毁了。 

# 线程

CreateThread

```c++
HANDLE WINAPI CreateThread(
    LPSECURITY_ATTRIBUTES   lpThreadAttributes, //线程安全相关的属性，常置为NULL
    SIZE_T                  dwStackSize,        //新线程的初始化栈大小，可设置为0
    LPTHREAD_START_ROUTINE  lpStartAddress,     //被线程执行的回调函数，也称为线程函数
    LPVOID                  lpParameter,        //传入线程函数的参数，不需传递参数时为NULL
    DWORD                   dwCreationFlags,    //控制线程创建的标志
    LPDWORD                 lpThreadId          //传出参数，用于获得线程ID，如果为NULL则不返回线程
);
```

dwCreationFlags：表示创建线程的运行状态，其中CREATE_SUSPEND表示挂起当前创建的线程，而0表示立即执行当前创建的进程； 

如果函数调用成功，则返回新线程的句柄，调用WaitForSingleObject函数等待所创建线程的运行结束。函数的格式如下： 

```c++

DWORD WaitForSingleObject(
                          HANDLE hHandle,
                          DWORD dwMilliseconds
                         );
```

