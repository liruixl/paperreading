# C++11实现

ThreadPoolManager.h

```c++
#pragma once

#include <thread>
#include <vector>
#include <queue>
#include <condition_variable>
#include <future>
#include <atomic>
#include <functional>
#include <stdexcept>

class ThreadPoolManager
{
public:
	ThreadPoolManager(size_t = 8);

	template<class F, class... Args>
	auto addTask(F&& f, Args&&... args)
		->std::future<typename std::result_of<F(Args...)>::type>;

	~ThreadPoolManager();

private:

	//std::function<void()> getTask() {};

	std::vector<std::thread> pool;
	std::queue<std::function<void()>> tasks;
	//std::queue<std::packaged_task<std::function<void()> > > tasks;
	

	// synchronization
	std::mutex queue_mutex;
	std::condition_variable condition;
	bool stop;
};

template<class F, class ...Args>
inline auto ThreadPoolManager::addTask(F && f, Args && ...args) -> std::future<typename std::result_of<F(Args ...)>::type>
{
	// don't allow enqueueing after stopping the pool
	if (stop)
		throw std::runtime_error("enqueue on stopped ThreadPool");
	
	using return_type = typename std::result_of<F(Args...)>::type;

	auto task = std::make_shared<std::packaged_task<return_type()>>(
		std::bind(std::forward<F>(f), std::forward<Args>(args)...) 
		);
	std::future<return_type> res = task->get_future();
	
	{  
		std::unique_lock<std::mutex> lock(queue_mutex);
		tasks.emplace([task]() { (*task)(); });
	}

	condition.notify_one(); 
	return res;
}

```

ThreadPoolManager.cpp

```c++
#include "ThreadPoolManager.h"

ThreadPoolManager::ThreadPoolManager(size_t threadsNum):
	stop(false)
{
	for (size_t i = 0; i < threadsNum; ++i)
	{
		pool.emplace_back(
			[this]
			{
				while(true)
				{
					std::function<void()> task;

					{
						std::unique_lock<std::mutex> lock(this->queue_mutex);
						this->condition.wait(lock,
							[this] { return this->stop || !this->tasks.empty(); });
						if (this->stop && this->tasks.empty()) 
							return;
						task = std::move(this->tasks.front());
						this->tasks.pop();
					}

					task();

				}
			}
		);
	}
}


ThreadPoolManager::~ThreadPoolManager()
{
	{
		std::unique_lock<std::mutex> lock(queue_mutex);
		stop = true;
	}
	condition.notify_all();
	for (std::thread &thread : pool)
		thread.join();
}

```

# 线程池代码解析

+ 模板
  + 可变参数函数模板
  + 模板实参推断与引用（引用折叠，和右值引用参数）
  + 尾置返回类型与类型转换
  + 转发（保持实参的所有性质）
+ future库（packaged_task模板类、future模板类）
+ functional库（std::fuction、std::bind）



一个一个来说：

## 可变参数函数模板

```c++
template<class F, class... Args>
inline auto ThreadPoolManager::addTask(F && f, Args &&... args)
```

有两种参数包：

+ 模板参数包，0或多个，例如：class ... 或 typename... ：就意味着接下来的参数表示零个或多个类型的列表。
+ 函数参数包，0或多个，函数参数列表中，如果一个参数的类型是一个模板参数包，则此参数是一个函数参数包。`Args && ...args`，表示`args`是函数参数包。`Args && ...`是包扩展。

就上面两行代码来说，声明了`addTask`是一个可变参数函数模板，有一个名为`F`的类型参数和一个名为`Args`模板参数包，表示0或多个额外的类型参数。

参数列表中，含有一个右值引用，指向`F`类型；另一个是函数参数包。

## 包扩展

我们通过在模式右边放一个省略号(...)，来触发扩展操作。

例如代码中的：

`Args &&... args`，省略号不是与后面的参数结合，而是扩展Args。编译器将模式`Args &&`应用到模板参数包`Args`中的每一个元素。

`rest...`，则是对函数参数包的扩展。

` std::forward<Args>(args)...`也是包扩展，不过它即扩展了模板参数包`Args`，也扩展了函数参数包`args`。相当于`std::forward<Ti>(ti)`。

## 模板实参推断与引用（引用折叠，和右值引用参数）

c++ primer16.2.5

```c++
template<class F, class... Args>
inline auto ThreadPoolManager::addTask(F && f, Args &&... args)    
//为什么这里传递的是参数的右值引用？？？
```

+ 从左值引用函数参数推断类型（形如 T&）
  + 只能传递给它一个左值（一个变量、一个返回引用的表达式），不能是右值，如果实参是const的，则T被推断为const类型。（注意：右值引用变量是变量，变量也是左值）
  + 如果函数参数是const T &，可以传递给它任何类型的实参。此时 T 的推断不会是const类型的。const已经是函数参数类型的一部分，他不会也是模板参数类型的一部分。
+ 从右值引用函数参数推断类型（T &&）
  + 正常绑定规则告诉我们可以传递给它一个右值。此时，推断出的T类型是该右值实参的类型。例如：f(99)，则推断模板参数 T 为 int。
  + 如果 i 是 一个 int 对象，我们认为`f(i)`这样的调用不合法，i 是左值，我们不能将右值引用绑定到左值。但是在这正常的绑定规则外，有两个例外规则，这两个例外规则是 move 这种标准库设施正确的基础。
+ 引用折叠和右值引用传参（T &&）
  + 例外1：**我们将一个左值传递给函数的右值引用参数**，如`f(i)`，变量 i 是int类型，函数的右值引用参数指向模板类型参数时，**编译器推断 T 为实参的左值引用类型**。即：T 推断为 int &。
    那么，此时，函数的参数为 int&  &&，是一个类型为 int & 的右值引用（通常我们不能定义一个引用的引用，但是通过**类型别名**、或者**模板类型参数**间接定义是可以的），这就引出了例外2的绑定规则。
  + 例外2：如果我们间接创建的引用的引用，这些“引用”形成了折叠。对于一个给定的类型X：
    + X&  &、X&  && 和 X&& & 都折叠成 X&
    + 类型X&& &&折叠成 X&&
  + **这两个规则导致了两个重要结果**：
    + 如果一个函数参数是一个——指向模板类型参数的右值引用（如，T&&），则它可以被绑定到一个左值；且
    + 如果实参是一个左值，则推断出模板实参类型将是一个左值引用，且函数将被实例化为一个（普通）左值引用参数（T&）。

所以，如果一个函数参数是指向模板参数类型的右值引用（T &&），则可以传递给它任意类型的实参。

## std::move

**std::move就是一个使用右值引用的模板的一个很好的例子哦。**

```c++
template<class _Ty>
	_NODISCARD constexpr remove_reference_t<_Ty>&&
		move(_Ty&& _Arg) noexcept
	{	// forward _Arg as movable
	return (static_cast<remove_reference_t<_Ty>&&>(_Arg));
	}
```

可以接受任意类型的实参：

+ 左值（推断为实参的左值引用类型）
  + int a = 42，传递 a ，T 推断为 `int &`，函数参数折叠为 `move(int &)`
  + int &b = a，传递b，T 推断为实参的左值引用类型，即`in& &` = `int &`，函数参数折叠为 `move(int &)`
+ 右值
  + 42，字面值。T 推断为 int，函数参数为 `move(int &&)`

永远返回右值引用`static_cast<remove_reference_t<_Ty>&&>(_Arg)`，这里可以看出，**虽然不能隐式地将一个左值转化为右值引用，但可以通过static_cast显示转换**。



但编写接受右值引用参数的函数模板可能有歧义，例如：

```c++
template<typename T>
void f3(T&& val)
{
    T t = val;
}
```

就有两种可能：

+ 传递右值，比如字面常量42，此时推断 T 为 int，变量 t 赋值语句，与val从此无瓜。
+ 传递左值，如左值 i = 42 ，T 推断为 int &，变量 t 是对val的引用。

由此当涉及到类型可能不定的时候，编写正确的代码就变得困难！！

**实际中，右值引用通常用于两种情况：**

+ 模板转发其实参
+ 模板重载

## 实参转发

16.2.7

某些函数需要将一个或多个实参连同类型不变地转发给其他函数。需要保持被转发实参的所有性质。

与move不同，forward必须通过**显式模板实参**来调用，其**返回该显式实参类型的右值引用**。

即`forward<T>`返回类型为 `T&&`。

```c++
template<class _Ty>
	_NODISCARD constexpr _Ty&& forward(remove_reference_t<_Ty>& _Arg) noexcept
	{	// forward an lvalue as either an lvalue or an rvalue
	return (static_cast<_Ty&&>(_Arg));
	}
```



```c++
template<class _Ty>
	_NODISCARD constexpr _Ty&& forward(remove_reference_t<_Ty>&& _Arg) noexcept
	{	// forward an rvalue as an rvalue
	static_assert(!is_lvalue_reference_v<_Ty>, "bad forward call");
	return (static_cast<_Ty&&>(_Arg));
	}
```

**当用于一个指向模板参数类型的右值引用函数参数（T &&）时，forward会保持实参类型的所有细节。**

线程池代码中，函数参数是右值引用，

```c++
template<class F, class... Args>
inline auto ThreadPoolManager::addTask(F && f, Args &&... args)    
```

参数转发部分，可以保留实参类型的所有细节：

```c++
auto task = std::make_shared<std::packaged_task<return_type()>>(
		std::bind(std::forward<F>(f), std::forward<Args>(args)...) 
		);
```

就用到了转发，这也解释了，为什么函数参数是右值引用。。



## 左右值引用

初始化时，右值引用一定要用一个右值表达式绑定；初始化之后，可以用左值表达式修改右值引用的所引用临时对象的值 。

```c++
int main()
{
	int a = 42;     //0x000000709e31fc34 {42}
	int &b = a;     //0x000000709e31fc34 {42}
    
    //int &&c = a;     //error
    
    //int &&c = std::move(a); //ok
	int &&c = 100;  //0x000000709e31fc94 {100}

	c = a;          //0x000000709e31fc94 {42}
    
	return 0;
}
```

## 尾置返回类型

```c++
 -> std::future<typename std::result_of<F(Args ...)>::type>
 //或者
 -> std::future<decltype(f(args...))>
```

## future

参见[thread](thread.md)

```c++
using return_type = typename std::result_of<F(Args...)>::type;
auto task = std::make_shared<std::packaged_task<return_type()>>(   //智能指针
    std::bind(std::forward<F>(f), std::forward<Args>(args)...) 
);
std::future<return_type> res = task->get_future();

{  
    std::unique_lock<std::mutex> lock(queue_mutex);
    tasks.emplace([task]() { (*task)(); });
}

condition.notify_one(); 
return res;
```

为什么传递智能指针？

？？？？拉姆达表达式捕获列表的原因？？

值捕获会拷贝，如果不用指针的话值拷贝会拷贝packaged_task对象，拷贝构造函数被delete。

如果是这样呢？

```c++
auto task = std::packaged_task<return_type()>( 
    std::bind(std::forward<F>(f), std::forward<Args>(args)...) 
);
tasks.emplace([&task]() { task(); });
//用[&task]() {task();}构造std::function<void()>对象


tasks.emplace([&task]() { (std::move(task))(); });//emplace模板函数，右值引用参数，里面参数转发
```



我不知道。

实验了一下，好像都tm的行唉>>>>>>>>>

```c++

int run()
{
	return 1;
}

int main()
{

	std::queue<std::function<void()>> tasks;

	auto task = std::packaged_task<int()>(run);
	std::future<int> res = task.get_future();

	tasks.emplace([&task]() { std::move(task)(); });

	auto task1 = std::move(tasks.front());
	tasks.pop();
	
	task1();

	cout << res.get() << endl;

	return 0;
}
```





## function

```c++
std::queue<std::function<void()>> tasks;
```

队列中，函数对象是没有返回值的。要得到线程的返回值，所以要用到future的特性：

**addTask()返回值是一个future对象，其绑定到packaged_task对象，packaged_task对象在线程中被某一线程取出运行。之后用户可以通过调用addTask返回值future的`get()`方法取回线程中执行任务的返回值。**



调用get()方法是会阻塞当前线程的。。。