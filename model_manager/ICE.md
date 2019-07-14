摘自Ice 3.7.1 Documentation 

# Writing an Ice Application with C++ (C++11) 

## Compiling a Slice Definition for C++ 

1. A Slice File： `Printer.ice`

   ```c++
   module Demo
   {
       interface Printer
       {
       	void printString(string s);
       }
   }
   ```

2. 编译slice定义

   `slice2cpp Printer.ice`产生两个c++源文件：`Printer.h`和`Printer.cpp`.

   Printer.h头文件包含我们的Printer接口的slice定义相应的C++类型定义。此头文件必须包含在客户端和服务器的源代码中。

   Printer.cpp文件包含Printer接口的源代码。包含了对客户端和服务器的特定类型的运行时支持。例如：它包含编组参数（marshal parameter data ）的代码在客户端，和解组数据在服务端。The Printer.cpp file must be compiled and linked into both client and server.

## Writing and Compiling a Server in C++ 

### 代码Server.cpp：

```c++
#include <Ice/Ice.h>
#include <Printer.h>
using namespace std;
using namespace Demo;

//这个类叫servant类
class PrinterI : public Printer
{
public:
    virtual void printString(string s, const Ice::Current&) override;
};
void PrinterI::printString(string s, const Ice::Current&)
{
    cout << s << endl;
}
int main(int argc, char* argv[])
{
    try
    {
        //持有Ice::CommunicatorPtr ic;
         Ice::CommunicatorHolder ich(argc, argv);
        //Ice::ObjectAdapterPtr 适配器类型
         auto adapter = ich->createObjectAdapterWithEndpoints(
        	"SimplePrinterAdapter", "default -p 10000");
        //Ice::ObjectPtr 服务类型，这里用了智能指针
         auto servant = make_shared<PrinterI>();
         adapter->add(servant, Ice::stringToIdentity("SimplePrinter"));
         adapter->activate();
         ich->waitForShutdown();
    }
    catch(const std::exception& e)
    {
    cerr << e.what() << endl;
    return 1;
    }
	return 0;
}
```

我们实现了a single printer servant，of type PrinterI.     继承子Printer.h中的Printer类：

```c++
//去掉了一些不相关的细节
namespace Demo
{
    class Printer : public virtual Ice::Object
{
    public:
    //纯虚函数，类不能被实例话
        virtual void printString(std::string, const Ice::Current&) = 0;
    };
}
```

这个Printer skeleton class 被Slice编译器生成，我们的servant类继承于此提供了纯虚方法实现（按照惯例，我们使用`I`后缀来表示该类实现了一个接口。）

方法中有第二个参数`Ice::Current`，这里我们先不管此参数。

### 主方法：

一般框架：

```c++
int
main(int argc, char* argv[])
{
    try
    {
        Ice::CommunicatorHolder ich(argc, argv);
        // Server implementation here ...
    }
    catch(const std::exception& e)
    {
        cerr << e.what() << endl;
        return 1;
    }
    return 0;
}
```

try/catch block。在栈上创建了一个CommunicatorHolder。我们将argc和argv传递给CommunicatorHolder，因为服务器可能具有运行时感兴趣的命令行参数; 对于这个例子，服务器不需要任何命令行参数。

1. 创建了一个适配器（an object adapter）通过调用Communicator实例上的方法createObjectAdapterWithEndpoints（通过调用CommunicatorHolder的重载箭头操作符）。此方法有两个参数：1）适配器名字，2）`"default -p 10000"`，指示适配器在端口号10000使用默认协议（TCP/IP）侦听传入请求。
2. 此时，服务端运行时( the server-side run time)被初始化。并且我们实例化`PrinterI`对象作为服务方。
3. 我们通知the object adapter存在一个新的servant通过调用适配器上的`add`方法。参数是：servant对象还有一个标识符。此例中，字符串“SimplePrinter”是Ice对象的名称。
4. 接下来，通过调用适配器的`activate`方法来激活适配器。适配器以一种holding状态初始化，这在有多个servant共享一个adapter，并且在实例化所有setvant不想处理请求的情况下会很有用。 The server starts to process incoming requests from clients as soon as the adapter is activated.
5. 最后，我们调用waitForShutdown。此调用将挂起调用线程直到服务器terminates，方法是调用关闭运行时或响应信号。（现在，当我们不再需要它时，我们将简单地在命令行上中断服务器。）

## Writing and Compiling a Client in C++ 

客户端代码看起来与服务端非常相似：

代码：

```c++
#include <Ice/Ice.h>
#include <Printer.h>
#include <stdexcept>
using namespace std;
using namespace Demo;
int main(int argc, char* argv[])
{
    try
	{
        //持有Ice::CommunicatorPtr ic;
        Ice::CommunicatorHolder ich(argc, argv);
        //Ice::ObjectPrx代理类型，不是服务器的Ice::ObjectPtr像指针的类型
        auto base = ich->stringToProxy("SimplePrinter:default -p 10000");
        //PrinterPrx
        auto printer = Ice::checkedCast<PrinterPrx>(base);
        if(!printer)
        {
            throw std::runtime_error("Invalid proxy");
        }
        printer->printString("Hello World!");
    }
    catch(const std::exception& e)
    {
         cerr << e.what() << endl;
            return 1;
    }
	return 0;
}
```

请注意，整个代码布局与服务器相同。头文件、try/catch块。

1. 与服务端一样，通过创建an Ice::CommunicatorHolder object来初始化ICE运行时（Ice run time），那创建并且持有了一个`Ice ::Communicator`。
2. 下一步是获得远程方法的代理（proxy），We create a proxy by calling `stringToProxy` on the communicator，参数是服务端使用过的对象标识符和端口号。在这里，硬编码不是一个好方法。
3. `stringToProxy` 返回的代理类型是`Ice::ObjectPrx`，但是我们需要的是一个`Printer`接口的代理，而不是一个Object 接口，我们需要down-cast来获得需要的代理。A checked cast sends a message to the server, effectively asking "is this a proxy for a Printer interface?" If so, the call returns a proxy to a Printer; otherwise, if the proxy denotes an interface of some other type, the call returns a null proxy.
4. 测试是否转化成功，假如没有，抛出运行时异常。
5. 我们现在得到了一个代理在我们的地址空间中，我们调用其方法`printString`，服务端在其终端打印出”Hello World!“字符串。

## Running Client and Server in C++ 

