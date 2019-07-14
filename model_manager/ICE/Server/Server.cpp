#include <Ice/Ice.h>
#include <Printer.h>

using namespace PrinterService;

#include <iostream>
using namespace std;

class PrinterI : public Printer {
public:
	virtual void printString(const string & s,
		const Ice::Current &);
};


void PrinterI::printString(const string & s, const Ice::Current &)
{
	cout << s << endl;
}
int main(int argc, char* argv[])
{

	cout<<"lirui"<<endl;
	int status = 0;
	Ice::CommunicatorPtr ic;
	try {
		//初始化ICE运行库
		ic = Ice::initialize(argc, argv);
		//建立ObjectAdapter（对象适配器）
		//名字，绑定端口：默认tcp 10000端口监听
		Ice::ObjectAdapterPtr adapter
			= ic->createObjectAdapterWithEndpoints(
			"SimplePrinterAdapter", "default -p 10000");

		//为我们的 Printer 接口创建一个 servant
		Ice::ObjectPtr object = new PrinterI;

		//调用适配器的 add，告诉它有了一个新的 servant ；
		//加入ObjectAdapter，并命名为SimplePrinter，即标识符
		adapter->add(object,ic->stringToIdentity("SimplePrinter"));
		//激活适配器
		//Ice run time 就会开始把请 求分派给适配器的 servants。
		adapter->activate();

		// //等待直到Communicator关闭
		ic->waitForShutdown();
	} catch (const Ice::Exception & e) {
		cerr << e << endl;
		status = 1;
	} catch (const char * msg) {
		cerr << msg << endl;
		status = 1;
	}
	if (ic)
		ic->destroy();
	return status;
}