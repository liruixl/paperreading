#include <Ice/Ice.h>
#include <Printer.h>

using namespace PrinterService;
using namespace std;

int main(int argc, char * argv[])
{
	int status = 0;
	Ice::CommunicatorPtr ic;
	try {
		ic = Ice::initialize(argc, argv);

		//获取远地打印机的代理。
		Ice::ObjectPrx base = ic->stringToProxy(
			"SimplePrinter:default -h 192.168.0.31 -p 10000");
		//进行向下转换。这个方法会发送一
		//条消息给服务器，实际询问 “这是 Printer 接口的代理吗？”
		PrinterPrx printer = PrinterPrx::checkedCast(base);
		if (!printer)
			throw "Invalid proxy";

		//我们的地址空间里有了一个活的代理！！
		cout<<"代理调用print"<<endl;
		printer->printString("Hello World!");
		cout<<"代理调用print完成"<<endl;
		
	} catch (const Ice::Exception & ex) {
		cerr << ex << endl;
		status = 1;
	} catch (const char * msg) {
		cerr << msg << endl;
		status = 1;
	}
	if (ic)
		ic->destroy();
	return status;
}