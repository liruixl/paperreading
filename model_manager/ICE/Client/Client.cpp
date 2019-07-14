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

		//��ȡԶ�ش�ӡ���Ĵ���
		Ice::ObjectPrx base = ic->stringToProxy(
			"SimplePrinter:default -h 192.168.0.31 -p 10000");
		//��������ת������������ᷢ��һ
		//����Ϣ����������ʵ��ѯ�� ������ Printer �ӿڵĴ����𣿡�
		PrinterPrx printer = PrinterPrx::checkedCast(base);
		if (!printer)
			throw "Invalid proxy";

		//���ǵĵ�ַ�ռ�������һ����Ĵ�����
		cout<<"�������print"<<endl;
		printer->printString("Hello World!");
		cout<<"�������print���"<<endl;
		
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