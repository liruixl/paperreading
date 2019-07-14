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
		//��ʼ��ICE���п�
		ic = Ice::initialize(argc, argv);
		//����ObjectAdapter��������������
		//���֣��󶨶˿ڣ�Ĭ��tcp 10000�˿ڼ���
		Ice::ObjectAdapterPtr adapter
			= ic->createObjectAdapterWithEndpoints(
			"SimplePrinterAdapter", "default -p 10000");

		//Ϊ���ǵ� Printer �ӿڴ���һ�� servant
		Ice::ObjectPtr object = new PrinterI;

		//������������ add������������һ���µ� servant ��
		//����ObjectAdapter��������ΪSimplePrinter������ʶ��
		adapter->add(object,ic->stringToIdentity("SimplePrinter"));
		//����������
		//Ice run time �ͻῪʼ���� ����ɸ��������� servants��
		adapter->activate();

		// //�ȴ�ֱ��Communicator�ر�
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