#include "SpinBox.h"
#include "delegate.h"
#include <QtWidgets/QApplication>

#include <QHeaderView>
#include <QStandardItemModel>
#include <QTableView>
#include <QtCore>
#include "model/CurrencyModel.h"

int main(int argc, char *argv[])
{
	QApplication app(argc, argv);

	QTranslator* trans = new QTranslator;
	trans->load("spinbox_zh.ts");
	app.installTranslator(trans);

	//With QMap, the items are always sorted by key.
	//底层数据改变，视图会自动刷新??
	//分页？
	QMap<QString, double> data;
	data["USD"] = 1.0000;
	data["CNY"] = 0.1628;
	data["GBP"] = 1.5361;
	data["EUR"] = 1.2992;
	data["HKD"] = 0.1289;

	QTableView view;
	CurrencyModel *model = new CurrencyModel();
	//CurrencyModel *model = new CurrencyModel(&view);
	model->setCurrencyMap(data);
	view.setModel(model);
	view.resize(400, 300);
	view.show();




	/*QStandardItemModel model(4, 2);
	QTableView tableView;
	tableView.setModel(&model);

	SpinBoxDelegate delegate;
	tableView.setItemDelegate(&delegate);

	tableView.horizontalHeader()->setStretchLastSection(false);

	for (int row = 0; row < 4; ++row) {
		for (int column = 0; column < 2; ++column) {
			QModelIndex index = model.index(row, column, QModelIndex());
			model.setData(index, QVariant((row + 1) * (column + 1)));
		}
	}

	tableView.setWindowTitle(QObject::tr("Spin Box Delegate"));
	tableView.show();*/

	return app.exec();
}
