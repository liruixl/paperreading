#pragma once

#include <QAbstractTableModel>

class CurrencyModel : public QAbstractTableModel
{
	Q_OBJECT

public:
	CurrencyModel(QObject *parent = 0);
	~CurrencyModel();

	//model
	int columnCount(const QModelIndex &parent) const override;
	int rowCount(const QModelIndex &parent = QModelIndex()) const override;
	QVariant data(const QModelIndex &index, int role) const override;
	QVariant headerData(int section, Qt::Orientation orientation, int role) const override;

	//edit
	Qt::ItemFlags flags(const QModelIndex &index) const override;
	bool setData(const QModelIndex &index, const QVariant &value, int role = Qt::EditRole);

	void setCurrencyMap(const QMap<QString, double> &map);


private:
	QString currencyAt(int offset) const;  //ÁÐÃû
	QMap<QString, double>  currencyMap;
};
