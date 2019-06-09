#include "CurrencyModel.h"

CurrencyModel::CurrencyModel(QObject *parent)
	: QAbstractTableModel(parent)
{
}

CurrencyModel::~CurrencyModel()
{
}

int CurrencyModel::columnCount(const QModelIndex &parent) const
{
	return currencyMap.count();
}

int CurrencyModel::rowCount(const QModelIndex & parent) const
{
	return currencyMap.count();
}

QVariant CurrencyModel::data(const QModelIndex & index, int role) const
{
	if (!index.isValid()) { return QVariant(); }

	

	if(role == Qt::TextAlignmentRole)
	{
		return int(Qt::AlignRight | Qt::AlignVCenter);//QVariant(int i);
	}
	//DisplayRole is what you show to the user and 
	//EditRole is what you will load in the editor, 
	//else if (role == Qt::DisplayRole || Qt::EditRole)
	
	else if (role == Qt::DisplayRole || role == Qt::EditRole)
	{
		int row = index.row();
		int col = index.column();

		double currencyRow = (currencyMap.begin() + row).value();
		double currencyCol = (currencyMap.begin() + col).value();

		if (currencyRow == 0)
		{
			return "###";
		}

		double amount = currencyCol / currencyRow;
		return  QString("%1").arg(amount, 0, 'f', 4);
	}

	return QVariant();
}

QVariant CurrencyModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	if (role != Qt::DisplayRole)
	{
		return QVariant();
	}
	return currencyAt(section);
}

Qt::ItemFlags CurrencyModel::flags(const QModelIndex & index) const
{
	
	Qt::ItemFlags flags = QAbstractItemModel::flags(index); //这函数也不是静态的。？怎么调用的
	if (index.row() != index.column()) {
		flags |= Qt::ItemIsEditable;
	}
	
	return flags;
}

bool CurrencyModel::setData(const QModelIndex & index, const QVariant & value, int role)
{
	if (index.isValid()
		&& index.row() != index.column()
		&& role == Qt::EditRole) {
		QString columnCurrency = headerData(index.column(),
			Qt::Horizontal, Qt::DisplayRole)
			.toString();
		QString rowCurrency = headerData(index.row(),
			Qt::Vertical, Qt::DisplayRole)
			.toString();
		currencyMap.insert(columnCurrency,
			value.toDouble() * currencyMap.value(rowCurrency));
		emit dataChanged(index, index);
		return true;
	}
	return false;
}

void CurrencyModel::setCurrencyMap(const QMap<QString, double> &map)
{
	beginResetModel();
	currencyMap = map;
	endResetModel();
}

QString CurrencyModel::currencyAt(int offset) const
{
	return (currencyMap.begin()+offset).key();
}
