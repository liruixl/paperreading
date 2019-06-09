/*
	delegate.cpp

	A delegate that allows the user to change integer values from the model
	using a spin box widget.
*/

#include "delegate.h"

#include <QSpinBox>

SpinBoxDelegate::SpinBoxDelegate(QObject *parent)
	: QStyledItemDelegate(parent)
{
}

QWidget *SpinBoxDelegate::createEditor(QWidget *parent,
	const QStyleOptionViewItem &/* option */,
	const QModelIndex &/* index */) const
{
	QSpinBox *editor = new QSpinBox(parent);
	editor->setFrame(false);
	editor->setMinimum(0);
	editor->setMaximum(100);

	return editor;
}

void SpinBoxDelegate::setEditorData(QWidget *editor,
	const QModelIndex &index) const
{
	int value = index.model()->data(index, Qt::EditRole).toInt()+20;

	//the view treats delegates as ordinary QWidget instances, 
	//we have to use a static cast before we can set the value in the spin box.
	QSpinBox *spinBox = static_cast<QSpinBox*>(editor);
	spinBox->setValue(value);
}

void SpinBoxDelegate::setModelData(QWidget *editor, QAbstractItemModel *model,
	const QModelIndex &index) const
{
	QSpinBox *spinBox = static_cast<QSpinBox*>(editor);
	spinBox->interpretText();//up to data
	int value = spinBox->value();

	model->setData(index, value, Qt::EditRole);
}

void SpinBoxDelegate::updateEditorGeometry(QWidget *editor,
	const QStyleOptionViewItem &option, const QModelIndex &/* index */) const
{
	editor->setGeometry(option.rect);
}
