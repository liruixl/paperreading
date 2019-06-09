# 注意

1. 所有的信号声明都是公共的，所以Qt规定不能在signals前面加public,private, protected。 
2. 所有的信号都没有返回值，所以返回值都用void。 
3. 所有的信号都不需要定义。 
4. 必须直接或间接继承自QOBject类，并且开头私有声明包含Q_OBJECT。 
5. 当一个信号发出时，会立即执行其槽函数，等待槽函数执行完毕后，才会执行后面的代码，如果一个信号链接了多个槽，那么会等所有的槽函数执行完毕后才执行后面的代码，槽函数的执行顺序是按照它们链接时的顺序执行的。 
6. 在链接信号和槽时，可以设置链接方式为：在发出信号后，不需要等待槽函数执行完，而是直接执行后面的代码。 
7. 发出信号使用emit关键字。 
8. 信号参数的个数不得少于槽参数的个数。 

# 控件随着窗口变化

一定要有顶级布局！！！！

点击设计页面的空白处，或者对象查看其中的QMainWindows，设置布局。

# 信号槽连接

5个重载：

```c++
QMetaObject::Connection connect(const QObject *, const char *,
                                const QObject *, const char *,
                                Qt::ConnectionType);

QMetaObject::Connection connect(const QObject *, const QMetaMethod &,
                                const QObject *, const QMetaMethod &,
                                Qt::ConnectionType);

QMetaObject::Connection connect(const QObject *, const char *,
                                const char *,
                                Qt::ConnectionType) const; //默认接收者为this

//指向成员函数的指针
QMetaObject::Connection connect(const QObject *, PointerToMemberFunction,
                                const QObject *, PointerToMemberFunction,
                                Qt::ConnectionType)

//最后一个参数是Functor类型。这个类型可以接受 static 函数、全局函数以及 Lambda 表达式。
QMetaObject::Connection connect(const QObject *, PointerToMemberFunction,
                                Functor);
```



1）Qt4和Qt5都可以使用这种连接方式

```c++
static QMetaObject::Connection connect(
    const QObject *sender, //信号发送对象指针
    const char *signal,    //信号函数字符串，使用SIGNAL()
    const QObject *receiver, //槽函数对象指针
    const char *member, //槽函数字符串，使用SLOT()
    Qt::ConnectionType = Qt::AutoConnection//连接类型，一般默认即可
);

//例如
connect(pushButton, SIGNAL(clicked()), dialog,  SLOT(close()));
```

2）Qt5，Qt5新增这种连接方式，这使得在编译期间就可以进行拼写检查，参数检查，类型检查，并且支持相容参数的兼容性转换。 

```c++
static QMetaObject::Connection connect(
    const QObject *sender, //信号发送对象指针
    const QMetaMethod &signal,//信号函数地址
    const QObject *receiver, //槽函数对象指针
    const QMetaMethod &method,//槽函数地址
    Qt::ConnectionType type = Qt::AutoConnection//连接类型，一般默认即可
);

//例如
connect(pushButton, QPushButton::clicked, dialog,  QDialog::close);
```

# QT中的Model/View

All item models are based on the [QAbstractItemModel](https://doc.qt.io/qt-5/qabstractitemmodel.html) class.   all subclasses of `QAbstractItemModel ` represent the data as a hierarchical structure containing **tables of items**.

![img](assets/modelview-models.png)

QAbstractItemModel提供了一个数据接口，该接口足够灵活，可以处理以表，列表和树形式表示数据的视图。但是，在为列表和类似表的数据结构实现新模型时，QAbstractListModel和QAbstractTableModel类是更好的起点，因为它们提供了常用函数的适当默认实现。

In the model/view architecture, the model provides **a standard interface** that views and delegates use to access data. 

## 名词解释

+ Model indexes：To obtain a model index that corresponds to an item of data, three properties must be specified to the model: **a row number, a column number, and the model index of a parent item**. 

+ Rows(行) and columns

  ```c++
  QModelIndex indexC = model->index(2, 1, QModelIndex());
  ```

  Top level items in a model are always referenced by specifying `QModelIndex()` as their parent item. 





## view->setModel(model);

```c++
#include <QApplication>
#include <QTableView>
#include "mymodel.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QTableView tableView;
    MyModel myModel;
    tableView.setModel(&myModel);
    tableView.show();
    return a.exec();
}
```

Here is the interesting part: We create an instance of MyModel and use tableView.setModel(&myModel); to pass a pointer of it to tableView. tableView will invoke the methods of the pointer it has received to find out two things调用它收到的指针的方法来找出两件事：

+ How many rows and columns should be displayed.
+ What content should be printed into each cell.

该模型需要一些代码响应这一点。

```c++
// mymodel.h
#include <QAbstractTableModel>

class MyModel : public QAbstractTableModel
{
    Q_OBJECT
public:
    MyModel(QObject *parent = nullptr);
    int rowCount(const QModelIndex &parent = QModelIndex()) const override;
    int columnCount(const QModelIndex &parent = QModelIndex()) const override;
    QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
};
```

[QAbstractTableModel](https://doc.qt.io/qt-5/qabstracttablemodel.html) requires the implementation of three abstract methods. 

## data/rowCount/columnCount

```c++
// mymodel.cpp
#include "mymodel.h"

MyModel::MyModel(QObject *parent)
    : QAbstractTableModel(parent)
{
}

int MyModel::rowCount(const QModelIndex & /*parent*/) const
{
   return 2;
}

int MyModel::columnCount(const QModelIndex & /*parent*/) const
{
    return 3;
}

QVariant MyModel::data(const QModelIndex &index, int role) const
{
    if (role == Qt::DisplayRole)
       return QString("Row%1, Column%2")
                   .arg(index.row() + 1)
                   .arg(index.column() +1);

    return QVariant();
}
```

在实际的应用程序中，`MyModel`将有一个被调用的成员`MyData`，它作为所有读写操作的目标。 

行和列的数量由MyModel :: rowCount（）和MyModel :: columnCount（）提供。当视图必须知道单元格的文本是什么时，它会调用方法MyModel :: data（）。使用参数指定行和列信息index，并将角色设置为Qt :: DisplayRole。

除了控制视图显示的文本外，模型(Model)还控制文本的外观。当我们稍微改变模型时，我们得到以下结果：



![img](assets/readonlytable_role.png)

设置字体，背景颜色，对齐方式和复选框。下面是生成上面显示的结果的[data（）](https://doc.qt.io/qt-5/qabstractitemmodel.html#data)方法。 这次我们使用参数int role根据其值返回不同的信息。 

```c++
// mymodel.cpp
QVariant MyModel::data(const QModelIndex &index, int role) const
{
    int row = index.row();
    int col = index.column();
    // generate a log message when this method gets called
    qDebug() << QString("row %1, col%2, role %3")
            .arg(row).arg(col).arg(role);

    switch (role) {
    case Qt::DisplayRole:
        if (row == 0 && col == 1) return QString("<--left");
        if (row == 1 && col == 1) return QString("right-->");

        return QString("Row%1, Column%2")
                .arg(row + 1)
                .arg(col +1);
    case Qt::FontRole:
        if (row == 0 && col == 0) { //change font only for cell(0,0)
            QFont boldFont;
            boldFont.setBold(true);
            return boldFont;
        }
        break;
    case Qt::BackgroundRole:
        if (row == 1 && col == 2)  //change background only for cell(1,2)
            return QBrush(Qt::red);
        break;
    case Qt::TextAlignmentRole:
        if (row == 1 && col == 1) //change text alignment only for cell(1,1)
            return Qt::AlignRight + Qt::AlignVCenter;
        break;
    case Qt::CheckStateRole:
        if (row == 1 && col == 0) //add a checkbox to cell(1,0)
            return Qt::Checked;
        break;
    }
    return QVariant();
}
```

## 设置Headers for Columns and Rows

通过View 方法隐藏垂直标题：

```c++
 tableView->verticalHeader()->hide();
```

但是，标题内容是通过模型设置的，因此我们重新实现了[headerData（）](https://doc.qt.io/qt-5/qabstractitemmodel.html#headerData)方法 ：

```c++
QVariant MyModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    if (role == Qt::DisplayRole && orientation == Qt::Horizontal) {
        switch (section) {
        case 0:
            return QString("first");
        case 1:
            return QString("second");
        case 2:
            return QString("third");
        }
    }
    return QVariant();
}
```

## 编辑

模型决定编辑功能是否可用。我们只需要修改模型，以便启用可用的编辑功能。这是通过重新实现以下虚拟方法来完成的：[setData（）](https://doc.qt.io/qt-5/qabstractitemmodel.html#setData)和[flags（）](https://doc.qt.io/qt-5/qabstractitemmodel.html#flags)。 

## 更换View

您可以将上面的示例转换为具有树视图的应用程序。只需用QTreeView替换QTableView，即可生成读/写树。不需要对模型进行任何更改。树将没有任何层次结构，因为模型本身没有任何层次结构。

## Delegates

一般地，视图将数据向用户进行展示并且处理通用的输入。但是，对于某些特殊要求（比如这里的要求必须输入数字），则交予委托完成。这些组件提供输入功能，同时也能渲染某些特殊数据项。委托的接口由`QAbstractItemDelegate`定义。在这个类中，委托通过`paint()`和`sizeHint()`两个函数渲染用户内容（也就是说，你必须自己将渲染器绘制出来）。为使用方便，从 4.4 开始，Qt 提供了另外的基于组件的子类：`QItemDelegate`和`QStyledItemDelegate`。默认的委托是`QStyledItemDelegate`。二者的区别在于绘制和向视图提供编辑器的方式。`QStyledItemDelegate`使用当前样式绘制，并且能够使用 Qt Style Sheet（我们会在后面的章节对 QSS 进行介绍），因此我们推荐在自定义委托时，使用`QStyledItemDelegate`作为基类。不过，除非自定义委托需要自己进行绘制，否则，二者的代码其实是一样的。 

继承`QStyledItemDelegate`需要实现以下几个函数：

+ `createEditor()`：返回一个组件。该组件会被作为用户编辑数据时所使用的编辑器，从模型中接受数据，返回用户修改的数据。
+ `setEditorData()`：提供上述组件在显示时所需要的默认值。
+ `setModelData()`：返回给模型用户修改过的数据。
+ `updateEditorGeometry()`：确保上述组件作为编辑器时能够完整地显示出来。

数据在单元格中显示为文本或复选框，并作为文本或复选框进行编辑。提供这些演示和编辑服务的组件称为*委托*。 View使用一个默认的委托。

需要重写：

```c++
  void paint(QPainter *painter, const QStyleOptionViewItem &option,
               const QModelIndex &index) const override;
```

The [paint()](https://doc.qt.io/qt-5/qabstractitemdelegate.html#paint) function is reimplemented from [QItemDelegate](https://doc.qt.io/qt-5/qitemdelegate.html) and is called whenever the view needs to repaint an item. The function **is invoked once for each item**, represented by a [QModelIndex](https://doc.qt.io/qt-5/qmodelindex.html) object from the model.

```c++
QWidget *createEditor(QWidget *parent,
                      const QStyleOptionViewItem &option,
                      const QModelIndex &index) const override;
//返回一个组件，用户在上面编辑
```

The [createEditor()](https://doc.qt.io/qt-5/qabstractitemdelegate.html#createEditor) function is called when the user **starts editing** an item. return a editor.

```c++
void setEditorData(QWidget *editor, const QModelIndex &index) const override;
//初始化组件的值
```

The [setEditorData()](https://doc.qt.io/qt-5/qabstractitemdelegate.html#setEditorData) function is called when an editor is created to **initialize it** with data from the model.

```c++
void setModelData(QWidget *editor, 
                  QAbstractItemModel *model, 
                  const QModelIndex &index) const override;
//提交数据给模型
```

The [setModelData()](https://doc.qt.io/qt-5/qabstractitemdelegate.html#setModelData) function is called to **commit data from the editor to the model** when editing is finished.



```c++
void updateEditorGeometry(QWidget *editor,
		const QStyleOptionViewItem &option, const QModelIndex &index) const override;
```

`editor->setGeometry(option.rect);`

由于我们的编辑器只有一个数字输入框，所以只是简单将这个输入框的大小设置为单元格的大小（由`option.rect`提供）。如果是复杂的编辑器，我们需要根据单元格参数（由`option`提供）、数据（由`index`提供）结合编辑器（由`editor`提供）计算编辑器的显示位置和大小。 





```c++
QSize sizeHint(const QStyleOptionViewItem &option, const QModelIndex &index) const override;
```

The `sizeHint()` function returns an **item's preferred size**.



# 事件

## Qt事件

Qt程序是事件驱动的, 程序的每个动作都是由幕后某个事件所触发
Qt事件的发生和处理成为程序运行的主线，存在于程序整个生命周期

## 常见Qt事件类型

键盘事件：按键按下和松开
鼠标事件：鼠标移动，鼠标按键的按下和松开
拖放事件：用鼠标进行拖放
滚轮事件：鼠标滚轮滚动
绘屏事件：重绘屏幕的某些部分
定时事件：定时器到时
焦点事件：键盘焦点移动
进入和离开事件：鼠标移入widget之内,或是移出
移动事件：widget的位置改变
大小改变事件：widget的大小改变
显示和隐藏事件：widget显示和隐藏
窗口事件：窗口是否为当前窗口

# QWidget

QWidget类是所有用户界面对象（User Interface Object）的基类。

## composite widget

当一个widget用作容器以对多个子窗口小部件进行分组时，它被称为复合widget。By default, composite widgets which do not provide a size hint will be sized according to the space requirements of their child widgets.

## Events

通常，widgets响应用户操作引起的事件。Qt会通过调用 specific event handler functions 将事件传递给widgets，事件是QEvent的子类的实例，其中包含了关于事件信息

### common events

+ paintEvent() 

  is called whenever the widget needs to be repainted. Every widget displaying custom content must implement it. Painting using a QPainter can only take place in a paintEvent() or a function called by a paintEvent().

+ resizeEvent() 

  is called when the widget has been resized.

+ mousePressEvent() 

  is called when a mouse button is pressed while the mouse cursor is inside the widget.

+ mouseReleaseEvent() 

  is called when a mouse button is released. A widget receives mouse release events when it has received the corresponding mouse press event. This means that if the user presses the mouse inside your widget, then drags the mouse somewhere else before releasing the mouse button, your widget receives the release event. There is one exception: if a popup menu appears while the mouse button is held down, this popup immediately steals the mouse events.

+ mouseDoubleClickEvent() 

  is called when the user double-clicks in the widget. If the user double-clicks, the widget receives a mouse press event, a mouse release event, (a mouse click event,) a second mouse press, this event and finally a second mouse release event. (Some mouse move events may also be received if the mouse is not held steady during this operation.) **It is not possible to distinguish a click from a double-click until the second click arrives**. (This is one reason why most GUI books recommend that double-clicks be an extension of single-clicks, rather than trigger a different action.)

### keyboard

+ keyPressEvent() is called whenever a key is pressed, and again when a key has been held down long enough for it to auto-repeat. The Tab and Shift+Tab keys are only passed to the widget if they are not used by the focus-change mechanisms. To force those keys to be processed by your widget, you must reimplement QWidget::event().
+ focusInEvent() is called when the widget gains keyboard focus (assuming you have called setFocusPolicy()). Well-behaved widgets indicate that they own the keyboard focus in a clear but discreet way.
+ focusOutEvent() is called when the widget loses keyboard focus.

### 不常见less common

+ mouseMoveEvent()只要按住鼠标按钮鼠标移动，就会调用mouseMoveEvent（）。 这在拖放操作期间非常有用。 如果你调用setMouseTracking（true），即使没有按下任何按钮，也会得到鼠标移动事件。
+ wheelEvent() is called whenever the user turns the mouse wheel while the widget has the focus.
+ enterEvent() is called when the mouse enters the widget's screen space. (This excludes screen space owned by any of the widget's children.)
+ leaveEvent() is called when the mouse leaves the widget's screen space. If the mouse enters a child widget it will not cause a leaveEvent().
+ moveEvent() is called when the widget has been moved relative to its parent.
+ **closeEvent()** is called when the user closes the widget (or when close() is called).
+ **showEvent()**

# 图片显示

Qt助手：`image view`关键字。

Label相关：scaledContents、resize()、asjustSize()、

# 鼠标画矩形

<https://blog.csdn.net/seanwang_25/article/details/18667871> 

<https://www.cnblogs.com/lifeng-blog/p/9057509.html> 

<https://blog.csdn.net/cutter_point/article/details/43087497> 

<http://www.qtcn.org/bbs/simple/?t53303.html> 

# 其他

<https://blog.csdn.net/u012627502/article/details/26814049> 