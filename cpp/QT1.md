# 对象树

![1554376724281](assets/1554376724281.png)

# 信号的overload

```c++
void (QSpinBox:: *spinBoxSignal)(int) = &QSpinBox::valueChanged;

//这个函数指针如果用以下2句代码替换：
void (*spinBoxSignal)(int);
spinBoxSignal = &QSpinBox::valueChanged;
```

运行时就会指示：
error: no matches converting function ‘valueChanged’ to type ‘void (*)(int)’spinBoxSignal = &QSpinBox::valueChanged;

为什么会这样呢？valueChanged的原型不就是 void valueChanged(int)吗？

答：

成员函数指针需要类实例去调用，因此成员函数指针和全局函数指针是不兼容的。你修改之后的代码：void (*spinBoxSignal)(int); 定义了一个全局函数指针，是不能用成员函数 QSpinBox::valueChanged 进行赋值的，这也就是为什么编译器会报无法转换的错误。 

# layout析构问题

QHBoxLayout对象也是new出来的，并且没有父对象，这种情况还需要delete吗？我自己的代码里在类里面定义了QHBoxLayout*成员对象，在类的构造函数里new出来，没有指定父对象，需要在类的析构函数中delete，并置空吗？ 

layout 这类对象，在调用 QWidget::setLayout() 函数时，会将这个 layout 的 parent 自动设置为该 widget（查阅文档，有 reparent 一段描述）。因此，如果你的 parent 能够一直追溯到一个可管理的组件（也就是能够被正确 delete 的对象），就不需要自己 delete 这个 layout，否则应该自己管理。 

# exec

为什么建立pushbutton等widget时就不需要exec(), show()等， QDialog就需要呢，是因为它比较特殊吗？ 

对话框是一个独立的窗口，需要有自己的事件循环，接收用户响应，而按钮之类只是组件，是由窗口的事件循环支持的。 

# tr

main函数中不能直接使用`tr()`？？

tr()函数是定义在QObject里面的，所有使用了Q_OBJECT宏的类都自动具有tr()的函数。和connect函数一样，只有继承了QObject所以能够直接使用。 

# &

通常使用含字符'&'的字符串为按钮的显示名称，如果设置按钮的text为"&Cancel", 即设置text, setText("&Cancel");或创建时QPushButton *pushButton = new QPushButton (QObject::tr("&Cancel")); Qt的编译器会将字符'&'后的'C'在显示时下方多一下划线，表明'C'为该按钮的快捷键，通过"Alt＋c"操作来实现对pushButton的点击。 

有可能是你没有添加 Q_OBJECT 宏。这个函数是 Q_OBJECT 宏展开的。或者你的类不是 QObject 子类，这样的话就要用 QObject::tr() 这样的静态函数。 

# [=]Lamada函数

```c++
connect(textEdit, &QTextEdit::textChanged, [=]() {
this->setWindowModified(true);
});//中的[=]()是什么意思？
```

# 事件处理

现在我们可以总结一下 Qt 的事件处理，实际上是有五个层次：

1. 重写`paintEvent()`、`mousePressEvent()`等事件处理函数。这是最普通、最简单的形式，同时功能也最简单。
2. 重写`event()`函数。`event()`函数是所有对象的事件入口，`QObject`和`QWidget`中的实现，默认是把事件传递给特定的事件处理函数。
3. 在特定对象上面安装事件过滤器。该过滤器仅过滤该对象接收到的事件。
4. 在`QCoreApplication::instance()`上面安装事件过滤器。该过滤器将过滤所有对象的所有事件，因此和`notify()`函数一样强大，但是它更灵活，因为可以安装多个过滤器。全局的事件过滤器可以看到 disabled 组件上面发出的鼠标事件。全局过滤器有一个问题：只能用在主线程。
5. 重写`QCoreApplication::notify()`函数。这是最强大的，和全局事件过滤器一样提供完全控制，并且不受线程的限制。但是全局范围内只能有一个被使用（因为`QCoreApplication`是单例的）。

```c++
class Label : public QWidget
{
public:
    Label()
    {
        installEventFilter(this);
    }

    bool eventFilter(QObject *watched, QEvent *event)
    {
        if (watched == this) {
            if (event->type() == QEvent::MouseButtonPress) {
                qDebug() << "eventFilter";
            }
        }
        return false;
    }

protected:
    void mousePressEvent(QMouseEvent *)
    {
        qDebug() << "mousePressEvent";
    }

    bool event(QEvent *e)
    {
        if (e->type() == QEvent::MouseButtonPress) {
            qDebug() << "event";
        }
        return QWidget::event(e);
    }
};

class EventFilter : public QObject
{
public:
    EventFilter(QObject *watched, QObject *parent = 0) :
        QObject(parent),
        m_watched(watched)
    {
    }

    bool eventFilter(QObject *watched, QEvent *event)
    {
        if (watched == m_watched) {
            if (event->type() == QEvent::MouseButtonPress) {
                qDebug() << "QApplication::eventFilter";
            }
        }
        return false;
    }

private:
    QObject *m_watched;
};

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    Label label;
    app.installEventFilter(new EventFilter(&label, &label));
    label.show();
    return app.exec();
}
```

结果：

```
QApplication::eventFilter 
eventFilter 
event 
mousePressEvent
```

# 头文件循环包含

这个问题我一般是这么解决的： 1.header1.h文件中进行类声明：class ClassA;（因为.h中只用到了类的指针，因此不需要类的具体定义）； 2.在source1.cpp中，包含ClassA的定义文件header2.h（因为.cpp里会用到类的函数等） 

<https://blog.csdn.net/hazir/article/details/38600419> 