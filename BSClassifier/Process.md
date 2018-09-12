# ConvertToDataTable

```c#
public DataTable ConvertToDataTable<T>(IEnumerable<T> varlist){
    //IEnumerable对象转换为datatable
}
```

1. 获取属性**名字及类型**的例子

```c#
var a = new { haha = 2, hahaha = "abc" };//假设a是其中一个元素
Console.WriteLine(a);  //{ haha = 2, hahaha = abc }
Console.WriteLine(a.GetType());
//<>f__AnonymousType1`2[System.Int32,System.String]

PropertyInfo[] myPropertyInfo = a.GetType().GetProperties();    //获取某个Type的属性信息

for (int i = 0; i < myPropertyInfo.Length; i++)
{
    Console.WriteLine(myPropertyInfo[i].ToString());
    Console.WriteLine(myPropertyInfo[i]);
    //Int32 haha
    //System.String hahaha
    Console.WriteLine(myPropertyInfo[i].Name);    //名字
    //haha
    //hahaha
    Console.WriteLine(myPropertyInfo[i].PropertyType);   //类型
    //System.Int32
	//System.String           
}
```

2. 反射，Type这个类对范型有许多特别的属性。 例如`IsGenericType`，[`IsGenericTypeDefinition`](https://msdn.microsoft.com/zh-cn/library/system.type.isgenerictypedefinition(v=vs.110).aspx)，`IsGenericParameter`，`ContainsGenericParameters`；方法：`GetGenericTypeDefinition()`，[`GetGenericArguments()`](https://msdn.microsoft.com/zh-cn/library/system.type.getgenericarguments(v=vs.110).aspx)。

关于方法例子参考：[1](https://blog.csdn.net/norsd/article/details/45871011)，[2](https://blog.csdn.net/lunasea0_0/article/details/6257395)  。

对于泛型和反射参考：[C#中的泛型](http://www.cnblogs.com/rush/archive/2011/06/11/2078493.html)和[泛型和反射](https://www.cnblogs.com/rush/archive/2012/09/30/2709113.html)。[使用反射检查和实例化泛型类型](https://docs.microsoft.com/zh-cn/dotnet/framework/reflection-and-codedom/how-to-examine-and-instantiate-generic-types-with-reflection)。

C#中提供五种泛型分别是：**classes, structs, interfaces, delegates, and methods**。 

> A generic type definition, from which other generic types can be constructed 
>
> 他们都是泛型，typeof(List\<int>).IsGenericType是true，但它不是泛型定义。typeof(List<>).IsGenericTypeDefinition是true 。



# loadPicture

出现异常用默认黑图替代

```c#
public Image loadPicture(string fp)
{
    Image image = null;
    System.IO.FileStream fs = null;
    try
    {
        fs = new System.IO.FileStream(fp, System.IO.FileMode.Open, System.IO.FileAccess.Read);
        image = System.Drawing.Image.FromStream(fs);
        //fs.Close();
    }
    catch (Exception e)
    {
        string unseenpath = System.Environment.CurrentDirectory.ToString();
        unseenpath = unseenpath + "\\data\\unseen.tif";
        fs = new System.IO.FileStream(unseenpath, System.IO.FileMode.Open, System.IO.FileAccess.Read);
        image = System.Drawing.Image.FromStream(fs);
        //fs.Close();
        LogHelper.WriteLog(string.Format("写入路径不正确：{0}", fp), e);
        System.Diagnostics.Debug.Write(e.Message.ToString());
    }
    finally
    {
        if (fs != null)
        {
            fs.Close();
        }
    }
    return image;
}
```





# manipulateMDBData

```c#
public  DataTable manipulateMDBData(string mdbPath, string sql, string selectFlag){
    //mdb使用sql语句进行查询,不再负责增删改,参数selectFlag没有用到啊。。。什么鬼？
    DataTable dt = new DataTable();
    try
    {
        //建立连接   
        string strConn
            = @"Provider=Microsoft.Jet.OLEDB.4.0;Data Source=" + mdbPath + "\\defectinfo.mdb;";
        OleDbConnection odcConnection = new OleDbConnection(strConn);
        //打开连接   
        odcConnection.Open();
        OleDbDataAdapter myAdapter = new OleDbDataAdapter(sql, odcConnection);
        myAdapter.Fill(dt);
        odcConnection.Close();
        odcConnection.Dispose();
        myAdapter.Dispose();
        return dt;
    }
    catch(Exception e)
    {
        LogHelper.WriteLog(string.Format("操作mdb文件失败,{0}",sql), e);
        return dt;
    }
}
```

关于`OleDbDataAdapter`和`OleDbCommand `的[一些例子](https://blog.csdn.net/wzk456/article/details/80609363)。

[C#与数据库访问技术总结（十四）之DataAdapter对象](https://www.cnblogs.com/zi-xing/p/4058090.html)





# ReadAllData

```c#
public DataTable ReadAllData(string mdbPath, ref bool success, string tablename)
     {
            DataTable dt = new DataTable();
            try
            {
                DataRow dr;
                string strConn = @"Provider=Microsoft.Jet.OLEDB.4.0;Data Source="
                                    + mdbPath + "\\defectinfo.mdb;";
                OleDbConnection odcConnection = new OleDbConnection(strConn);//1.数据库连接
                odcConnection.Open();//2.要打开数据库连接
                OleDbCommand odCommand = odcConnection.CreateCommand();//3.通过连接创建命令对象
                odCommand.CommandText = "select * from " + tablename;//4.设置SQL语句
                //5.将发送CommandText到Connection并生成OleDbDataReader
                //是从数据源读取数据行的只进流
                OleDbDataReader odrReader = odCommand.ExecuteReader();
                int size = odrReader.FieldCount;//获取列数
                for (int i = 0; i < size; i++)
                {
                    DataColumn dc;
                    dc = new DataColumn(odrReader.GetName(i));//新建列，第二个参数可以指定类型
                    dt.Columns.Add(dc);//1.先把列弄好
                }
                while (odrReader.Read())
                {
                    dr = dt.NewRow();//2.新建空行
                    for (int i = 0; i < size; i++)
                    {
                        //3.赋值
                        dr[odrReader.GetName(i)] = odrReader[odrReader.GetName(i)].ToString();
                    }
                    dt.Rows.Add(dr);//4.添加行
                }
                //dt.DefaultView.Sort = "缺陷类id";
                odrReader.Close();
                odrReader.Dispose();//释放资源
                odcConnection.Close();
                success = true;
                return dt;
            }
            catch (Exception e)
            {
                // GC.Collect();
                success = false;
                Console.WriteLine("读取失败");
                Console.WriteLine(e);
                return dt;
            }
        }
```

函数涉及三个知识点：

1. ref与out都是传引用，[区别参考这里](https://www.cnblogs.com/windinsky/archive/2009/02/13/1390071.html)。

2. DataTable

   > 可以把DataTable和DataSet看做是数据容器，比如你查询数据库后得到一些结果，可以放到这种容器里，那你可能要问：我不用这种容器，自己读到变量或数组里也一样可以存起来啊，为什么用容器？
   >
   > 原因是，这种容器的功能比较强大，除了可以存数据，还可以有更大用途。举例：在一个c/s结构的桌面数据库系统里，你可以把前面存放查询结果的容器里的数据显示到你客户端界面上，用户在界面上对数据进行添加、删除、修改，你可以把用户的操作更新到容器，等用户操作完毕了，要求更新，然后你才把容器整个的数据变化更新到中心数据库，这样做的好处是什么？就是减少了数据库操作，客户端速度提高了，数据库压力减小了。
   >
   > DataSet可以比作一个内存中的数据库，DataTable是一个内存中的数据表，DataSet里可以存储多个DataTable。
   >
   > DataSet：数据集。一般包含多个DataTable，用的时候，dataset["表名"]得到DataTable  
   >
   > DataTable：数据表。  

   简单操作参见[C# DataTable 详解](https://www.cnblogs.com/Sandon/p/5175829.html)

3. mdb数据库文件读取按照代码操作

   ```c#
   OleDbCommand cmd = new OleDbCommand(sql, connection); //这样也可以执行
   cmd.ExecuteNonQuery();
   ```

# searchData

   ```c#
public DataTable searchData(string mdbPath, string sql){
    //查询，调用manipulateMDBData
}
   ```

# ShowListViewByDt

```c#
public void ShowListViewByDt(ListView listView, ArrayList filepath, string groupname,
                             Form classifier){
    //缺陷缩略图的显示
    //涉及到ListView类的使用
    //涉及到图片的读取：loadPicture方法
    //还有进度条progressbar类
}
```

[ListView类的使用](https://blog.csdn.net/chen_zw/article/details/7910324)

关键代码（省略）

```c#
ImageList il = new ImageList();//管理一套Image对象
listView.LargeImageList = il;
ListViewItem[] lvi = new ListViewItem[filepath.Count];
for (int i = 0; i < totallength; i++){
	lvi[i] = new ListViewItem(new string[] { filename, groupname, fp }, i, groupsArray);
    //...
    Image image = getImg(filepath);
    il.Images.Add(image);
}
listView.Items.Add(lvi[i]);
```

上边`new ListViewItem`：

> Initializes a new instance of the ListViewItem class with the specified item text and the image index position of the item's icon, and assigns the item to the specified group

image index应该对应`listView.LargeImageList`里的`ImageList`。

