```c#
public static class BSClassifierHelper{}
```

# changeTypeIdToName

```c#
public static DataTable changeTypeIdToName(ref DataTable dt){
    //将DataTable中缺陷子形状，由数字改为真正缺陷名称
}
```

# dataTableSort

```c#
//重新排序，专用于FirstTable数据排序：sortString写死的。用于显示左下角缺陷列表
public static DataTable dataTableSort(ref DataTable dt)
    {
        string sortString = "轧批 ASC,序号 ASC,炉号 ASC,dpu_id asc,imageLabel asc";
        DataTable dtTemp = dt.Clone();
        DataView dvTemp = dt.DefaultView;    //视角可进行排序在转化成排序后的DataTable
        dvTemp.Sort = sortString;
        dtTemp = dvTemp.ToTable();
        return dtTemp;
    }
```

