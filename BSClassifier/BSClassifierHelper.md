```c#
public static class BSClassifierHelper{}
```

# changeTypeIdToName

```c#
public static DataTable changeTypeIdToName(ref DataTable dt){
    //将DataTable中缺陷字形状的由数字改为真正缺陷名称
}
```

# dataTableSort

```c#
//重新排序，专用于FirstTable数据排序：sortString写死的。
public static DataTable dataTableSort(ref DataTable dt)
    {
        string sortString = "轧批 ASC,序号 ASC,炉号 ASC,dpu_id asc,imageLabel asc";
        DataTable dtTemp = dt.Clone();
        DataView dvTemp = dt.DefaultView;
        dvTemp.Sort = sortString;
        dtTemp = dvTemp.ToTable();
        return dtTemp;
    }
```

