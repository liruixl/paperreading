# Window文件目录遍历和WIN32_FIND_DATA结构

WIN32_FIND_DATA结构

```c++
typedef struct _WIN32_FIND_DATA 
{
   DWORD dwFileAttributes; //文件属性
   FILETIME ftCreationTime; // 文件创建时间
   FILETIME ftLastAccessTime; // 文件最后一次访问时间
   FILETIME ftLastWriteTime; // 文件最后一次修改时间
   DWORD nFileSizeHigh; // 文件长度高32位
   DWORD nFileSizeLow; // 文件长度低32位
   DWORD dwReserved0; // 系统保留
   DWORD dwReserved1; // 系统保留
   TCHAR cFileName[ MAX_PATH ]; // 长文件名（最多可达 255 个字符的长文件名），带句点和扩展名
   TCHAR cAlternateFileName[ 14 ]; //8.3格式文件名（句点前最多有8个字符，而扩展名最多可以有3个字符）
} WIN32_FIND_DATA, *PWIN32_FIND_DATA;
```

我们经常用到的有长文件名、和文件的属性。

经常用到有关文件夹的函数：

```c++
_access(char *);     //判断文件或文件夹路径是否合法  <direct.h>
_chdir(char *);      //设置当前目录   <direct.h>
_mkdir(char *)       //创建文件夹
```

下面给出一个简单的例子：

```c++
#include "stdafx.h"
#include <string>
#include <windows.h>
#include <iostream>
using namespace std;

int main() {
	string path = "F:\\pic_test\\20180914 73.03-5.51-1124\\1\\1";
	string str_find = path + "\\*.*";  //必须是这种带通配符*的形式

	WIN32_FIND_DATA ffd;
	HANDLE hFind = FindFirstFile(str_find.c_str(), &ffd); //报错
    
	while (hFind != INVALID_HANDLE_VALUE) {
		if ((ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
			cout << ffd.cFileName << endl;;
		}
		if (!FindNextFile(hFind, &ffd))
			break;
	}
	FindClose(hFind);

	cin.get();
	return 0;
}
```

+ 路径字符串必须是带通配符*的形式
+ str_find.c_str()这里报错：char* 类型形参与LPWSTR 类型的实参不兼容，这是字符编码的问题，主要是因为Unicode字符集，所以对于string的字宽度不一样，导致LPWSTR类型不能直接定义string类型的变量。修改字符集为多字节字符集。

我发现：

```c++
//fileapi.h
#ifdef UNICODE
#define FindFirstFile  FindFirstFileW
#else
#define FindFirstFile  FindFirstFileA
#endif // !UNICODE
```

当为Unicode字符集

```c++
HANDLE __stdcall FindFirstFileW(
    _In_ LPCWSTR lpFileName,         //这里为LPCWSTR类型,CONST WCHAR *
    _Out_ LPWIN32_FIND_DATAW lpFindFileData
    );
```

当为多字节字符集时

```c++

HANDLE __stdcall FindFirstFileA(
    _In_ LPCSTR lpFileName,   //CONST CHAR *
    _Out_ LPWIN32_FIND_DATAA lpFindFileData
    );
```

输出：

![1540545481199](assets/1540545481199.png)

可以看到有当前目录以及上一级目录（一定要注意）。