下载：[MinGW-w64 - for 32 and 64 bit Windows](https://link.zhihu.com/?target=https%3A//sourceforge.net/projects/mingw-w64/files/) 往下稍微翻一下，选最新版本中的`x86_64-posix-seh`。

官方docs：<https://code.visualstudio.com/docs/cpp/config-mingw> 、



1. Add your source code

2. you will create a `tasks.json` file to tell VS Code how to build (compile) the program. 

   ```json
   
   // https://code.visualstudio.com/docs/editor/tasks
   {
       "version": "2.0.0",
       "tasks": [{
           "label": "Compile", // 任务名称，与launch.json的preLaunchTask相对应
           "command": "g++",   // 要使用的编译器，C++用g++
           "args": [
               "${file}",
               "-o",    // 指定输出文件名，不加该参数则默认输出a.exe，Linux下默认a.out
               "${fileDirname}/${fileBasenameNoExtension}.exe",
               "-g",    // 生成和调试有关的信息
               "-Wall", // 开启额外警告
               "-static-libgcc",     // 静态链接libgcc，一般都会加上
               "-fexec-charset=GBK", // 生成的程序使用GBK编码，不加这一条会导致Win下输出中文乱码
               //"-std=c11", // C++最新标准为c++17，或根据自己的需要进行修改
           ], // 编译的命令，其实相当于VSC帮你在终端中输了这些东西
           "type": "process", // process是vsc把预定义变量和转义解析后直接全部传给command；shell相当于先打开shell再输入命令，所以args还会经过shell再解析一遍
           "group": {
               "kind": "build",
               "isDefault": true // 不为true时ctrl shift B就要手动选择了
           },
           "presentation": {
               "echo": true,
               "reveal": "always", // 执行任务时是否跳转到终端面板，可以为always，silent，never。具体参见VSC的文档
               "focus": false,     // 设为true后可以使执行task时焦点聚集在终端，但对编译C/C++来说，设为true没有意义
               "panel": "shared"   // 不同的文件的编译信息共享一个终端面板
           },
           // "problemMatcher":"$gcc" // 此选项可以捕捉编译时终端里的报错信息；但因为有Lint，再开这个可能有双重报错
       }]
   }
   ```

   Modifying tasks.json

   You can modify your `tasks.json` to build multiple C++ files by using an argument like `"${workspaceFolder}\\*.cpp"` instead of `${file}`. This will build all `.cpp` files in your current folder. You can also modify the output filename by replacing `"${fileDirname}\\${fileBasenameNoExtension}.exe"` with a hard-coded filename (for example `"${workspaceFolder}\\myProgram.exe"`)

3. Next, you'll create a `launch.json` file to configure VS Code to launch the GDB debugger when you press F5 to debug the program. From the main menu, choose **Run** > **Add Configuration...** and then choose **C++ (GDB/LLDB)**. 

   ```json
   {
     "version": "0.2.0",
     "configurations": [
       {
         "name": "g++.exe build and debug active file",
         "type": "cppdbg",
         "request": "launch",
         "program": "${fileDirname}\\${fileBasenameNoExtension}.exe",
         "args": [],
         "stopAtEntry": false,
         "cwd": "${workspaceFolder}",
         "environment": [],
         "externalConsole": false,
         "MIMode": "gdb",
         "miDebuggerPath": "C:\\mingw-w64\\i686-8.1.0-posix-dwarf-rt_v6-rev0\\mingw32\\bin\\gdb.exe", //配置PATH填写 gbd.exe即可
         "setupCommands": [
           {
             "description": "Enable pretty-printing for gdb",
             "text": "-enable-pretty-printing",
             "ignoreFailures": true
           }
         ],
         "preLaunchTask": "g++.exe build active file" //task.json label 字段
       }
     ]
   }
   ```

4. 多文件编译

   如果你想进行少量的多文件编译，C语言直接用gcc 源文件1.c 源文件2.c 头文件1.h这样就好，C++用g++。默认生成a.exe，加-o可指定输出文件名，其余选项百度gcc使用教程。如果需要多次编译可以写一个批处理。 

   如果你想进行大量的多文件编译，请学习如何写makefile或使用cmake。然后把tasks的命令改成调用make等。

   如果你想使用别人的库，比如ffmpeg，可能需要在命令中指定-I、-l（小写的L）、-L。具体参数阅读那个库的文档。还可能需要把路径添加到c_cpp_properties.json里来配置Intellisense。

   这些情况下可以考虑单独建一个工作区，不要和单文件编译的共用。其实不新建工程(Project)、只是单文件就能调试，是不利于以后使用和理解大型IDE的。不过初学也不用掌握那么多，不要觉得建工程很麻烦、不建工程就能编译很强就是了。

   总之这些和VSC无关，用其它IDE或是手动编译也会遇到差不多的问题，也有点复杂。本文就不多讨论这些了，自行解决。

   

    

    

    

    

    