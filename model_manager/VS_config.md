配置好各种目录可打造一个清爽的目录结构

# model_manager项目

就记录一下此项目配置

SDK
10.0.14393.

## 输出目录

$(SolutionDir)bin\$(PlatformTarget)\$(Configuration)\$(ProjectName)\
E:\Judgment\bin\x64\Debug\model_manager
.exe 文件目录

## 中间目录

$(SolutionDir)tmp\$(ProjectName)\$(PlatformTarget)\$(Configuration)\
E:\Judgment\tmp\model_manager\x64\Debug
.obj文件目录

## 平台工具集

141

## 生成前事件

```c++
copy "$(SolutionDir)src\terminal\terminal_zh.qm" "$(SolutionDir)config\lang\" /Y

md "$(OutputPath)config"
copy /d "$(SolutionDir)config\terminal*" "$(OutputPath)config\"

md "$(OutputPath)config\lang"
copy /d "$(SolutionDir)config\lang\*" "$(OutputPath)config\lang\"

if "$(Configuration)" == "Debug" goto Debug
if "$(Configuration)" == "Release" goto Release

:Debug
copy "$(SolutionDir)3rdparty\ice\ice 3.7.0\bin\vc141\$(Platform)\Debug\ice37d.dll" "$(OutputPath)" /Y
copy "$(SolutionDir)3rdparty\ice\ice 3.7.0\bin\vc141\$(Platform)\Debug\icestorm37d.dll" "$(OutputPath)" /Y
copy "$(SolutionDir)3rdparty\ice\ice 3.7.0\bin\vc141\$(Platform)\Debug\bzip2d.dll" "$(OutputPath)" /Y
copy "$(SolutionDir)3rdparty\libconfig\bin\libconfig++.dll" "$(OutputPath)" /Y
copy "$(QTDIR)\bin\Qt5Cored.dll" "$(OutputPath)" /Y
copy "$(QTDIR)\bin\Qt5Guid.dll" "$(OutputPath)" /Y
copy "$(QTDIR)\bin\Qt5Widgetsd.dll" "$(OutputPath)" /Y
copy "$(SolutionDir)3rdparty\opencv\opencv 3.2.0\bin\opencv_world320d.dll" "$(OutputPath)" /Y
md "$(OutputPath)\iconengines"
copy "$(QTDIR)\plugins\iconengines\qsvgicond.dll" "$(OutputPath)\iconengines\" /Y
md "$(OutputPath)\imageformats"
copy "$(QTDIR)\plugins\imageformats\qgifd.dll" "$(OutputPath)\imageformats\" /Y
copy "$(QTDIR)\plugins\imageformats\qicnsd.dll" "$(OutputPath)\imageformats\" /Y
copy "$(QTDIR)\plugins\imageformats\qicod.dll" "$(OutputPath)\imageformats\" /Y
copy "$(QTDIR)\plugins\imageformats\qjpegd.dll" "$(OutputPath)\imageformats\" /Y
copy "$(QTDIR)\plugins\imageformats\qsvgd.dll" "$(OutputPath)\imageformats\" /Y
copy "$(QTDIR)\plugins\imageformats\qtgad.dll" "$(OutputPath)\imageformats\" /Y
copy "$(QTDIR)\plugins\imageformats\qtiffd.dll" "$(OutputPath)\imageformats\" /Y
copy "$(QTDIR)\plugins\imageformats\qwbmpd.dll" "$(OutputPath)\imageformats\" /Y
copy "$(QTDIR)\plugins\imageformats\qwebpd.dll" "$(OutputPath)\imageformats\" /Y
md "$(OutputPath)\platforms"
copy "$(QTDIR)\plugins\platforms\qwindowsd.dll" "$(OutputPath)\platforms\" /Y
md "$(OutputPath)\styles"
copy "$(QTDIR)\plugins\styles\qwindowsvistastyled.dll" "$(OutputPath)\styles\" /Y
exit

:Release
copy "$(SolutionDir)3rdparty\ice\ice 3.7.0\bin\vc141\$(Platform)\Release\ice37.dll" "$(OutputPath)" /Y
copy "$(SolutionDir)3rdparty\ice\ice 3.7.0\bin\vc141\$(Platform)\Release\icestorm37.dll" "$(OutputPath)" /Y
copy "$(SolutionDir)3rdparty\ice\ice 3.7.0\bin\vc141\$(Platform)\Release\bzip2.dll" "$(OutputPath)" /Y
copy "$(SolutionDir)3rdparty\libconfig\bin\libconfig++.dll" "$(OutputPath)" /Y
copy "$(QTDIR)\bin\Qt5Core.dll" "$(OutputPath)" /Y
copy "$(QTDIR)\bin\Qt5Gui.dll" "$(OutputPath)" /Y
copy "$(QTDIR)\bin\Qt5Widgets.dll" "$(OutputPath)" /Y
copy "$(SolutionDir)3rdparty\opencv\opencv 3.2.0\bin\opencv_world320.dll" "$(OutputPath)" /Y
md "$(OutputPath)\iconengines"
copy "$(QTDIR)\plugins\iconengines\qsvgicon.dll" "$(OutputPath)\iconengines\" /Y
md "$(OutputPath)\imageformats"
copy "$(QTDIR)\plugins\imageformats\qgif.dll" "$(OutputPath)\imageformats\" /Y
copy "$(QTDIR)\plugins\imageformats\qicns.dll" "$(OutputPath)\imageformats\" /Y
copy "$(QTDIR)\plugins\imageformats\qico.dll" "$(OutputPath)\imageformats\" /Y
copy "$(QTDIR)\plugins\imageformats\qjpeg.dll" "$(OutputPath)\imageformats\" /Y
copy "$(QTDIR)\plugins\imageformats\qsvg.dll" "$(OutputPath)\imageformats\" /Y
copy "$(QTDIR)\plugins\imageformats\qtga.dll" "$(OutputPath)\imageformats\" /Y
copy "$(QTDIR)\plugins\imageformats\qtiff.dll" "$(OutputPath)\imageformats\" /Y
copy "$(QTDIR)\plugins\imageformats\qwbmp.dll" "$(OutputPath)\imageformats\" /Y
copy "$(QTDIR)\plugins\imageformats\qwebp.dll" "$(OutputPath)\imageformats\" /Y
md "$(OutputPath)\platforms"
copy "$(QTDIR)\plugins\platforms\qwindows.dll" "$(OutputPath)\platforms\" /Y
md "$(OutputPath)\styles"
copy "$(QTDIR)\plugins\styles\qwindowsvistastyle.dll" "$(OutputPath)\styles\" /Y
exit
```

### 计算的值

```c++
copy "E:\Judgment\src\terminal\terminal_zh.qm" "E:\Judgment\config\lang\" /Y

md "E:\Judgment\bin\x64\Debug\terminal\config"
copy /d "E:\Judgment\config\terminal*" "E:\Judgment\bin\x64\Debug\terminal\config\"

md "E:\Judgment\bin\x64\Debug\terminal\config\lang"
copy /d "E:\Judgment\config\lang\*" "E:\Judgment\bin\x64\Debug\terminal\config\lang\"

if "Debug" == "Debug" goto Debug
if "Debug" == "Release" goto Release

:Debug
copy "E:\Judgment\3rdparty\ice\ice 3.7.0\bin\vc141\x64\Debug\ice37d.dll" "E:\Judgment\bin\x64\Debug\terminal\" /Y
copy "E:\Judgment\3rdparty\ice\ice 3.7.0\bin\vc141\x64\Debug\icestorm37d.dll" "E:\Judgment\bin\x64\Debug\terminal\" /Y
copy "E:\Judgment\3rdparty\ice\ice 3.7.0\bin\vc141\x64\Debug\bzip2d.dll" "E:\Judgment\bin\x64\Debug\terminal\" /Y
copy "E:\Judgment\3rdparty\libconfig\bin\libconfig++.dll" "E:\Judgment\bin\x64\Debug\terminal\" /Y
copy "D:\Qt\5.12.2\msvc2017_64\bin\Qt5Cored.dll" "E:\Judgment\bin\x64\Debug\terminal\" /Y
copy "D:\Qt\5.12.2\msvc2017_64\bin\Qt5Guid.dll" "E:\Judgment\bin\x64\Debug\terminal\" /Y
copy "D:\Qt\5.12.2\msvc2017_64\bin\Qt5Widgetsd.dll" "E:\Judgment\bin\x64\Debug\terminal\" /Y
copy "E:\Judgment\3rdparty\opencv\opencv 3.2.0\bin\opencv_world320d.dll" "E:\Judgment\bin\x64\Debug\terminal\" /Y
md "E:\Judgment\bin\x64\Debug\terminal\\iconengines"
copy "D:\Qt\5.12.2\msvc2017_64\plugins\iconengines\qsvgicond.dll" "E:\Judgment\bin\x64\Debug\terminal\\iconengines\" /Y
md "E:\Judgment\bin\x64\Debug\terminal\\imageformats"
copy "D:\Qt\5.12.2\msvc2017_64\plugins\imageformats\qgifd.dll" "E:\Judgment\bin\x64\Debug\terminal\\imageformats\" /Y
copy "D:\Qt\5.12.2\msvc2017_64\plugins\imageformats\qicnsd.dll" "E:\Judgment\bin\x64\Debug\terminal\\imageformats\" /Y
copy "D:\Qt\5.12.2\msvc2017_64\plugins\imageformats\qicod.dll" "E:\Judgment\bin\x64\Debug\terminal\\imageformats\" /Y
copy "D:\Qt\5.12.2\msvc2017_64\plugins\imageformats\qjpegd.dll" "E:\Judgment\bin\x64\Debug\terminal\\imageformats\" /Y
copy "D:\Qt\5.12.2\msvc2017_64\plugins\imageformats\qsvgd.dll" "E:\Judgment\bin\x64\Debug\terminal\\imageformats\" /Y
copy "D:\Qt\5.12.2\msvc2017_64\plugins\imageformats\qtgad.dll" "E:\Judgment\bin\x64\Debug\terminal\\imageformats\" /Y
copy "D:\Qt\5.12.2\msvc2017_64\plugins\imageformats\qtiffd.dll" "E:\Judgment\bin\x64\Debug\terminal\\imageformats\" /Y
copy "D:\Qt\5.12.2\msvc2017_64\plugins\imageformats\qwbmpd.dll" "E:\Judgment\bin\x64\Debug\terminal\\imageformats\" /Y
copy "D:\Qt\5.12.2\msvc2017_64\plugins\imageformats\qwebpd.dll" "E:\Judgment\bin\x64\Debug\terminal\\imageformats\" /Y
md "E:\Judgment\bin\x64\Debug\terminal\\platforms"
copy "D:\Qt\5.12.2\msvc2017_64\plugins\platforms\qwindowsd.dll" "E:\Judgment\bin\x64\Debug\terminal\\platforms\" /Y
md "E:\Judgment\bin\x64\Debug\terminal\\styles"
copy "D:\Qt\5.12.2\msvc2017_64\plugins\styles\qwindowsvistastyled.dll" "E:\Judgment\bin\x64\Debug\terminal\\styles\" /Y
exit

:Release
copy "E:\Judgment\3rdparty\ice\ice 3.7.0\bin\vc141\x64\Release\ice37.dll" "E:\Judgment\bin\x64\Debug\terminal\" /Y
copy "E:\Judgment\3rdparty\ice\ice 3.7.0\bin\vc141\x64\Release\icestorm37.dll" "E:\Judgment\bin\x64\Debug\terminal\" /Y
copy "E:\Judgment\3rdparty\ice\ice 3.7.0\bin\vc141\x64\Release\bzip2.dll" "E:\Judgment\bin\x64\Debug\terminal\" /Y
copy "E:\Judgment\3rdparty\libconfig\bin\libconfig++.dll" "E:\Judgment\bin\x64\Debug\terminal\" /Y
copy "D:\Qt\5.12.2\msvc2017_64\bin\Qt5Core.dll" "E:\Judgment\bin\x64\Debug\terminal\" /Y
copy "D:\Qt\5.12.2\msvc2017_64\bin\Qt5Gui.dll" "E:\Judgment\bin\x64\Debug\terminal\" /Y
copy "D:\Qt\5.12.2\msvc2017_64\bin\Qt5Widgets.dll" "E:\Judgment\bin\x64\Debug\terminal\" /Y
copy "E:\Judgment\3rdparty\opencv\opencv 3.2.0\bin\opencv_world320.dll" "E:\Judgment\bin\x64\Debug\terminal\" /Y
md "E:\Judgment\bin\x64\Debug\terminal\\iconengines"
copy "D:\Qt\5.12.2\msvc2017_64\plugins\iconengines\qsvgicon.dll" "E:\Judgment\bin\x64\Debug\terminal\\iconengines\" /Y
md "E:\Judgment\bin\x64\Debug\terminal\\imageformats"
copy "D:\Qt\5.12.2\msvc2017_64\plugins\imageformats\qgif.dll" "E:\Judgment\bin\x64\Debug\terminal\\imageformats\" /Y
copy "D:\Qt\5.12.2\msvc2017_64\plugins\imageformats\qicns.dll" "E:\Judgment\bin\x64\Debug\terminal\\imageformats\" /Y
copy "D:\Qt\5.12.2\msvc2017_64\plugins\imageformats\qico.dll" "E:\Judgment\bin\x64\Debug\terminal\\imageformats\" /Y
copy "D:\Qt\5.12.2\msvc2017_64\plugins\imageformats\qjpeg.dll" "E:\Judgment\bin\x64\Debug\terminal\\imageformats\" /Y
copy "D:\Qt\5.12.2\msvc2017_64\plugins\imageformats\qsvg.dll" "E:\Judgment\bin\x64\Debug\terminal\\imageformats\" /Y
copy "D:\Qt\5.12.2\msvc2017_64\plugins\imageformats\qtga.dll" "E:\Judgment\bin\x64\Debug\terminal\\imageformats\" /Y
copy "D:\Qt\5.12.2\msvc2017_64\plugins\imageformats\qtiff.dll" "E:\Judgment\bin\x64\Debug\terminal\\imageformats\" /Y
copy "D:\Qt\5.12.2\msvc2017_64\plugins\imageformats\qwbmp.dll" "E:\Judgment\bin\x64\Debug\terminal\\imageformats\" /Y
copy "D:\Qt\5.12.2\msvc2017_64\plugins\imageformats\qwebp.dll" "E:\Judgment\bin\x64\Debug\terminal\\imageformats\" /Y
md "E:\Judgment\bin\x64\Debug\terminal\\platforms"
copy "D:\Qt\5.12.2\msvc2017_64\plugins\platforms\qwindows.dll" "E:\Judgment\bin\x64\Debug\terminal\\platforms\" /Y
md "E:\Judgment\bin\x64\Debug\terminal\\styles"
copy "D:\Qt\5.12.2\msvc2017_64\plugins\styles\qwindowsvistastyle.dll" "E:\Judgment\bin\x64\Debug\terminal\\styles\" /Y
exit
```

