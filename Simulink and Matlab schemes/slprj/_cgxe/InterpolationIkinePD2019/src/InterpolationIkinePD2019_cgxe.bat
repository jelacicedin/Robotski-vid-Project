@echo off
set COMPILER=cl
                set COMPFLAGS=/c /Zp8 /GR /W3 /EHs /nologo /MD /D_CRT_SECURE_NO_DEPRECATE /D_SCL_SECURE_NO_DEPRECATE /D_SECURE_SCL=0   /DMATLAB_MEX_FILE  /DMATLAB_MEX_FILE
                set OPTIMFLAGS=/O2 /Oy- /DNDEBUG
                set DEBUGFLAGS=/Z7
                set LINKER=link
                set LINKFLAGS=/nologo /manifest  /export:%ENTRYPOINT% /DLL /LIBPATH:"F:\Program Files\MATLAB\R2019b\extern\lib\win64\microsoft" libmx.lib libmex.lib libmat.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /EXPORT:mexFunction
                set LINKDEBUGFLAGS=/debug /PDB:"%OUTDIR%%MEX_NAME%.mexw64.pdb"
                set NAME_OUTPUT=/out:"%OUTDIR%%MEX_NAME%%MEX_EXT%"
set PATH=D:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.23.28105\bin\HostX64\x64\;D:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\IDE\VC\vcpackages;D:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\IDE;D:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools;D:\Windows Kits\10\\Bin\10.0.18362.0\x64;D:\Windows Kits\10\\Bin\10.0.18362.0\x86;D:\Windows Kits\10\\Bin\x64;D:\Windows Kits\10\\Bin\x86;;%MATLAB_BIN%;%PATH%
set INCLUDE=D:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.23.28105\include;D:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.23.28105\atlmfc\include;D:\Windows Kits\10\\include\10.0.18362.0\ucrt;D:\Windows Kits\10\\include\10.0.18362.0\shared;D:\Windows Kits\10\\include\10.0.18362.0\um;D:\Windows Kits\10\\include\10.0.18362.0\winrt;F:\Program Files\MATLAB\R2019b\extern\include;%INCLUDE%
set LIB=D:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.23.28105\lib\x64;D:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.23.28105\atlmfc\lib\x64;D:\Windows Kits\10\\Lib\10.0.18362.0\ucrt\x64;D:\Windows Kits\10\\lib\10.0.18362.0\um\x64;F:\Program Files\MATLAB\R2019b\lib\win64;%LIB%
set LIBPATH=D:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.23.28105\lib\x64;D:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.23.28105\atlmfc\lib\x64;%LIBPATH%

nmake -f InterpolationIkinePD2019_cgxe.mak
