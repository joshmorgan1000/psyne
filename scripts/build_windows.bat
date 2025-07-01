@echo off
REM Build script for Windows

echo Building Psyne tests for Windows...

REM Set up Visual Studio environment
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" 2>nul
if errorlevel 1 call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" 2>nul
if errorlevel 1 call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" 2>nul
if errorlevel 1 call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" 2>nul
if errorlevel 1 (
    echo ERROR: Could not find Visual Studio installation
    exit /b 1
)

echo.
echo Compiling test_windows.cpp...
cl /std:c++20 /O2 /EHsc /I"include" tests\test_windows.cpp /Fe:test_windows.exe
if errorlevel 1 (
    echo ERROR: Compilation failed
    exit /b 1
)

echo.
echo Running tests...
test_windows.exe
if errorlevel 1 (
    echo ERROR: Tests failed
    exit /b 1
)

echo.
echo SUCCESS: All Windows tests passed!