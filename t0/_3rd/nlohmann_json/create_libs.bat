:: create_libs.bat debug && create_libs.bat rel
@echo off
REM Build fmt library as a static library (.lib)
REM Usage: create_libs.bat [debug|release|rel]
REM Default: debug

REM Parse command line argument
set BUILD_TYPE=debug
if "%1"=="" goto :set_config
if /i "%1"=="debug" set BUILD_TYPE=debug
if /i "%1"=="release" set BUILD_TYPE=release
if /i "%1"=="rel" set BUILD_TYPE=release

:set_config
REM Define compiler options for both configurations
REM Debug: Keep debug info but optimize for size
set VS_CL_OPT_DEBUG=/c /Zi /nologo /W3 /O1 /Oy /Gy /Gw /D DEBUG /D _CONSOLE /EHsc /MDd /Zc:inline /permissive- /TP
set VS_LIB_OPT_DEBUG=/NOLOGO /MACHINE:X64
set VS_LINK_OPT_DEBUG=/MANIFEST /manifest:embed /DEBUG:FASTLINK /OPT:REF /OPT:ICF /SUBSYSTEM:CONSOLE /MACHINE:X64

REM Release: Maximum optimization for smallest size
REM /O1 = Minimize size, /Oi = Intrinsic functions, /Oy = Omit frame pointers
REM /GL = Whole program optimization, /GS- = Disable security checks, /Gy = Function-level linking
REM /Gw = Optimize global data, /GF = String pooling

set VS_CL_OPT_RELEASE=/c /nologo /W3 /O1 /Oi /Oy /GS- /Gy /Gw /GF /D NDEBUG /D _CONSOLE /EHsc /MD /Zc:inline /permissive- /TP
set VS_LIB_OPT_RELEASE=/NOLOGO /MACHINE:X64
set VS_LINK_OPT_RELEASE=/OPT:REF /OPT:ICF /SUBSYSTEM:CONSOLE /MACHINE:X64

set VS_CL_OPT_RELEASE=/c /nologo /W3 /O1 /Oi /Oy /GS- /Gy /Gw /GF /D NDEBUG /D _CONSOLE /EHsc /MD /Zc:inline /permissive- /TP
set VS_LIB_OPT_RELEASE=/NOLOGO /MACHINE:X64
set VS_LINK_OPT_RELEASE=/OPT:REF /OPT:ICF /SUBSYSTEM:CONSOLE /MACHINE:X64


REM Set active configuration
if "%BUILD_TYPE%"=="debug" (
    set VS_CL_OPT=%VS_CL_OPT_DEBUG%
    set VS_LIB_OPT=%VS_LIB_OPT_DEBUG%
    set VS_LINK_OPT=%VS_LINK_OPT_DEBUG%
    set CONFIG_NAME=Debug
) else (
    set VS_CL_OPT=%VS_CL_OPT_RELEASE%
    set VS_LIB_OPT=%VS_LIB_OPT_RELEASE%
    set VS_LINK_OPT=%VS_LINK_OPT_RELEASE%
    set CONFIG_NAME=Release
)

set INC=%CD%\single_include

REM Check if Visual Studio environment is set up
where cl.exe >nul 2>&1
if not %errorlevel% equ 0 (
    echo ERROR: Please setup your VS2022 environment
    echo Run: "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    exit /b -1
)

echo ============================================
echo Building nlohmann_json %CONFIG_NAME% Library (Minimal Size)
echo ============================================
echo.

REM Clean up old files
del *.obj *.exe *.pdb *.ilk 2>nul

echo Step 1: Compiling nlohmann_json.cpp [%CONFIG_NAME%]...
CL.exe %VS_CL_OPT% /I"%INC%" nlohmann_json.cpp

if not %errorlevel% equ 0 (
    echo ERROR: Failed to compile nlohmann_json.cpp
    exit /b 1
)

echo.
echo Step 2: Creating static library nlohmann_json.lib...
lib.exe %VS_LIB_OPT% /OUT:"nlohmann_json.lib" nlohmann_json.obj

if not %errorlevel% equ 0 (
    echo ERROR: Failed to create library
    exit /b 1
)

echo.
echo Step 3: Compiling demo.cpp [%CONFIG_NAME%]...
CL.exe %VS_CL_OPT% /I"%INC%" demo.cpp

if not %errorlevel% equ 0 (
    echo ERROR: Failed to compile demo.cpp
    exit /b 1
)

echo.
echo Step 4: Linking demo.exe...
link.exe %VS_LINK_OPT% /out:"demo.exe" demo.obj nlohmann_json.lib

if not %errorlevel% equ 0 (
    echo ERROR: Failed to link demo.exe
    exit /b 1
)

echo.
echo ============================================
echo SUCCESS! %CONFIG_NAME% Library and demo created
echo ============================================
echo.
echo Files created:
echo   - nlohmann_json.lib (static library - %CONFIG_NAME%)
echo   - nlohmann_json.h (header file for your projects)
echo   - demo.exe (test program)
echo.
echo Running demo...
echo.
.\demo.exe

echo.
echo.
echo To use in your projects:
echo   1. Copy nlohmann_json.lib to your project
echo   2. Include: #include "nlohmann_json.h"
echo   3. Link with: nlohmann_json.lib
echo   4. Add include path: /I"%INC%"
echo.
if 0 == 1 (
echo Usage: create_libs.bat [debug^|release^|rel]
echo   Default: debug
echo.
echo Done!
)
