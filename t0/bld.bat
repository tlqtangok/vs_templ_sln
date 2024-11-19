:: verify vs env 
@echo off 
where cl 2>nul
if %errorlevel% equ 1 (
echo no VS env !!!
goto EOF_
)
@echo on 


@echo off
pushd %0\..

set /a "paramCount=0"

for %%i in (%*) do (
    set /a "paramCount+=1"
)


set "msg= "
if %paramCount%  equ 0 (
echo must run one of following...
echo __________________________
echo %msg%  bld Scan.sln
echo %msg%  bld ClassifyReport.sln
echo %msg%  bld Analysis.sln
echo %msg%  bld t0.sln
echo __________________________

goto EOF_
)


echo bld %1 start
msbuild  /p:Configuration=Debug %1 
::msbuild  /p:Configuration=Release %1 
if %errorlevel% equ 0 (
echo .
echo %1 build ok
) else (
echo ____________________________________
echo ______ %1 build error ERROR ______
echo ____________________________________
)
echo bld %1 end

:EOF_

@echo on
