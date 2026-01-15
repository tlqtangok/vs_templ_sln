:: change config to Debug, only for Debug
@echo off

setlocal
cd |repl "^.*\\" ""  > %tmp%\basename.txt

set /p projname=<%tmp%\basename.txt

dir /s /b /AD |grepw /V _3rd | grepw "%projname%$ .vs$ x64$ .vscode$" > txt.txt 
dir /a /s /b *.pdb  *.aps tags *.swp >> txt.txt
dir /a /s /b *.obj  *.idb  *.ilk  *.exp *.pdb *.tlog *.exe.recipe  *.lastbuildstate  t0.exe >> txt.txt



type txt.txt | tol | repl "^" "del /Q " |cmd /k 

del txt.txt
endlocal
@echo on
