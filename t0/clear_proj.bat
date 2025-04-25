:: change config to Debug, only for Debug
@echo off

setlocal
cd |repl "^.*\\" ""  > %tmp%\basename.txt

set /p projname=<%tmp%\basename.txt

dir /s /b /AD |grepw /V _3rd | grepw "%projname%$ .vs$ x64$ .vscode$" > txt.txt 
dir /a /s /b *.pdb  *.aps tags *.swp >> txt.txt


type txt.txt | tol | repl "^" "rm -rf "

del txt.txt
endlocal
@echo on
