:: change config to Debug, only for Debug
@echo off

setlocal
cd |repl "^.*\\" ""  > %tmp%\basename.txt

set projname=<%tmp%\basename.txt

dir /s /b /AD |grepw /V _3rd | grepw "%projname%$ .vs$ x64$ .vscode$" > txt.txt 
dir /s /b *.pdb >> txt.txt


type txt.txt | tol | repl "^" "rm -rf "

del txt.txt
endlocal
@echo on
