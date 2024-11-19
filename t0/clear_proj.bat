:: change config to Debug, only for Debug
@echo off
dir /s /b /AD |grepw /V _3rd | grepw "t0$ .vs$ x64$ .vscode$" > txt.txt 
dir /s /b *.pdb >> txt.txt


type txt.txt | tol | repl "^" "rm -rf "

del txt.txt
@echo on
