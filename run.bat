@echo off
setlocal

REM 
if not exist "build\CRUSHBLAS.exe" (
    echo [RUN CRUSHBLAS] Error: build\CRUSHBLAS.exe not found.
    echo Build it first with: cmake --build --preset mingw-debug-build
    exit /b 1
)

echo [RUN CRUSHBLAS] Running CRUSHBLAS.exe...
"build\CRUSHBLAS.exe"

endlocal

