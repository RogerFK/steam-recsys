@echo off
:: https://stackoverflow.com/a/203116
for /F "usebackq tokens=1,2 delims==" %%i in (`wmic os get LocalDateTime /VALUE 2^>NUL`) do if '.%%i.'=='.LocalDateTime.' set start_time=%%j
set start_time=%start_time:~0,4%-%start_time:~4,2%-%start_time:~6,2% %start_time:~8,2%:%start_time:~10,2%:%start_time:~12,6%
@echo on
C:\Users\User\anaconda3\envs\env\python.exe experiments.py
@echo off
for /F "usebackq tokens=1,2 delims==" %%i in (`wmic os get LocalDateTime /VALUE 2^>NUL`) do if '.%%i.'=='.LocalDateTime.' set end_time=%%j
set end_time=%end_time:~0,4%-%end_time:~4,2%-%end_time:~6,2% %end_time:~8,2%:%end_time:~10,2%:%end_time:~12,6%
echo Start time: [%start_time%], end time [%end_time%]

PAUSE