

RATING_MULTIPLIER = 5.0

import os
import sys
# import win32com.shell.shell as shell
# ASADMIN = 'asadmin'
# 
# if sys.argv[-1] != ASADMIN:
#     script = os.path.abspath(sys.argv[0])
#     params = ' '.join([script] + sys.argv[1:] + [ASADMIN])
#     print(sys.executable, params)
#     input("Press Enter to continue...")
#     shell.ShellExecuteEx(lpVerb='runas', lpFile=sys.executable, lpParameters=params)
#     sys.exit(0)

import sys
try:
    sys.getwindowsversion()
except AttributeError:
    isWindows = False
else:
    isWindows = True

if isWindows:
    print("Set process priority to REALTIME_PRIORITY_CLASS")
    import psutil
    psutil.Process().nice(psutil.REALTIME_PRIORITY_CLASS)