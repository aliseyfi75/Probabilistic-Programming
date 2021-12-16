import subprocess
import time

print("hellow")
start = time.time()
subprocess.run(['discotress'])
end = time.time()
print("done, took ", end - start, "seconds")


# change input so
# 146 seconds for 1000 paths WRAPPER BTOA, TRAJ BKL