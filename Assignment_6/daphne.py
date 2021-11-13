import json
import subprocess


def daphne(args, cwd='/Users/aliseyfi/Documents/UBC/Semester3/Probabilistic-Programming/HW/Probabilistic-Programming/Assignment_6/daphne'):
    proc = subprocess.run(['lein','run']  + args,
                          capture_output=True, cwd=cwd)
    if(proc.returncode != 0):
        raise Exception(proc.stdout.decode() + proc.stderr.decode())
    return json.loads(proc.stdout)
