import atexit
import os
import subprocess
import threading
import time
import traceback
from threading import Lock
_lock = Lock()
_run_log = time.strftime('log/run/%Y%m%d-%H%M%S')
_run_id = 0
os.makedirs(_run_log, exist_ok=True)


def run(cmd):
    global _run_id
    with _lock:
        _id = _run_id
        _run_id += 1

    def output(s):
        with open(f'{_run_log}/{_id}.log', 'a', encoding='utf8') as f:
            f.write(s)
        if '%|' in s or 'it/s]' in s:
            print(s.rstrip(), end='\r', flush=True)
        else:
            print(s, end='', flush=True)
    try:
        output(time.strftime('[%Y%m%d %H:%M:%S] ')+cmd+'\n\n')
        p = subprocess.Popen(cmd, shell=True, bufsize=1, encoding='utf8', stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for l in p.stdout: # type: ignore
            output(l)
        p.communicate()
    except:
        err = traceback.format_exc()
        output(err)


def worker(jobs, device):
    while True:
        with _lock:
            if not jobs:
                return
            job = jobs.pop()
        run(f'CUDA_VISIBLE_DEVICES={device} {job}')


def main():
    os.setpgrp()
    atexit.register(os.killpg, 0, 9)
    e_type = "moss"
    jobs = [
        f'python3 algo_{algo}.py --data ./data/{e_type}_{city} --start 25200 --steps {3*3600} --training_step {30*3600//30+1} --training_start {360} --exp {city}_3h_moss --engine_type {e_type}'
        for algo in ['colight',]
        for city in ['newyork',]
    ]
    jobs = jobs[::-1]
    _devices = [0,1]
    ts = [threading.Thread(target=worker, args=(jobs, device), daemon=True) for device in _devices]
    for i in ts:
        i.start()
    while True:
        if not jobs:
            break
        time.sleep(2)
    for i in ts:
        i.join()


main()
