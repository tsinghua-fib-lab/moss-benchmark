import subprocess
import time
import argparse
import os
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", default="")
    parser.add_argument("--duration",type=int,default=900)
    return parser.parse_args()
def is_target_process_running(target_part, exclude_pid):
    try:
        # 使用 pgrep -f 查找命令行中包含 target_part 的进程，但不包含自己的 PID
        output = subprocess.check_output(["pgrep", "-f", target_part])
        pids = output.decode().strip().split('\n')
        filtered_pids = [pid for pid in pids if pid != str(exclude_pid)]
        return len(filtered_pids) > 0
    except subprocess.CalledProcessError:
        return False

if __name__ == "__main__":
    args = get_args()
    command_to_check = args.command
    current_pid = os.getpid()
    while True: 
        # 检查指定名称的部分是否在运行，同时排除当前脚本的 PID
        if is_target_process_running(command_to_check, current_pid):
            print(f"The command '{command_to_check}' is currently running. Sleeping for {args.duration} seconds.")
            time.sleep(args.duration)
        else:
            print(f"The command '{command_to_check}' is not running.")
            break
