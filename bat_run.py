from subprocess import Popen
import argparse

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_list",type=str,nargs="+",default=["config/slip/cable1.json","config/slip/key3.json","config/slip/nail1.json","config/slip/skewdriver1.json"])
    parser.add_argument("--phase",type=str,choices=['test','train'],default='test')
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    for config in args.config_list:
        process = Popen(['python','run.py','-p','test','-c',config])
        process.wait()
        if process.returncode != 0:
            print("Uncessrun for {} config.".format(config))