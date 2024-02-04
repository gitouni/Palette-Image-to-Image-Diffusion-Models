from subprocess import Popen
import argparse

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_list",type=str,nargs="+",default=["config/nmk_rand/nmk_m1_patch.json","config/nmk_rand/nmk_m2_patch.json"])
    parser.add_argument("--phase",type=str,choices=['test','train','semi'],default='test')
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    for config in args.config_list:
        process = Popen(['python','run.py','-p',args.phase,'-c',config])
        process.wait()
        if process.returncode != 0:
            print("Uncessrun for {} config.".format(config))
        else:
            print("Sucessfully run config:{}.".format(config))