from subprocess import Popen
import argparse

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_list",type=str,nargs="+",default=["config/nmk_rand.json","config/taxim_m1.json","config/taxim_m2.json","config/mk_rand.json"])
    parser.add_argument("--phase",type=str,choices=['test','train'],default='test')
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    for config in args.config_list:
        process = Popen(['python','run.py','-p','test','-c',config])
        process.wait()
        if process.returncode != 0:
            print("Uncessrun for {} config.".format(config))