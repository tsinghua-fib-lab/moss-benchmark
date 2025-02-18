import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', choices='Beijing Nanchang Changchun Jinan Shenzhen Hangzhou Shanghai eff_10inter eff_1e2inter eff_1e2veh eff_1e3inter eff_1e3veh eff_1e4inter_Shanghai eff_1e4veh eff_1e5veh eff_1e6veh_1e4.5inter'.split(), required=True)
    parser.add_argument('--steps', type=int, default=3600)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--threads', type=int, default=32)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()
