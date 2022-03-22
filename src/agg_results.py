import json
import argparse
import os
import os.path as osp
import numpy as np
from collections import defaultdict


def agg_results(log_dir):
    results = defaultdict(list)
    for seed in os.listdir(log_dir):
        seed_dir = osp.join(log_dir, seed)
        if not osp.isdir(seed_dir):
            continue
        with open(osp.join(seed_dir, 'final.json')) as f:
            for k, v in json.loads(f.readlines()[-1]).items():
                results[k].append(v)
    agg_res = {}
    for k, v in results.items():
        agg_res[k] = np.mean(v)
        agg_res['{}_std'.format(k)] = np.std(v)
    with open(osp.join(log_dir, 'agg_results.json'), 'w+') as f:
        json.dump(agg_res, f, indent=4)
    return agg_res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=str, required=True)
    args = parser.parse_args()
    agg_res = agg_results(args.log_dir)
