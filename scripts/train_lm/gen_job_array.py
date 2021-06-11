import itertools
import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--joblist_filename", type=str, default="jobs")
parser.add_argument("--sbatch_filename", type=str, default="run_jobs")
parser.add_argument("--parallel_jobs", "-p", type=int, default=100)
args = parser.parse_args()

CONFIG_LM = {
    "lr": [40., 30., 20., 10., 1., 0.1, 0.01, 0.001],
    "model": ["transformer", "lstm"],
    "optimizer": ["adam", "sgd"],
    "nlayers": [2, 3, 4],
}

COMMAND_TEMPLATE = "python scripts/train_lm/main.py with "

SBATCH_1 = """#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition="""

SBATCH_2 = """#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --array=0-"""

SBATCH_3 = """#SBATCH --output="slogs/slurm-%A_%a.out"
#SBATCH -c 2

IFS=$"\\n" read -d "" -r -a lines < """

SBATCH_4 = r"""echo "Starting task $SLURM_ARRAY_TASK_ID: ${lines[SLURM_ARRAY_TASK_ID]}"
eval ${lines[SLURM_ARRAY_TASK_ID]}
"""


def build_sbatch_script(partitions, total_jobs, parallel_jobs, joblist_filename):
    filestr = SBATCH_1
    filestr += partitions + "\n"
    filestr += SBATCH_2
    filestr += total_jobs + r"%" + parallel_jobs + "\n"
    filestr += SBATCH_3
    filestr += joblist_filename + "\n"
    return filestr + SBATCH_4


class ConfigIterator:
    def __init__(self, conf):
        self.conf = conf

    def __iter__(self):
        return itertools.product(*[self.conf[key] for key in self.conf])


def generate_job_strings(config):
    jobs = []
    for setting in ConfigIterator(config):
        command = COMMAND_TEMPLATE
        for i, k in enumerate(config):
            command += "'{}={}' ".format(k, setting[i])
        command += "\n"
        jobs.append(command)
    return jobs


if __name__ == "__main__":
    jobs = generate_job_strings(CONFIG_LM)
    with open(args.joblist_filename, "w") as f:
        f.writelines(jobs)
    sbatch = build_sbatch_script("t4v1,t4v2,p100,rtx6000", str(len(jobs)), str(args.parallel_jobs),
                                 args.joblist_filename)
    with open(args.sbatch_filename, "w") as f:
        f.write(sbatch)
