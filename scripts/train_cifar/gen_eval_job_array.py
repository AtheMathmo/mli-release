import os
import itertools
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("evaldir")
parser.add_argument("joblist_filename")
parser.add_argument("sbatch_filename")
parser.add_argument("--parallel_jobs", '-p', type=int, default=12)
args = parser.parse_args()

COMMAND_TEMPLATE = 'python scripts/train_cifar/eval/eval_model.py '
COMMAND_ARGS = '-d 5000 -a 50 --compute_gl'

SBATCH_1 = """#!/bin/bash
#SBATCH --account=deadline
#SBATCH --qos=deadline
#SBATCH --partition="""

SBATCH_2 = """#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --array=0-"""

SBATCH_3 = """#SBATCH --output="slogs/slurm-%A_%a.out"
#SBATCH -c 2

IFS=$'\\n' read -d '' -r -a lines < """

SBATCH_4 = r"""echo "Starting task $SLURM_ARRAY_TASK_ID: ${lines[SLURM_ARRAY_TASK_ID]}"
eval ${lines[SLURM_ARRAY_TASK_ID]}
"""


def build_sbatch_script(partitions, total_jobs, parallel_jobs, joblist_filename):
  filestr = SBATCH_1
  filestr += partitions + '\n'
  filestr += SBATCH_2
  filestr += total_jobs + r'%' + parallel_jobs + '\n'
  filestr += SBATCH_3
  filestr += joblist_filename + '\n'
  return filestr + SBATCH_4

def generate_job_strings(evaldir):
    jobs = []
    alldirs = os.listdir(evaldir)
    for rundir in alldirs:
        # Sacred saves all source code
        if rundir == '_sources':
            continue
        dirpath = os.path.join(evaldir, rundir)
        if not os.path.isdir(dirpath):
            continue
        config_f = os.path.join(dirpath, 'config.json')
        metrics_f = os.path.join(dirpath, 'metrics.json')
        init_f = os.path.join(evaldir, rundir, 'init.pt')
        final_f = os.path.join(evaldir, rundir, 'final.pt')
        # Check directory for completeness:
        valid = True
        valid = valid and os.path.isfile(config_f)
        valid = valid and os.path.isfile(metrics_f)
        valid = valid and os.path.isfile(init_f)
        valid = valid and os.path.isfile(final_f)
        if not valid:
            print("Incomplete experiment output in {}".format(dirpath))
        else:
            jobs.append(COMMAND_TEMPLATE + dirpath + ' ' + dirpath + ' ' + COMMAND_ARGS + '\n')
    return jobs

if __name__ == '__main__':
  jobs = generate_job_strings(args.evaldir)
  with open(args.joblist_filename, 'w') as f:
      f.writelines(jobs)
  sbatch = build_sbatch_script('t4v1,t4v2,p100', str(len(jobs)), str(args.parallel_jobs), args.joblist_filename)
  with open(args.sbatch_filename, 'w') as f:
    f.write(sbatch)
