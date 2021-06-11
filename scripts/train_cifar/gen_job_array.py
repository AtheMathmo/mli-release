import itertools
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("joblist_filename")
parser.add_argument("sbatch_filename")
parser.add_argument("--parallel_jobs", '-p', type=int, default=12)
args = parser.parse_args()

CONFIG_CIFAR10 = {
  'lr': [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003],
  'model_name': [
      'resnet-20', 'fixup_resnet-20', 'resnet-20-nobn',
      'resnet-32', 'fixup_resnet-32', 'resnet-32-nobn',
      'resnet-44', 'fixup_resnet-44', 'resnet-44-nobn',
      'resnet-56', 'fixup_resnet-56', 'resnet-56-nobn',
    ],
  'optim_name': ['sgd', 'adam'],
  'dset_name': ['cifar10'],
  'num_classes': [10]
}

CONFIG_CIFAR100 = {
  'lr': [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003],
  'model_name': [
      'resnet-20', 'fixup_resnet-20', 'resnet-20-nobn',
      'resnet-32', 'fixup_resnet-32', 'resnet-32-nobn',
      'resnet-44', 'fixup_resnet-44', 'resnet-44-nobn',
      'resnet-56', 'fixup_resnet-56', 'resnet-56-nobn',
    ],
  'optim_name': ['sgd', 'adam'],
  'dset_name': ['cifar100'],
  'num_classes': [100],
  'min_loss_threshold': [3.0]
}

CONFIG_CIFAR10_INIT = {
  'lr': [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003],
  'model_name': [
      'resnet-20',
      'resnet-32',
      'resnet-44',
      'resnet-56',
    ],
  'identity_init': [False],
  'optim_name': ['sgd', 'adam'],
  'dset_name': ['cifar10'],
  'num_classes': [10]
}

CONFIG_CIFAR100_INIT = {
  'lr': [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003],
  'model_name': [
      'resnet-20',
      'resnet-32',
      'resnet-44',
      'resnet-56',
    ],
  'identity_init': [False],
  'optim_name': ['sgd', 'adam'],
  'dset_name': ['cifar100'],
  'num_classes': [100]
}


COMMAND_TEMPLATE = 'python scripts/train_cifar/train.py with '

SBATCH_1 = """#!/bin/bash
#SBATCH --account=deadline
#SBATCH --qos=deadline
#SBATCH --partition="""

SBATCH_2 = """#SBATCH --gres=gpu:1
#SBATCH --mem=16G
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
        command += '\n'
        jobs.append(command)
    return jobs


if __name__ == '__main__':
    jobs = generate_job_strings(CONFIG_CIFAR10)
    with open(args.joblist_filename, 'w') as f:
        f.writelines(jobs)
    sbatch = build_sbatch_script('t4v1,t4v2,p100', str(len(jobs)), str(args.parallel_jobs), args.joblist_filename)
    with open(args.sbatch_filename, 'w') as f:
        f.write(sbatch)
