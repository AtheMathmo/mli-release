# Analyzing Monotonic Linear Interpolation in Neural Network Loss Landscapes

This repository contains the code for running and visualizing the experiments in the paper [Analyzing Monotonic Linear Interpolation in Neural Network Loss Landscapes](https://arxiv.org/abs/2104.11044).

The `lib` folder contains the core of this project, and includes two packages. First, `mli` for building the models and computing the metrics that we use in the paper. And second, `mli_eval` that contains useful evaluation utility functions.

Next, the `scripts` folder contains the actual scripts that we used to train and evaluate the models from our paper. These are included primarily for reproducibility (and are generally a little less organized than the main library).


## Installation

We provide an `environment.yml` file that details the package dependencies of this project, and can be installed via `conda env create -f environment.yml`. The command installs a Python package called `mli`.

**Note: The packages listed here depend on CUDA11, you may need to adjust for your hardware.**

Short summary of necessary packages:
- pytorch
- [sacred](https://github.com/IDSIA/sacred)
- tqdm
- numpy
- scipy

To test if the module was installed successfully, execuate all files listed in `lib/tests`.

## Running Training Scripts

The training scripts utilize [sacred](https://github.com/IDSIA/sacred) for config management. An example syntax for passing config arguments is as follows:

```
python scripts/train_fcnet/train.py with epochs=10 hsizes=[1024,1024,1024,10] use_batchnorm=True
```

#### Supported tasks

- Fully-connected autoencoders on MNIST/FashionMNIST
    - `python scripts/train_ae_fc/train.py`
- Fully-connected network classification on MNIST/FashionMNIST
    - `python scripts/train_fcnet/train.py`
- Convolutional network classification on CIFAR-10 & CIFAR-100
    - `python scripts/train_cifar/train.py`
- LSTM and Transformer Language Modelling on WikiText
    - `python scripts/train_lm/train.py`

For each of these training scripts, we can generate an [sbatch](https://slurm.schedmd.com/sbatch.html) training script to execute a grid search. These are generated via the `gen_job_array.py` scripts in each task folder.
An example command is as follows:

```
python scripts/train_fcnet/gen_job_array.py jobs.txt run_jobs_sbatch --parallel_jobs 4
```

Note that this generates a batch array for our particular SLURM environment. You will likely need to tweak these scripts for your own environment. Furthermore, the grid search configs are currently set manually within the `gen_job_array.py` scripts.

After executing the above script, use the following command to run a batch experiment:

```
sbatch run_jobs_sbatch
```

## Running Evaluation Scripts

After executing the training script, it will create a folder named `/runs`. To generate a summary of the batch experiment, execute the evaluation script with the following command:

```
python scripts/train_fcnet/eval/eval_runs.py runs my/output/path
```

The name for the script is the same for each supported task. For additional summarization and visualization for each supported task, you can execute the scripts located in `/eval` and `/visualization`. 

## Citation

To cite this work, please use:
```bibtex
@article{lucas2021analyzing,
  title={Analyzing Monotonic Linear Interpolation in Neural Network Loss Landscapes},
  author={Lucas, James and Bae, Juhan and Zhang, Michael R and Fort, Stanislav and Zemel, Richard and Grosse, Roger},
  journal={arXiv preprint arXiv:2104.11044},
  year={2021}
}
```

## Contributors

Please contact James Lucas for any questions on the code. 

- [James Lucas](https://www.cs.toronto.edu/~jlucas/)
- [Juhan Bae](http://www.juhanbae.com/)
- [Michael Zhang](https://michaelrzhang.github.io/)
- [Stanislav Fort](http://stanford.edu/~sfort1/)
