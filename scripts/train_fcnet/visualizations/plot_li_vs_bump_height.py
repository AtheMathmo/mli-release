import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import argparse

colors = sns.color_palette()
plt.style.use('seaborn')

parser = argparse.ArgumentParser()
parser.add_argument("summary_path")
parser.add_argument("--thresh", type=float, default=0.1)
parser.add_argument("--show", action="store_true")
args = parser.parse_args()

def main(summary_path, threshold=0.1):
    with open(summary_path) as f:
        data = json.load(f)
    lin_integrals = []
    bump_height = []
    loss = []
    color = []
    
    print(colors[1])
    print(colors[0])
    for run in data:
        min_loss = run['min_train_loss']
        if min_loss < threshold:
            loss.append(min_loss)
            bump = run['max_bump']
            lin_integrals.append(run['linearity_integral'])
            bump_height.append(bump)
            color.append(colors[1] if bump > 0 else colors[0])
    
    plt.figure(figsize=(6,4))
    # plt.title(r"Max $\Delta$ vs distance from initialization", fontsize=18)
    plt.xlabel("Linearity measure", fontsize=14)
    plt.ylabel(r"Max $\Delta$", fontsize=14)
    
    plt.scatter(np.log(lin_integrals), bump_height, color = color)
    plt.tight_layout()
    if args.show:
        plt.show()
    else:
        plt.savefig("dist_vs_delta.pdf")


if __name__ == '__main__':
    main(args.summary_path, args.thresh)