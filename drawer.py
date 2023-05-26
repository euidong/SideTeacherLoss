import matplotlib.pyplot as plt
import numpy as np
import torch
from model import Model
import os
from dataLoader import get_data_loader


# 1. read all files in result directory
dataset = "cifar100"
alpha = "0.005"
dir = f"result/{dataset}/alpha=={alpha}"
files = os.listdir(dir)
files = [file for file in files if file.endswith('.npy')]
result = {}
for file in files:
    result[file] = np.load(os.path.join(dir, file))

train_loader, test_loader = get_data_loader(1, dataset)

# 2. define out_c, in_c, in_h, in_w
# - out_c = # output class
# - in_c = # input channel
# - in_h = input height
# - in_w = input width
out_c = len(train_loader.dataset.classes)
if (len(train_loader.dataset.data.shape) == 3):
    _, in_h, in_w = train_loader.dataset.data.shape
    in_c = 1
else:
    _, in_h, in_w, in_c = train_loader.dataset.data.shape

# 2. draw graph
metric = ["test_acc", "test_loss", "train_acc", "train_loss"]
models = ["baseline", "teacher", "student"]
model_names = {
    "teacher": [f"teacher{i}" for i in range(10)],
    "student": ["fro-neg", "l1-recp", "l2-recp", "fro-recp"],
    "baseline": ["default", "weight-decay"]
}

def draw_each():
    for m in metric:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(m)
        for i, model in enumerate(models):
            axes[i].set_title(model)
            target = model_names[model]
            for t in target:
                v = result[f'{m}_{t}.npy']
                axes[i].plot(v, label=t)
            axes[i].legend()
        fig.savefig(f'{dir}/{m}.png')

def draw_base_and_student():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("test metrics")
    for i, m in enumerate(["test_acc", "test_loss"]):
        axes[i].set_title(m)
        for t in ["baseline", "student"]:
            target = model_names[t]
            for t in target:
                v = result[f'{m}_{t}.npy']
                axes[i].plot(v, label=t)
        axes[i].legend()
    fig.savefig(f'{dir}/test_metrics.png')

def draw_distance_between_students_and_baselines():
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("distance with default")
    default = Model(in_c, in_w, in_h, out_c)
    default.load(f"param/{dataset}/alpha=={alpha}/default")
    for i, m in enumerate(metric):
        axe = axes[i//2][i%2]
        axe.set_title(m)
        for t in ["baseline", "student"]:
            target = model_names[t]
            for t in target:
                model = Model(in_c, in_w, in_h, out_c)
                model.load(f"param/{dataset}/alpha=={alpha}/{t}")
                dist = 0
                for name, d_p in default.layers.named_parameters():
                    dist += ((d_p - model.layers.get_parameter(name)) ** 2).mean()
                height = result[f'{m}_{t}.npy'][-1]
                dist = dist.detach().numpy()
                axe.bar(x=dist, height=height, label=t)
        axe.legend()
    # axes[0][0].set_ylim(0.97, 1.0)
    # axes[1][0].set_ylim(0.97, 1.0)
    fig.savefig(f'{dir}/dist_metrics.png')

if __name__ == '__main__':
    draw_each()
    draw_base_and_student()
    draw_distance_between_students_and_baselines()