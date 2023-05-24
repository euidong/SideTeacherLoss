import matplotlib.pyplot as plt
import numpy as np
import os


# 1. read all files in result directory
dir = 'result/fashion_mnist/alpha==0.005'
files = os.listdir(dir)
files = [file for file in files if file.endswith('.npy')]
result = {}
for file in files:
    result[file] = np.load(os.path.join(dir, file))

# 2. draw graph
metric = ["test_acc", "test_loss", "train_acc", "train_loss"]
models = ["baseline", "teacher", "student"]
model_names = {
    "teacher": [f"teacher{i}" for i in range(10)],
    "student": ["l1-neg", "l2-neg", "nuc-neg", "fro-neg", "l1-recp", "l2-recp", "nuc-recp", "fro-recp"],
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

if __name__ == '__main__':
    draw_each()
    draw_base_and_student()
