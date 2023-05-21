import torch
from model import Model
from dataLoader import get_data_loader
from lossFn import SideTeacherLoss
import json

device = ('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 64
epoch_num = 300
learning_rate = 0.005
alpha = 0.005
num_teachers = 10
dataset='cifar100'
train_loader, test_loader = get_data_loader(batch_size, dataset)
out_c = len(train_loader.dataset.classes)
if (len(train_loader.dataset.data.shape) == 3):
    _, in_h, in_w = train_loader.dataset.data.shape
    in_c = 1
else:
    _, in_h, in_w, in_c = train_loader.dataset.data.shape
teachers = [Model(in_c, in_w, in_h, out_c).to(device) for _ in range(num_teachers)]
students = {
    "l2-neg": Model(in_c, in_w, in_h, out_c).to(device),
    "l1-neg": Model(in_c, in_w, in_h, out_c).to(device),
    "fro-neg": Model(in_c, in_w, in_h, out_c).to(device),
    "nuc-neg": Model(in_c, in_w, in_h, out_c).to(device),
    "l2-recp": Model(in_c, in_w, in_h, out_c).to(device),
    "l1-recp": Model(in_c, in_w, in_h, out_c).to(device),
    "fro-recp": Model(in_c, in_w, in_h, out_c).to(device),
    "nuc-recp": Model(in_c, in_w, in_h, out_c).to(device),
}
baselines = {
    "default": Model(in_c, in_w, in_h, out_c).to(device),
    "weight-decay": Model(in_c, in_w, in_h, out_c).to(device),
}
dir = f"param/{dataset}/alpha=={alpha}"
for i in range(num_teachers):
    teachers[i].load_state_dict(torch.load(f"{dir}/teacher{i}.pt"))

for s_name, student in students.items():
    student.load_state_dict(torch.load(f"{dir}/{s_name}.pt"))

for b_name, baseline in baselines.items():
    baseline.load_state_dict(torch.load(f"{dir}/{b_name}.pt"))

teacher_loss_fn = torch.nn.CrossEntropyLoss()
student_loss_fns = {s_name: SideTeacherLoss(student, torch.nn.CrossEntropyLoss(), device, alpha=alpha, teachers=teachers,dist=s_name) for s_name, student in students.items()}
baseline_loss_fn = torch.nn.CrossEntropyLoss()

tot_batch_num = len(train_loader)
teacher_batch_num = tot_batch_num // num_teachers
# check teacher accuracy and whether overfitting or not
# - 1. print train accuracy
teachers_train_loss = [0 for _ in range(num_teachers)]
teachers_train_correct = [0 for _ in range(num_teachers)]
with torch.no_grad():
    for cur_batch_num, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        teacher_idx = cur_batch_num // teacher_batch_num
        if teacher_idx == num_teachers:
            break
        pred = teachers[teacher_idx](X)
        teachers_train_loss[teacher_idx] += teacher_loss_fn(pred, y).item()
        teachers_train_correct[teacher_idx] += (pred.argmax(1) == y).type(torch.float).sum().item()
size = batch_size * teacher_batch_num
for i in range(num_teachers):
    teachers_train_loss[i] /= size
    teachers_train_correct[i] /= size
    print(f"Teacher{i+1} Train Error: \n Accuracy: {(100*teachers_train_correct[i]):>0.1f}%, Avg loss: {teachers_train_loss[i]:>8f} \n")
# - 2. print test accuracy
teachers_test_loss = [0 for _ in range(num_teachers)]
teachers_test_correct = [0 for _ in range(num_teachers)]
with torch.no_grad():
    for batch_num, (X, y) in enumerate(test_loader):
        X, y = X.to(device), y.to(device)
        for i in range(num_teachers):
            pred = teachers[i](X)
            teachers_test_loss[i] += teacher_loss_fn(pred, y).item()
            teachers_test_correct[i] += (pred.argmax(1) == y).type(torch.float).sum().item()

size = len(test_loader.dataset)
for i in range(num_teachers):
    teachers_test_loss[i] /= size
    teachers_test_correct[i] /= size
    print(f"Teacher{i+1} Test Error: \n Accuracy: {(100*teachers_test_correct[i]):>0.1f}%, Avg loss: {teachers_test_loss[i]:>8f} \n")

# test student, base_model, weight_decay_model
# - 1. print train accuracy
size = len(train_loader.dataset)
student_train_losses = {s_name: 0 for s_name in students.keys()}
student_train_correct = {s_name: 0 for s_name in students.keys()}
baseline_train_losses = {b_name: 0 for b_name in baselines.keys()}
baseline_train_correct = {b_name: 0 for b_name in baselines.keys()}
with torch.no_grad():
    for batch_num, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        
        for s_name, student in students.items():
            pred = student(X)
            student_train_losses[s_name] += student_loss_fns[s_name](pred, y).item()
            student_train_correct[s_name] += (pred.argmax(1) == y).type(torch.float).sum().item()

        for b_name, baseline in baselines.items():
            pred = baseline(X)
            baseline_train_losses[b_name] += baseline_loss_fn(pred, y).item()
            baseline_train_correct[b_name] += (pred.argmax(1) == y).type(torch.float).sum().item()

for s_name in students.keys():
    student_train_losses[s_name] /= size
    student_train_correct[s_name] /= size
    print(f"{s_name} Train Error: \n Accuracy: {(100*student_train_correct[s_name]):>0.1f}%, Avg loss: {student_train_losses[s_name]:>8f} \n")

for b_name in baselines.keys():
    baseline_train_losses[b_name] /= size
    baseline_train_correct[b_name] /= size
    print(f"{b_name} Train Error: \n Accuracy: {(100*baseline_train_correct[b_name]):>0.1f}%, Avg loss: {baseline_train_losses[b_name]:>8f} \n")

# - 2. print test accuracy
size = len(test_loader.dataset)
stundet_test_losses = {s_name: 0 for s_name in students.keys()}
student_test_correct = {s_name: 0 for s_name in students.keys()}
baseline_test_losses = {b_name: 0 for b_name in baselines.keys()}
baseline_test_correct = {b_name: 0 for b_name in baselines.keys()}
with torch.no_grad():
    for batch_num, (X, y) in enumerate(test_loader):
        X, y = X.to(device), y.to(device)

        for s_name, student in students.items():
            pred = student(X)
            stundet_test_losses[s_name] += student_loss_fns[s_name](pred, y).item()
            student_test_correct[s_name] += (pred.argmax(1) == y).type(torch.float).sum().item()

        for b_name, baseline in baselines.items():
            pred = baseline(X)
            baseline_test_losses[b_name] += baseline_loss_fn(pred, y).item()
            baseline_test_correct[b_name] += (pred.argmax(1) == y).type(torch.float).sum().item()

result = {
    "test-loss": {},
    "test-accuracy": {},
    "train-loss": {},
    "train-accuracy": {},
}




for s_name in students.keys():
    stundet_test_losses[s_name] /= size
    student_test_correct[s_name] /= size
    print(f"{s_name} Test Error: \n Accuracy: {(100*student_test_correct[s_name]):>0.1f}%, Avg loss: {stundet_test_losses[s_name]:>8f} \n")

for b_name in baselines.keys():
    baseline_test_correct[b_name] /= size
    baseline_test_losses[b_name] /= size
    print(f"{b_name} Test Error: \n Accuracy: {(100*baseline_test_correct[b_name]):>0.1f}%, Avg loss: {baseline_test_losses[b_name]:>8f} \n")


for t_idx in range(num_teachers):
    result["test-loss"][f'teacher{t_idx}'] = teachers_test_loss[t_idx]
    result["test-accuracy"][f'teacher{t_idx}'] = 100 * teachers_test_correct[t_idx]
    result["train-loss"][f'teacher{t_idx}'] = teachers_train_loss[t_idx]
    result["train-accuracy"][f'teacher{t_idx}'] = 100 * teachers_train_correct[t_idx]


for s_name in students.keys():
    result["test-loss"][s_name] = stundet_test_losses[s_name]
    result["test-accuracy"][s_name] = 100 * student_test_correct[s_name]
    result["train-loss"][s_name] = student_train_losses[s_name]
    result["train-accuracy"][s_name] = 100 * student_train_correct[s_name]

for b_name in baselines.keys():
    result["test-loss"][b_name] = baseline_test_losses[b_name]
    result["test-accuracy"][b_name] = 100 * baseline_test_correct[b_name]
    result["train-loss"][b_name] = baseline_train_losses[b_name]
    result["train-accuracy"][b_name] = 100 * baseline_train_correct[b_name]

json.dump(result, open(f"{dataset}_{alpha}_result.json", "w"))