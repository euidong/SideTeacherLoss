import torch
from model import Model
from dataLoader import get_data_loader
from lossFn import SideTeacherLoss
import torch.optim as optim
import os

device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def main():
    batch_size = 64
    epoch_num = 300
    learning_rate = 0.005
    alpha = 0.01
    num_teachers = 10
    dataset = "fashion_mnist"

    train_loader, test_loader = get_data_loader(batch_size, dataset)
    if (len(train_loader.dataset.data.shape) == 3):
        _, in_h, in_w = train_loader.dataset.data.shape
        in_c = 1
    else:
        _, in_h, in_w, in_c = train_loader.dataset.data.shape

    teachers = [Model(in_c, in_w, in_h).to(device) for _ in range(num_teachers)]
    students = {
        "l2-neg": Model(in_c, in_w, in_h).to(device),
        "l1-neg": Model(in_c, in_w, in_h).to(device),
        "fro-neg": Model(in_c, in_w, in_h).to(device),
        "nuc-neg": Model(in_c, in_w, in_h).to(device),
        "l2-recp": Model(in_c, in_w, in_h).to(device),
        "l1-recp": Model(in_c, in_w, in_h).to(device),
        "fro-recp": Model(in_c, in_w, in_h).to(device),
        "nuc-recp": Model(in_c, in_w, in_h).to(device),
    }
    baselines = {
        "default": Model(in_c, in_w, in_h).to(device),
        "weight-decay": Model(in_c, in_w, in_h).to(device),
    }
        
    teacher_optimizers = [optim.SGD(teachers[i].parameters(), lr=learning_rate)for i in range(num_teachers)]
    student_optimizers = {s_name: optim.SGD(student.parameters(), lr=learning_rate) for s_name, student in students.items()}
    baseline_optimizers = {
        "default": optim.SGD(baselines["default"].parameters(), lr=learning_rate),
        "weight-decay": optim.SGD(baselines["weight-decay"].parameters(), lr=learning_rate, weight_decay=alpha)
    }

    teacher_loss_fn = torch.nn.CrossEntropyLoss()
    student_loss_fns = {s_name: SideTeacherLoss(student, torch.nn.CrossEntropyLoss(), device, alpha=alpha, teachers=teachers,dist=s_name) for s_name, student in students.items()}
    baseline_loss_fn = torch.nn.CrossEntropyLoss()

    tot_batch_num = len(train_loader)
    teacher_batch_num = tot_batch_num // num_teachers
    # train teacher
    for epoch in range(epoch_num):
        print(f"Epoch: {epoch+1}====================")
        for cur_batch_num, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            teacher_idx = cur_batch_num // teacher_batch_num
            if teacher_idx == num_teachers:
                break
            pred = teachers[teacher_idx](X)
            loss = teacher_loss_fn(pred, y)
            teacher_optimizers[teacher_idx].zero_grad()
            loss.backward()
            teacher_optimizers[teacher_idx].step()

            if cur_batch_num % teacher_batch_num == teacher_batch_num - 1:
                print(f"teacher {teacher_idx} finished")
                loss, current = loss.item(), (cur_batch_num - teacher_idx * teacher_batch_num) * len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{teacher_batch_num * len(X):>5d}]")

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
    
    size = len(train_loader)
    # train student, base_model, weight_decay_model
    for epoch in range(epoch_num):
        print(f"Epoch: {epoch+1}====================")

        for batch_num, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            student_loss = {s_name: 0 for s_name in students.keys()}
            baseline_loss = {b_name: 0 for b_name in baselines.keys()}
            for s_name, student in students.items():
                pred = student(X)
                student_loss[s_name] = student_loss_fns[s_name](pred, y)
                student_optimizers[s_name].zero_grad()
                student_loss[s_name].backward()
                student_optimizers[s_name].step()
            for b_name, baseline in baselines.items():
                pred = baseline(X)
                baseline_loss[b_name] = baseline_loss_fn(pred, y)
                baseline_optimizers[b_name].zero_grad()
                baseline_loss[b_name].backward()
                baseline_optimizers[b_name].step()

            if batch_num % 100 == 0:
                for s_name in students.keys():
                    loss, current = student_loss[s_name].item(), batch_num * len(X)
                    print(f"{s_name} loss: {loss:>7f} [{current:>5d}/{size * len(X):>5d}]")
                for b_name in baselines.keys():
                    loss, current = baseline_loss[b_name].item(), batch_num * len(X)
                    print(f"{b_name} loss: {loss:>7f} [{current:>5d}/{size * len(X):>5d}]")


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
    baseline_test_correct = {b_name: 0 for b_name in baselines.keys()}
    with torch.no_grad():
        for batch_num, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)

            for s_name, student in students.items():
                pred = student(X)
                stundet_test_losses[s_name] += student_loss_fns[s_name](pred, y).item()
                student_train_correct[s_name] += (pred.argmax(1) == y).type(torch.float).sum().item()

            for b_name, baseline in baselines.items():
                pred = baseline(X)
                baseline_test_correct[b_name] += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    for s_name in students.keys():
        stundet_test_losses[s_name] /= size
        student_train_correct[s_name] /= size
        print(f"{s_name} Test Error: \n Accuracy: {(100*student_train_correct[s_name]):>0.1f}%, Avg loss: {stundet_test_losses[s_name]:>8f} \n")
    
    for b_name in baselines.keys():
        baseline_test_correct[b_name] /= size
        print(f"{b_name} Test Error: \n Accuracy: {(100*baseline_test_correct[b_name]):>0.1f}%, Avg loss: {baseline_test_correct[b_name]:>8f} \n")

    dir = f"params/{dataset}/alpha=={alpha}"
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print ('Error: Creating directory. ' +  dir)

    for i in range(num_teachers):
        teachers[i].eval()
        torch.save(teachers[i].state_dict(), f"{dir}/teacher{i}.pt")
    for s_name, student in students.items():
        student.eval()
        torch.save(student.state_dict(), f"{dir}/{s_name}.pt")
    for b_name, baseline in baselines.items():
        baseline.eval()
        torch.save(baseline.state_dict(), f"{dir}/{b_name}.pt")
    
    print("Done!")

if __name__ == "__main__":
    main()
        