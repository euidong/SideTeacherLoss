import torch
from model import Model
from dataLoader import get_data_loader
from lossFn import SideTeacherLoss
import torch.optim as optim

device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def train(dataloader, model, loss_fn, optimizer, start_batch_num=0, stop_batch_num=-1):
    size = len(dataloader.dataset)
    stop_batch_num = size if stop_batch_num == -1 else stop_batch_num

    for batch_num, (X, y) in enumerate(dataloader):
        if start_batch_num > batch_num:
            continue
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_num % 100 == 0:
            loss, current = loss.item(), (batch_num - start_batch_num) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{(stop_batch_num - start_batch_num) * len(X):>5d}]")

        # for over fitting
        if batch_num == stop_batch_num:
            break
    
def test(dataloader, model, loss_fn, start_batch_num=0, stop_batch_num=-1):
    size = 0
    test_loss, correct = 0, 0

    with torch.no_grad():
        for batch_num, (X, y) in enumerate(dataloader):
            if start_batch_num > batch_num:
                continue
            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            size += len(X)
            if batch_num == stop_batch_num:
                break
    
    test_loss /= size
    correct /= size

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    batch_size = 64
    epoch_num = 300
    learning_rate = 0.005
    alpha = 0.0001
    start_batch_num = 0
    stop_batch_num = -1

    train_loader, test_loader = get_data_loader(batch_size, "fashion_mnist")
    model = Model().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = SideTeacherLoss(model, torch.nn.CrossEntropyLoss(), device, alpha=alpha)

    for epoch in range(epoch_num):
        print(f"Epoch: {epoch+1}====================")
        train(train_loader, model, loss_fn, optimizer, start_batch_num, stop_batch_num)
        if epoch % 10 == 0:
            test(train_loader, model, loss_fn, start_batch_num, stop_batch_num)
            test(test_loader, model, loss_fn)
    print("Done!")

    model.save()

if __name__ == "__main__":
    main()
        