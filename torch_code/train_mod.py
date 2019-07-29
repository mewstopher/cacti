from imports import *
from model_simple import *

def train_mod(net,train_loader, test_loader,error, optimizer, num_iterations, num_epochs):
    """
    function for training model

    """
    loss_list = []
    iteration_list = []
    accuracy_list = []
    count = 0
    num_epochs = 2

    for e in range(num_epochs):
        for i, data in enumerate(train_loader):
            batch = data['image'].to(torch.float32)
            batch = Variable(batch)
            lables = Variable(data['cacti'].view(-1))
            optimizer.zero_grad()
            outputs = net(batch)
            loss = error(outputs, lables)
            loss.backward()
            optimizer.step()
            count += 1
            loss_list.append(loss.item())
            iteration_list.append(count)

            if count % 50 == 0:
                correct = 0
                total = 0
                for data in test_loader:
                    test = Variable(data['image'].to(torch.float32))
                    test_lables = Variable(data['cacti'].view(-1))
                    outputs = net(test)
                    predicted = torch.max(outputs.data, 1)[1]
                    total += len(test_lables)
                    correct += (predicted == test_lables).sum()

                accuracy = 100 * correct / float(total)
                accuracy_list.append(accuracy)
            if count % 100 == 0:
                sys.stdout.write('iteration: {} loss: {} accuracy {}'.format(count,
                                                                   loss.item(),
                                                                   accuracy))
                sys.stdout.flush()
    return iteration_list, loss_list, accuracy_list

