from imports import *
from preprocess import *
from model_simple import *
from train_mod import *
from torchvision import models
from torch.optim import lr_scheduler
# get data into loaders
cacti_dataset = CactiDataset(csv_file="../input/train.csv", root_dir="../input/train/",
                             transform=transforms.Compose([Rescale(224), ToTensor()]))

train_size = int(0.8 * cacti_dataset.__len__())
test_size = cacti_dataset.__len__() - train_size
train_dataset, test_dataset = torch.utils.data.random_split(cacti_dataset,
                                                            [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=5,
                                           shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=5,
                                          shuffle=False, num_workers=2)

# initialize model class
# re-write fc layer of pretrained model
model = models.vgg16(pretrained=True)
model.classifier[6] = nn.Linear(4096, 2)


# define loss and hyperparameters
error = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=.00001, momentum=.9)
num_iterations = 2500
num_epochs = 2
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# train model
if __name__ == '__main__':
    iteration, loss, accuracy = train_mod(model, train_loader, test_loader,
                                          error, optimizer, num_iterations, num_epochs)

    sys.exit()

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
        outputs = model(batch)
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
                test_outputs = model(test)
                predicted = torch.max(test_outputs.data, 1)[1]
                total += len(test_lables)
                correct += (predicted == test_lables).sum()

            accuracy = 100 * correct / float(total)
            accuracy_list.append(accuracy)
        if count % 100 == 0:
            print('iteration: {} loss: {} accuracy {}'.format(count,loss.item(),accuracy))


# load trained model
model = Net()
model.load_state_dict(torch.load("../output/model_simple"))
model.eval()

# visualization loss 
plt.plot(iteration_list,loss_list)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("ANN: Loss vs Number of iteration")
plt.show()

# visualization accuracy 
plt.plot(iteration_list,accuracy_list,color = "red")
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("ANN: Accuracy vs Number of iteration")
plt.show()


