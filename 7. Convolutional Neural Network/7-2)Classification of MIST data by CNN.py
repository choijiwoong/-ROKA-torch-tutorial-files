#import and define of input
import torch
import torch.nn as nn

inputs=torch.Tensor(1,1,28,28)#batch size x channel x height x widht
print('size of tensor: {}'.format(inputs.shape))

#define conv and pooling
conv1=nn.Conv2d(1,32,3, padding=1)
print(conv1)

conv2=nn.Conv2d(32, 64, kernel_size=3, padding=1)
print(conv2)

pool=nn.MaxPool2d(2)
print(pool)

#making model
out=conv1(inputs)
print(out.shape)

out=pool(out)
print(out.shape)

out=conv2(out)
print(out.shape)

out=pool(out)
print(out.shape)

print(out.size(0))
print(out.size(1))
print(out.size(2))
print(out.size(3))

print("****END PRACTICE****")
print()


out=out.view(out.size(0),-1)
print(out.shape)

fc=nn.Linear(3136,10)
out=fc(out)
print(out.shape)

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)

if device == 'cuda':
    torch.cuda.manual_seeld_all(777)

#Set parameters for training
learning_rate=0.001
training_epochs=15
batch_size=100

mnist_train=dsets.MNIST(root='MNIST_data/',
                        train=True,
                        transform=transforms.ToTensor(),
                        download=True)

mnist_test=dsets.MNIST(root='MNIST_data/',
                       train=False,
                       transform=transforms.ToTensor(),
                       download=True)

data_loader=torch.utils.data.DataLoader(dataset=mnist_train,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        drop_last=True)

class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        #first layer
        self.layer1=torch.nn.Sequential(
            torch.nn.Conv2d(1,32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        #second layer
        self.layer2=torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        #Fully-Conncected layer
        self.fc=torch.nn.Linear(7*7*64,10,bias=True)
        
        #Initialization of weight on Fully-Connected layer
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out=self.layer1(x)
        out=self.layer2(out)
        out=out.view(out.size(0),-1)
        out=self.fc(out)
        return out

model=CNN().to(device)
criterion=torch.nn.CrossEntropyLoss().to(device)
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)

total_batch=len(data_loader)
print('amount of total batch: {}'.format(total_batch))

#Start training
for epoch in range(training_epochs):
    avg_cost=0

    for X, Y in data_loader:#get mini batch. X=mini batch, Y=label
        X=X.to(device)
        Y=Y.to(device)

        optimizer.zero_grad()
        hypothesis=model(X)
        cost=criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost+=cost/ total_batch
    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch+1, avg_cost))
    
with torch.no_grad():
    X_test=mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test=mnist_test.test_labels.to(device)

    prediction=model(X_test)
    correct_prediction=torch.argmax(prediction, 1)==Y_test
    accuracy=correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())
