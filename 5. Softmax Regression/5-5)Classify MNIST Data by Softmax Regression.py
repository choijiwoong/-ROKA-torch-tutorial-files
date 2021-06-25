import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random

USE_CUDA=torch.cuda.is_available()#if we can use GPU, then return True
device=torch.device("cuda" if USE_CUDA else "cpu")#if we can use GPU, then use GPU
print("next device will be used:",device)

#random.seed(777)
#torch.manual_seed(777)
#if device == 'cuda':
    #torch.cuda.manual_seed_all(777)

#hyperperameter
training_epochs=15
batch_size=100

mnist_train=dsets.MNIST(root='MNIST_data/',#data download location
                        train=True,#if no train data, then download train data
                        transform=transforms.ToTensor(),#data to tytorch Tensor
                        download=True)#if data is not in location, then download
mnist_test=dsets.MNIST(root='MNIST_data/',
                       train=False,
                       transform=transforms.ToTensor(),
                       download=True)

data_loader=DataLoader(dataset=mnist_train, batch_size=batch_size,shuffle=True,drop_last=True)#target to load, ,, if last batch delete_loss data get accuracy

#MNIST data image of shape 28*28=784
linear=nn.Linear(784,10,bias=True).to(device)

criterion=nn.CrossEntropyLoss().to(device)#cost function define
optimizer=torch.optim.SGD(linear.parameters(),lr=0.1)#set optimizer

for epoch in range(training_epochs):# 15times repeat
    avg_cost=0
    total_batch=len(data_loader)#data loader's len_size

    for X, Y in data_loader:#batch_size=100
        X=X.view(-1,28*28).to(device)#be tensor of (10,784)
        Y=Y.to(device)#label is not state of one_hot_encoding but 0~9 integral num

        optimizer.zero_grad()
        hypothesis=linear(X)
        cost=criterion(hypothesis,Y)
        cost.backward()
        optimizer.step()

        avg_cost+=cost/total_batch

    print('Epoch:', '%04d' %(epoch+1), 'cost=','{:.9f}'.format(avg_cost))
print('Learning finished')#503 error occur

with torch.no_grad():
    X_test=mnist_test.test_data.view(-1,28*28).float().to(device)
    Y_test=mnist_test.test_labels.to(device)

    prediction=linear(X_test)
    correct_prediction=torch.argmax(prediction,1)==Y_test
    accuracy=correct_prediction.float().mean()
    print('Accuracy:',accuracy.item())

    r=random.randint(0,len(mnist_test)-1)
    X_single_data=mnist_test.test_data[r:r+1].view(-1,28*28).float().to(device)
    Y_single_data=mnist_test.test_labels[r:r+1].to(device)

    print('Label: ',Y_single_data.item())
    single_prediction=linear(X_single_data)
    print('Prediction: ',torch.argmax(single_prediction,1).item())

    plt.imshow(mnist_test.test_data[r:r+1].view(28,28),cmap='Greys',interpolation='nearest')
    plt.show()
