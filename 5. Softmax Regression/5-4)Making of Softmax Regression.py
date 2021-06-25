import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)

#initial data_8 sample
x_train=[[1,2,1,1],
         [2,1,3,2],
         [3,1,3,4],
         [4,1,5,5],
         [1,7,5,5],
         [1,2,5,6],
         [1,6,6,6],
         [1,7,7,7]]
y_train=[2,2,2,1,1,1,0,0]#We can know there are 3 class

#data transforming
x_train=torch.FloatTensor(x_train)
y_train=torch.LongTensor(y_train)

print(x_train.shape)#8x4
print(y_train.shape)#8
#class is 3. so one_hot_encoding(y_train) has to be 8x3

y_one_hot=torch.zeros(8,3)#make storage fot result of one_hot_encoding(y_train)
y_one_hot.scatter_(1,y_train.unsqueeze(1),1)#y_train data inputting to y_one_hot
print(y_one_hot.shape)

#W's size has to be 4x3_weight because of 4 to 3
#model initializing

#
#low_level making
print()
print('#low_level making')
W=torch.zeros((4,3),requires_grad=True)
b=torch.zeros(1,requires_grad=True)
optimizer=optim.SGD([W,b],lr=0.1)

nb_epochs=1000
for epoch in range(nb_epochs+1):
    hypothesis=F.softmax(x_train.matmul(W)+b,dim=1)#.matmul array's multiple
    
    cost=(y_one_hot*-torch.log(hypothesis)).sum(dim=1).mean()
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch%100==0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))

#
#high_level making
print()
print('#high_level making')
W=torch.zeros((4,3),requires_grad=True)
b=torch.zeros(1,requires_grad=True)
optimizer=optim.SGD([W,b],lr=0.1)

nb_epochs=1000
for epoch in range(nb_epochs+1):
    z=x_train.matmul(W)+b
    cost=F.cross_entropy(z,y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))

#
#Making Softmax Regression by nn.Module
print()
print('#Making Softmax Regression by nn.Module')
model=nn.Linear(4,3)#4 signitures, 3 class->input 4 signitures in one sample, classifing by 3 class
optimizer=optim.SGD(model.parameters(),lr=0.1)

nb_epochs=1000
for epoch in range(nb_epochs+1):
    prediction=model(x_train)

    cost=F.cross_entropy(prediction,y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch %100==0:
        print('Epoch {:4d}/{} cost: {:.6}'.format(epoch, nb_epochs, cost.item()))

#Making Softmax Regression by Class
print()
print('#Making Softmax Regression by Class')
class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(4,3)

    def forward(self,x):
        return self.linear(x)

model=SoftmaxClassifierModel()

optimizer=optim.SGD(model.parameters(),lr=0.1)

nb_epochs=1000
for epoch in range(nb_epochs+1):
    prediction=model(x_train)

    cost=F.cross_entropy(prediction,y_train)
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs,cost.item()))
