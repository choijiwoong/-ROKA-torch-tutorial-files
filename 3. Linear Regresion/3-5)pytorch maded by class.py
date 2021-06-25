import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

model=nn.Linear(1,1)

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(1,1)

    def forward(self,x):
        return self.lnear(x)
model=LinearRegressionModel()



class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(3,1)

    def forward(self,x):
        return self.linear(x)
model=MultivariateLinearRegressionModel()


# up room is example for conception
x_train=torch.FloatTensor([[1],[2],[3]])
y_train=torch.FloatTensor([[2],[4],[6]])

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(1,1)
    def forward(self,x):
        return self.linear(x)
model=LinearRegressionModel()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

nb_epochs=2000
for epoch in range(nb_epochs+1):
    prediction=model(x_train)
    cost=F.mse_loss(prediction,y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch%100==0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))

#
x_train=torch.FloatTensor([[73,80,75],
                           [93,88,93],
                           [89,91,90],
                           [96,98,100],
                           [73,66,70]])
y_train=torch.FloatTensor([[152],[185],[180],[196],[142]])

class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(3,1)

    def forward(x):
        return self.linear(x)
    
model=MultivariateLinearRegressionModel()
optimizer=torch.optim.SGD(model.parameters(),lr=1e-5)

nb_epochs=2000
for epoch in range(nb_epochs+1):
    prediction=model(x_train)#*********error occur***********
    cost=F.mse_loss(prediction,y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
if epoch%100==0:
    print('Epoch {:4d}/{} cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))
