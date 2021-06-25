import torch#main namespace
import torch.nn as nn#for making nural map
import torch.nn.functional as F
import torch.optim as optim#Stochastic Gradient Descent

torch.manual_seed(1)#for same result value_random seed fixing

x_train=torch.FloatTensor([[1],[2],[3]])#define variable
y_train=torch.FloatTensor([[2],[4],[6]])

print(x_train)#value show
print(x_train.shape)#print size
print(y_train)
print(y_train.shape)

W=torch.zeros(1,requires_grad=True)#Weight setting to Zero and setting variable values by cases
print(W)#Weight show

b=torch.zeros(1,requires_grad=True)#bias setting to Zero
print(b)
#now Y=0X+0
hypothesis=x_train*W+b#make hypothesis maded by linear calculation
print(hypothesis)

cost=torch.mean((hypothesis-y_train)**2)#setting Mean Squeared Error_cost function
print(cost)

optimizer=optim.SGD([W,b],lr=0.01)#SGD is one of Stochastic Gradient Descent, lr is learning rate

nb_epochs=1999#repeats num
for epoch in range(nb_epochs+1):
    hypothesis=x_train*W+b
    cost=torch.mean((hypothesis-y_train)**2)

    optimizer.zero_grad()#for preventing add gradiented values
    cost.backward()
    optimizer.step()

    if epoch%100==0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(epoch, nb_epochs,W.item(),b.item(),cost.item()))
