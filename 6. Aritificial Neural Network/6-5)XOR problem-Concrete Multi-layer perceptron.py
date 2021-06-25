import torch
import torch.nn as nn

device='cuda' if torch.cuda.is_available else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
device='cpu'
X=torch.FloatTensor([[0,0],[0,1],[1,0],[1,1]]).to(device)
Y=torch.FloatTensor([[0],[1],[1],[0]]).to(device)

model=nn.Sequential(
    nn.Linear(2,10,bias=True),
    nn.Sigmoid(),
    nn.Linear(10,10,bias=True),

    nn.Sigmoid(),
    nn.Linear(10,10,bias=True),

    nn.Sigmoid(),
    nn.Linear(10,1,bias=True),
    nn.Sigmoid()
    ).to(device)

criterion=torch.nn.BCELoss().to(device)
optimizer=torch.optim.SGD(model.parameters(),lr=1) #mofified learning rate from 0.1 to 1

for epoch in range(10001):
    optimizer.zero_grad()
    hypothesis=model(X)

    cost=criterion(hypothesis,Y)
    cost.backward()
    optimizer.step()

    if epoch%100==0:
        print(epoch, cost.item())
        
with torch.no_grad():
    hypothesis=model(X)
    predicted=(hypothesis>0.5).float()
    accuracy=(predicted==Y).float().mean()
    print('output of model(Hypothesis): ',hypothesis.detach().cpu().numpy())
    print('predicted value of model(Predicted): ',predicted.detach().cpu().numpy())
    print('real value: ',Y.cpu().numpy())
    print('Accuracy: ',accuracy.item())
