import torch
import numpy

device='cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

X=torch.FloatTensor([[0,0],[0,1],[1,0],[1,1]]).to(device)
Y=torch.FloatTensor([[0],[1],[1],[0]]).to(device)

linear=torch.nn.Linear(2,1,bias=True)
sigmoid=torch.nn.Sigmoid()
model=torch.nn.Sequential(linear,sigmoid).to(device)

criterion=torch.nn.BCELoss().to(device)#CrossEntropyFunction in Binary Claccification_define of cost function
optimizer=torch.optim.SGD(model.parameters(),lr=1)#define optimizer

for step in range(10001):
    optimizer.zero_grad()
    hypothesis=model(X)

    cost=criterion(hypothesis,Y)#X->model->criterion
    cost.backward()
    optimizer.step()

    if step%100==0:
        print(step, cost.item())
        #single-perceptron cannot solve XOR

with torch.no_grad():
    hypothesis=model(X)
    predicted=(hypothesis>0.5).float()
    accuracy=(predicted==Y).float().mean()
    print('model\'s print(Hypothesis): ',hypothesis.detach().cpu().numpy())
    print('model\'s prediction(Predicted): ', predicted.detach().cpu().numpy())
    print('real value(Y): ',Y.cpu().numpy())
    print('Accuracy: ',accuracy.item())
