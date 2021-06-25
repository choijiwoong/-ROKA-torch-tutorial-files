import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

#1) pretreatment
sentence=("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")#one sentence
char_set=list(set(sentence))
char_dic={c: i for i, c in enumerate(char_set)}
print(char_dic)

dic_size=len(char_dic)
print('문자 집합의 크기 : {}'.format(dic_size))

#set hyperparameter
hidden_size=dic_size
sequence_length=10#임의의 숫자 지정, 10의 단위로 샘플을 잘라서 데이터 만들 예정
learning_rate=0.1

x_data=[]
y_data=[]

for i in range(0, len(sentence)- sequence_length):
    x_str=sentence[i:i+sequence_length]
    y_str=sentence[i+1:i+sequence_length+1]
    print(i,x_str,'->',y_str)

    x_data.append([char_dic[c] for c in x_str])
    y_data.append([char_dic[c] for c in y_str])

print(x_data[0])
print(y_data[0])

x_one_hot=[np.eye(dic_size)[x] for x in x_data]
X=torch.FloatTensor(x_one_hot)
Y=torch.LongTensor(y_data)

print('size of training data: {}'.format(X.shape))
print('size of label: {}'.format(Y.shape))

print(X[0])
print(Y[0])

#2) create model
class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super(Net, self).__init__()
        self.rnn=torch.nn.RNN(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc=torch.nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x):
        x, _status=self.rnn(x)
        x=self.fc(x)
        return x
    
net=Net(dic_size, hidden_size, 2)#2 layer
criterion=torch.nn.CrossEntropyLoss()
optimizer=optim.Adam(net.parameters(), learning_rate)
outputs=net(X)
print(outputs.shape)

print(outputs.view(-1,dic_size).shape)#2차원 텐서로 변환
print(Y.shape)
print(Y.view(-1).shape)#마찬가지로 정확도 측정시 펼쳐서 계산 예정

for i in range(100):
    optimizer.zero_grad()
    outputs=net(X)
    loss=criterion(outputs.view(-1, dic_size), Y.view(-1))
    loss.backward()
    optimizer.step()

    results=outputs.argmax(dim=2)
    predict_str=""
    for j, result in enumerate(results):
        if j==0:
            predict_str+=''.join([char_set[t] for t in result])
        else:
            predict_str+=char_set[result[-1]]
    print(predict_str)
