import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

#1) Pretreatment
input_str='apple'
label_str='pple!'
char_vocab=sorted(list(set(input_str+label_str)))
vocab_size=len(char_vocab)
print("Size of char set: {}".format(vocab_size))

input_size=vocab_size#입력의 크기는 문자 집합의 크기
hidden_size=5
output_size=5
learning_rate=0.1

char_to_index=dict((c, i) for i, c in enumerate(char_vocab))#매핑
print(char_to_index)

index_to_char={}
for key, value in char_to_index.items():
    index_to_char[value]=key
print(index_to_char)

x_data=[char_to_index[c] for c in input_str]
y_data=[char_to_index[c] for c in label_str]
print(x_data)
print(y_data)

#batch tensor add
x_data=[x_data]
y_data=[y_data]
print(x_data)
print(y_data)

x_one_hot=[np.eye(vocab_size)[x] for x in x_data]
print(x_one_hot)

X=torch.FloatTensor(x_one_hot)
Y=torch.LongTensor(y_data)

print('size of trainning data: {}'.format(X.shape))
print('size of label: {}'.format(Y.shape))

#2) create model
class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.rnn=torch.nn.RNN(input_size, hidden_size, batch_first=True)#cell
        self.fc=torch.nn.Linear(hidden_size, output_size, bias=True)#출력층 fully-connected layer

    def forward(self, x):
        x, _status=self.rnn(x)
        x=self.fc(x)
        return x

net=Net(input_size, hidden_size, output_size)
outputs=net(X)
print(outputs.shape)#배치차원, 시점, 출력크기

#정확도 측정
print(outputs.view(-1, input_size).shape)

print(Y.shape)
print(Y.view(-1).shape)

criterion=torch.nn.CrossEntropyLoss()
optimizer=optim.Adam(net.parameters(), learning_rate)

for i in range(100):
    optimizer.zero_grad()
    outputs=net(X)
    loss=criterion(outputs.view(-1, input_size), Y.view(-1))
    loss.backward()
    optimizer.step()

    result=outputs.data.numpy().argmax(axis=2)
    result_str=''.join([index_to_char[c] for c in np.squeeze(result)])
    print(i, "loss: ", loss.item(), "prediction: ", result, "true Y: ", y_data, "prediction str: ", result_str)
