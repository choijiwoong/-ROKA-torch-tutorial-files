#Sequence model. != Recursive Neural Network
#memory cell or RNN cell
#hidden state
#one-to-many_image captioning, many-to-one_sentiment classfication || spam detection, many-to-many_chat bot

#2) create RNN in python
import numpy as np

timesteps=10#시점의 수 _문장의 길이
input_size=4#입력의 차원_단어벡터의 차원
hidden_size=8#메모리 셀의 용량(은닉상태의 크기)

inputs=np.random.random((timesteps, input_size))#입력에 해당하는 2D텐서
hidden_state_t=np.zeros((hidden_size,))#jiddensize로 은닉상태 만들고 0초기화
print(hidden_state_t)

Wx=np.random.random((hidden_size, input_size))#입력 가중치
Wh=np.random.random((hidden_size, hidden_size))#은닉 가중치
b=np.random.random((hidden_size,))

print(np.shape(Wx))
print(np.shape(Wh))
print(np.shape(b))

total_hidden_states=[]

#memory cell work
for input_t in inputs:
    output_t=np.tanh(np.dot(Wx,input_t)+np.dot(Wh,hidden_state_t)+b)
    total_hidden_states.append(list(output_t))#각 시점의 은닉상태값을 축적
    print(np.shape(total_hidden_states))
    hidden_state_t=output_t

total_hidden_states=np.stack(total_hidden_states, axis=0)#깨끗한 출력
print(total_hidden_states)

#3) nn.RNN() in pytorch
import torch
import torch.nn as nn

input_size=5#입력 크기
hidden_size=8#은닉상태의 크기

inputs=torch.Tensor(1, 10, 5)#배치크기 1 10번의 시점동안 5차원의 입력벡터
cell=nn.RNN(input_size, hidden_size, batch_first=True)#입력텐서의 첫번째 차원이 배치크기
outputs, _status=cell(inputs)#2개의 입력을 리턴. 모든시점의 은닉상태들, 마시막시점의 은닉상태
print(outputs.shape)

#4) Deep Recurrent Neural Network
inputs=torch.Tensor(1, 10, 5)
cell=nn.RNN(input_size=5, hidden_size=8, num_layers=2, batch_first=True)# 은닉층 2개(cell)
print(outputs.shape)
print(_status.shape)#층개수, 배치크기, 은닉상태의 크기

#5) Bidirectional Recurrent Neural Network
inputs=torch.Tensor(1, 10, 5)
cell=nn.RNN(input_size=5, hidden_size=8, num_layers=2, batch_first=True, bidirectional=True)#양방향순환
outputs, _status=cell(inputs)
print(outputs.shape)#연결되었기에 은닉상태크기2배
print(_status.shape)#층의개수2배
