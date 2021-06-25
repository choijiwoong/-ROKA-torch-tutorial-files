#Word Embedding = Express of word to vector

#1) Sparse Representation: 행렬의 대부분의 값을 0으로 표현하느 방법_ex) one-hot vector
#유사도를 구별할 수 없는 치명적 단점
import torch
dog=torch.FloatTensor([1,0,0,0,0])
cat=torch.FloatTensor([0,1,0,0,0])
computer=torch.FloatTensor([0,0,1,0,0])
netbook=torch.FloatTensor([0,0,0,1,0])
book=torch.FloatTensor([0,0,0,0,1])

print(torch.cosine_similarity(dog, cat, dim=0))
print(torch.cosine_similarity(cat, computer, dim=0))
print(torch.cosine_similarity(computer, netbook, dim=0))
print(torch.cosine_similarity(netbook, book, dim=0))

#2) Dense Representation: 대충 희소표현과 반대로 사용자가 설정 한 값으로 단어의 벡터 표현의 차원을 맞춰서 실수값을 갖음

#3) Word Embedding: express of word to Dense Repredentation & result -> embedding vector
#단어를 랜덤한 값을 가지는 밀집 벡터로 전환하고 인공 신경망의 가중치르 학습시킴
