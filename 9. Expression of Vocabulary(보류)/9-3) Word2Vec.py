#원핫벡터에서 유사도를 판별할 수 없기에 대신 사용되는 대표적인 방법이 워드투벡터
#1) Sparse Representation
#단어의 의미를 다차원 공간에 벡터화하는 표현->Distributed representation

#2) Distributed Representation (based on distributional hypothesis)
#Ex) 강아지 = [ 0 0 0 0 1 0 0 0 0 0 0 0 ... 중략 ... 0]
#Ex) 강아지 = [0.2 0.3 0.5 0.7 0.2 ... 중략 ... 0.2]--->여러 차원에 분산 for similarity 이러한 학습방법 Word2Vec

#Word2Vec=CBOW_주변단어로 중간단어 에측 || Skip-Gram_중간단어로 주변단어 예측

#3) CBOW(Continuous Bag of Words)
#Shallow Neural Network(Only one hidden layer), projection layer
#output y에 softmax함수를 이용하여 0과1사이의 실수로, 합이 1이되게 상태를 변환함->>스코어 벡터==중심단어일 확률
#output y 와 실제 y의 오차를 줄이기 위해 손실함수 cross-entropy를 사용
#정확하게 예측했다면 손실함수의 값이 0이 되기에 손실함수의 값을 최소화하는 방향으로 학습.
#역전파를 수행하면 학습이 진행되는데, 임베딩벡터를 선택하면 됨(W와 W'중에서_평균을 쓰기도 함)

#4) Skip-gram_투사층에서 벡터들의 평균을 구하는 과정이 없기에 성능이 더 좋음
#CBOW는 주변단어를 통해 중심단어를 예측 || Skip-gram은 중심단어를 통해 주변단어를 예측

#5) Negative Sampling
#Word2Vec을 사용하면 SGNS(Skip-Gram with Negative Sampling)을 사용하는데 이는 Skip-gram에서 Negative Sampling을 추가로 사용한다는 것이다.
#쓸데없는 항목까지 다 계산하는 속도의 비효율성을 고려하여 훨씬 작은 단어집합을 만들고 시작(ㅈ주변을 긍정 랜덤추출단어를 부정으로 두고 학습 진행)
