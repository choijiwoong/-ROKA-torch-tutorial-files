#컴퓨터는 문자보다는 숫자를 더 잘 처리할 수 있기에 대표적으로 단어를 숫자로 바꾸는 기법인  원-핫 인코딩 사용
#단어집합은 텍스트의 모든 단어를 중복 허용하지 않고 모아놓으면 완성
#정수맵핑 후 숫자로 바뀐 단어를 벡터로 다루고 싶다면

#1) One-hot-encoding
from konlpy.tag import Okt
okt=Okt()#ERROR*********자바 skd문제로 코드만 따라 작성
token=okt.morphs("나는 자연어 처릴 배운다")
print(token)
#complete tokenizing

word2index={}
for voca in token:
    if voca not in word2index.keys():
        word2index[voca]=len(word2index)
print(word2index)
#complete integral mapping

def one_hot_encoding(word, word2index):#function make one-hot-vector when input token
    one_hot_vector=[0]*(len(word2index))
    index=word2index[word]
    one_hot_vector[index]=1
    return one_hot_vector
print(one_hot_encoding("자연어", word2index))

#2) Limit of one-hot encoding
#amount of voca 정비례 tensor of vector_unefficient
#can't express similarity.
