#Tokenization
en_text="A Dog Run back corner near spare bedrooms"

#1)use spaCy****ERROR OCCUR****
import spacy

#spacy_en=spacy.load('en')
def tokenize(en_text):
    return [tok.text for tok in apacy_en.tokenizer(en_text)]
#print(tokenize(en_text))


#2)use NLTK
import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize
print(word_tokenize(en_text))


#3)tokenize by spacing
print(en_text.split())


#4)Tokenizing by spacing in Korean
kor_text="사과의 놀라운 효능이라는 글을 봤어. 그래서 오늘 사과를 먹으려고 했는데 사과가 썩어서 슈퍼에 가서 사과랑 오렌지 사왔어"
print(kor_text.split())

#4)형태소 토큰화****ERROR OCCUR****
from konlpy.tag import Mecab
#tokenizer=Mecab()
#print(tokenizer.morphs(kor_text))

#5)문자 토큰화
print(list(en_text))


#*******************************************************
print()
print()

#Make Vocabulary
import urllib.request
import pandas as pd
from konlpy.tag import Mecab
from nltk import FreqDist
import numpy as np
import matplotlib.pyplot as plt

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")
data=pd.read_table('ratings.txt')
print(data[:10])
print('전체 샘플의 수: {}'.format(len(data)))
sample_data=data[:100]
sample_data['document']=sample_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣]","")
print(sample_data[:10])


stopwords=['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
