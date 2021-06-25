#Module import
import nltk
nltk.download('punkt')
import urllib.request
import zipfile
from lxml import etree
import re
from nltk.tokenize import word_tokenize, sent_tokenize
#1) Understanding of data _download data
urllib.request.urlretrieve("https://raw.githubusercontent.com/GaoleMeng/RNN-and-FFNN-textClassification/master/ted_en-20160408.xml", filename="ted_en-20160408.xml")
#이는 xml파일이기에 <context>만 추출해야함

#2) Data Pretreatment
targetXML=open('ted_en-20160408.xml', 'r', encoding='UTF8')
target_text=etree.parse(targetXML)
parse_text='\n'.join(target_text.xpath('//content/text()'))
#Xml에서 <content>와 </content>사이의 내용만 가져온다.

content_text=re.sub(r'\([^)]*\)', '', parse_text)
#sub로 (Audio), (Laughter)등의 배경음 부분 제거, 괄호 제거

sent_text=sent_tokenize(content_text)
#NLTK를 이용하여 문장 토큰화를 수행

normalized_text=[]
for string in sent_text:
    tokens=re.sub(r"[^a-z0-9]+", " ", string.lower())
    normalized_text.append(tokens)
    
#구두점을 제거하고, 모두 소문자화
    
result=[]
result=[word_tokenize(sentence) for sentence in normalized_text]
#각 문장에 대하여 단어 토큰화

print('Amount of all samples: {}'.format(len(result)))

for line in result[:3]:
    print(line)

#3) Training Word2Vec
from gensim.models import Word2Vec, KeyedVectors
model=Word2Vec(sentences=result, vector_size=100, window=5, min_count=5, workers=4, sg=0)#argument of size is vector_size
#입력한 단어에 대해 가장 유사한 단어를 출력하는 model.ww.most_similar을 지원
model_result=model.wv.most_similar("man")
print(model_result)

#4) Save Word2Vec model ans load
model.wv.save_word2vec_format('./eng_w2v')#save model
loaded_model=KeyedVectors.load_word2vec_format("eng_w2v")#load model
#잠깐 쉬자
model_result=loaded_model.most_similar("man")
print(model_result)

#*****한국어 Word2Vec 만들기******#
#python -m C:\Users\admin0!\AppData\Local\Programs\Python\Python39\Scripts\wikiextractor\wikiextractor\WikiExtractor C:\Users\admin0!\AppData\Local\Programs\Python\Python39\Scripts\wikiextractor\wikiextractor\kowiki-latest-pages-articles.xml.bx2
