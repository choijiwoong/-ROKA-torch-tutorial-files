#1) Seperate of training data and test data
import urllib.request
import pandas as pd

#download training data and test data & Tranforming 
urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", filename="IMDb_Reviews.csv")
df=pd.read_csv('IMDb_Reviews.csv', encoding='latin1')
print('Amount of Samples: {}'.format(len(df)))

#data slicing               
train_df=df[:25000]
test_df=df[25000:]

#df to csv
train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

#2) Defining field
from torchtext.legacy import data#torchtext.data->torchtext.legacy.data


TEXT=data.Field(sequential=True,
                use_vocab=True,
                tokenize=str.split,
                lower=True,
                batch_first=True,#Whether calling of data with mini batch tensor to first
                fix_length=20)

LABEL=data.Field(sequential=False,
                 use_vocab=False,
                 batch_first=False,
                 is_target=True)

#3) Make dataset, vocabulary set, dataloader
from torchtext.legacy.data import TabularDataset
from torchtext.legacy.data import Iterator
#dataload & tokenize to field
train_data, test_data=TabularDataset.splits(
    path='.', train='train_data.csv', test='test_data.csv', format='csv',
    fields=[('text', TEXT), ('label', LABEL)], skip_header=True)

#make vocabulary set in difined field
TEXT.build_vocab(train_data, min_freq=10, max_size=10000)#with 10000voca

batch_size=5
train_loader=Iterator(dataset=train_data, batch_size=batch_size)
batch=next(iter(train_loader))

print(batch.text)#first mini batch
print(batch.text.shape)

#4) Redefine Field without batch_first=True
TEXT=data.Field(sequential=True,
                use_vocab=True,
                tokenize=str.split,
                lower=True,
                fix_length=20)

LABEL=data.Field(sequential=False,
                 use_vocab=False,
                 batch_first=False,
                 is_target=True)

train_data, test_data=TabularDataset.splits(
    path='.', train='train_data.csv', test='test_data.csv', format='csv',
    fields=[('text', TEXT), ('label', LABEL)], skip_header=True)

#make vocabulary set in difined field
TEXT.build_vocab(train_data, min_freq=10, max_size=10000)#with 10000voca

batch_size=5
train_loader=Iterator(dataset=train_data, batch_size=batch_size)
batch=next(iter(train_loader))

#5) Size of tnesor when batch_first=False
print(batch.text)
print(batch.text.shape)

#Plus Learning
#Field_어떻게 전처리를 진행할 것인지 클래스와 비슷한 개념 , TabularDataset_데이터를 불러오면서 필드에서 정의했던 토큰화 방식으로 토큰화 자동 수행,
#TEXT.build_vocab_토큰화 전처리를 끝낸 후 각 단어에 고유한 정수를 맵핑해주는 정수 인코딩 이전에 단어집합 을 만들기 ,
#Iterator_torchtrxt에서 미니 배치만큼 데이터를 로드해주는역활 
