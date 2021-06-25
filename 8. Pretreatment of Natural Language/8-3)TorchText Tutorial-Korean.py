#2. download train data and test data
import urllib.request
import pandas as pd

#download naver movie review data->ratings_train.txt & retings_test.txt
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

train_df=pd.read_table('ratings_train.txt')
test_df=pd.read_table('ratings_test.txt')

print(train_df.head())
print(test_df.head())#id: not needed data, document: movie review, label: good or not
print('Amount of training data samples: {}'.format(len(train_df)))
print('Amount of test data samples: {}'.format(len(train_df)))

print()
#3. set field(torchtext.legacy.data)
from torchtext.legacy import data
from konlpy.tag import Mecab

tokenizer=Mecab()#ERROR
