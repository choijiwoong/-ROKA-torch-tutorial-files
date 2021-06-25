#1. Seperating of training data and test data
import urllib.request
import pandas as pd

urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", filename="IMDb_Reviews.csv")
df=pd.read_csv('IMDb_Reviews.csv', encoding='latin1')
df.head()

print('Amount of all samples: {}'.format(len(df)))

train_df=df[:25000]
test_df=df[25000:]
test_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)#not save index

print()
#2. Set field
from torchtext.legacy import data#torchtext.data->torchtext.legacy.data

TEXT=data.Field(sequential=True,#Whether of sequence data
                use_vocab=True,#Whether of making work set
                tokenize=str.split,#set tokenizer function
                lower=True,#Make upper to lower
                batch_first=True,#Whether of getting mini_batch layer to first
                fix_length=20)#Maximum length

LABEL=data.Field(sequential=False,
                 use_vocab=False,
                 batch_first=False,
                 is_target=True)

print()
#3. Make dataset
from torchtext.legacy.data import TabularDataset

train_data, test_data=TabularDataset.splits(
    path='.', train='train_data.csv', test='test_data.csv', format='csv',#directory
    fields=[('text', TEXT), ('label', LABEL)], skip_header=True)#set field name

print('Amount of training samples: {}'.format(len(train_data)))
print('Amount of test samples: {}'.format(len(test_data)))
print(vars(train_data[0]))
print(train_data.fields.items())
print()
#4. Make Vocabulary
TEXT.build_vocab(train_data, min_freq=10, max_size=10000)#Make vocabulary set, minimum showing condition, max size
print('Amount of vocabulary set: {}'.format(len(TEXT.vocab)))#auto adding <unk> & <pad>

print()
#5. Make dataloader of torchtext
from torchtext.legacy.data import Iterator#name of dataloader

batch_size=5

train_loader=Iterator(dataset=train_data, batch_size=batch_size)
test_loader=Iterator(dataset=test_data, batch_size=batch_size)
print('Amount of mini_batch in training data: {}'.format(len(train_loader)))
print('Amount of mini_batch in test data: {}'.format(len(test_loader)))

batch=next(iter(train_loader))#first mini_batch
print(type(batch))#not Tensor, get class "torchtext.legacy.data.batch.Batch"

print(batch.text)#num 0 is <unk> that is not in vocabulary set

print()
#6. Case of using <pad>
#In defining field, if we set fix_length 150 not 20, then empty storage set 1 that means <pad>(padding)
