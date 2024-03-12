import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
filename = '../THUCNews/data/dev.txt'
train_df = pd.read_csv(filename,sep='\t',nrows=1000)
print(train_df.head())
# tfidf = TfidfVectorizer(ngram_range=(1,3),max_features=3000)
# train_test = tfidf.fit_transform(train_df['text'])
#
# clf = RidgeClassifier()
# clf.fit(train_test[:10000],train_df['label'].values[:10000])
# val_pred = clf.predict(train_test[:10000])
# print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))
