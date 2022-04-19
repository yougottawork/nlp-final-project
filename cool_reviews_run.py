from tokenize import cool
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
import string
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.feature_extraction
from sklearn.utils import resample

temp_file = open("cool_with_weight.txt", "w")
with open("no_header_review_yelp.txt", errors = "ignore") as f:
    for line in f: 
        temp = line.split('\t')
        weight = 0
        cool = temp[3].strip("\t")
        if cool.isdigit() == True:
            if int(cool) >= 3:
                weight = int(cool)
                cool = 1   
        new = temp[4].replace('\t', ' ')
        temp_file.write(str(cool) + "\t" + new + "\t" + str(weight) + "\n")
temp_file.close()

cool_df = pd.read_csv("cool_with_weight.txt", sep="\t", names=["cool", "review", "weight"], header=None)
cool_df.head()
# We can only use the ones with 0 and 1 for value counts, discard rest
# cool_df["cool"].value_counts()
cool_df = cool_df[(cool_df["cool"] == str(0)) | ((cool_df["cool"] == str(1)) & (cool_df["weight"] >=3))]
cool_df["cool"].value_counts()

df_majority = cool_df[cool_df["cool"] == str(0)]
df_minority = cool_df[cool_df["cool"] == str(1)]

# Downsample majority 
df_majority_downsampled = resample(df_majority, replace=False, n_samples = 6316)#, random_state=123)

# Combine minority class with downsampled majority class
cool_df = pd.concat([df_majority_downsampled, df_minority])

cool_df["cool"].value_counts()

train, test= train_test_split(cool_df)
train

# Categorial Value for cool Column (Not needed honestly)

labelencoder = LabelEncoder()
train['cool'] = labelencoder.fit_transform(train['cool'])
test['cool'] = labelencoder.fit_transform(test['cool'])

#Remove special character. 

pattern = re.compile(r'<br\s*/><br\s*/>>*|(\-)|(\\)|(\/)|($)')
def preprocess_reviews(reviews):
    reviews = [pattern.sub(" ",item) for item in reviews]
    return reviews

train_clean = preprocess_reviews(train['review'])
test_clean = preprocess_reviews(test['review'])
train['review'] = train_clean
test['review'] = test_clean

train

def remove_punctuation(input):
    table = str.maketrans('','',string.punctuation)
    return input.translate(table)
train['review'] = train['review'].apply(remove_punctuation)
test['review'] = test['review'].apply(remove_punctuation)
train

# Convert all text to lowercase
train['review'] = train['review'].str.lower()
test['review'] = test['review'].str.lower()
train

# Remove line breaks
def remove_linebreaks(input):
    text = re.compile(r'\n')
    return text.sub(r' ',input)
train['review'] = train['review'].apply(remove_linebreaks)
test['review'] = test['review'].apply(remove_linebreaks)

train

nltk.download('punkt')

train['review'] = train['review'].apply(word_tokenize)
test['review'] = test['review'].apply(word_tokenize)

train

nltk.download('stopwords')

def remove_stopwords(input1):
    words = []
    for word in input1:
        if word not in stopwords.words('english'):
            words.append(word)
    return words
train['review'] = train['review'].apply(remove_stopwords)
test['review'] = test['review'].apply(remove_stopwords)

nltk.download('wordnet')
nltk.download('omw-1.4')

# Lemmartize
lem = WordNetLemmatizer()
def lemma_wordnet(input):
    return [lem.lemmatize(w) for w in input]
train['review'] = train['review'].apply(lemma_wordnet)
test['review'] = test['review'].apply(lemma_wordnet)

def combine_text(input):
    combined = ' '.join(input)
    return combined
train['review'] = train['review'].apply(combine_text)
test['review'] = test['review'].apply(combine_text)

import scipy
import scipy.sparse
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=1600, norm = None)
vectorizer.fit(train)
X_train_tfidf = vectorizer.fit_transform(train['review'])
X_test_tfidf = train['cool']
Y_train_tdidf =vectorizer.transform(test['review'])

X_train_tfidf

from sklearn.metrics import accuracy_score

clf = sklearn.linear_model.LogisticRegression()
clf.fit(X_train_tfidf, X_test_tfidf)
yhat = clf.predict(Y_train_tdidf)
y_true = test['cool']

accuracy_score(y_true, yhat)

from sklearn.metrics import classification_report,confusion_matrix 
print(classification_report(y_true, yhat, target_names = ['Bad Reviews','Good Reviews']))

import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(y_true, yhat)
plt.figure(figsize = (5,5))
sns.heatmap(cm,cmap= "Blues", 
            linecolor = 'black', 
            linewidth = 1, 
            annot = True, 
            fmt='', 
            xticklabels = ['Bad Reviews','Good Reviews'], 
            yticklabels = ['Bad Reviews','Good Reviews'])
plt.xlabel("Predicted")

from sklearn import svm 
svm = svm.SVC()
X_test_tfidf = train['cool']
svm.fit(X_train_tfidf, X_test_tfidf)
yhat = svm.predict(Y_train_tdidf)
y_true = test['cool']
accuracy_score(y_true, yhat)

from sklearn.metrics import classification_report,confusion_matrix 
print(classification_report(y_true, yhat, target_names = ['Bad Reviews','Good Reviews']))

import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(y_true, yhat)
plt.figure(figsize = (5,5))
sns.heatmap(cm,cmap= "Blues", 
            linecolor = 'black', 
            linewidth = 1, 
            annot = True, 
            fmt='', 
            xticklabels = ['Bad Reviews','Good Reviews'], 
            yticklabels = ['Bad Reviews','Good Reviews'])
plt.xlabel("Predicted")

from sklearn import linear_model
from sklearn.model_selection import cross_val_score
alpha = [0.001, 0.01, 0.1, 1]
for a in alpha:
    ridge = linear_model.RidgeClassifier(a)
    scores = cross_val_score(ridge, X_train_tfidf, X_test_tfidf, cv=5, scoring='f1')
    print("alpha: ",a)
    print(scores)
    print(np.mean(scores))
    print('\n')


from sklearn.metrics import accuracy_score
ridge = linear_model.RidgeClassifier(0.001)
ridge.fit(X_train_tfidf, X_test_tfidf)
test['cool_pred'] = ridge.predict(Y_train_tdidf)
y_true = test['cool']
y_pred = test['cool_pred']
accuracy_score(y_true, y_pred)

# SVM Classifier... Don't Run this... 

from sklearn import svm 
svm = svm.SVC()
svm.fit(X_train_tfidf, X_test_tfidf)
test["cool_pred"] = svm.predict(Y_train_tdidf)
y_pred = test['cool_pred']
accuracy_score(y_true, y_pred)

from sklearn.metrics import classification_report,confusion_matrix 
print(classification_report(y_true, y_pred, target_names = ['Bad Reviews','Good Reviews']))

import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize = (5,5))
sns.heatmap(cm,cmap= "Blues", 
            linecolor = 'black', 
            linewidth = 1, 
            annot = True, 
            fmt='', 
            xticklabels = ['Bad Reviews','Good Reviews'], 
            yticklabels = ['Bad Reviews','Good Reviews'])
plt.xlabel("Predicted")
