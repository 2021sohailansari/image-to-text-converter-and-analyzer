from os import closerange
from PIL import Image
import pytesseract as tess
from nltk.util import pr
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

tess.pytesseract.tessetact_cmd = r'G:\Program Files\Tesseract'

image = r'I:\Sohail Ansari backup3\PROJ MCA\mini proj\images\comments-meme.jpeg'
text = tess.image_to_string(Image.open(image), lang="eng")
#print(text)

data = pd.read_csv("twitter.CSV")
#print(data.head())

data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "No Hate and Offensive"})
#print(data.head())

data = data[["tweet", "labels"]]
#print(data.head())

import re
import nltk
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
data["tweet"] = data["tweet"].apply(clean)
#print(data.head())

x = np.array(data["tweet"])
y = np.array(data["labels"])

cv = CountVectorizer()
X = cv.fit_transform(x) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

def hate_speech_detection(sample):
    data = cv.transform([sample]).toarray()
    a = clf.predict(data)
    print(a)

def isPositive(sample,fname):
    with open(fname) as f:
        l1 = f.read().split("\n")
        flag = False
        for i in sample.split(" "):
            if i in ["never","no","not","none"]:
                flag =  True
            elif i in l1 and not flag:
                return False
    return True

sample = text
print(sample)
if(isPositive(sample,"negWord.txt")):
    print("Neutral")
else:
    hate_speech_detection(sample)

