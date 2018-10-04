# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 19:30:55 2018

@author: Priyal Narang
"""
import re
import nltk
import csv
import numpy as np
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize


def preprocess_word(word):
    # Remove punctuation
    word = word.strip('\'"?!,.():;')

    # funnnnny --> funny
    word = re.sub(r'(.)\1+', r'\1\1', word)
    # Remove - & '
    word = re.sub(r'(-|\')', '', word)
    return word




def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', 'Smiley', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', 'Laughing', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', 'Love', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', 'Wink', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', 'Sad', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', 'Cry', tweet)
    return tweet


def preprocess_tweet(tweet):
    # Convert to lower case
    tweet = tweet.lower()
    # Replaces URLs with the word URL
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', '', tweet)
    # Replace @handle with the word USER_MENTION
    tweet = re.sub(r'@[\S]+', '', tweet)
    # Replaces #hashtag with hashtag
    tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
    # Remove RT (retweet)
    tweet = re.sub(r'\brt\b', '', tweet)
    # Replace 2+ dots with space
    tweet = re.sub(r'\.{2,}', ' ', tweet)
    # Replace emojis with their type
    tweet = handle_emojis(tweet)
    # Strip space, " and ' from tweet
    tweet = tweet.strip(' "\'')
    # funnnnny --> funny
    tweet = re.sub(r'(.)\1+', r'\1\1', tweet)
    # Removing punctuations
    tweet = re.sub(r'[^\w\s]', ' ', tweet)
     # Removing numbers
    tweet = re.sub(r'\w*\d\w*', ' ', tweet)
    # Replace multiple spaces with a single space
    tweet = re.sub(r'\s+', ' ', tweet)
    tweet = re.sub('[^A-Za-z0-9]+', ' ', tweet)
    tweet= re.sub(" \d+", " ", tweet)
    #words = tweet.split()

    return tweet


def replace_slang(text):
    with open('slang.txt') as file:
        slang_map = dict(map(str.strip, line.partition('\t')[::2]) for line in file if line.strip())

    def replace(match):
        return slang_map[match.group(0)]

    return(re.sub('|'.join(r'\b%s\b' % re.escape(s) for s in slang_map), replace, text))
    # for word in text.split():
    #     if word in slang_map.keys():
    #         print(word + ' ' + slang_map[word])
    #         text = text.replace(' ' + word, ' ' + slang_map[word])
    #         print(text)
    # return text

def removestopwords(tweet):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tweet)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    #print(filtered_sentence)
    return filtered_sentence


def tokenize(tweet):
    tokens = word_tokenize(tweet)  # Generate list of tokens
    tokens_pos = pos_tag(tokens)
    return tokens_pos


def expandContractions(text):
    cList = {
        "ain't": "am not", "aren't": "are not", "can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
        "he's": "he is", "how'd": "how did", "how'll": "how will", "how's": "how is", "I'd": "I would", "I'll": "I will", "I'm": "I am", "I've": "I have", "isn't": "is not", "it'd": "it had", "it'll": "it will", "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not",
        "might've": "might have", "mightn't": "might not", "must've": "must have", "mustn't": "must not", "needn't": "need not", "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "she'd": "she would", "she'll": "she will", 
        "she's": "she is", "should've": "should have", "shouldn't": "should not", "so've": "so have", "so's": "so is", "that'd": "that would", "that's": "that is", "there'd": "there had", "there's": "there is", "they'd": "they would", "they'll": "they will", "they're": "they are", 
        "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we had", "we'll": "we will", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is", 
        "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", 
        "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "y'all": "you all", "y'alls": "you alls", "you'd": "you had", "you'll": "you you will", "you're": "you are", "you've": "you have"}
    c_re = re.compile('(%s)' % '|'.join(cList.keys()))

    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text)
def unique_list(l):
    ulist = []
    [ulist.append(x) for x in l if x not in ulist]
    return ulist

i=0
res=[]#list of all preprocessed tweets
temp=['']#list of list of last tockenised record
f = open("twitter4242.txt", 'r')
f1=open("result.txt","w")
f2=open("preprocessed.csv","w")

wr=csv.writer(f2)
for line in f:
    cols = line.split("\t")
    # res.append(cells[2])
    if(i>0):
        for i in range(2):
            f1.write(cols[i] + "\t")
            
        text = cols[2]
        text=replace_slang(text)
        text=expandContractions(text)
        text = preprocess_tweet(text)
        text = ' '.join(unique_list(text.split()))
        text = removestopwords(text)
        #res = ' '.join(text)
        #print(text)

        temp[0]=text
        tweet=' '.join(text)
        res.append(tweet)
        #f1.write(cols[0]+" "+cols[1]+" "+tweet+"\n")
        f1.write(tweet+"\n")
        wr.writerow(temp)
        
        #f1.write(res+"\n")
        
       
        
    i=i+1

from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(res)
print(bow)


#tfidf
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# TF-IDF feature matrix
tfidf = tfidf_vectorizer.fit_transform(res)
print(tfidf)
#gloVe
'''file = "glove.6B.50d.txt"

def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    
     
    with open(gloveFile, encoding="utf8" ) as f:
       content = f.readlines()
    model = {}
    for line in content:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model
     
     
model= loadGloveModel(file)   
 
print (model['hello'])
from gensim.scripts.glove2word2vec import glove2word2vec
glove2word2vec(glove_input_file=file, word2vec_output_file="gensim_glove_vectors.txt")
 
###Finally, read the word2vec txt to a gensim model using KeyedVectors:
 
from gensim.models.keyedvectors import KeyedVectors
glove_model = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)
'''
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
## Plotly

# Others
import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

#embeddings
embeddings_index = dict()
f = open("glove.6B.100d.txt",encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))
t=Tokenizer()
t.fit_on_texts(res)
vocabulary_size=len(t.word_index)+1
encoded_docs = t.texts_to_sequences(X)#encode words to integers
print(encoded_docs)
# pad documents to a max length of 4 words
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length)#pad these words to same length
print(padded_docs)
embedding_matrix = np.zeros((vocabulary_size, 100))
for word, index in t.word_index.items():
    if index >vocabulary_size-1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
X=[]
y=[]
f=open("result.txt","r")
for line in f:
    cols=line.split("\t")
    X.append(cols[2])
    y.append(cols[0])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
## create model
'''model = Sequential()
e = Embedding(vocabulary_size, 100, weights=[embedding_matrix], input_length=4, trainable=False)
model.add(e)
model.add(Flatten())
model.add(Dense(units=1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())
# fit the model

model.fit(padded_docs,y,batch_size=128,epochs=25)
loss, accuracy = model.evaluate(padded_docs,np.array(y), verbose=0)
print('Accuracy: %f' % (accuracy))
'''

'''#glove with logistic regression
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
tk = Tokenizer()
tk.fit_on_texts(X)
index_list = tk.texts_to_sequences(X)
X_train = pad_sequences(index_list, maxlen=4)
y_train = pad_sequences(index_list, maxlen=4)

from sklearn.tree import DecisionTreeClassifier
model_dt=DecisionTreeClassifier()
model_dt.fit(X_train,y_train)
pred_dt=model_dt.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,pred_dt))
'''