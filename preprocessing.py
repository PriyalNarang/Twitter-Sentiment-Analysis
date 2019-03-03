# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 19:30:55 2018

@author: Priyal Narang
"""
import re
import nltk
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
    
def removestopwords(tweet):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tweet)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    
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


for line in f:
    cols = line.split("\t")
    
    if(i>0):
        for i in range(2):
            f1.write(cols[i] + "\t")
            
        text = cols[2]
        text=replace_slang(text)
        text=expandContractions(text)
        text = preprocess_tweet(text)
        text = ' '.join(unique_list(text.split()))
        text = removestopwords(text)
        temp[0]=text
        tweet=' '.join(text)
        f1.write(tweet+"\n")
       
        
       
        
    i=i+1

