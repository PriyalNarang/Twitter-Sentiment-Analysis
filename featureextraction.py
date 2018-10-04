from sklearn import tree
from scipy.sparse import hstack
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier


f = open("result.txt", "r")
corpus = []
data=[]
for line in f:
    cols = line.split("\t")
    corpus.append(cols[2])
    data.append(cols[0])
f.close()


# -----------------------------------LOGISTIC RERGRESSION-----------------------------------

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(corpus)
#test_bow = bow[2112:, :]
#train_bow = bow[:2112, :]
#data=data[:2112]
X_train, X_test, y_train, y_test = train_test_split(bow,data, train_size=0.80, random_state=1234)
log_model = LogisticRegression()
log_model = log_model.fit(X_train, y_train)
y_pred = log_model.predict(X_test)
print("----------------------------------------------------------------------------------")
print("________LOGISTIC RERGRESSION________")
print("BagOfWords ----->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus)
X_train, X_test, y_train, y_test = train_test_split(tfidf, data, test_size=0.20, train_size=0.80, random_state=1234)
log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)
print("TF-IDF --------->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")

#--------------------------------BOW+TFIDF--------------------------------------------------
hybrid = hstack([tfidf, bow])
hy = hybrid.tocsr()
X_train, X_test, y_train, y_test = train_test_split(hy, data, test_size=0.20, train_size=0.80, random_state=1234)
log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)
print("Bow + Tf-idf --->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


# ---------------------------------PASSIVE AGGRESSIVE CLASSIFIER--------------------------------

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(corpus)
X_train, X_test, y_train, y_test = train_test_split(bow, data, test_size=0.20, train_size=0.80, random_state=1234)
pac = PassiveAggressiveClassifier(max_iter=1000, random_state=0)
pac = pac.fit(X=X_train, y=y_train)
y_pred = pac.predict(X_test)
print("----------------------------------------------------------------------------------")
print("_____PASSIVE AGGRESSIVE CLASSIFIER_____")
print("BagOfWords ----->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus)
X_train, X_test, y_train, y_test = train_test_split(tfidf, data, test_size=0.20, train_size=0.80, random_state=1234)
pac = PassiveAggressiveClassifier(max_iter=1000, random_state=0)
pac = pac.fit(X=X_train, y=y_train)
y_pred = pac.predict(X_test)
print("TF-IDF --------->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")

bow_vectorizer = CountVectorizer()
# bow = bow_vectorizer.fit_transform(corpus)
tfidf_vectorizer = TfidfVectorizer()
# tfidf = tfidf_vectorizer.fit_transform(corpus)
hy_vectorizer = FeatureUnion([('bow', bow_vectorizer), ('tfidf', tfidf_vectorizer)])
hy = hy_vectorizer.fit_transform(corpus)
# hybrid = hstack([tfidf, bow])
# hy = hybrid.tocsr()
X_train, X_test, y_train, y_test = train_test_split(hy, data, train_size=0.80, test_size=0.20, random_state=1234)
pac = PassiveAggressiveClassifier(max_iter=1000, random_state=0)
pac = pac.fit(X=X_train, y=y_train)
y_pred = pac.predict(X_test)
print("Bow + Tf-idf --->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")

# -------------------------------------MULTINOMIAL NB-------------------------------------

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(corpus)
X_train, X_test, y_train, y_test = train_test_split(bow, data,test_size=0.20, train_size=0.80, random_state=1234)
nb = MultinomialNB()
nb = nb.fit(X=X_train, y=y_train)
y_pred = nb.predict(X_test)
print("----------------------------------------------------------------------------------")
print("__________MULTINOMIAL NB__________")
print("BagOfWords ----->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus)
X_train, X_test, y_train, y_test = train_test_split(tfidf, data, test_size=0.20, train_size=0.80, random_state=1234)
nb = MultinomialNB()
nb = nb.fit(X=X_train, y=y_train)
y_pred = nb.predict(X_test)
print("TF-IDF --------->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


bow_vectorizer = CountVectorizer()
# bow = bow_vectorizer.fit_transform(corpus)
tfidf_vectorizer = TfidfVectorizer()
# tfidf = tfidf_vectorizer.fit_transform(corpus)
hy_vectorizer = FeatureUnion([('bow', bow_vectorizer), ('tfidf', tfidf_vectorizer)])
hy = hy_vectorizer.fit_transform(corpus)
# hybrid = hstack([tfidf, bow])
# hy = hybrid.tocsr()
X_train, X_test, y_train, y_test = train_test_split(hy, data, train_size=0.80, test_size=0.20, random_state=1234)
nb = MultinomialNB()
nb = nb.fit(X=X_train, y=y_train)
y_pred = nb.predict(X_test)
print("Bow + Tf-idf --->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


# -------------------------------------PERCEPTRON-------------------------------------

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(corpus)
X_train, X_test, y_train, y_test = train_test_split(bow, data, test_size=0.20, train_size=0.80, random_state=1234)
per = Perceptron(tol=1e-3, random_state=0)
per = per.fit(X=X_train, y=y_train)
y_pred = per.predict(X_test)
print("----------------------------------------------------------------------------------")
print("__________PERCEPTRON__________")
print("BagOfWords ----->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus)
X_train, X_test, y_train, y_test = train_test_split(tfidf, data, test_size=0.20, train_size=0.80, random_state=1234)
per = Perceptron(tol=1e-3, random_state=0)
per = per.fit(X=X_train, y=y_train)
y_pred = per.predict(X_test)
print("TF-IDF --------->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


bow_vectorizer = CountVectorizer()
# bow = bow_vectorizer.fit_transform(corpus)
tfidf_vectorizer = TfidfVectorizer()
# tfidf = tfidf_vectorizer.fit_transform(corpus)
hy_vectorizer = FeatureUnion([('bow', bow_vectorizer), ('tfidf', tfidf_vectorizer)])
hy = hy_vectorizer.fit_transform(corpus)
# hybrid = hstack([tfidf, bow])
# hy = hybrid.tocsr()
X_train, X_test, y_train, y_test = train_test_split(hy, data, train_size=0.80, test_size=0.20, random_state=1234)
per = Perceptron(tol=1e-3, random_state=0)
per = per.fit(X=X_train, y=y_train)
y_pred = per.predict(X_test)
print("Bow + Tf-idf --->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


# -----------------------------------RIDGE CLASSIFIER------------------------------------

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(corpus)
X_train, X_test, y_train, y_test = train_test_split(bow, data, test_size=0.20, train_size=0.80, random_state=1234)
rc = RidgeClassifier()
rc = rc.fit(X=X_train, y=y_train)
y_pred = rc.predict(X_test)
print("----------------------------------------------------------------------------------")
print("__________RIDGE CLASSIFIER__________")
print("BagOfWords ----->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus)
X_train, X_test, y_train, y_test = train_test_split(tfidf, data, test_size=0.20, train_size=0.80, random_state=1234)
rc = RidgeClassifier()
rc = rc.fit(X=X_train, y=y_train)
y_pred = rc.predict(X_test)
print("TF-IDF --------->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


bow_vectorizer = CountVectorizer()
# bow = bow_vectorizer.fit_transform(corpus)
tfidf_vectorizer = TfidfVectorizer()
# tfidf = tfidf_vectorizer.fit_transform(corpus)
hy_vectorizer = FeatureUnion([('bow', bow_vectorizer), ('tfidf', tfidf_vectorizer)])
hy = hy_vectorizer.fit_transform(corpus)
# hybrid = hstack([tfidf, bow])
# hy = hybrid.tocsr()
X_train, X_test, y_train, y_test = train_test_split(hy, data, train_size=0.80, test_size=0.20, random_state=1234)
rc = RidgeClassifier()
rc = rc.fit(X=X_train, y=y_train)
y_pred = rc.predict(X_test)
print("Bow + Tf-idf --->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


# -------------------------------------LINEAR SVC-------------------------------------

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(corpus)
X_train, X_test, y_train, y_test = train_test_split(bow, data, test_size=0.20, train_size=0.80, random_state=1234)
LSVC = LinearSVC()
LSVC = LSVC.fit(X_train, y_train)
y_pred = LSVC.predict(X_test)
print("----------------------------------------------------------------------------------")
print("__________LINEAR SVC__________")
print("BagOfWords ----->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus)
X_train, X_test, y_train, y_test = train_test_split(tfidf, data, test_size=0.20, train_size=0.80, random_state=1234)
LSVC = LinearSVC()
LSVC = LSVC.fit(X_train, y_train)
y_pred = LSVC.predict(X_test)
print("TF-IDF --------->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


bow_vectorizer = CountVectorizer()
# bow = bow_vectorizer.fit_transform(corpus)
tfidf_vectorizer = TfidfVectorizer()
# tfidf = tfidf_vectorizer.fit_transform(corpus)
hy_vectorizer = FeatureUnion([('bow', bow_vectorizer), ('tfidf', tfidf_vectorizer)])
hy = hy_vectorizer.fit_transform(corpus)
# hybrid = hstack([tfidf, bow])
# hy = hybrid.tocsr()
X_train, X_test, y_train, y_test = train_test_split(hy, data, train_size=0.80, test_size=0.20, random_state=1234)
LSVC = LinearSVC()
LSVC = LSVC.fit(X_train, y_train)
y_pred = LSVC.predict(X_test)
print("Bow + Tf-idf --->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


# -----------------------------------DECISION TREE CLASSIFIER-----------------------------------

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(corpus)
X_train, X_test, y_train, y_test = train_test_split(bow, data, test_size=0.20, train_size=0.80, random_state=1234)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X=X_train, y=y_train)
y_pred = clf.predict(X_test)
print("----------------------------------------------------------------------------------")
print("_____DECISION TREE CLASSIFIER_____")
print("BagOfWords ----->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus)
X_train, X_test, y_train, y_test = train_test_split(tfidf, data, test_size=0.20, train_size=0.80, random_state=1234)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X=X_train, y=y_train)
y_pred = clf.predict(X_test)
print("TF-IDF --------->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


bow_vectorizer = CountVectorizer()
# bow = bow_vectorizer.fit_transform(corpus)
tfidf_vectorizer = TfidfVectorizer()
# tfidf = tfidf_vectorizer.fit_transform(corpus)
hy_vectorizer = FeatureUnion([('bow', bow_vectorizer), ('tfidf', tfidf_vectorizer)])
hy = hy_vectorizer.fit_transform(corpus)
# hybrid = hstack([tfidf, bow])
# hy = hybrid.tocsr()

X_train, X_test, y_train, y_test = train_test_split(hy, data, train_size=0.80, test_size=0.20, random_state=1234)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X=X_train, y=y_train)
y_pred = clf.predict(X_test)
print("Bow + Tf-idf --->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


# print(y_pred)
# test_pred = log_model.predict(test_bow)  ### PREDICTING TEST DATA SET ###
# print(test_pred)

# x_train, x_test, y_train, y_test = train_test_split(data2, data_labels, train_size=0.80)
# tweet_w2v = Word2Vec(size=200, min_count=10)
# tweet_w2v.build_vocab([x.split() for x in tqdm(x_train)])
# tweet_w2v.train([x.split() for x in tqdm(x_train)])
