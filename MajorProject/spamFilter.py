from nltk.corpus.reader.wordlist import WordListCorpusReader
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import  WordNetLemmatizer
from math import log
import pandas as pd
import numpy as np


def process_mails(mail, lower_case=True, lemma=True, stem=True, stop_words=True, gram=2):
    if lower_case:
        mail = mail.lower()
    words = word_tokenize(mail) #tokenize words
    words = [w for w in words if len(w) > 2] #only consider words with len>2
    w = []
    if gram > 1:
        for i in range(len(words)-gram+1):
            w += [' '.join(words[i:i+gram])] #create list with pairs of words (gram=2)
        return w
    if stop_words:
        sw = stopwords.words('english')
        words = [word for word in words if word not in sw]
    if lemma:
        lemmatizer=WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
    return w+words


class SpamClassifier(object):
    def __init__(self, trainData):
        self.mail = trainData['message']
        self.label = trainData['label']

    def train(self):
        self.calc_TF_and_IDF()
        self.calc_TF_IDF()

    def calc_TF_and_IDF(self):
        noOfMessages = self.mail.shape[0] #no of messages in trainData
        self.spam_mails = self.label.value_counts()[1] #no of spam mails
        self.ham_mails = self.label.value_counts()[0] #no of ham mails
        self.total_mails = self.spam_mails+self.ham_mails #total no of mails
        self.spam_words = 0
        self.ham_words = 0
        self.tf_spam = dict()
        self.tf_ham = dict()
        self.idf_spam = dict()
        self.idf_ham = dict()
        for i in range(noOfMessages):
            mail_processed = process_mails(self.mail[i])
            count = list()
            for word in mail_processed:
                if self.label[i]:
                    self.tf_spam[word] = self.tf_spam.get(word, 0)+1 #tf of spam words
                    self.spam_words += 1 #no of spam words
                else:
                    self.tf_ham[word] = self.tf_ham.get(word, 0)+1 #tf of ham words
                    self.ham_words+1 #no of ham words
                if word not in count:
                    count += [word] #list of unique words in a message
            #add +1 if word exists in doc
            for word in count:
                if self.label[i]:
                    self.idf_spam[word] = self.idf_spam.get(word, 0)+1 
                else:
                    self.idf_ham[word] = self.idf_ham.get(word, 0)+1

    def calc_TF_IDF(self):
        self.prob_spam = dict()
        self.prob_ham = dict()
        self.sum_tf_idf_spam = 0
        self.sum_tf_idf_ham = 0
        for word in self.tf_spam:
            self.prob_spam[word] = (self.tf_spam[word])*log((self.spam_mails+self.ham_mails)/(self.idf_spam[word]+self.idf_ham.get(word, 0)))
            self.sum_tf_idf_spam += self.prob_spam[word]
        for word in self.tf_spam:
            self.prob_spam[word] = (self.prob_spam[word]+1)/(self.sum_tf_idf_spam+len(list(self.prob_spam.keys())))
        for word in self.tf_ham:
            self.prob_ham[word] = (self.tf_ham[word])*log((self.spam_mails+self.ham_mails)/(self.idf_spam.get(word, 0)+self.idf_ham[word]))
            self.sum_tf_idf_ham += self.prob_ham[word]
        for word in self.tf_ham:
            self.prob_ham[word] = (self.prob_ham[word]+1)/(self.sum_tf_idf_ham+len(list(self.prob_ham.keys())))
        self.prob_spam_mail = self.spam_mails/self.total_mails
        self.prob_ham_mail = self.ham_mails/self.total_mails

    def classify(self, processed_mail):
        pSpam = 0
        pHam = 0
        for word in processed_mail:
            if word in self.prob_spam:
                pSpam += log(self.prob_spam[word])
            else:
                pSpam -= log(self.sum_tf_idf_spam + len(list(self.prob_spam.keys())))
            if word in self.prob_ham:
                pHam += log(self.prob_ham[word])
            else:
                pHam -= log(self.sum_tf_idf_ham + len(list(self.prob_ham.keys())))
            pSpam += log(self.prob_spam_mail)
            pHam += log(self.prob_ham_mail)
        return pSpam >= pHam

    def predict(self, testData):
        result = dict()
        for (i, mail) in enumerate(testData):
            processed_mail = process_mails(mail)
            result[i] = int(self.classify(processed_mail))
        return result


def metrics(labels, predictions):
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    for i in range(len(labels)):
        true_pos += int(labels[i] == 1 and predictions[i] == 1)
        true_neg += int(labels[i] == 0 and predictions[i] == 0)
        false_pos += int(labels[i] == 0 and predictions[i] == 1)
        false_neg += int(labels[i] == 1 and predictions[i] == 0)
    precision = true_pos/(true_pos+false_pos)
    recall = true_pos/(true_pos+false_neg)
    Fscore = 2*precision*recall/(precision+recall)
    accuracy = (true_pos+true_neg)/(true_pos+true_neg+false_pos+false_neg)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F-score: ", Fscore)
    print("Accuracy: ", accuracy)


mails = pd.read_csv('spam.csv', encoding='latin-1')
# print(mails.head())
mails.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
mails.rename(columns={'v1': 'labels', 'v2': 'message'}, inplace=True)
mails['label'] = mails['labels'].map({'ham': 0, 'spam': 1})
mails.drop(['labels'], axis=1, inplace=True)
totalMails = 5587
trainIndex, testIndex = list(), list()
for i in range(mails.shape[0]):
    if np.random.uniform(0, 1) < 0.75:
        trainIndex += [i]
    else:
        testIndex += [i]
trainData = mails.loc[trainIndex]
testData = mails.loc[testIndex]
trainData.reset_index(inplace=True)
trainData.drop(['index'], axis=1, inplace=True)
testData.reset_index(inplace=True)
testData.drop(['index'], axis=1, inplace=True)
