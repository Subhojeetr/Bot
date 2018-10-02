
id=[1,2,3,4]
data=['how to resolve error for Server Load','How to resolve error for No file found','job abended or failed  for script','how to execute a script','efgh ']
target=['run top to idntify the process.kill it','specify the correct directory','check the logs','execute a sh/source','abcd']
test=['abcd']

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')
stopword_list.remove('or')


# # Lemmatizing text
def lemmatize_text(text):
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text


# # Removing Stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


def normalize_corpus(corpus,  text_lower_case=True, text_lemmatization=True,stopword_removal=True):
    
    normalized_corpus = []
    
    for doc in corpus:
        
  
        if text_lower_case:
            doc = doc.lower()
        
#        if text_lemmatization:
  #          doc = lemmatize_text(doc)
            
           
        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        
        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=text_lower_case)
            
        normalized_corpus.append(doc)
        
    return normalized_corpus


corpus=normalize_corpus(data)
#print(corpus)
def question(corpus):
    tv = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0, ngram_range=(1,2),sublinear_tf=True)
    tfidf_matrix_train = tv.fit_transform(corpus)
    #tfidf_matrix_train=tfidf_matrix_train.toarray()
    #vocab=tv.get_feature_names()
    #df=pd.DataFrame(tfidf_matrix_train,columns=vocab)
    #print(df)
    return tv, tfidf_matrix_train

def reply(test,tv,tfidf_matrix_train):
    #print(tv)
    test=(test,)
    tfidf_matrix_test = tv.transform(test)
    cosine = cosine_similarity(tfidf_matrix_test, tfidf_matrix_train)
    #print(cosine)


    minimum_score=0.3
    #cosine = np.delete(cosine, 0)  #not required
    #print(cosine)
    maxa = cosine.max()
    #print(maxa)
    response_index=-99999
    if (maxa >= minimum_score):
        #print("hello")
        #new_max = maxa - 0.01
        alist = np.where(cosine > minimum_score)[1]
        #alist = np.where(cosine > new_max)
        # print ("number of responses with 0.01 from max = " + str(list[0].size))
        #response_index = random.choice(alist[0])
        #print(alist)
        #print(response_index)
        for index in alist:
            return target[index]
    

 
 

if __name__=="__main__":
    corpus=normalize_corpus(data)
    tv, tfidf_matrix_train=question(corpus)
    while True:
        ques=input("rudra:" )
        print(reply(ques,tv,tfidf_matrix_train))



    

