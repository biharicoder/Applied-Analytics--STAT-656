
# coding: utf-8

# In[4]:


# classes provided for the course
import sys
sys.path.append('/Users/Pawan/')
from Class_replace_impute_encode import ReplaceImputeEncode

from Class_tree import DecisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix, classification_report 
from pydotplus import graph_from_dot_data
import graphviz

import pandas as pd
import numpy  as np
import string

import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF


# In[5]:


# my_analyzer replaces both the preprocessor and tokenizer
# it also replaces stop word removal and ngram constructions

def my_analyzer(s):
    # Synonym List
    syns = {'veh': 'vehicle', 'car': 'vehicle','air bag': 'airbag',               'seat belt':'seatbelt', "n't":'not', 'to30':'to 30',               'wont':'would not', 'cant':'can not', 'cannot':'can not',               'couldnt':'could not', 'shouldnt':'should not',               'wouldnt':'would not'}
    
    # Preprocess String s
    s = s.lower()
    s = s.replace(',', '. ')
    # Tokenize 
    tokens = word_tokenize(s)
    tokens = [word.replace(',','') for word in tokens ]
    tokens = [word for word in tokens if ('*' not in word) and               ("''" != word) and ("``" != word) and               (word!='description') and (word !='dtype')               and (word != 'object') and (word!="'s")]
    
    # Map synonyms
    for i in range(len(tokens)):
        if tokens[i] in syns:
            tokens[i] = syns[tokens[i]]
            
    # Remove stop words
    punctuation = list(string.punctuation)+['..', '...']
    pronouns = ['i', 'he', 'she', 'it', 'him', 'they', 'we', 'us', 'them']
    stop = stopwords.words('english') + punctuation + pronouns
    filtered_terms = [word for word in tokens if (word not in stop) and                   (len(word)>1) and (not word.replace('.','',1).isnumeric())                   and (not word.replace("'",'',2).isnumeric())]
    
    # Lemmatization & Stemming - Stemming with WordNet POS
    # Since lemmatization requires POS need to set POS
    tagged_words = pos_tag(filtered_terms, lang='eng')
    # Stemming with for terms without WordNet POS
    stemmer = SnowballStemmer("english")
    wn_tags = {'N':wn.NOUN, 'J':wn.ADJ, 'V':wn.VERB, 'R':wn.ADV}
    wnl = WordNetLemmatizer()
    stemmed_tokens = []
    for tagged_token in tagged_words:
        term = tagged_token[0]
        pos  = tagged_token[1]
        pos  = pos[0]
        try:
            pos   = wn_tags[pos]
            stemmed_tokens.append(wnl.lemmatize(term, pos=pos))
        except:
            stemmed_tokens.append(stemmer.stem(term))
    return stemmed_tokens


# In[6]:


# Further Customization of Stopping and Stemming using NLTK
def my_preprocessor(s):
    #Vectorizer sends one string at a time
    s = s.lower()
    s = s.replace(',', '. ')
    print("preprocessor")
    return(s)


# In[7]:


def my_tokenizer(s):
    # Tokenize
    print("Tokenizer")
    tokens = word_tokenize(s)
    tokens = [word.replace(',','') for word in tokens ]
    tokens = [word for word in tokens if word.find('*')!=True and               word != "''" and word !="``" and word!='description'               and word !='dtype']
    return tokens


# In[8]:


def display_topics(lda, terms, n_terms=15):  
    for topic_idx, topic in enumerate(lda):  
        if topic_idx > 10:   
            break  
        message  = "Topic #%d: " %(topic_idx+1)  
        print(message)  
        abs_topic = abs(topic)  
        topic_terms_sorted = [[terms[i], topic[i]]
                     for i in abs_topic.argsort()[:-n_terms - 1:-1]]  
        k = 5  
        n = int(n_terms/k)  
        m = n_terms - k*n  
        for j in range(n):  
            l = k*j  
            message = ''  
            for i in range(k):  
                if topic_terms_sorted[i+l][1]>0:  
                    word = "+"+topic_terms_sorted[i+l][0]  
                else:  
                    word = "-"+topic_terms_sorted[i+l][0]  
                message += '{:<15s}'.format(word)  
            print(message)  
        if m> 0:  
            l = k*n  
            message = ''  
            for i in range(m):  
                if topic_terms_sorted[i+l][1]>0:  
                    word = "+"+topic_terms_sorted[i+l][0]  
                else:  
                    word = "-"+topic_terms_sorted[i+l][0]  
                message += '{:<15s}'.format(word)  
            print(message)  
        print("")  
    return  


# In[9]:


# Increase Pandas column width to let pandas read large text columns
pd.set_option('max_colwidth', 32000)

# Read GMC Ignition Recall Comments from NTHSA Data
df = pd.read_excel("HondaComplaints.xlsx")
sw = pd.read_excel("afinn_sentiment_words.xlsx") 
# Setup program constants and reviews
n_reviews  = len(df['description'])
s_words    = 'english'
ngram = (1,2)
reviews = df['description']

# Constants
m_features      = None # default is None
n_topics        =  7   # number of topics
max_iter        = 10   # maximum number of iterations
max_df          = 0.95  # max proportion of docs/reviews allowed for a term
min_df          = 2
learning_offset = 10.      # default is 10
learning_method = 'online' # alternative is 'batch' for large files
tf_matrix='tfidf'


# In[10]:


# Create Word Frequency by Review Matrix using Custom Analyzer
cv = CountVectorizer(max_df=max_df, min_df=min_df, max_features=m_features,                     analyzer=my_analyzer, ngram_range=ngram)
tf = cv.fit_transform(reviews)


# In[11]:


terms = cv.get_feature_names()
print('{:.<22s}{:>6d}'.format("Number of Reviews", len(reviews)))
print('{:.<22s}{:>6d}'.format("Number of Terms",   len(terms)))

term_sums = tf.sum(axis=0)
term_counts = []
for i in range(len(terms)):
    term_counts.append([terms[i], term_sums[0,i]])
def sortSecond(e):
    return e[1]
term_counts.sort(key=sortSecond, reverse=True)
print("\nTerms with Highest Frequency:")
for i in range(10):
    print('{:<15s}{:>5d}'.format(term_counts[i][0], term_counts[i][1]))


# In[12]:


# Construct the TF/IDF matrix from the Term Frequency matrix 
print("\nConstructing Term/Frequency Matrix using TF-IDF")
# Default for norm is 'l2', use norm=None to supress
tfidf_vect = TfidfTransformer(norm=None, use_idf=True) #set norm=None
# tf matrix is (n_reviews)x(m_terms)
tf = tfidf_vect.fit_transform(tf) 

# Display the terms with the largest TFIDF value
term_idf_sums = tf.sum(axis=0)
term_idf_scores = []
for i in range(len(terms)):
    term_idf_scores.append([terms[i], term_idf_sums[0,i]])
print("The Term/Frequency matrix has", tf.shape[0], " rows, and",      tf.shape[1], " columns.")
print("The Term list has", len(terms), " terms.")
term_idf_scores.sort(key=sortSecond, reverse=True)
print("\nTerms with Highest TF-IDF Scores:")
for i in range(10):
    j = i
    print('{:<15s}{:>8.2f}'.format(term_idf_scores[j][0],           term_idf_scores[j][1]))


# In[13]:


# In sklearn, SVD is synonymous with LSA (Latent Semantic Analysis)
uv = TruncatedSVD(n_components=n_topics, algorithm='arpack',                                    tol=0, random_state=12345)

U = uv.fit_transform(tf)
# Display the topic selections
print("\n********** GENERATED TOPICS **********")
display_topics(uv.components_, terms, n_terms=15)

# Store topic selection for each doc in topics[]
topics = [0] * n_reviews
for i in range(n_reviews):
    max       = abs(U[i][0])
    topics[i] = 0
    for j in range(n_topics):
        x = abs(U[i][j])
        if x > max:
            max = x
            topics[i] = j


# In[14]:


# Review Scores  
# Normalize LDA Weights to probabilities  
uv_norm = uv.components_ / uv.components_.sum(axis=1)[:, np.newaxis]
  
#***** SCORE REVIEWS *****  
rev_scores = [[0]*(n_topics+1)] * n_reviews  
# Last topic count is number of reviews without any topic words  
topic_counts = [0] * (n_topics+1)  


# In[16]:


for r in range(n_reviews):  
    idx = n_topics  
    max_score = 0  
    # Calculate Review Score  
    j0 = tf[r].nonzero()  
    nwords = len(j0[1])  
    rev_score = [0]*(n_topics+1)  
    # get scores for rth doc, ith topic  
    for i in range(n_topics):  
        score = 0  
        for j in range(nwords):  
            j1 = j0[1][j]  
            if tf[r,j1] != 0:  
                score += uv_norm[i][j1] * tf[r,j1]  
        rev_score [i+1] = score  
        if score>max_score:  
            max_score = score  
            idx = i  
    # Save review's highest scores  
    rev_score[0] = idx  
    rev_scores [r] = rev_score  
    topic_counts[idx] += 1  
      
print('{:<6s}{:>8s}{:>8s}'.format("TOPIC", "REVIEWS", "PERCENT"))  
for i in range(n_topics):  
    print('{:>3d}{:>10d}{:>8.1%}'.format((i+1), topic_counts[i], 
                                         topic_counts[i]/n_reviews))  
  
      
  
  
sentiment_dic = {}  
for i in range(len(sw)):  
    sentiment_dic[sw.iloc[i][0]] = sw.iloc[i][1]  
min_sentiment = +5  
max_sentiment = -5  
avg_sentiment, min, max = 0,0,0  
min_list, max_list = [],[]  
sentiment_score = [0]*n_reviews  
for i in range(n_reviews):  
    # n_sw = number of sentiment words in a review  
    n_sw = 0  
    # Pick non zero terms with non-zero score for each review  
    term_list = tf[i].nonzero()[1]  
      
    if len(term_list)>0:  
        for t in np.nditer(term_list):  
            score = sentiment_dic.get(terms[t])  
            if score != None:  
                sentiment_score[i] += score * tf[i,t]  
                n_sw += tf[i,t]  
    if n_sw>0:  
        sentiment_score[i] = sentiment_score[i]/n_sw  
    if sentiment_score[i]==max_sentiment and n_sw>3:  
        max_list.append(i)  
    if sentiment_score[i]>max_sentiment and n_sw>3:  
        max_sentiment=sentiment_score[i]  
        max = i  
        max_list = [i]  
    if sentiment_score[i]==min_sentiment and n_sw>3:  
        min_list.append(i)  
    if sentiment_score[i]<min_sentiment and n_sw>3:  
        min_sentiment=sentiment_score[i]  
        min = i  
        min_list = [i]  
    avg_sentiment += sentiment_score[i]  
avg_sentiment = avg_sentiment/n_reviews  
print("\nCorpus Average Sentiment: ", avg_sentiment)  
print("\nMost Negative Reviews with 4 or more Sentiment Words:")  
for i in range(len(min_list)):  
    print("{:<s}{:<d}{:<s}{:<5.2f}".format(" Review ", min_list[i], " Sentiment is ", min_sentiment))  
print("\nMost Positive Reviews with 4 or more Sentiment Words:")  
for i in range(len(max_list)):  
    print("{:<s}{:<d}{:<s}{:<5.2f}".format(" Review ", max_list[i], " Sentiment is ", max_sentiment))  
      


# In[17]:


Topic_1 = []  
Topic_2 = []  
Topic_3 = []  
Topic_4 = []  
Topic_5 = []  
Topic_6 = []  
Topic_7 = []   
for i in range(len(rev_scores)):  
    if rev_scores[i][0] == 0:  
        Topic_1.append(i)  
    elif rev_scores[i][0] == 1:  
        Topic_2.append(i)  
    elif rev_scores[i][0] == 2:  
        Topic_3.append(i)  
    elif rev_scores[i][0] == 3:  
        Topic_4.append(i)  
    elif rev_scores[i][0] == 4:  
        Topic_5.append(i)  
    elif rev_scores[i][0] == 5:  
        Topic_6.append(i)  
    elif rev_scores[i][0] == 6:  
        Topic_7.append(i) 


# In[18]:


Topics = [Topic_1,Topic_2,Topic_3,Topic_4,Topic_5,Topic_6,Topic_7]  
k=1  
for j in Topics:  
  
    corpus_sentiment = {}  
    n_sw = 0  
    for i in range(len(j)):  
        # Iterate over the terms with nonzero scores  
        term_list = tf[j[i]].nonzero()[1]  
        if len(term_list)>0:  
            for t in np.nditer(term_list):  
                score = sentiment_dic.get(terms[t])  
                if score != None:  
                    n_sw += tf[j[i],t]  
                    current_count = corpus_sentiment.get(terms[t])  
                    if current_count == None:  
                        corpus_sentiment[terms[t]] = tf[j[i],t]  
                    else:  
                        corpus_sentiment[terms[t]] += tf[j[i],t]  
    print("The Topic %i contains a total of "%k, len(corpus_sentiment), " unique sentiment words")  
    print("The total number of sentiment words in the Topic %i is"%k, n_sw,"\n")
    k+=1


# In[20]:


rev_scores = pd.DataFrame(rev_scores)
rev_scores.columns = ['Topic','T1','T2','T3','T4','T5','T6','T7']

sentiment_score = pd.DataFrame(sentiment_score)
sentiment_score.columns = ['Sentiment_Score']
df1 = pd.concat([df, sentiment_score, rev_scores], axis=1)


# In[21]:


# Data Map for these Reviews
attribute_map = {
    'description':[3,(''),[0,0]],
    'Make':[2,('HONDA', 'ACURA'),[0,0]],
    'Model':[2,('TL','ODYSSEY','CR-V','CL','CIVIC', 'ACCORD'),[0,0]],
    'Year':[2,(2001,2002,2003),[0,0]],
    'abs':[2,('Y','N'),[0,0]],
    'cruise':[2,('Y','N'),[0,0]],
    #'crash':[2,('Y','N'),[0,0]],
    'mph':[0,(0,80),[0,0]],
    'mileage':[0,(0,200000),[0,0]],
    'Sentiment_Score':[0,(-4,4),[0,0]],
    'Topic':[2,(0,1,2,3,4,5,6),[0,0]],
    'T1':[0,(-1e+8,1e+8),[0,0]],
    'T2':[0,(-1e+8,1e+8),[0,0]],
    'T3':[0,(-1e+8,1e+8),[0,0]],
    'T4':[0,(-1e+8,1e+8),[0,0]],
    'T5':[0,(-1e+8,1e+8),[0,0]],
    'T6':[0,(-1e+8,1e+8),[0,0]],
    'T7':[0,(-1e+8,1e+8),[0,0]]}


# In[22]:


target   = 'crash'
# Drop data with missing values for target (price)
drops= []
for i in range(df1.shape[0]):
    if pd.isnull(df1['crash'][i]):
        drops.append(i)
df1 = df1.drop(drops)


# In[23]:


encoding = 'one-hot' 
scale    = None  # Interval scaling:  Use 'std', 'robust' or None
# drop=False - do not drop last category - used for Decision Trees
rie = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding=encoding,                           interval_scale = scale, drop=False, display=True)


# In[24]:


df1.drop('crash', axis=1, inplace = True)


# In[25]:


encoded_df = rie.fit_transform(df1)


# In[26]:


#varlist = [target, 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9','points']
X = encoded_df.drop(['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7'], axis=1)
y = [1 if df.crash[i] == "Y" else 0 for i in df.index]
np_y = np.ravel(y) #convert dataframe column to flat array
col  = rie.col


# In[27]:


X_train, X_validate, y_train, y_validate = train_test_split(X,y,test_size = 0.3, random_state=7)


# In[29]:


# Cross Validation
depth_list = [5, 6, 7, 8, 10, 12, 15, 20, 25]
score_list = ['accuracy', 'recall', 'precision', 'f1']
for d in depth_list:
    print("\nMaximum Tree Depth: ", d)
    dtc = DecisionTreeClassifier(max_depth=d, min_samples_leaf=5,                                  min_samples_split=5)
    dtc = dtc.fit(X,y)
    scores = cross_validate(dtc, X, y, scoring=score_list,                             return_train_score=False, cv=10)
    
    print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
    for s in score_list:
        var = "test_"+s
        mean = scores[var].mean()
        std  = scores[var].std()
        print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))


# In[30]:


dtc = DecisionTreeClassifier(criterion='gini', max_depth = 8,min_samples_split=5, min_samples_leaf=5)
dtc = dtc.fit(X_train,y_train)

features = X.columns.values.tolist()
classes = [0,1]
#FI = list();
#for i in range(len(X.columns)):
#FI.append(dtc.feature_importances_[i])
#len(FI)
FI_df = pd.DataFrame({'Feature':features,'Importance':dtc.feature_importances_})
predictions = dtc.predict(X_validate)

print('\n****************Decision Tree with Depth = 8 branches****************\n')
DecisionTree.display_binary_split_metrics(dtc, X_train, y_train, X_validate, y_validate)

