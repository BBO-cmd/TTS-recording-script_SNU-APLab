# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 07:52:42 2021

@author: jadeb
"""

### 2.3.낭독체 문장수집
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import requests
from bs4 import BeautifulSoup
import pandas as pd

req = requests.get('url') # 기사 url
raw = req.text
html = BeautifulSoup(raw, 'html.parser')

corpus = [element.text for element in html.find_all("p", "gnt_ar_b_p")] # 해당 기사의 텍스트를 담고있는 부분 지정
corpus = ' '.join(corpus)
corpus.replace("\xa0", "")

sentences = nltk.sent_tokenize(corpus)

df = pd.DataFrame(sentences, columns =['Sentence']) 
df['Count'] = df['Sentence'].str.count(' ') + 1
df = df.sort_values(["Count"])
df



### 2.4.대화체 문장수집

## <방법1>
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
 
file = open('Corpus_all.txt', 'r', encoding = 'UTF8') # txt파일 불러오기(영화 전체 코퍼스)
text = file.read()
 
sent_token = sent_tokenize(text) # 문장 토큰화
df = pd.DataFrame(sent_token, columns =['Sentence'])
df["lower"] = df['Sentence'].str.lower() # 소문자화
 
dups_sent = df.pivot_table(index=['lower'], aggfunc='size')  # 문장 빈도수(중복) 체크
dups_sent = dups_sent.to_frame().reset_index()
dups_sent.columns = ['lower', 'Count']
dups_sent = dups_sent.sort_values(['Count'], ascending=False)

## <방법2>
from nltk import collocations
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.metrics import TrigramAssocMeasures
from nltk.corpus import stopwords
 
file = open('Corpus_domain.txt', 'r', encoding = 'UTF8') # txt파일 불러오기(도메인별 영화 코퍼스)
text = file.read()
 
text = text.lower() # 소문자화
word_token = nltk.word_tokenize(text) # 단어 토큰화
stop_words = stopwords.words("english") # 기본 불용어
stop_words.extend(["'re", "You", "n't", "'ve", "'ll"]) # 불용어 추가
stop_words = set(stop_words)
filter_stops = lambda w: len(w) <3 or w in stop_words
 
# bigram 분석
bigram = BigramCollocationFinder.from_words(word_token)
bigram.apply_word_filter(filter_stops)
bigram.apply_freq_filter(3) #세 번 이상 등장하는 bigram
bigram.nbest(BigramAssocMeasures.likelihood_ratio, 30) # 우도비 기준, 상위 30개
 
# trigram 분석
trigram = TrigramCollocationFinder.from_words(word_token)
trigram.apply_word_filter(filter_stops)
trigram.apply_freq_filter(3) #세 번 이상 등장하는 trigram
trigram.nbest(TrigramAssocMeasures.likelihood_ratio, 30) # 우도비 기준, 상위 30개
 
# 특정 단어나 구가 포함된 문장 추출
sample = df["lower"].str.contains("long time ago") # 소문자로 입력
sample = df[sample]

##<방법 3>
# 데이터 준비(코퍼스, 단어 토큰 리스트, 불용어 등은 <방법 2>에서 설정한 값 그대로)
filtered_word = [w for w in word_token if not w in stop_words] # 불용어 제거
df2 = pd.DataFrame(filtered_word, columns =['Word'])
df2["lower"] = df2['Word'].str.lower()
 
# 단어 빈도수(중복) 체크
dups_word = df2.pivot_table(index=['lower'], aggfunc='size') # 중복 확인
dups_word = dups_word.to_frame().reset_index()
dups_word.columns = ['lower', 'Count']
dups_word = dups_word.sort_values(['Count'], ascending=False)



### 3.1.전체 결과

# 총 단어 수 계산
with open('(문서명).txt', 'r', encoding = 'UTF-8') as data:
   lines = data.readlines()
  
tokenized_sents = [word_tokenize(i) for i in lines]

flatten = []
for item in tokenized_sents:
        flatten.extend(item)

lower = []
for i in flatten:
    lower.append(i.lower())

list_all = []
for i in lower:
    text = re.sub('[^a-zA-Z0-9-]',' ', i).strip()
    if(text != ''):
        list_all.append(text)
print(list_all)
print ("\n")
print ("Number of Words: " , len(list_all))

# 고유 단어 수 계산
list_unique = []
for n in list_all:
    if n not in list_unique:
        list_unique.append(n)
print(list_unique)
print ("\n")
print ("Number of Words: " , len(list_unique))



### 3.5.커버리지

## 단어 커버리지
news_articles = pd.read_csv('news_articles_new.csv', encoding = 'utf-8').astype(str)
news_articles.head()
 
# data selection
corpus = news_articles[["TEXT", "TITLE"]].copy()
corpus.to_csv('corpus.csv', index=False, encoding = 'utf-8-sig')
 
corpus["lower"] = corpus["TEXT"].fillna("")
 
# 전체 단어에 대한 소문자 변환
corpus["lower"] = corpus["lower"].apply(lambda x: x.lower())
corpus["lower"] = corpus["lower"].str.replace("[^a-zA-Z0-9'’]"," ")
corpus["lower"] = corpus["lower"].dropna()
 
# word tokenize
corpus['lower_nltk'] = corpus['lower'].apply(lambda x: word_tokenize(x))
corpus.to_csv('corpus_nlp.csv', index=False, encoding = 'utf-8-sig')
 

## 발성목록 단어 리스트 만들기 ##
# 발성목록 파일 불러오기
nangdok = open('nangdok_20205.txt', 'r', encoding = 'utf-8')
nangdok = nangdok.read()
 
# tokenize
nangdok.lower()
nangdok_list = nltk.word_tokenize(nangdok)
nangdok_list
 
# 발성목록 단어 리스트
database_list = []
for word in nangdok_list:
  if word not in database_list:
    database_list.append(word)
 
nltk.download('averaged_perceptron_tagger')
d = []
for sents in corpus['lower_nltk']: 
#     print(pos_tag(sents))
    for word, pos in pos_tag(sents): 
        param = [word, pos]
        headers = ['word', 'pos'] 
        dict_ = {k:v for k, v in zip(headers, param)}
        d.append(dict_)
 
corpus_token_df = pd.DataFrame(d)[headers] 
corpus_token_df



## biphone, triphone 커버리지

import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from g2p_en import G2p
import re
p = re.compile("[^0-9]")
import os
os.getcwd()
os.chdir("C:\\DATA") # 작업 디렉토리 변경
file = open('nangdok_20205.txt', 'r', encoding = 'UTF8') #txt파일 불러오기
text = file.read()


##### Phoneset 제작
# biphoneset
bi_vow = ["AA", "AE", "AH", "AO", "AW", "AX", "AXR", "AY", "EH", "ER", "EY", "IH", "IX", "IY", "OW", "OY", "UH", "UW"]
bi_con = ["B", "CH", "D", "DH", "DX", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH"]

preset_left = bi_vow + bi_con + ["SIL1", "PAU"]
preset_right = bi_vow + bi_con + ["SIL2", "PAU"]
biphoneset = []
for i in range(len(preset_left)):
    for j in range(len(preset_right)):
        biphoneset.append(str(preset_left[i] + ", " + preset_right[j]))

biphoneset.remove("PAU, PAU")
biphoneset.remove("PAU, SIL2")
biphoneset.remove("SIL1, SIL2")
biphoneset.remove("SIL1, PAU")


# triphoneset
preset_1 = bi_vow + bi_con + ["SIL1", "PAU"]
preset_2 = bi_vow + bi_con + ["PAU"]
preset_10 = []
for i in range(len(preset_1)):
    for j in range(len(preset_2)):
        preset_10.append(str(preset_1[i] + ", " + preset_2[j]))

preset_10.remove("PAU, PAU")
preset_10.remove("SIL1, PAU")

preset_3 = bi_vow + bi_con + ["SIL2", "PAU"]
preset_11 = []
for i in range(len(preset_10)):
    for j in range(len(preset_3)):
        preset_11.append(str(preset_10[i] + ", " + preset_3[j]))
       
triphoneset = []
for i in range(len(preset_11)):
    if "PAU, SIL2" in preset_11[i]:
        None
    elif "PAU, PAU" in preset_11[i]:
        None
    else:
        triphoneset.append(preset_11[i])


##### biphone 생성
sent_token = sent_tokenize(text)
g2p = G2p()
sent_g2p = []
for i in sent_token:
    sent_g2p.append(g2p(i)) #sent_g2p는 Two-dimensional lists
    
all_bi = []
for i in sent_g2p:
    i.insert(0, 'SILone') # SIL1 추가
    i.append("SILtwo") # SIL2 추가
    for j in range(len(i)):
        if i[j] == ",":
            i[j] = "PAU"
        elif i[j] == "-":
            i[j] = "PAU"
        elif i[j] == "?":
            i[j] = "NONE"
        elif i[j] == "!":
            i[j] = "NONE"
        elif i[j] == "\'":
            i[j] = "NONE"
        elif i[j] == "..":
            i[j] = "NONE"
        elif i[j] == ".":
            i[j] = "NONE"            
        elif i[j] == " ":
            i[j] = "NONE"
    while "NONE" in i:
        i.remove("NONE")
    for m in range(len(i)-1):
        i[m] = "".join(p.findall(i[m]))
        i[m+1] = "".join(p.findall(i[m+1]))
        all_bi.append([i[m], i[m+1]])
for i in range(len(all_bi)):
    all_bi[i] = ", ".join(all_bi[i])
    all_bi[i] = all_bi[i].replace("one", "1")
    all_bi[i] = all_bi[i].replace("two", "2")
 

##### biphone 분석
sent_bi = all_bi # sent_bi는 코퍼스에 있는 biphone들
biphoneset_all = sent_bi + biphoneset # 분석용 셋

biphone_df = pd.DataFrame(biphoneset_all, columns =['corpus_biphone'])
biphone_df = biphone_df.pivot_table(index=['corpus_biphone'], aggfunc='size')
biphone_df = biphone_df.to_frame().reset_index()
biphone_df.columns = ['corpus_biphone', 'n']
biphone_df = biphone_df.sort_values(['n'], ascending=False) #빈도수 높은 순으로 문장 정렬
biphone_df["count"] = biphone_df["n"] -1 
del biphone_df["n"]
# count열이 0이면 코퍼스에 등장하지 않음


##### triphone 생성 
sent_token = sent_tokenize(text)
g2p = G2p()
sent_g2p = []
for i in sent_token:
    sent_g2p.append(g2p(i)) #sent_g2p는 Two-dimensional lists

all_tri = []
for i in sent_g2p:
    i.insert(0, 'SILone') # SIL1 추가
    i.append("SILtwo") # SIL2 추가
    for j in range(len(i)):
        if i[j] == ",":
            i[j] = "PAU"
        elif i[j] == "-":
            i[j] = "PAU"
        elif i[j] == "?":
            i[j] = "NONE"
        elif i[j] == "!":
            i[j] = "NONE"
        elif i[j] == "\'":
            i[j] = "NONE"
        elif i[j] == "..":
            i[j] = "NONE"
        elif i[j] == ".":
            i[j] = "NONE"            
        elif i[j] == " ":
            i[j] = "NONE"
    while "NONE" in i:
        i.remove("NONE")
    for m in range(len(i)-2):
        i[m] = "".join(p.findall(i[m]))
        i[m+1] = "".join(p.findall(i[m+1]))
        i[m+2] = "".join(p.findall(i[m+2]))         
        all_tri.append([i[m], i[m+1], i[m+2]])
for i in range(len(all_tri)):
    all_tri[i] = ", ".join(all_tri[i])
    all_tri[i] = all_tri[i].replace("one", "1")
    all_tri[i] = all_tri[i].replace("two", "2")


##### triphone 분석
sent_tri = all_tri # sent_tri는 코퍼스에 있는 triphone들
triphoneset_all = sent_tri + triphoneset # 분석용 셋

triphone_df = pd.DataFrame(triphoneset_all, columns =['corpus_biphone'])
triphone_df = triphone_df.pivot_table(index=['corpus_biphone'], aggfunc='size')
triphone_df = triphone_df.to_frame().reset_index()
triphone_df.columns = ['corpus_triphone', 'n']
triphone_df = triphone_df.sort_values(['n'], ascending=False) #빈도수 높은 순으로 문장 정렬
triphone_df["count"] = triphone_df["n"] -1 
del triphone_df["n"]
# count열이 0이면 코퍼스에 등장하지 않음


##### biphone, triphone 빈도 엑셀 파일
biphone_df.to_excel('biphone_count_none.xlsx')
triphone_df.to_excel('triphone_count_none.xlsx')


##### 문장 - g2p 엑셀 파일
df2 = pd.DataFrame({"Sentence" : sent_token, "G2P" : sent_g2p})
df2.to_excel('sent_g2p_biphone.xlsx')




### 3.6. TN모델 평가
import operator
import pandas as pd
import numpy as np
import os
import pickle
from num2words import num2words
import xgboost as xgb
import numpy as np
import re
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
os.getcwd()
os.chdir("C:\\DATA") # 작업 디렉토리 변경

df = pd.read_csv('en_train.csv') # Kaggle train set
test = pd.read_excel('en_test_35.xlsx', encoding = 'utf-8') # test set


# Step 1. XGboost Model
max_num_features = 20
max_size = 200000
x_data = []
y_data = pd.factorize(df['class'])
labels = y_data[1]
y_data = y_data[0]
for x in df['before'].values:
    x_row = np.zeros(max_num_features, dtype=int)
    for xi, i in zip(list(str(x)), np.arange(max_num_features)):
        x_row[i] = ord(xi) - ord('a')
    x_data.append(x_row)

print('Total number of samples:', len(x_data))
print('Use: ', max_size)
x_data = np.array(x_data[:max_size])
y_data = np.array(y_data[:max_size])

print('x_data sample:')
print(x_data[0])
print('y_data sample:')
print(y_data[0])
print('labels:')
print(labels)

x_train = x_data
y_train = y_data
del x_data
del y_data

x_train, x_valid, y_train, y_valid= train_test_split(x_train, y_train,
                                                      test_size=0.1, random_state=2017)
num_class = len(labels)
dtrain = xgb.DMatrix(x_train, label=y_train)
dvalid = xgb.DMatrix(x_valid, label=y_valid)
watchlist = [(dvalid, 'valid'), (dtrain, 'train')]

param = {'objective':'multi:softmax',
         'eta':'0.3', 'max_depth':10,
         'silent':1, 'nthread':-1,
         'num_class':num_class,
         'eval_metric':'merror'}
model = xgb.train(param, dtrain, 60, watchlist, early_stopping_rounds=20,
                  verbose_eval=10)

model.save_model('xgb_model')


# Step 2. Make our label predictions
pred = model.predict(dvalid)
pred = [labels[int(x)] for x in pred]
y_valid = [labels[x] for x in y_valid]
x_valid = [ [ chr(x + ord('a')) for x in y] for y in x_valid]
x_valid = [''.join(x) for x in x_valid]
x_valid = [re.sub('a+$', '', x) for x in x_valid]

df_pred = pd.DataFrame(columns=['data', 'predict', 'target'])
df_pred['data'] = x_valid
df_pred['predict'] = pred
df_pred['target'] = y_valid
df_pred.to_csv('pred.csv')

df_errors = df_pred.loc[df_pred['predict'] != df_pred['target']]
df_errors.to_csv('errors.csv')

x_test = []
for x in test['before'].values:
    x_row = np.zeros(max_num_features, dtype=int)
    for xi, i in zip(list(str(x)), np.arange(max_num_features)):
        x_row[i] = ord(xi) - ord('a')
    x_test.append(x_row)

x_test = np.array(x_test)
dtest = xgb.DMatrix(x_test)
pred_test = model.predict(dtest)
pred_test2 = [labels[int(x)] for x in pred_test]
test['class_pred']=pred_test2
test["after_pred"] = ""

print(set(test['class_pred'])) # possible labels


# Step 3. Functions needed to normalize each label type
def verbatim(x):
    #this appears to be mainly deals with symbols that will be in the trained dictionary. 
    #only other thing is letters we need to separate
    if len(x)>1:
        x_list = [i for i in x]
        return " ".join(x_list)
    else:
        return(x)

def time(x):
    x = re.sub('\.','',x)
    if len(x.split(':')) == 2:
        x = re.sub(':',' ',x)
        x = re.sub('([^0-9])', r' \1 ', x)
        x = re.sub('\s{2,}', ' ', x)
        time_list = x.split(' ')
        t_list = [i for i in time_list if i != ""] 
        for i,v in enumerate(t_list):
            if v == '00':
                t_list[i] = ''
            elif v.isdigit():
                t_list[i] = num2words(int(v))
            else:
                t_list[i] = v.lower()
        t = " ".join(t_list)
    elif len(x.split(':')) == 3:
        x_list = x.split(':')
        time_list = [num2words(int(num)) for num in x_list]
        time_units = []
        if int(x_list[0]) != 1:
            time_units.append('hours')
        else:
            time_units.append('hour')
        if int(x_list[1]) != 1:
            time_units.append('minutes')
        else:
            time_units.append('minute')
        if int(x_list[2]) != 1:
            time_units.append('seconds')
        else:
            time_units.append('second')
        t_list = [time_list[0],time_units[0],time_list[1],time_units[1],time_list[2],time_units[2]]
        t = " ".join(t_list)
    else:
        x = re.sub('([^0-9])', r' \1 ', x)
        x = re.sub('\s{2,}', ' ', x)
        time_list = x.split(' ')
        t_list = [i for i in time_list if i != ""] 
        for i,v in enumerate(t_list):
            if v == '00':
                t_list[i] = ''
            elif v.isdigit():
                t_list[i] = num2words(int(v))
            else:
                t_list[i] = v.lower()
        t = " ".join(t_list)       
        
    time = re.sub(',',"",t)
    time_final = re.sub('-'," ",time)
    return(time_final)

def measure(x):
    x = re.sub(',','',x)
    sdict = {}
    sdict['km2'] = 'square kilometers'
    sdict['km'] = 'kilometers'
    sdict['kg'] = 'kilograms'
    sdict['lb'] = 'pounds'
    sdict['dr'] = 'doctor'
    sdict['sq'] = 'square'
    sdict['m²'] = 'square meters'
    sdict['in'] = 'inch'
    sdict['oz'] = 'ounce'
    sdict['gal'] = 'gallon'
    sdict['m'] = 'meter'
    sdict['m2'] = 'square meters'
    sdict['m3'] = 'cubic meters'
    sdict['mm'] = 'millimeters'
    sdict['ft'] = 'feet'
    sdict['mi'] = 'miles'
    sdict['ha'] = "hectare"
    sdict['mph'] = "miles per hour"
    sdict['%'] = 'percent'
    sdict['GB'] = 'gigabyte'
    sdict['MB'] = 'megabyte'
    SUB = {ord(c): ord(t) for c, t in zip(u"₀₁₂₃₄₅₆₇₈₉", u"0123456789")}
    SUP = {ord(c): ord(t) for c, t in zip(u"⁰¹²³⁴⁵⁶⁷⁸⁹", u"0123456789")}
    OTH = {ord(c): ord(t) for c, t in zip(u"፬", u"4")}

    x = x.translate(SUB)
    x = x.translate(SUP)
    x = x.translate(OTH)
    
    if len(x.split(' ')) > 1:
        m_list = x.split(' ')
    elif "/" in x:
        m_list = x.split('/')
        m_list.insert(1,'per')
    else:
        x = re.sub('([^0-9\.])', r' \1 ', x)
        x = re.sub('\s{2,}', ' ', x)
        measure_list = x.split(' ')
        measure_list = [i for i in measure_list if i != ""]
        m_list = [measure_list[0],"".join(measure_list[1:])]
    for i,v in enumerate(m_list):
        if v.isdigit():
            srtd = v.translate(SUB)
            srtd = srtd.translate(SUP)
            srtd = srtd.translate(OTH)
            m_list[i] = num2words(float(srtd))
        elif is_float(v):
            m_list[i] = decimal(v)
        elif v in sdict:
            m_list[i] = sdict[v]
    measure = " ".join(m_list)
    measure = re.sub(',',"",measure)
    measure_final = re.sub('-'," ",measure)
    return(measure_final)

def decimal(x):
    x = re.sub(',','',x)
    deci_list = x.split('.')
    deci_list.insert(1,'point')
    if deci_list[0] =="":
        deci_list = deci_list[1:]
    else:
        deci_list[0] = num2words(int(deci_list[0]))
    #dealing with decimals after . (ex: .91 = point nine one)
    decimals_list = [num2words(int(num)) for num in deci_list[-1]]
    decimals = " ".join(decimals_list)
    decimals = re.sub('zero','o',decimals)
    deci_list[-1] = decimals
    return(" ".join(deci_list))

def year(x):
    year_list = [num for num in str(x)]
    if len(year_list)==4 and (year_list[0] == "1" or (year_list[0] == "2" and year_list[2]!='0')):
        year_list.insert(2, " ")
        year = "".join(year_list)
        year = year.split(' ')
        year = [num2words(int(num)) for num in year]
        year = " ".join(year)
    elif len(year_list)==4 and (year_list[0] == "1" or (year_list[0] == "2" and year_list[2]=='0')):
        year = "".join(year_list)
        year = num2words(int(year))
    elif  len(year_list)==2 and year_list[0]=='9':
        new_year_list = ["nineteen"]
        new_year_list.append("".join(year_list))
        new_year_list[1] = num2words(int(new_year_list[1]))
        year = " ".join(new_year_list)        
    elif len(year_list)==2 and year_list[0]!='0':
        new_year_list = ["twenty"]
        new_year_list.append("".join(year_list))
        new_year_list[1] = num2words(int(new_year_list[1]))
        year = " ".join(new_year_list)
    elif len(year_list)==2 and year_list[0]=='0':
        new_year_list = ['o']
        num = num2words(int(year_list[1]))
        new_year_list.append(num)
        year = " ".join(new_year_list)
    year = re.sub(',',"",year)
    year_final = re.sub('-'," ",year)
    return(year_final)

def date(x):
    x = re.sub(',','',x)
    months = ["january","febuary","march","april","may","june","july","august","september","october","november","december"]
    
    day = {"01":'first', "02":"second" , "03":'third', "04":'fourth', "05":'fifth', "06":'sixth', "07":'seventh', "08":'eighth', "09":'ninth', "10":'tenth', "11":'eleventh',
    "12":'twelfth', "13":'thirteenth', "14":'fourteenth', "15":'fifteenth', "16":'sixteenth', "17":'seventeenth', "18":'eighteenth', "19":'nineteenth', "20":'twentieth', "21":'twenty-first',
    "22":'twenty-second', "23":'twenty-third', "24":'twenty-fourth', "25":'twenty-fifth', "26":'twenty-sixth', "27":'twenty-seventh', "28":'twenty-eighth', "29":'twenty-ninth', "30":'thirtieth', "31":'thirty-first',"1":'first', "2":"second" , "3":'third', "4":'fourth', "5":'fifth', "6":'sixth', "7":'seventh', "8":'eighth', "9":'ninth'}

    month = {"01":"January","02":"February","03":"March","04":"April","05":"May","06":"June",
         "07":"July", "08":"August","09":"September","1":"January","2":"February","3":"March","4":"April","5":"May","6":"June",
         "7":"July", "8":"August", "9":"September","10":"October","11":"November", "12":"December"}

    ord_days = {"1st":'first', "2nd":"second" , "3rd":'third', "4th":'fourth', "5th":'fifth', "6th":'sixth', "7th":'seventh', "8th":'eighth', "9th":'ninth', "10th":'tenth', "11th":'eleventh',
    "12th":'twelfth', "13th":'thirteenth', "14th":'fourteenth', "15th":'fifteenth', "16th":'sixteenth', "17th":'seventeenth', "18th":'eighteenth', "19th":'nineteenth', "20th":'twentieth', "21th":'twenty-first',
    "22nd":'twenty-second', "23rd":'twenty-third', "24th":'twenty-fourth', "25th":'twenty-fifth', "26th":'twenty-sixth', "27th":'twenty-seventh', "28th":'twenty-eighth', "29th":'twenty-ninth', "30th":'thirtieth', "31st":'thirty-first'}
    x = re.sub(',','',x)
    #Changing dates in form month/day/year
    if len(x.split("/")) == 3:
        date = x.split("/")
        date[0] = month[date[0]]
        date[1] = day[date[1]]
        date[2] = year(date[2])
        x_final = " ".join(date).lower() 
    #Changing dates in form day.month.year
    elif len(x.split(".")) == 3:
        date = x.split(".")
        date[1] = month[date[1]]
        date[0] = day[date[0]]+" of"
        date[2] = year(date[2])
        x_final = " ".join(date).lower() 
    # Dates written out
    elif len(x.split(' ')) > 1:  #testing for words (well sentences) like days and numbers with units
        date_list = x.split(' ')
        for i,v in enumerate(date_list):
            if v in ord_days:  #checking for date case 15th OF Jan.
                if i == 0:
                    date_list[i] = "the "+ord_days[v]+" of"
                else:
                    date_list[i] = ord_days[v]
            if v.isdigit():
                if i == 0 and len(v)<3:
                    date_list[i] = "the "+day[v]+" of"
                elif len(v)<3:
                    date_list[i] = day[v]
                elif len(v)==4:
                    date_list[i] = year(v)
            x_final = " ".join(date_list)
    elif len(x) == 4:
        x_final = year(x)
    else:
        #in case we missed some (take a loss)
        x_final = x
        
    x_final = re.sub(',',"",x_final)
    x_final = re.sub('-'," ",x_final)
    return(x_final.lower())

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def money(x):
    money = re.sub('([$£€])', r'\1 ', x)
    money = re.sub('\s{2,}', ' ', money)
    money = re.sub(r',', '', money)
    money_list = money.split(' ')
    if money_list[0] == '$':
        money_list.append('dollars')
        money_list = money_list[1:]
    elif money_list[0] == '£':
        money_list.append('pounds')
        money_list = money_list[1:]
    elif money_list[0] == '€':
        money_list.append('euros')
        money_list = money_list[1:]    
    for i in range(len(money_list)):
        if money_list[i].isdigit():
            money_list[i] = num2words(int(money_list[i]))
        elif is_float(money_list[i]):
            money_list[i] = num2words(float(money_list[i]))
            money_list[i] = re.sub(r' zero', '', money_list[i])
    x = ' '.join(money_list)
    x = re.sub(r',', '', x)
    x = re.sub(r'-', ' ', x)
    return(x)

def digit(x): 
    try:
        x = re.sub('[^0-9]', '',x)
        result_string = ''
        for i in x:
            result_string = result_string + cardinal(i) + ' '
        result_string = result_string.strip()
        return result_string
    except:
        return(x) 
    
def telephone(x):
    try:
        result_string = ''
        for i in range(0,len(x)):
            if re.match('[0-9]+', x[i]):
                result_string = result_string + cardinal(x[i]) + ' '
            else:
                result_string = result_string + 'sil '
        return result_string.strip()    
    except:    
        return(x)   
    

def fraction(x):
    try:
        y = x.split('/')
        result_string = ''
        y[0] = cardinal(y[0])
        y[1] = ordinal(y[1])
        if y[1] == 4:
            result_string = y[0] + ' quarters'
        else:    
            result_string = y[0] + ' ' + y[1] + 's'
        return(result_string)
    except:    
        return(x)
    

def ordinal(x):
    try:
        result_string = ''
        x = x.replace(',', '')
        x = x.replace('[\.]$', '')
        if re.match('^[0-9]+$',x):
            x = num2words(int(x), ordinal=True)
            return(x.replace('-', ' '))
        if re.match('.*V|X|I|L|D',x):
            if re.match('.*th|st|nd|rd',x):
                x = x[0:len(x)-2]
                x = rom_to_int(x)
                result_string = re.sub('-', ' ',  num2words(x, ordinal=True))
            else:
                x = rom_to_int(x)
                result_string = 'the '+ re.sub('-', ' ',  num2words(x, ordinal=True))
        else:
            x = x[0:len(x)-2]
            result_string = re.sub('-', ' ',  num2words(float(x), ordinal=True))
        return(result_string)  
    except:
        return x
    
def cardinal(x):
    try:
        if re.match('.*[A-Za-z]+.*', x):
            return x
        x = re.sub(',', '', x, count = 10)

        if(re.match('.+\..*', x)):
            x = num2words(float(x))
        elif re.match('\..*', x): 
            x = num2words(float(x))
            x = x.replace('zero ', '', 1)
        else:
            x = num2words(int(x))
        x = x.replace('zero', 'o')    
        x = re.sub('-', ' ', x, count=10)
        x = re.sub(' and','',x, count = 10)
        return x
    except:
        return x
    
def address(x):
    try:
        x = re.sub('[^0-9a-zA-Z]+', '', x)
        x_list = [char for char in x]
        for i in range(len(x_list)):
            if re.match('[A-Z]|[a-z]',x_list[i]):
                x_list[i] = f'{x_list[i].lower()} '
            else:
                continue
        x = "".join(x_list)
        x_list2 = x.split(' ')
        for i in range(len(x_list2)):
            if re.match('[0-9]',x_list2[i]):                        
                x_list2[i]=(num2words(int(x_list2[i])))
        x = " ".join(x_list2)
        return(x)
    except:
        return(x)
    
def letters(x):
    try:
        x = re.sub('[^a-zA-Z]', '', x)
        x = x.lower()
        result_string = ''
        for i in range(len(x)):
            result_string = result_string + x[i] + ' '
        return(result_string.strip())  
    except:
        return x
    
def electronic(x):
    try:
        replacement = {'.' : 'dot', ':' : 'colon', '/':'slash', '-' : 'dash', '#' : 'hash tag', }
        result_string = ''
        if re.match('.*[A-Za-z].*', x):
            for char in x:
                if re.match('[A-Za-z]', char):
                    result_string = result_string + letters(char) + ' '
                elif char in replacement:
                    result_string = result_string + replacement[char] + ' '
                elif re.match('[0-9]', char):
                    if char == 0:
                        result_string = result_string + 'o '
                    else:
                        number = cardinal(char)
                        for n in number:
                            result_string = result_string + n + ' ' 
            return result_string.strip()                
        else:
            return(x)
    except:    
        return(x)
    

# Step 4. Creating a dictionary of before and afters from our train
diffs = dict() #dictionary to save results of differences; it will be key: before word value: d
total = 0
not_same = 0


for row in df[["before","after"]].values: #goes through all of the rows in the columns before and after
    total += 1
    if row[0] != row[1]:    #checks if the before and after are the same
        not_same += 1       #keeps track of how many are different
    if row[0] not in diffs:  #checks if word already in the dictionary
        diffs[row[0]] = dict() #if its not it adds it as a key, and gives it a dictionary as a value
        diffs[row[0]][row[1]] = 1  #in that created dictionary, it add the after word as a key, and 1 #times seen
    else:
        if row[1] in diffs[row[0]]:    #checking that it is in the key - 
            diffs[row[0]][row[1]] += 1  #add to the number of times word has been seen
        else:
            diffs[row[0]][row[1]] = 1
print('Train File:\tTotal: {} Have diff value: {}'.format(total, not_same))


# Step 6. Normalizing text by label and creating output
total = 0
changes = 0

out_list = [] #outout dataframe to be filled in

for row in test[["sentence_id","token_id","before","class_pred"]].values:
    i1 = row[0]
    i2 = row[1]
    before = row[2]
    label = row[3]
    
    #if it is in the training dictionary we created, then that will be the normalized
    if before in diffs:
        norm = sorted(diffs[before].items(), key=operator.itemgetter(1), reverse=True)
        out_list.append(('%d_%d'%(i1,i2), norm[0][0]))
    #'ADDRESS'
    elif label == 'ADDRESS':
        try:
            norm = address(before)
            out_list.append(('%d_%d'%(i1,i2), norm))
            changes += 1
        except:
            out_list.append(('%d_%d'%(i1,i2), before))
    #'CARDINAL'
    elif label == 'CARDINAL':
        try:
            norm = cardinal(before)
            out_list.append(('%d_%d'%(i1,i2), norm))
            changes += 1
        except:
            out_list.append(('%d_%d'%(i1,i2), before))
    #'DATE'
    elif label == 'DATE':
        try:
            norm = date(before)
            out_list.append(('%d_%d'%(i1,i2), norm))
            changes += 1
        except:
            out_list.append(('%d_%d'%(i1,i2), before))
    #'DECIMAL'
    elif label == 'DECIMAL':
        try:
            norm = decimal(before)
            out_list.append(('%d_%d'%(i1,i2), norm))
            changes += 1
        except:
            out_list.append(('%d_%d'%(i1,i2), before))
    #'DIGIT',
    elif label == 'DIGIT':
        try:
            norm = digit(before)
            out_list.append(('%d_%d'%(i1,i2), norm)) 
            changes += 1
        except:
            out_list.append(('%d_%d'%(i1,i2), before))
    #'ELECTRONIC',
    elif label == 'ELECTRONIC':
        try:
            norm = electronic(before)
            out_list.append(('%d_%d'%(i1,i2), norm)) 
            changes += 1
        except:
            out_list.append(('%d_%d'%(i1,i2), before))
    #'FRACTION',
    elif label == 'FRACTION':
        try:
            norm = fraction(before)
            out_list.append(('%d_%d'%(i1,i2), norm)) 
            changes += 1
        except:
            out_list.append(('%d_%d'%(i1,i2), before))
    #'LETTERS',
    elif label == 'LETTERS':
        try:
            norm = letters(before)
            out_list.append(('%d_%d'%(i1,i2), norm)) 
            changes += 1
        except:
            out_list.append(('%d_%d'%(i1,i2), before))
    #'MEASURE',
    elif label == 'MEASURE':
        try:
            norm = measure(before)
            out_list.append(('%d_%d'%(i1,i2), norm))
            changes += 1
        except:
            out_list.append(('%d_%d'%(i1,i2), before))
    #'MONEY',
    elif label == 'MONEY':
        try:
            norm = money(before)
            out_list.append(('%d_%d'%(i1,i2), norm)) 
            changes += 1
        except:
            out_list.append(('%d_%d'%(i1,i2), before))
    #'ORDINAL',
    elif label == 'ORDINAL':
        try:
            norm = ordinal(before)
            out_list.append(('%d_%d'%(i1,i2), norm))
            changes += 1
        except:
            out_list.append(('%d_%d'%(i1,i2), before))
    #'PLAIN' nothing changes
    elif label == 'PLAIN':
        norm = before
        out_list.append(('%d_%d'%(i1,i2), norm))        
    #'PUNCT',
    elif label == 'PUNCT':
        norm = before
        out_list.append(('%d_%d'%(i1,i2), norm))        
    #'TELEPHONE',
    elif label == 'TELEPHONE':
        try:
            norm = telephone(before)
            out_list.append(('%d_%d'%(i1,i2), norm))
            changes += 1
        except:
            out_list.append(('%d_%d'%(i1,i2), before))
    #'TIME',
    elif label == 'TIME':
        try:
            norm = time(before)
            out_list.append(('%d_%d'%(i1,i2), norm))
            changes += 1
        except:
            out_list.append(('%d_%d'%(i1,i2), before))
    #'VERBATIM'
    elif label == 'VERBATIM':
        try:
            norm = verbatim(before)
            out_list.append(('%d_%d'%(i1,i2), norm))
            changes += 1
        except:
            out_list.append(('%d_%d'%(i1,i2), before))
    else:
        out_list.append(('%d_%d'%(i1,i2), "#ERROR"))
    total += 1

labels = ["id","after"] #headers for the output dataframe
out_df = pd.DataFrame.from_records(out_list, columns = labels)
print('Total: {} Changed: {}'.format(total, changes))

out_df.to_csv('my_output_new2.csv', index = False)  #making the output dataframe a csv file without and index column

test["after_pred"] = out_df["after"]
test = test[["sentence_id", "token_id", "before", "class_pred", "after_pred"]]
test.to_excel("final_pred.xlsx")
