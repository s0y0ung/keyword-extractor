from konlpy.tag import Kkma
from konlpy.tag import Okt
from konlpy.corpus import kolaw
from konlpy.utils import pprint
from nltk import collocations
import re
from soynlp.word import WordExtractor
from soynlp.utils import DoublespaceLineCorpus
from soynlp.tokenizer import LTokenizer
from soynlp.noun import LRNounExtractor_v2
from math import log10
from khaiii import KhaiiiApi
import sys
#책 데이터 가져오기
def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
#책 데이터 가져오기
import pickle as pk
with open('/content/drive/My Drive/test/data.pickle','rb') as f:
    corpus = pk.load(f)
#6번 책을 대상으로 키워드 추출 (원하는 텍스트 넣으시면 돼요)
target=corpus[0]
target= re.sub('[0-9\n\'\"?!.]', '',target)
#soynlp
noun_extractor = LRNounExtractor_v2(verbose=True, extract_compound=True)
nouns = noun_extractor.train_extract(corpus)
noun_scores = {noun:score.score for noun, score in nouns.items()}
word_extractor = WordExtractor(
    min_frequency=3, # example
    min_cohesion_forward=0.05,
    min_right_branching_entropy=0.0
)
word_extractor.train(corpus)
words = word_extractor.extract()
cohesion_score = {word:score.cohesion_forward for word, score in words.items()}
combined_scores = {noun:score + cohesion_score.get(noun, 0)
    for noun, score in noun_scores.items()}
combined_scores.update(
    {subword:cohesion for subword, cohesion in cohesion_score.items()
    if not (subword in combined_scores)}
)
tokenizer = LTokenizer(scores=combined_scores)
#복합명사 추출
bigram_measures = collocations.BigramAssocMeasures()
bigram=[]
words=tokenizer.tokenize(target)
ignored_words = ['에게','으로','들이','우리','다운', '에서', '하는', '야', '있게', '스러운', '뿐만', '아니라', '부터', '로만', '라고', '로', '바를', '나게']
finder = collocations.BigramCollocationFinder.from_words(words)
finder.apply_word_filter(lambda w: len(w) < 1 or w in ignored_words)    #ignored_words 제외
finder.apply_freq_filter(5) # only bigrams that appear 5+ times
bigram=finder.nbest(bigram_measures.pmi,15)     #상위 15개 복합 명사
words2=Okt().morphs(target)
finder = collocations.BigramCollocationFinder.from_words(words2)
finder.apply_word_filter(lambda w: len(w) < 2 or w in ignored_words)    #한 글자, ignored_words 제외
finder.apply_freq_filter(5) # only bigrams that appear 5+ times
bigram += finder.nbest(bigram_measures.pmi,10)   #상위 10개 복합 명사
bigram=list(set(bigram))
# — TF-IDF function
# =======================================
def f(t, d):
    return d.count(t)
def tf(t, d):
    # d is document == tokens
    return 0.5 + 0.5*f(t,d)/max([f(w,d) for w in d])
def idf(t, D):
    # D is documents == document list
    numerator = len(D)
    denominator = 1 + len([ True for d in D if t in d])
    return log10(numerator/denominator)
def tfidf(t, d, D):
    return f(t,d)*idf(t, D)
def tokenizer_1(d):
    #한글만
    d=re.sub('[0-9\n\'\"?!.()=,<>-|~/]', '',d)
    tokens = tokenizer.tokenize(d, remove_r=True)
    return tokens
#일단 stop_words빼기    result=[word for word in tokens if not word in stop_words]
#    return result
# =================================================
#tf-idf 구하기
tokenized_D=[tokenizer_1(d) for d in corpus]
d=tokenizer_1(target)
#temp = 단어만 (중복없이)
#score_temp = 단어의 tf-idf 값만
#tuple_list = (단어, tf-idf)쌍
#tuple_list[i] = (temp[i], score_temp[i])
temp=[] #temp=list(set(d))라서 사실 temp없어도 될거야,,
score_temp=[]
for t in d:
    if t not in temp:
        temp.append(t)
        score_temp.append(tfidf(t,d,tokenized_D))
tuple_list=[]
for i in range(len(temp)):
    tuple_list.append((temp[i], score_temp[i]))
#tmp = tf-idf값 기준 상위 30개
#(단어, tf-idf) 형태
tmp=[]
tmp=sorted(tuple_list,key=lambda x:x[1], reverse=True)[:30]
# ================================================
#khaill를 이용해서 일반명사, 고유명사, 관형사, 일반부사 추출
#soynlp에는 품사가 없기 때문
kt=[]
api=KhaiiiApi()
for word in api.analyze(target):
    for morph in word.morphs:
        if morph.tag in ['NNG', 'NNP', 'MAG', 'MM']:
            kt.append(morph.lex)
#answer = tmp의 단어만
answer=[]
for i in range(len(tmp)):
    answer.append(tmp[i][0])
#현재 answer에는 여러 품사가 있기 때문에 kt를 이용해 명사, 관형사, 부사 외의 품사는 제거
#del_arr는 제거할 answer의 index를 담음
del_arr = []
api = KhaiiiApi()
for i in range(len(answer)-1,-1,-1):
    if answer[i] not in kt:
        del_arr.append(i)
#명사, 관형사, 부사 외의 품사 제거
for i in del_arr:
    if i > 3 :
        tmp.pop(i)
        answer.pop(i)
# =======================================
#bigram 추가 및 중복 제거
result=[]
delete=[]
#복합명사 중복 제거: bigram의 복합 명사의 두 단어 모두 answer에 있는 경우, 이 복합 명사를 result에 추가하고 두 단어는 delete에 추가
#delete는 복합 명사와의 중복 제거를 위한 list
for i in range(len(bigram)):
    if bigram[i][0] in answer and bigram[i][1] in answer:
        if f(bigram[i][0],d) > f(bigram[i][1],d) + 5:   #앞 단어가 뒷 단어보다 5번 이상 더 등장했을 경우, 앞 단어는 주인공으로 간주하고 중복 제거 안함.
            delete.append(bigram[i][1])                 # EX) '무민 아빠'의 경우, '무민 아빠'도 등장인물이지만 '무민'이 주인공이기 때문에 중복 제거하면 안됨
        else:
            delete.append(bigram[i][0])     #일반적으로는 복합 명사를 이루는 두 단어 모두 delete에 추가
            delete.append(bigram[i][1])
        if bigram[i] not in result:     #복합 명사를 이루는 두 단어 중 tf-idf가 높은 값이 복합명사의 tf-idf값이 됨. ex) '이득춘(100)', '대감'(50) -> '이득춘 대감'의 tf-idf값은 100
            max = tmp[answer.index(bigram[i][0])][1] if tmp[answer.index(bigram[i][0])][1] > tmp[answer.index(bigram[i][1])][1] else tmp[answer.index(bigram[i][1])][1]
            result.append((bigram[i],max))
#복합 명사 중복 제거
for i in range(len(delete)):
    if delete[i] in answer:
        del tmp[answer.index(delete[i])]
        answer.remove(delete[i])
#현재 result에는 복합명사만 있으므로 tmp (단일 명사, 복합 명사와는 중복 제거됨)추가
result += tmp
result=sorted(result,key=lambda x:x[1], reverse=True)
result=result[:10]  #상위 10개만
#정리
print ('here!!!')
'''
for i in result:
    for j in i:
        if(type(j) == tuple):
            leng = len(j) - 1
            for k in range(0,leng):
                print(j[k],end=' ')
            print(j[leng],end='')
        elif(isfloat(j)):
            print('|' + str(round(float(j),2)))
        else:
            print(j, end = '')
'''
