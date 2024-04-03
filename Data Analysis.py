#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('data1.csv')


# In[3]:


print(df.head())


# In[4]:


df = df.drop(['Unnamed: 0'],axis = 1)


# In[5]:


print(df.head())


# # Sentiment Analysis
# 1.1 : Cleaning using stopwords

# In[6]:


# Stopwords provided in the foldre
StopWords_Generic=pd.read_csv('StopWords_Generic.txt', sep=' ', header=None, names=['stop_words'])


# In[7]:


StopWords_Generic_List = list(StopWords_Generic['stop_words'])


# In[8]:


print(len(StopWords_Generic_List))


# In[9]:


StopWords_GenericLong=pd.read_csv('StopWords_GenericLong.txt', sep=' ', header=None, names=['stop_words_Long'])


# In[10]:


StopWords_GenericLong_List = StopWords_GenericLong['stop_words_Long']


# In[11]:


print(len(StopWords_GenericLong_List))


# In[12]:


import string
print(string.punctuation)


# In[13]:


import nltk
from nltk.corpus import stopwords


# In[14]:


# creating a function which will clean the textual data according to the stopwords (in nltk library and the provided folder) and removing punctuations
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in StopWords_Generic_List and i not in StopWords_GenericLong_List and i not in string.punctuation and i not in stopwords.words('english'):
            y.append(i)
            
    return " ".join(y)


# In[15]:


df['cleaned_text'] = df['Article_text'].apply(transform_text)


# In[16]:


sentences = df['Article_text'].apply(nltk.sent_tokenize)


# In[17]:


sentence_count = []
for i in sentences:
    sentence_count.append(len(i))


# In[18]:


df["SENTENCE COUNT"] = sentence_count


# In[19]:


df = df.drop(['Article_text'],axis = 1)


# In[20]:


print(df.head())


# 1.2 CREATING DICIONARY OF POSITIVE AND NEGATIVE WORDS

# In[21]:


positive=pd.read_csv('positive-words.txt', sep=' ', header=None, names=['positive_words'])


# In[22]:


positive_list = list(positive['positive_words'])
print(positive_list[:5])


# In[23]:


negative=pd.read_csv('negative-words.txt', sep=' ', header=None, names=['negative_words'],encoding='latin-1')


# In[24]:


negative_list = list(negative['negative_words'])
print(negative_list[:5])


# In[25]:


word_dict = {'positive':positive_list,
            'negative':negative_list}


# 1.3 EXTRACTING DERIVED VARIABLES

# In[26]:


def positive_counter(text):
    
    p_count = 0
    
    for i in positive_list:
        if i in text:
            p_count+=1

    return p_count


# In[27]:


df['POSITIVE SCORE'] = df['cleaned_text'].apply(positive_counter)


# In[28]:


def negative_counter(text):
    
    n_count = 0
    
    for i in negative_list:
        if i in text:
            n_count-=1

    return n_count*-1


# In[29]:


df['NEGATIVE SCORE'] = df['cleaned_text'].apply(negative_counter)


# In[30]:


print(df.head())


# In[31]:


df['POLARITY SCORE'] = (df['POSITIVE SCORE'] - df['NEGATIVE SCORE']) / ((df['POSITIVE SCORE'] + df['NEGATIVE SCORE']) + 0.000001)


# In[32]:


print(df.head())


# In[33]:


df['WORD COUNT'] = df['cleaned_text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[34]:


print(df.head())


# In[35]:


df['SUBJECTIVITY SCORE'] = (df['POSITIVE SCORE'] + df['NEGATIVE SCORE'])/ ((df['WORD COUNT']) + 0.000001)


# In[36]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')

spam_wc = wc.generate(df['cleaned_text'].str.cat(sep=" "))

plt.figure(figsize=(15,6))
plt.title('Most Used Words (200)', fontsize = 10)
print(plt.imshow(spam_wc))


# In[37]:


# creating a corpus for all the words
word_corpus = []
for msg in df['cleaned_text'].tolist():
    for word in msg.split():
        word_corpus.append(word)

print(len(word_corpus))


# In[38]:


import seaborn as sns
from collections import Counter


# In[39]:


sns.barplot(x = pd.DataFrame(Counter(word_corpus).most_common(30))[0], y = pd.DataFrame(Counter(word_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.title('Most Common 30 Words Ranked')
print(plt.show())


# In[40]:


df['Average Sentence Length'] = df['WORD COUNT'] / df['SENTENCE COUNT']


# In[41]:


print(df.head())


# In[42]:


import re

def count_syllables(text):
    tex = nltk.word_tokenize(text)
    count = 0
    comp_count = 0
    for word in tex:
        syllable_pattern = r'(?!e$)[aeiouy]+'
        syllables = re.findall(syllable_pattern, word, re.I)
        #print(syllables)

        exceptions = ["es", "ed"]
        for exception in exceptions:
            if word.endswith(exception):
                syllables.pop()
        if len(syllables) > 2:
            comp_count+=1
        count += len(syllables)

    return count


# In[43]:


df['Syllable Count'] = df['cleaned_text'].apply(count_syllables)


# In[44]:


print(df.head())


# In[45]:


def count_complex(text):
    tex = nltk.word_tokenize(text)
    comp_count = 0
    for word in tex:
        syllable_pattern = r'(?!e$)[aeiouy]+'
        syllables = re.findall(syllable_pattern, word, re.I)
        #print(syllables)

        exceptions = ["es", "ed"]
        for exception in exceptions:
            if word.endswith(exception):
                syllables.pop()
        if len(syllables) > 2:
            comp_count+=1

    return comp_count


# In[46]:


df['Complex Count'] = df['cleaned_text'].apply(count_complex)


# In[47]:


df['Fog Index'] = 0.4 * (df['Average Sentence Length'] + (df['Complex Count'] / df['WORD COUNT']))


# In[48]:


print(df.head())


# In[49]:


df['Percentage of Complex words '] = df['Complex Count'] / df['WORD COUNT']


# In[50]:


def count_personal_pronouns(text):
    
    pronouns = ["I", "we", "my", "ours", "us"]

    pattern = r'\b(?:' + '|'.join(pronouns) + r')\b'
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    # excluding the country US
    filtered_matches = [match for match in matches if match!= "US"]
    pronoun_count = len(filtered_matches)

    return pronoun_count


# In[51]:


df['PERSONAL PRONOUNS'] = df['cleaned_text'].apply(count_personal_pronouns)


# In[52]:


print(df.head())


# In[53]:


def no_of_characters(text):
    count = 0
    text = nltk.word_tokenize(text)
    for word in text:
        count+= len(word)
        
    return count


# In[54]:


df['Number of Characters'] = df['cleaned_text'].apply(no_of_characters)


# In[55]:


df['AVERAGE WORD LENGTH'] = df['Number of Characters'] / df['WORD COUNT']


# In[56]:


# Dataframe after all the analysis
print(df.head())


# In[57]:


df.columns


# In[58]:


# Creating another dataframe to convert it according to the output file
df2 = df[['URL_ID', 'URL','POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE',
       'SUBJECTIVITY SCORE','Average Sentence Length','Percentage of Complex words ',
       'Fog Index', 'Average Sentence Length', 'Complex Count', 'WORD COUNT','Syllable Count',
       'PERSONAL PRONOUNS', 'AVERAGE WORD LENGTH']]


# In[64]:


print(df2.head())


# In[60]:


df2.rename(columns = {'Average Sentence Length':'AVERAGE SENTENCE LENGTH', 'Percentage of Complex words' : 'PERCENTAGE OF COMPLEX WORD','Fog Index':'FOG INDEX', 'Syllable Count':'SYLLABLE PER WORD'}, inplace = True)


# In[61]:


print(df2.columns)


# In[62]:


df2.set_axis(['URL_ID', 'URL', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE',
       'SUBJECTIVITY SCORE', 'AVERAGE SENTENCE LENGTH',
       'PERCENTAGE OF COMPLEX WORDS ', 'FOG INDEX', 'AVERAGE NUMBER OF WORDS PER SENTENCE',
       'COMPLEX WORD COUNT', 'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS',
       'AVERAGE WORD LENGTH'], axis='columns')


# In[63]:


df2.to_excel('OUTPUT DATA STRUCTURE.xlsx')


# In[ ]:





# In[ ]:




