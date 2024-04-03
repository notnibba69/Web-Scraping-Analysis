#!/usr/bin/env python
# coding: utf-8

# Making a csv file with extracted data 

# In[1]:


import pandas as pd


# In[2]:


website_dataframe = pd.read_excel('input.xlsx')


# In[3]:


web2 = website_dataframe.copy()


# In[4]:


websites_list = list(web2['URL'])


# In[5]:


print(len(websites_list))


# In[6]:


import requests
from bs4 import BeautifulSoup


# In[7]:


web_list = []
class_name = 'td-post-content tagdiv-type'
pic_class="tdb-block-inner td-fix-index"


def extract_text_from_url(websites_list): #LIST OVER HERE OF WEBSITES
    for url in websites_list: 
        text_for_each_website = ''
        try:
            response = requests.get(url)
        
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.content, 'html.parser')
            # Find and extract the textual data
            mydivs = soup.find_all("div", class_=class_name)
            for div in mydivs:
                text_for_each_website+= div.text
                
            # for url with error 404
            if url == web2['URL'][35] or url == web2['URL'][48]:
                web_list.append(text_for_each_website)
                continue
            
            # for websites which have different classes
            if text_for_each_website == '':
                mydivs = soup.find_all("div", class_=pic_class)
                for div in mydivs:
                    text_for_each_website+= div.text
                
            web_list.append(text_for_each_website) #APPEND THE LIST
            
        except requests.exceptions.RequestException as e:
            print("Error:", e)
            return None

extract_text_from_url(websites_list)


# In[8]:


empty_elements_with_indexes = [(index, element) for index, element in enumerate(web_list) if not element]

print("Empty elements with their indexes:")
for index, element in empty_elements_with_indexes:
    print(f"Index: {index}, Value: {element}")


# In[9]:


print(len(web_list))


# In[10]:


web2['article_text'] = web_list
print(web2.head())


# In[11]:


from numpy import nan


# In[12]:


web2['article_text'][web2['article_text'] == ''] = nan


# In[13]:


print(web2.isnull().sum())


# In[14]:


web2 = web2.dropna()


# In[20]:


# Since the extracted text had some extra text towards the end, a list is made of the extra text and then altered the main article text
# also made a list of titles of all the articles since we were using the url to access the website again
class_name = 'wp-block-preformatted'
list_of_text_remove = []
title_list = []
num = 0
for url in web2['URL']:
    
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')    
    # get the title
    title = soup.title
    title_list.append(title.string)
    # Find and extract the textual data
    text_to_strip = ''
    try:
        mydivs = soup.find("pre", class_=class_name)
        text_to_strip+= mydivs.text
    except AttributeError:
        list_of_text_remove.append('')
        continue
    list_of_text_remove.append(text_to_strip)


# In[21]:


print(len(list_of_text_remove))


# In[22]:


web2['text_to_delete'] = list_of_text_remove


# In[23]:


article_text_list = list(web2['article_text'])


# In[24]:


print(len(article_text_list))


# In[39]:


text_to_delete_list = list_of_text_remove.copy()


# In[26]:


print(len(text_to_delete_list))


# In[27]:


# Stripping '\n' because the 'strip' function was not detecting the text_to_delete_list function's data
for i in range(len(article_text_list)):
    article_text_list[i] = article_text_list[i].rstrip('\n')
    article_text_list[i] = article_text_list[i].lstrip('\n')


# In[28]:


final_data = []
for i in range(len(article_text_list)):
    final_data.append(article_text_list[i].strip(text_to_delete_list[i]))


# In[29]:


web2['Article_text'] = final_data


# In[30]:


web2['Title'] = title_list


# In[31]:


web2 = web2.drop(['article_text', 'text_to_delete'], axis = 1)


# In[40]:


print(web2.head())


# In[ ]:


web2.to_csv('data1.csv')


# In[ ]:




