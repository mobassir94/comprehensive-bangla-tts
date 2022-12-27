#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %%capture

# !pip install pyspellchecker==0.7.0
# !pip install -q transformers

# !pip install sentencepiece
# !pip install git+https://github.com/csebuetnlp/normalizer

# ! pip install bangla==0.0.2
# # !pip install num2words
get_ipython().system('nvidia-smi')


# In[2]:



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import re
# from num2words import num2words
import os
from tqdm.auto import tqdm
tqdm.pandas()

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import transformers

import torch
import bangla
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from normalizer import normalize # pip install git+https://github.com/csebuetnlp/normalizer

from spellchecker import SpellChecker

import warnings
warnings.filterwarnings("ignore")

# from pandarallel import pandarallel
# pandarallel.initialize(progress_bar=True,nb_workers=8)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

print(torch.__version__)
spell = SpellChecker()
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[3]:


get_ipython().system('pwd')


# In[4]:


tafsir_en = pd.read_csv('/home/ansary/Shabab/tafsir_ibn_kathir/tafsir_ibn_kathir.csv')
tafsir_en.head()


# In[5]:


get_ipython().run_cell_magic('time', '', '\ndef tafsir_cleaner(tafsir):\n    res = re.sub(\'\\\'\',\'\',tafsir)\n    res = re.sub(\'\\n\',\'.\',res)\n    \n    res = re.sub(\'﴾\',\'\',res)\n    res = re.sub(\'﴿\',\'\',res)\n    \n    res = re.sub(\'«\',\'\',res)\n    res = re.sub(\'»\',\'\',res)\n    \n    res = res.replace(\'Prev.Next\',\'\')\n    return res\n    \ntafsir_en["Tafsir"]=tafsir_en.Tafsir.progress_apply(lambda tafsir: tafsir_cleaner(tafsir))')


# In[6]:


def tag_arabic_text(text,ar_pattern=u'[\u0600-\u06FF]+',english_only = False):
    # remove multiple spaces
    data=re.sub(' +', ' ',text)
    texts=[]
    if "।" in data:punct="।"
    elif "." in data:punct="."
    else:punct="\n"
    for text in data.split(punct):    
        # create start and end
        text="start"+text+"end"
        # tag text
        parts=re.split(ar_pattern, text)
        parts=[p for p in parts if p.strip()]
        parts=set(parts)
        for m in parts:
            if len(m.strip())>1:text=text.replace(m,f"</ar>{m}<ar>")
        # clean-up invalid combos
        text=text.replace("</ar>start",'')
        text=text.replace("end<ar>",'')
        texts.append(text)
    text=f"{punct}".join(texts)
    if(english_only):
        #https://stackoverflow.com/questions/55656429/replace-or-remove-html-tag-content-python-regex
        return re.sub(r'(?s)<ar>.*?</ar>', '', text)
    return text


# In[7]:



tafsir_en["Tafsir_en"]=tafsir_en.Tafsir.progress_apply(lambda tafsir_eng: tag_arabic_text(tafsir_eng,english_only=True))


# In[8]:


#tafsir_en["Tafsir_en"][0]


# In[9]:


#https://stackoverflow.com/questions/71620622/is-there-a-way-to-identify-and-create-a-list-of-all-acronyms-in-a-dataframe

# tafsir_en["detected_misspelled"] = tafsir_en.assign(
#     misspelled=tafsir_en["Tafsir_en"].str.split().map(spell.unknown),
#     acronyms=tafsir_en["Tafsir_en"].str.findall(r"([A-Z]{2,})").map(set)
# )[["misspelled", "acronyms"]].apply(lambda row: set.union(*row), axis=1)

def find_unknown_words(text):
    list_of_words = text.split()
    list_of_words = [re.sub(r'[^A-Za-z0-9-]+', ' ', x) for x in list_of_words]
    
    word_list = []
    for i in range(len(list_of_words)):
        temp = list_of_words[i].split()
        if(len(temp) == 1):
            word_list.append(temp[0])
        else:
            for j in range(len(temp)):
                word_list.append(temp[j])
            
    return ' '.join(list(spell.unknown(word_list)))


# In[10]:


tafsir_en["detected_unknowns"] = tafsir_en.Tafsir_en.progress_apply(lambda tafsir_eng: find_unknown_words(tafsir_eng))


# In[11]:


tafsir_en.head()


# In[12]:


unknown_words = list(tafsir_en['detected_unknowns'].str.split(' ', expand=True).stack().unique())
len(unknown_words)


# In[13]:


count_unknown = tafsir_en['detected_unknowns'].str.split(' ', expand=True).stack().value_counts()


# In[14]:


count_unknown


# In[15]:




# en_text = "all praise to him."

# model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
# tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

# # translate en to bn
# tokenizer.src_lang = "en"
# encoded_en = tokenizer(en_text, return_tensors="pt")
# generated_tokens = model.generate(**encoded_en, forced_bos_token_id=tokenizer.get_lang_id("bn"))
# tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


# from code above, it seems like the performance of m2m100_418M is not good enough, nllb should be much better than m2m100_418M.

# In[16]:



print(torch.__version__)
transformers.__version__


# # single sample inference test

# thanks to [banglanmt](https://huggingface.co/csebuetnlp/banglat5_nmt_en_bn/discussions)

# In[17]:


torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(torch_device)
model = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/banglat5_nmt_en_bn").to(torch_device)
tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglat5_nmt_en_bn",use_fast=True)

def translate_en_bn(input_sentence):
    input_ids = tokenizer(normalize(input_sentence), return_tensors="pt").input_ids
    input_ids = input_ids.to(torch_device)
    generated_tokens = model.generate(input_ids)#max_length=1536
    decoded_tokens = tokenizer.batch_decode(generated_tokens)[0]
    decoded_tokens=decoded_tokens.replace("<pad>","").replace("</s>","")
    sen=decoded_tokens.split()
    words=[w for w in sen if w.strip()]
    sen="".join(words)
    return decoded_tokens

text=translate_en_bn("alhamdulillah for everything.")
print(text)


# In[18]:


#tag_arabic_text(tafsir_en.Tafsir[0],english_only=False)


# In[19]:


get_ipython().run_cell_magic('time', '', '\nen_text = \'\'\'In the Name of Allah, the Most Gracious, the Most Merciful. <ar> ﴿الْحَمْدُ للَّهِ رَبِّ الْعَـلَمِينَ﴾ </ar>\n(Allah, the Exalted, said, `I have divided the prayer (Al-Fatihah) into two halves between \nMyself and My servant, and My servant shall have what he asks for.\nIf he says,\' <ar> بِسْمِ اللَّهِ الرَّحْمَـنِ الرَّحِيمِ </ar> Allah says, `My servant has praised Me.\' \'\'\'\n\ndef EN_AR_to_BN_AR_Translator(en_text,tag_text = False):\n    \'\'\'\n    translates multilingual english-arabic code mixed text into \n    multilingual bengali-arabic code mixed text\n    \'\'\' \n    if(tag_text):\n        en_text = tag_arabic_text(en_text,english_only=False)\n    \n    sentenceEnders = re.compile(\'[.!?]\')\n    sentences = sentenceEnders.split(en_text)\n    main_list = []\n    for i in range(len(sentences)):\n        \n        list_str = sentences[i].split(\'<ar>\')\n        if(len(list_str) == 1):\n            main_list.append(list_str[0])\n        else:\n            for j in range(len(list_str)):\n                if(\'</ar>\' in list_str[j]):\n                    list_str1 = list_str[j].split(\'</ar>\')\n                    main_list.append("<ar>"+list_str1[0]+"</ar>")\n                    main_list.append(list_str1[1])\n                else:\n                    main_list.append(list_str[j])\n\n    while(" " in main_list):\n        main_list.remove(" ")\n    for idx in range(len(main_list)):\n        if(\'<ar>\' not in main_list[idx] or \'</ar>\' not in main_list[idx]):\n            \n            output_sentence = []\n            for word in main_list[idx].split():\n                output_sentence.append(word)\n     \n            main_list[idx] = \' \'.join(output_sentence)\n            #numerizer\n            main_list[idx] = bangla.convert_english_digit_to_bangla_digit(main_list[idx])\n            # multilingual english-arabic to multilingual bengali-arabic\n            try:\n                if len(main_list[idx])>1:\n                    main_list[idx]=translate_en_bn(main_list[idx])\n                            \n            except:\n                print("failed -> ",main_list[idx])\n    \n    bn_mlt = " ".join(main_list)\n    bn_mlt = re.sub(\' ্ \',\'\',bn_mlt)\n    bn_mlt = re.sub("\\\\\'","",bn_mlt)#replace \\\'\n    bn_mlt = re.sub(\'<unk>\',\'\',bn_mlt)\n    return bn_mlt\n        \nEN_AR_to_BN_AR_Translator(en_text,tag_text = False)')


# quality of translation looks much better!

# In[20]:


get_ipython().run_cell_magic('time', '', 'tafsir_en["Tafsir_bn"]=tafsir_en.Tafsir.progress_apply(lambda tafsir_eng: EN_AR_to_BN_AR_Translator(tafsir_eng,tag_text = True))\n\n#tafsir_en["Tafsir_bn"]=tafsir_en.Tafsir.parallel_apply(lambda tafsir_eng: EN_AR_to_BN_AR_Translator(tafsir_eng,tag_text = True))\n\n# tafsir_en["Tafsir_bn"] = tafsir_en["Tafsir"]\n# for i in range(5):\n#     tafsir_en["Tafsir_bn"][i] = EN_AR_to_BN_AR_Translator(tafsir_en.Tafsir[i],tag_text = True)')


# In[21]:


tafsir_en["Tafsir_bn"][0]


# In[22]:



tafsir_en.to_csv('/home/ansary/Shabab/tafsir_ibn_kathir/Multilingual_bn_ar_tafsir_ibn_kathir.csv',index = False)


# In[23]:


tafsir_en.head()


# In[ ]:





# In[ ]:




