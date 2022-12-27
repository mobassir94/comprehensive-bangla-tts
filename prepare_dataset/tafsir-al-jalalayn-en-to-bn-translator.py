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

import transformers

import torch
import bangla
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer,pipeline
from normalizer import normalize # pip install git+https://github.com/csebuetnlp/normalizer

import warnings
warnings.filterwarnings("ignore")

# from pandarallel import pandarallel
# pandarallel.initialize(progress_bar=True,nb_workers=8)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

print(torch.__version__)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[3]:


get_ipython().system('pwd')


# In[4]:


get_ipython().system('ls')


# In[5]:


tafsir_en = pd.read_csv('/home/ansary/Shabab/tafsir_jalalayn/en_Tafsir al-Jalalayn.csv')
tafsir_en.head()


# In[6]:



print(torch.__version__)
transformers.__version__


# # single sample inference test

# thanks to banglanmt and  meta's nllb-200

# In[7]:


torch_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(torch_device)
model1 = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/banglat5_nmt_en_bn").to(torch_device)
tokenizer1 = AutoTokenizer.from_pretrained("csebuetnlp/banglat5_nmt_en_bn",use_fast=True)



def banglanmt_translate_en_bn(input_sentence):
    input_ids = tokenizer1(normalize(input_sentence), return_tensors="pt").input_ids
    input_ids = input_ids.to(torch_device)
    generated_tokens = model1.generate(input_ids)#max_length=1536
    decoded_tokens = tokenizer1.batch_decode(generated_tokens)[0]
    decoded_tokens=decoded_tokens.replace("<pad>","").replace("</s>","")
    sen=decoded_tokens.split()
    words=[w for w in sen if w.strip()]
    sen="".join(words)
    return decoded_tokens

# !pip install -q transformers

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")


translation_pipeline = pipeline('translation', 
                                model=model, 
                                tokenizer=tokenizer, 
                                src_lang="eng_latn",  #arb_Arab,eng_latn
                                tgt_lang='ben_Beng',
                                device = torch_device,
                                max_length = 1024)

def translate_en_bn(input_sentence):
    sentences = re.split(r'[।!.,?]', input_sentence)
    bn = []
    for i in range(len(sentences)):
        result = translation_pipeline(sentences[i])
        bn.append(result[0]['translation_text'])
    bn = " ".join(bn)
    bn = re.sub(' ্ ','',bn)
    bn = re.sub("\\'","",bn)#replace \'
    bn = re.sub('<unk>','',bn)
    return bn

en_text = "Which was revealed in Makkah"
print("banglanmt -> \n",banglanmt_translate_en_bn(en_text))
print("\nnllb-200 -> \n",)
translate_en_bn(en_text)


# In[8]:


get_ipython().run_cell_magic('time', '', '\nen_text = \'\'\'In the Name of Allah, the Most Gracious, the Most Merciful. <ar> ﴿الْحَمْدُ للَّهِ رَبِّ الْعَـلَمِينَ﴾ </ar>\n(Allah, the Exalted, said, `I have divided the prayer (Al-Fatihah) into two halves between \nMyself and My servant, and My servant shall have what he asks for.\nIf he says,\' <ar> بِسْمِ اللَّهِ الرَّحْمَـنِ الرَّحِيمِ </ar> Allah says, `My servant has praised Me.\' \'\'\'\n\ndef EN_AR_to_BN_AR_Translator(en_text):\n    \'\'\'\n    translates multilingual english-arabic code mixed text into \n    multilingual bengali-arabic code mixed text\n    \'\'\' \n\n    sentenceEnders = re.compile(\'[.,!?]\')\n    sentences = sentenceEnders.split(en_text)\n    main_list = []\n    for i in range(len(sentences)):\n        list_str = sentences[i].split(\'<ar>\')\n        if(len(list_str) == 1):\n            main_list.append(list_str[0])\n        else:\n            for j in range(len(list_str)):\n                if(\'</ar>\' in list_str[j]):\n                    list_str1 = list_str[j].split(\'</ar>\')\n                    main_list.append("<ar>"+list_str1[0]+"</ar>")\n                    main_list.append(list_str1[1])\n                else:\n                    main_list.append(list_str[j])\n\n    while(" " in main_list):\n        main_list.remove(" ")\n        \n    for idx in range(len(main_list)):\n        if(\'<ar>\' not in main_list[idx] or \'</ar>\' not in main_list[idx]):\n            \n            output_sentence = []\n            for word in main_list[idx].split():\n                output_sentence.append(word)\n     \n            main_list[idx] = \' \'.join(output_sentence)\n            #numerizer\n            main_list[idx] = bangla.convert_english_digit_to_bangla_digit(main_list[idx])\n            # multilingual english-arabic to multilingual bengali-arabic\n            try:\n                main_list[idx] = banglanmt_translate_en_bn(main_list[idx])\n            except:\n                main_list[idx] = translate_en_bn(main_list[idx])\n                print("banglanmt failed for -> ",main_list[idx])\n    bn_mlt = " ".join(main_list)\n    bn_mlt = re.sub(\' ্ \',\'\',bn_mlt)\n    bn_mlt = re.sub("\\\\\'","",bn_mlt)#replace \\\'\n    bn_mlt = re.sub(\'<unk>\',\'\',bn_mlt)\n    return bn_mlt\n        \nEN_AR_to_BN_AR_Translator(en_text)')


# quality of translation looks much better!

# In[9]:


get_ipython().run_cell_magic('time', '', 'tafsir_en["Tafsir_bn"]=tafsir_en.en_tafsir_jalalayn.progress_apply(lambda tafsir_eng: EN_AR_to_BN_AR_Translator(tafsir_eng))')


# In[10]:


tafsir_en["Tafsir_bn"][0]


# In[11]:



tafsir_en.to_csv('/home/ansary/Shabab/tafsir_jalalayn/nllb-200_Multilingual_bn_ar_tafsir_al_jalalayn.csv',index = False)


# In[12]:


tafsir_en.head()


# In[ ]:





# In[ ]:




