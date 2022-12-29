
# Mission and Vision

With infinite kindness,mercy and blessings of Allah, we are launching  an open source Islamic book reader system today for everyone that knows/speaks Bangla and arabic. Even though spoken by more than 210 million people as a first or second language,Bangla is still a low resource language. It is also a very difficult language because of its many sounds and spelling rules. Additionally, the script is vastly different from English and other Latin Languages.

The main purpose of making Comprehensive Multilingual Speech synthesis was to reach people through Bengali Hadith and Glorious Quran in the Bengali language. 

# Our Contributions 

  * Collect/Scrape various important bangla-arabic or english-arabic hadith,tafsir and seerah books from the internet and translate english-arabic to bangla-arabic using powerful bangla neural machine translator. you will find our scrapper with comprehensive documentation here : https://github.com/mnansary/hadith-srcapper
  
  * To the best of our knowledge (from our extensive google search and research and extensive human validation) we’ve discovered that the Bangla Vits TTS (text to speech) system that we trained and used for reading various bangla tafsir / hadith is the highest performing State of the Art (SOTA) Bangla neural voice cloning system that’s ever released publicly for Bangla language for free and it beats past TTS systems like gtts,silero-tts,indic-tts by large margin in terms of quality.
  
  * First ever multilingual book reading pipeline that can read Bangla+Arabic code mixed books with ease.
  
  * We read all the books or sources chapter by chapter and made audiobooks.
  
  * performed audiobooks to videobooks conversion using ffmpeg


The entire process may not be 100% accurate. English to Bengali translation may contain errors in many cases, or because it is not read by humans (which is very time-consuming and expensive). It sometimes makes critical pronunciation mistakes as well, but we hope that these problems will be solved by the subsequent improvement of this work InSha'Allah. 

# Training and inference

we used fantastic coqui-ai🐸💬 - toolkit for bangla Text-to-Speech training with IITM dataset converted in ljspeech format. we've trained 4 models and they are : glowtts(male),glowtts(female),vits(male) and vits(female). glowtts didn't perform as well as expected because the coqui-ai used attached vocoder. in order to improve the glowtts performance one need to train spectrogram models and vocoder seperately and used a powerful vocoder instead like hifi gan 2. 
vits male and female variants are our best model that we used for making most of the audiobooks. from this [Comprehensive_Bangla_Text_to_Speech_(TTS)](https://github.com/mobassir94/comprehensive-bangla-tts/blob/main/Comprehensive_Bangla_Text_to_Speech_(TTS).ipynb) demo notebook you can see the sound quality of the vits model is almost as good as the training dataset which can be found here : https://www.kaggle.com/datasets/mobassir/comprehensive-bangla-tts that means e2e vits can clone human voice with high quality and it's attached vocoder is doing enough good job,one way to improve its performance could be to make robust G2P model for bangla and use phonemes.

each directory in this repo contains .txt file describing what that particular folders codes are doing.

for multilingual (bangla+arabic) inference demo you can check this colab tutorial [Multilingual_(ben+ara)_tts_inference_colab_demo.ipynb](https://github.com/mobassir94/comprehensive-bangla-tts/blob/main/mlt_TTS_inference_demo/Multilingual_(ben%2Bara)_tts_inference_colab_demo.ipynb)
and video tutorial of the API version of it is available here

Check out some of the samples generated by our system :

# Multilingual (Bangla+arabic) Audiobooks

|Books|Total_Hadiths/Surahs'|is english to bangla Neural Machine Translated?|Neural Speech synthesized Multilingual (Bangla+arabic) Audiobooks| 
|:---:|:---:|:---:|:---:|
|তাফসীর ইবনে কাসীর|114(surah)|Yes|https://www.youtube.com/playlist?list=PLsHVxzxNumvPOOnpy0om5F8uSm66gEbwF| 
|বাংলা সীরাহঃনবীজি সাল্লাল্লাহু আলাইহি ওয়াসাল্লাম এর জীবনী by  Dr. Yasir Qadhi|101 lectures|Yes|https://www.youtube.com/playlist?list=PLsHVxzxNumvPSbuqcL8oSWoxCPpZ2A3HT| 
|তাফসীরে জাকারিয়া (Tafsir Abu Bakar Zakaria)|114(surah)|No|https://www.youtube.com/playlist?list=PLsHVxzxNumvOintrZMeFFubL5132E72Yl|
|তাফসীরে আহসানুল বায়ান|114(surah)|No|https://www.youtube.com/playlist?list=PLsHVxzxNumvOT0a1ioq5fubqnDAjZ7PVj|
|তাফসীরে জালালাইন (Tafsir AL Jalalain)|114(surah)|Yes|https://www.youtube.com/playlist?list=PLsHVxzxNumvNbYBLhNoAIxw7BaS3yY2XB|
||||
|সহিহ বুখারী|7563|No|https://www.youtube.com/playlist?list=PLsHVxzxNumvNIlU0TjaQaAUAWr9DuZedv| 
|সহিহ মুসলিম|7500|No|https://www.youtube.com/playlist?list=PLsHVxzxNumvOmpGmZKy38RvOYwAWKssDu| 
|সুনানে আন-নাসায়ী|5758|No|https://www.youtube.com/playlist?list=PLsHVxzxNumvNoGSguLsp3ePTR4WOUhZvT|
|সুনানে আবু দাউদ|5274|No|https://www.youtube.com/playlist?list=PLsHVxzxNumvNZ2QPues46JtcRrwK4QF0I|
|জামে' আত-তিরমিজি|3956|No|https://www.youtube.com/playlist?list=PLsHVxzxNumvMb31g0oeJLxmYlufZrYC0X|
|সুনানে ইবনে মাজাহ|4341|No|https://www.youtube.com/playlist?list=PLsHVxzxNumvMqBsYZ1U5Z3uCg1hUF6pA_|
||||
||||
|মুয়াত্তা ইমাম মালিক|1832|No|https://www.youtube.com/playlist?list=PLsHVxzxNumvPn_8D5bn86OTQ9WFRocGGj|
|রিয়াদুস সলেহিন|1905|No|https://www.youtube.com/playlist?list=PLsHVxzxNumvON9GuH8N28YbJiJV0c-abc|
|বুলুগুল মারাম|1568|No|https://www.youtube.com/playlist?list=PLsHVxzxNumvMx11DlgONaLej3IeyVTXig|
|আল লু'লু ওয়াল মারজান|1906|No|https://www.youtube.com/playlist?list=PLsHVxzxNumvOELySX1jhuO2tlzpmnmrvq|
|হাদিস সম্ভার|2013|No|https://www.youtube.com/playlist?list=PLsHVxzxNumvPCCit-aKpSjls4KgabZOhb|
|সিলসিলা সহিহা|60|No|https://www.youtube.com/watch?v=geVWWA8RX3Q&list=PLsHVxzxNumvOsZibj3sZRJxt1uZH_k6n7&index=11|
|জাল জয়িফ হাদিস সিরিজ|102|No|https://www.youtube.com/watch?v=R1CU0AAiB7Y&list=PLsHVxzxNumvOsZibj3sZRJxt1uZH_k6n7|
|মিশকাতুল মাসাবিহ|2758|No|https://www.youtube.com/playlist?list=PLsHVxzxNumvNnnPeAOIhxBWcmlrvblKxb|
|৪০ হাদিস|42|No|https://www.youtube.com/watch?v=ROMcvpPpvoE&list=PLsHVxzxNumvOsZibj3sZRJxt1uZH_k6n7&index=2|
|আদাবুল মুফরাদ|1336|No|https://www.youtube.com/playlist?list=PLsHVxzxNumvNQfGS2d0DQHRX6_TYlnO1m|
|জুজ'উল রাফায়েল ইয়াদাইন|56|No|https://www.youtube.com/watch?v=mQtAo_xEhgs&list=PLsHVxzxNumvOsZibj3sZRJxt1uZH_k6n7&index=9|
|সহিহ হাদিসে কুদসি|163|No|https://www.youtube.com/watch?v=mqUIy6d6UfI&list=PLsHVxzxNumvOsZibj3sZRJxt1uZH_k6n7&index=8|
|১০০ সুসাব্যস্ত হাদিস|101|No|https://www.youtube.com/watch?v=ZBs-ZyI3brw&list=PLsHVxzxNumvOsZibj3sZRJxt1uZH_k6n7&index=3|
|মিশকাতে জয়িফ হাদিস |106|No|https://www.youtube.com/playlist?list=PLsHVxzxNumvOsZibj3sZRJxt1uZH_k6n7|
|শামায়েলে তিরমিযি|320|No|https://www.youtube.com/playlist?list=PLsHVxzxNumvNduizMiAvAVWRUu0UEx2Bw|
|সহিহ তারগিব ওয়াত তাহরিব|200|No|https://www.youtube.com/playlist?list=PLsHVxzxNumvPI5tg5cDoWBCWLQjXcSlBg|
|সহিহ ফাযায়েলে আমল |151|No|https://www.youtube.com/playlist?list=PLsHVxzxNumvMMUpTUsJykeFyWzttWKZyw|
|ঊপদেশ|234|No|https://www.youtube.com/playlist?list=PLsHVxzxNumvPzHcoR1gDHv0fm-6je0GvM|
|রমযান বিষয়ে জাল ও দুর্বল হাদিসসমূহ|36|No|https://www.youtube.com/watch?v=MJi1V7e5ai8&list=PLsHVxzxNumvOsZibj3sZRJxt1uZH_k6n7&index=10|
||||
|মুসনাদে আহমাদ|||
|জুজ'উল কিরাত|||
|সুনান আদ-দারিমী|||
|তাহাবী শরিফ|||
|সুনান দারাকুতনী |||


# issues

* GitHub automatically eliminates html like tags from python code written in jupyter notebook,please check this issue https://github.com/mobassir94/comprehensive-bangla-tts/issues/1
* [grutt](https://github.com/rhasspy/gruut) doesn't have support for bangla. if possible,build a strong G2P model for bangla and it should help improve the performance of our bangla TTS

References : 

1. https://aclanthology.org/2020.lrec-1.789.pdf
2. https://arxiv.org/pdf/2106.06103.pdf 
3. https://arxiv.org/abs/2005.11129

# Acknowledgements
[Apsis Solutions Ltd.](https://apsissolutions.com/)

[bengali.ai](https://bengali.ai/)
