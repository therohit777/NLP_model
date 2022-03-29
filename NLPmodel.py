import torch
from transformers import T5Tokenizer,T5ForConditionalGeneration,T5Config
from langdetect import detect
from textblob import TextBlob




#intialising the pretrained model
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')


#input Text
text = input("Enter your Article: ")


#Language Detection
lang_detect = detect(text)


#Language Translation
blob = TextBlob(text)
if(lang_detect!='en'):
    lang_translate = blob.translate(to='en')
else:
    lang_translate = text
lang_translate1 = str(lang_translate)





#Preprocessing input text.
preprocessed_text =  lang_translate1.strip().replace('\n','')
t5_input_text = preprocessed_text
print("Number of words before summarization: ",len(text.split()))





#Bugs to be fixed.
tokenized_text =  tokenizer.encode(t5_input_text,return_tensors='pt').to(device)
summary_ids = model.generate(tokenized_text,min_length=30,max_length=120)
summary = tokenizer.decode(summary_ids[0],skip_special_tokens=True)
if(lang_detect!="en"):
    blob2 = TextBlob(summary)
    Final_summary = blob2.translate(to=lang_detect)
else:
    Final_summary=summary
print("Summarized:", Final_summary )
print("Number of words After summarization: ",len(Final_summary.split()))  


