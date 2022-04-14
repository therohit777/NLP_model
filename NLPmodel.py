import torch
from transformers import T5Tokenizer,T5ForConditionalGeneration,T5Config
from langdetect import detect
from textblob import TextBlob


class Abstractive_Summarizer:
    
    def __init__(self):
        #intialising the pretrained model
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.device = torch.device('cpu')
    
    def main(self,abs_txt):  
        #input Text
        text = abs_txt
        
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
        
        #Summarizing and converting the article back to its original language.
        tokenized_text =  self.tokenizer.encode(t5_input_text,return_tensors='pt').to(self.device)
        summary_ids = self.model.generate(tokenized_text,min_length=30,max_length=120)
        summary = self.tokenizer.decode(summary_ids[0],skip_special_tokens=True)
        if(lang_detect!="en"):
            blob2 = TextBlob(summary)
            Final_summary = blob2.translate(to=lang_detect)
        else:
            Final_summary=summary
        return [len(text.split()),Final_summary,len(Final_summary.split())]


n=input("Enter the Article: ")
p = Abstractive_Summarizer()
k = p.main(n)
print(k)



