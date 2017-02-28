import nltk
import math
import re
import string
import data_provider

'''
Basically we apply Multinomial Naive Bayes model to classify each file.

The classifier requires the input data to be a dict (cvs form), whose key is class name and value a list,
a list of doc names under this class.

'''

class NaiveBayesClassifier():
    
    def __init__(self,training_data):
        self.data_dict = training_data
        self.stopwords = open("stopwords.txt","r").read().split()
      
    def text_tokenizer(self,doc):
        return [word.lower() for word in re.compile(r'\W+').split(open("file_directory/"+doc+".txt","r").read()) 
                if word.isalpha() if word.lower() not in self.stopwords if word not in string.letters]
    
    def get_tag_token_list(self,tag): # this is a token_list of each tag/class
        tag_token_list = []
        for doc in self.data_dict[tag]:
            tag_token_list += self.text_tokenizer(doc)
        return tag_token_list
    
    def get_total_token_list(self): # this is a token_list of total token in the whole given data set
        total_token_list = []
        for key in self.data_dict.keys():
            total_token_list += self.get_tag_token_list(key)
        return total_token_list
    
    def get_tag_prob(self,tag,tag_token_list,total_token_list): # this is the probability of class, P(C)
        return float(len(tag_token_list)) / float(len(total_token_list))
    
    def get_freq_dict(self,tag,tag_token_list): # this is a dict recording frequency of each word type under one class
        return nltk.FreqDist(tag_token_list)
    
    def get_parameter(self): # this method returns a dict, which contains all necessary info about the given data set
        print "Setting parameters..."
        model_parameter = {}
        total_token_list = self.get_total_token_list()
        model_parameter["total_token_num"] = float(len(total_token_list))
        model_parameter["total_type_num"] = float(len(set(total_token_list)))
        for tag in self.data_dict.keys():
            model_parameter[tag] = {}
            temp_dict=model_parameter[tag]
            temp_dict["tag_token_list"] = self.get_tag_token_list(tag)
            temp_dict["tag_token_num"] = float(len(temp_dict["tag_token_list"]))
            temp_dict["tag_prob"] = self.get_tag_prob(tag, temp_dict["tag_token_list"], total_token_list)
            temp_dict["tag_freq_dict"] = self.get_freq_dict(tag,temp_dict["tag_token_list"])
            temp_dict["cond_prob"] = temp_dict["tag_prob"]
        return model_parameter
    
    def classify(self,doc_list): # this method tests given doc_list and tells each one's class
        result_dict = {}
        model_parameter = self.get_parameter()
        for doc in doc_list:
            print "Testing "+doc
            compare_dict = {}
            for tag in self.data_dict.keys():
                cond_prob = math.log(model_parameter[tag]["cond_prob"])
                tag_freq_dict = model_parameter[tag]["tag_freq_dict"]
                tag_vol = model_parameter[tag]["tag_token_num"] + model_parameter["total_type_num"]
                for token in self.text_tokenizer(doc):
                        cond_prob += math.log(float(tag_freq_dict.get(token,0.0)+1.0)/tag_vol)
                compare_dict[tag] = cond_prob
            result_dict[doc] = max(compare_dict, key=lambda k: compare_dict[k])
            print "OK"
        return result_dict 
