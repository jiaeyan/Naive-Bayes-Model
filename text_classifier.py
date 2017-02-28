# Jiajie (Sven) Yan
# svenyan234@gmail.com

import csv,nltk,math,re,random,string;

'''
Basically we apply Multinomial Naive Bayes model to classify each file.

The classifier requires the input data to be a dict, whose key is class name and value a list,
a list of doc names under this class.

'''

class NaiveBayesClassifier():
    
    def __init__(self,training_data):
        self.data_dict = training_data
        self.stopwords = open("stopwords.txt","r").read().split()
      
    def text_tokenizer(self,doc):
        return [word.lower() for word in re.compile(r'\W+').split(open(doc+".txt","r").read()) 
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

class DataProvider():
    '''
    This class feeds all kinds of necessary lists and training data to the classifier.
    '''
    
    def get_reader(self):
        print "Reading files..."
        return csv.reader(file('Interview_Mapping.csv','rb'))
    
    def get_data_dict(self):  # this is the whole data set, containing 899 docs
        data_dict = {}
        for line in self.get_reader():
            if line[1] != "To be Tested" and line[1]!="Area.of.Law":
                if line[1] not in data_dict.keys():
                    data_dict[line[1]] = []
                data_dict[line[1]].append(line[0])
        return data_dict
    
    def get_doc_tag_dict(self):  # this is a dict whose key is doc and value its tag/class
        doc_tag_dict = {}           
        for line in self.get_reader():
            if line[1] != "To be Tested" and line[1] != "Area.of.Law":
                doc_tag_dict[line[0]] = line[1]
        return doc_tag_dict            
                    
    def get_test_doc_list(self): # this is the to be tested doc list, containing 100 docs.
        test_doc_list = []
        for line in self.get_reader():
            if line[1] == "To be Tested":
                test_doc_list.append(line[0])
        return test_doc_list        
                
    def get_tested_doc_list(self): # this is a list of 899 tested docs.
        tested_doc_list = []
        for line in self.get_reader():
            if line[1] != "To be Tested" and line[1] != "Area.of.Law":
                tested_doc_list.append(line[0])
        random.shuffle(tested_doc_list)
        return tested_doc_list
    
    def get_train_list(self): # this is a train list from sub-tested_doc_list for training
        return self.get_tested_doc_list()[0:int(math.floor(0.85*len(self.get_tested_doc_list())))]
    
    def get_test_list(self): # this is a hold-out list from sub-tested_doc_list for testing
        return self.get_tested_doc_list()[int(math.floor(0.85*len(self.get_tested_doc_list()))):]
    
    def get_training_data(self): # this function provides a traing data, containing 85 % of the given data set
        training_data = {}
        doc_tag_dict = self.get_doc_tag_dict()
        for doc in self.get_train_list():
            if doc_tag_dict[doc] not in training_data.keys():
                training_data[doc_tag_dict[doc]] = []
            training_data[doc_tag_dict[doc]].append(doc)
        return training_data

def evaluate():
    '''
    This function evaluates the accuracy of the classifier, with 85 % of the input data set as train set
    and  15 % as the hold-out set. 
    '''
    print "[Process: Evaluating]\n"
    dp = DataProvider()
    nbc = NaiveBayesClassifier(dp.get_training_data())
    result_dict = nbc.classify(dp.get_test_list())
    count = 0
    for doc in result_dict.keys():
        if result_dict[doc] == dp.get_doc_tag_dict()[doc]:
            count += 1
    return "Accuracy: " + str(float(count)/float(len(result_dict)))

def predict():
    '''
    This function predicts the to-be-tested docs and classify them with given data set.
    '''
    print "[Process: Predicting]\n"
    dp = DataProvider()
    nbc = NaiveBayesClassifier(dp.get_data_dict())
    result_dict = nbc.classify(dp.get_test_doc_list())
    f = open("result_doc.txt","w")
    for doc in result_dict.keys():
        f.write(doc+": "+result_dict[doc]+"\n")
        print doc+": "+result_dict[doc]
    return result_dict 

#print evaluate()
predict()
input()
