import csv
import math
import random

'''
This class feeds all kinds of necessary lists and training data to the classifier.
'''

class DataProvider():
    
    def get_reader(self):
        print "Reading files..."
        return csv.reader(file('filename.csv','rb'))
    
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
    