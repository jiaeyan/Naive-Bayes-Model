import data_provider
import text_classifier

def evaluate():
    '''
    This function evaluates the accuracy of the classifier, with 85 % of the input data set as train set
    and  15 % as the hold-out set. 
    '''
    print "[Process: Evaluating]\n"
    dp = data_provider.DataProvider()
    nbc = text_classifier.NaiveBayesClassifier(dp.get_training_data())
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
    dp = data_provider.DataProvider()
    nbc = text_classifier.NaiveBayesClassifier(dp.get_data_dict())
    result_dict = nbc.classify(dp.get_test_doc_list())
    f = open("result_doc.txt","w")
    for doc in result_dict.keys():
        f.write(doc+": "+result_dict[doc]+"\n")
        print doc+": "+result_dict[doc]
    return result_dict 

if __name__ == "__main__":
    predict()
    input()
