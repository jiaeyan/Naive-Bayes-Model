# naive_bayes_text_classifier

A classic naive bayes text classifier with add-1 smoothing.

Basically we apply Multinomial Naive Bayes model to classify each input file.

The classifier requires the input data to be a dict (as csv form), whose key is class name and value a list, a list of doc names under this class.
