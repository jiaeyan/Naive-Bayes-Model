# Naive Bayes Model

A Multinomial/Bernoulli Naive Bayes classifier with add-0.01 smoothing, implemented with Numpy matrix techniques.

■ The Prior probability matrix, shape = 1 * len(C):
c1|c2|c3|...|Cn
0.1

■ The MLE matrix, shape = len(F) * len(C):
     c1 |c2|c3|...|cn
     
  f1 0.5
  
  f2 0.1
  
  f3 0.1
  
  ...
  
  fm 0.03
  

■ The doc's feature vector, shape = 1 * len(d)

f1|f2|f3|...|fk
1  0  14     9  --> Multinomnial
1  0  1      1  --> Bernoulli
