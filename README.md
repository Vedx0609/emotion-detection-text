# emotion-detection-text
A ML project created to recognise emotions from text using sentiment analysis trained on the twitter dataset and the ISEAR dataset. 
Created during my ML workshop with CIE in PES University.

The backup file is using Models Like Random Forest , SVM , Logistic Regression and a few other models which is trained on the twitter dataset split into train , test and validation.

The Sentimental_analysis_using_word2vec.ipynb file uses Word Embedding with Word2Vec and uses Bi-Directional LSTM to train the model on the ISEAR dataset.

The initial_test.ipynb file uses word embedding with Glove and again uses Bi-Directional LSTM to train the model on the ISEAR dataset

modules that are required to be installed in the system 
1) pip install numpy
2) pip install matplotlib
3) pip install seaborn
4) pip install nltk
5) pip install scikit-learn
6) pip install tensorflow
7) pip install gensim

Glove requirements : 
1) https://nlp.stanford.edu/projects/glove/
2) Select glove.42B.300d.zip
