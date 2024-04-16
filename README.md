# emotion-detection-text
A ML project created to recognise emotions from text using sentiment analysis trained on the twitter dataset and the ISEAR dataset. 
Created during my ML workshop with CIE in PES University.

The backup file is using Models Like Random Forest , SVM , Logistic Regression and a few other models which is trained on the twitter dataset split into train , test and validation.

The Sentimental_analysis_using_word2vec.ipynb file uses Word Embedding with Word2Vec and uses Bi-Directional LSTM to train the model on the ISEAR dataset.

The initial_test.ipynb file uses word embedding with Glove and again uses Bi-Directional LSTM to train the model on the ISEAR dataset


modules that are required to be installed in the system : 
=> pip install numpy
=> pip install matplotlib
=> pip install seaborn
=> pip install nltk
=> pip install scikit-learn
=> pip install tensorflow
=> pip install gensim

Glove requirements : 
=> https://nlp.stanford.edu/projects/glove/
=> Select glove.42B.300d.zip
