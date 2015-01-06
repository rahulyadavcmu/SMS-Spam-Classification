#Read in the data file using pandas
import pandas as pd
df = pd.read_csv('smsspamcollection/SMSSpamCollection', delimiter='\t', header=None)
# print df.head()
# print 'Number of spam messages:', df[df[0] == 'spam'][0].count()
# print 'Number of ham messages:', df[df[0] == 'ham'][0].count()

#Split the data into training and test sets
from sklearn.cross_validation import train_test_split
Features_train_raw, Features_test_raw, Labels_train, Labels_test = train_test_split(df[1],df[0])

#Create a TfidfVectorizer, fit it with the training messages
#and transform both the training and test messages 
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
Features_train = vectorizer.fit_transform(Features_train_raw)
Features_test = vectorizer.transform(Features_test_raw)

#Create a model and train it
from sklearn.linear_model.logistic import LogisticRegression
classifier = LogisticRegression()
classifier.fit(Features_train, Labels_train)
predictions = classifier.predict(Features_test)
# for i, prediction in enumerate(predictions[:5]):
# 	print 'Prediction: %s. Message: %s' %(prediction, Features_test_raw[i])

#Calculate the classifier's accuracy
from sklearn.cross_validation import cross_val_score
import numpy as np
scores = cross_val_score(classifier, Features_train, Labels_train, cv=5)
print 'Scores', np.mean(scores), scores

#Calculate the classifier's Precision, Recall, and F1 score
from sklearn.metrics import precision_score, recall_score, f1_score
precisions = precision_score(Labels_test, predictions, average = None)
print 'Precision', np.mean(precisions), precisions

recalls = recall_score(Labels_test, predictions, average = None)
print 'Recall', np.mean(recalls), recalls

f1scores = f1_score(Labels_test, predictions, average = None)
print 'F1Scores', np.mean(f1scores), f1scores

#Plot the ROC curve for the classifier
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc
# false_positive_rate, recall, thresholds = roc_curve(Labels_test, predictions[:,1])
# roc_auc = auc(false_positive_rate, recall)
# plt.title('Receiver Operating Characteristic')
# plt.plot(false_positive_rate, recall, 'b', label='AUC=%0.2f' %roc_auc)
# plt.legend(loc='lower right')
# plt.plot([0,1], [0,1], 'r--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
# plt.ylabel('Recall')
# plt.xlabel('Fall-out')
# plt.show()
