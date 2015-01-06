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

#Calculate the classifier's Accuracy, Precision, Recall, and F1 score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
accuracy = accuracy_score(Labels_test, predictions)
print 'Accuracy: ', accuracy

precisions = precision_score(Labels_test, predictions, average = None)
print 'Precision: ', np.mean(precisions), precisions

recalls = recall_score(Labels_test, predictions, average = None)
print 'Recall: ', np.mean(recalls), recalls

f1scores = f1_score(Labels_test, predictions, average = None)
print 'F1Scores: ', np.mean(f1scores), f1scores

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

#Use grid search to find best hyperparameters for the classifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
pipeline = Pipeline([('vect', TfidfVectorizer(stop_words='english')), ('clf', LogisticRegression())])
parameters = {
'vect__max_df' : (0.25, 0.5, 0.75),
'vect__stop_words' : ('english', None),
'vect__max_features' : (2500, 5000, 10000, None),
'vect__ngram_range' : ((1,1,), (1,2)),
'vect__use_idf' : (True, False),
'vect__norm' : ('l1', 'l2'),
'clf__penalty' : ('l1', 'l2'),
'clf__C' : (0.01, 0.1, 1, 10)
}
#GridSearchCV() takes an estimator, a parameter space, and performance measure.
#The argument n_jobs specifies max number of concurrent jobs; set it to -1 
#to use all CPU cores. fit() must be called in the main block in order to 
# fork additional processes.
if __name__ == "__main__":
	grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='accuracy', cv=3)
	Features_train, Features_test, Labels_train, Labels_test = train_test_split(df[1],df[0])	
	grid_search.fit(Features_train, Labels_train)
	print 'Best score: %0.3f' %grid_search.best_score_
	print 'Best parameters set:'
	best_parameters = grid_search.best_estimator_.get_params()
	for param_name in sorted(parameters.keys()):
		print '\t%s: %r' %(param_name, best_parameters[param_name])
	predictions = grid_search.predict(Features_test)
	print 'Accuracy: ', accuracy_score(Labels_test, predictions)
	print 'Precision: ', precision_score(Labels_test, predictions, average = None)
	print 'Recall: ', recall_score(Labels_test, predictions, average = None)
