
# pro tip : before bothering with cross-validation, check to see if the algorithm is able to fit your training data as well as you like

# we will try dimensionality reduction techniques to see if log-reg's train data performance can be improved
# we will use hyper-optimization to pick DRT parameters 

# .6914 is standard for a single dataset
# .6915 is standard for all data, one set actually came back .6931!

# numpy vs. pandas : linear algebra speed

# first things first : how will we benchmark?

# want to test L1 and L2 regularization? Later?

# 1.) 
# 10-fold cross-validation on each individual dataset

# 2.) 
# use remaining 7 to predict the eighth

# 3.)
# use 10 fold CV for 7.75 datasets to predict the last .25 chunk 



# 1.)
# for each folder in Dropbox/numerai
# import training data
# 10-fold CV using logistic regression
# report average training and validatione error 

import os
import numpy
import sklearn
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import ExtraTreesRegressor


model = LogisticRegression(C = 1000000, max_iter = 100000)

train_log_losses = []

data_directory = '/home/Damian/Dropbox/numerai'

weeks = os.listdir(data_directory)

all_weeks_data = []

combined_data = None

'''
data = numpy.genfromtxt("/home/damian/Dropbox/numerai/Week_8/numerai_training_data.csv", delimiter = ",", skip_header = 1)
predictors = data[:,0:21]
target = data[:,21]
'''
'''
kernel_types = ['linear','poly','rbf','sigmoid']

for week in weeks:
	week_index += 1
	week_path = data_directory + "/" + week + "/" + "numerai_training_data.csv"
	data = numpy.genfromtxt(week_path, delimiter = ",", skip_header = 1)
	predictors = data[:,0:21]
	target = data[:,21]	
	
	for kernel in kernel_types:
		for num_components in range(1,22):	
			pca = KernelPCA(n_components = num_components, kernel = kernel)
			transformed_predictors = pca.fit_transform(predictors)
			fitted = model.fit(transformed_predictors, target)
			probs = fitted.predict_proba(transformed_predictors)
			probs_1 = probs[:,1]
			LL = sklearn.metrics.log_loss(target,probs_1)
			print "data set",week_index,"training error with", num_components,"components and a",kernel,"kernel:", LL
'''
week_9 = numpy.genfromtxt('/home/Damian/Dropbox/numerai/Week_9/numerai_training_data.csv', delimiter = ',', skip_header = 1)
week_9_predictors = week_9[:,0:21]
week_9_target = week_9[:,21]


week_index = 0
split_index = 0
for week in weeks:
	#week_index += 1
	week_path = data_directory + "/" + week + "/" + "numerai_training_data.csv"
	data = numpy.genfromtxt(week_path, delimiter = ",", skip_header = 1)
	if week == "Week_9":
		continue 	
	all_weeks_data.append(data)
	if combined_data is None:
		combined_data = data
	else:
		combined_data = numpy.concatenate([combined_data,data])
predictors = combined_data[:,0:21]
target = combined_data[:,21]

'''
fitted = model.fit(predictors,target)
probs = fitted.predict_proba(predictors)
probs_1 = probs[:,1]
'''
extra = sklearn.ensemble.ExtraTreesRegressor(n_jobs = -1)
extra.fit(predictors, target)
extra_test_preds = extra.predict_proba(week_9_predictors)
extra_test_LL = sklearn.metrics.log_loss(week_9_target, extra_test_preds)
extra_train_preds = extra.predict(predictors)	
extra_train_LL = sklearn.metrics.log_loss(target,extra_train_preds)
print "Extra Trees Train Loss", extra_train_LL, "\n"
print "Extra Trees Train Loss", extra_test_LL, "\n"

'''
for week in weeks:
	data = all_weeks_data[week_index]
	week_index += 1
	predictors = data[:,0:21]
	target = data[:,21]
	probs = fitted.predict_proba(predictors)
	probs_1 = probs[:,1]
	LL = sklearn.metrics.log_loss(target,probs_1)	
	print "Week",week_index,"Logarithmic Loss:",LL
'''
'''
	predictors = data[:,0:21]
	target = data[:,21]
	fitted = model.fit(predictors,target)
	probs = fitted.predict_proba(predictors)
	probs_1 = probs[:,1]
	LL = sklearn.metrics.log_loss(target,probs_1)	
	train_log_losses.append(LL)
	print "Week",week_index,"Logarithmic Loss:",LL
	train_val_splits = KFold(n = len(target), n_folds = 10, shuffle = True)	
	for train_index, val_index in train_val_splits:
		split_index += 1
		predictor_train, predictor_val = predictors[train_index], predictors[val_index]
		target_train, target_val = target[train_index], target[val_index]
'''			

