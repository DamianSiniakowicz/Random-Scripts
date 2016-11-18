# should i be using the classifier or regressor version of the model

# Goal : train a super-learner on every single algorithm in scikit-learn

# bash scripts for automating the stuff I do over and over again

# goal for the day : train a neon deep NN super learner using all sckit ensemble learners as base learners. Ensemble learners in turn use trees as base learners

# later must hyperoptimize each model

# import latest data set

# why do all 5 algorithms have the same training loss, but different test losses?

import sklearn.ensemble, sklearn.metrics, sklearn.svm
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
week_10 = numpy.genfromtxt('/home/damian/Dropbox/numerai/Week_10/numerai_training_data.csv', delimiter = ',', skip_header = 1)
end = len(week_10) 
'''
cum_adaboost_train = 0
cum_adaboost_test = 0
cum_bagging_train = 0
cum_bagging_test = 0
cum_gradient_train = 0
cum_gradient_test = 0
cum_extra_train = 0
cum_extra_test = 0
cum_random_train = 0
cum_random_test = 0
'''

# bagged Neural Networks

# we're gonna do something with random forests

# which parameters of RF can be optimized?
# int: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes, 
# other: criterion, max_features, class_weight

# which package(s) will we use to optimize them?

runs = 5

cumulative_train_error = 0
cumulative_test_error = 0

for run in range(runs):
	run_id = run + 1
	numpy.random.shuffle(week_10)
	train_predictors = week_10[0:80000,0:21]
	train_target = week_10[0:80000,21]
	train_target.astype(int)
	test_predictors = week_10[80000:end,0:21]
	test_target = week_10[80000:end,21]
	test_target.astype(int)
	print "\n","\n","!?!?!??!?!??!?!?!??!?!??!?!??!?!??!?!??!?!??!?!?!?","\n","ROUND",run_id,"!?!?!?!?!!!?!?!??!?!?!?!?!??!?!?!?!??!??!?!?!","\n"

	RF_machine = RandomForestClassifier(max_depth = 15,n_estimators = 500,oob_score = True, n_jobs = 2, verbose = 1)
	RF_machine.fit(train_predictors, train_target)
	train_predictions = RF_machine.predict_proba(train_predictors)
	train_LL = sklearn.metrics.log_loss(train_target, train_predictions)	
	test_predictions = RF_machine.predict_proba(test_predictors)
	test_LL = sklearn.metrics.log_loss(test_target, test_predictions)

	cumulative_train_error += train_LL
	cumulative_test_error += test_LL

mean_train_error = cumulative_train_error / runs
mean_test_error = cumulative_test_error / runs

print mean_train_error, mean_test_error


# notes on Adaboost

# 'strong' prediction is a linear combination of 'weak' predictions'
# the features of each observation are the hypotheses of the weak train_predictors
# subsequent learners focus on observations least-understood by previous learners
# examples of weak learners : logistic regression, decision trees, NN's, anything really

'''
	####### 1 : Adaboost with Logistic Regression

	#TE : 1.something
	# keeps giving me 0 and 1's as output, dunno wtf is up
	ada = sklearn.ensemble.AdaBoostRegressor(LogisticRegression(C = 1, max_iter = 10000),n_estimators = 50, loss = 'linear')
	ada.fit(train_predictors,train_target)
	preds = ada.predict(train_predictors)
	print preds[0:100]
	LL = sklearn.metrics.log_loss(train_target,preds)
	print LL
	
	###### 2 : AdaBoost Regressor with Decision Trees
	# TE : .689
	ada = sklearn.ensemble.AdaBoostRegressor(DecisionTreeRegressor(max_depth = 7),n_estimators = 100)
	ada.fit(train_predictors, train_target)
	ada_test_preds = ada.predict(test_predictors)	
	ada_test_LL = sklearn.metrics.log_loss(test_target,ada_test_preds)
	ada_train_preds = ada.predict(train_predictors)	
	ada_train_LL = sklearn.metrics.log_loss(train_target,ada_train_preds)
	print "AdaBoost Train Loss", ada_train_LL, "\n"	
	print "AdaBoost Test Loss", ada_test_LL, "\n"
	cum_adaboost_train += ada_train_LL
	cum_adaboost_test += ada_test_LL	
	
	###### 1337 : Logistic Regression
	
	logreg = LogisticRegression(C=10000, max_iter = 1000, n_jobs = -1)
	logreg.fit(train_predictors, train_target)
	logreg_test_preds = logreg.predict_proba(test_predictors)
	logreg_test_preds = logreg_test_preds[:,1]
	logreg_test_LL = sklearn.metrics.log_loss(test_target,logreg_test_preds)
	logreg_train_preds = logreg.predict_proba(train_predictors)
	logreg_train_preds = logreg_train_preds[:,1]	
	logreg_train_LL = sklearn.metrics.log_loss(train_target,logreg_train_preds)
	print "log reg Train Loss", logreg_train_LL, "\n"	
	print "log reg Test Loss", logreg_test_LL, "\n"

	###### 3 : Bagging Regressor with Decision Trees
	# TE : .215
	bag = sklearn.ensemble.BaggingRegressor()
	bag.fit(train_predictors, train_target)
	bag_test_preds = bag.predict(test_predictors)
	bag_test_LL = sklearn.metrics.log_loss(test_target, bag_test_preds)
	bag_train_preds = bag.predict(train_predictors)	
	bag_train_LL = sklearn.metrics.log_loss(train_target,bag_train_preds)
	print "Bagging Train Loss", bag_train_LL, "\n"
	print "Bagging Train Loss", bag_test_LL, "\n"
	cum_bagging_train += bag_train_LL
	cum_bagging_test += bag_test_LL

	###### 4 : Gradient Boosting Regressor with Decision Trees
	# TE : .686
	grad = sklearn.ensemble.GradientBoostingRegressor()
	grad.fit(train_predictors, train_target)
	grad_test_preds = grad.predict(test_predictors)
	grad_test_LL = sklearn.metrics.log_loss(test_target, grad_test_preds)
	grad_train_preds = grad.predict(train_predictors)	
	grad_train_LL = sklearn.metrics.log_loss(train_target, grad_train_preds)
	print "Gradient Boosting Train Loss", grad_train_LL, "\n"
	print "Gradient Boosting Train Loss", grad_test_LL, "\n"
	cum_gradient_train += grad_train_LL
	cum_gradient_test += grad_test_LL

	###### 5 : Extra Trees Regressor  
	# TE : basically 0
	extra = sklearn.ensemble.ExtraTreesRegressor(n_jobs = -1)
	extra.fit(train_predictors, train_target)
	extra_test_preds = extra.predict(test_predictors)
	extra_test_LL = sklearn.metrics.log_loss(test_target, extra_test_preds)
	extra_train_preds = extra.predict(train_predictors)	
	extra_train_LL = sklearn.metrics.log_loss(train_target,extra_train_preds)
	print "Extra Trees Train Loss", extra_train_LL, "\n"
	print "Extra Trees Train Loss", extra_test_LL, "\n"
	cum_extra_train += extra_train_LL
	cum_extra_test += extra_test_LL	

	###### 6 : Random Forest Regressor
	# TE : .215
	randomf = sklearn.ensemble.RandomForestRegressor(n_jobs = -1)
	randomf.fit(train_predictors, train_target)
	randomf_test_preds = randomf.predict(test_predictors)
	randomf_test_LL = sklearn.metrics.log_loss(test_target, randomf_test_preds)
	randomf_train_preds = randomf.predict(train_predictors)	
	randomf_train_LL = sklearn.metrics.log_loss(train_target,randomf_train_preds)
	print "Random Forest Train Loss", randomf_train_LL, "\n"
	print "Random Forest Train Loss", randomf_test_LL, "\n"
	cum_random_train += randomf_train_LL
	cum_random_test += randomf_test_LL
'''
'''
	###### 7 : Random Trees Embedding
	# TE : not a prediction algorithm. This is a preprocessing algorithm
	bag = sklearn.ensemble.RandomTreesEmbedding(n_jobs = -1)
	bag.fit(train_predictors, train_target)
	preds = bag.predict(train_predictors)
	LL = sklearn.metrics.log_loss(train_target, preds)
	print LL
'''

###### 8 : SVM
	

'''
cum_adaboost_train /= 10
cum_adaboost_test /= 10
cum_bagging_train /= 10
cum_bagging_test /= 10
cum_gradient_train /= 10
cum_gradient_test /= 10
cum_extra_train /= 10
cum_extra_test /= 10
cum_random_train /= 10
cum_random_test /= 10

print "MEAN TRAIN AND TEST LOSSES", "\n"
print "mean adaboost train loss", cum_adaboost_train, "\n"
print "mean adaboost test loss", cum_adaboost_test, "\n"
print "mean bagging train loss", cum_bagging_train, "\n"
print "mean bagging test loss", cum_bagging_test, "\n"
print "mean gradient train loss", cum_gradient_train, "\n"
print "mean gradient test loss", cum_gradient_test, "\n"
print "mean extra train loss", cum_extra_train, "\n"
print "mean extra test loss", cum_extra_test, "\n"
print "mean random train loss", cum_random_train, "\n"
print "mean random test loss", cum_random_test, "\n"
'''









