# -*- coding: utf-8 -*-

# The aim of the script is to:
# 1. Load Kaggle dataset with credit information of a sample of borrowers
# 2. Visualize the data, get a better understanding of its characteristics
# 3. Resample the dataset to account for an imbalance present (much less defaults than no defualts)
# 4. Fit the Logistic Regression model
# 5. Inspect the model fit and other classification metrics (precision, recall, confusion matrix etc)

# SOURCES:
# DATA SOURCE
# KAGGLE - Credit Risk Dataset
# https://www.kaggle.com/datasets/laotse/credit-risk-dataset?resource=download
# More on the oversampling in case of an imbalanced dataset
# https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/

###################################################################################################
# LOAD PACKAGES
###################################################################################################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import statsmodels.api as sm
from imblearn.over_sampling import SMOTE
###################################################################################################
# IMPORT DATA
###################################################################################################
data=pd.read_csv('C:\\Users\\macie\\OneDrive\\Desktop\\ML & DS\\credit risk\\credit_risk_dataset.csv')
data = data.dropna()
# Print basic information regarding a dataset
print(data.shape)
print(list(data.columns))
# Check the values in different columns
print(data['person_home_ownership'].unique())
print(data['loan_intent'].unique())
print(data['loan_grade'].unique())
print(data['loan_status'].unique())
print(data['cb_person_default_on_file'].unique())
# Count the number of defaults and no defualts
print(data['loan_status'].value_counts())
###################################################################################################
# VISUALIZE
###################################################################################################
plt.rc('font', size=14)
sns.set(style='white')
sns.set(style='whitegrid', color_codes=True)
# Dependent variable - a loan status (default / no-default)
sns.countplot(x='loan_status', data=data, palette='hls')
plt.show()
# Count the number of defaults and no defualts
count_no_default = len(data[data['loan_status'] == 0])
count_default = len(data[data['loan_status'] == 1])
pct_of_no_default = count_no_default / (count_no_default + count_default)
print('\033[1m percentage of no default is', pct_of_no_default * 100)
pct_of_default = count_default/(count_no_default+count_default)
print('\033[1m percentage of default', pct_of_default*100)
# Print data grouped by different categories
print(data.groupby('loan_status').mean())
print(data.groupby('person_home_ownership').mean())
# Plot the distribution of interest rates, split by the loan status
# One can observae the defaulted loans tend to have generally a higher rate 
sns.set(style='white')
sns.kdeplot( data['loan_int_rate'].loc[data['loan_status'] == 0], hue=data['loan_status'], shade=True)
sns.kdeplot( data['loan_int_rate'].loc[data['loan_status'] == 1], hue=data['loan_status'], shade=True)
# Visualize a proportion of defaults/no defaults split by a Home Ownership status
table=pd.crosstab(data.person_home_ownership,data.loan_status)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Home Ownership vs Default')
plt.xlabel('Home Ownership')
plt.ylabel('Proportion of Applicants')
plt.savefig('Home Ownership_vs_def_stack')
# Visualize a proportion of defaults/no defaults split by a Loan Grade
table=pd.crosstab(data.loan_grade,data.loan_status)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of loan_grade vs Default')
plt.xlabel('loan_grade')
plt.ylabel('Proportion of Applicants')
plt.savefig('loan_grade_vs_def_stack')
# Visualize a proportion of defaults/no defaults split by a Loan Intent
table=pd.crosstab(data.loan_intent,data.loan_status)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of loan_intent vs Default')
plt.xlabel('loan_intent')
plt.ylabel('Proportion of Applicants')
plt.savefig('loan_intent_vs_def_stack')
###################################################################################################
# ADD DUMMIES, REBALANCES THE DATASET
###################################################################################################
# Create dummy variables (0-1) as separate columns
# Ther dummies are intended to replace the columns with categorical values (needed for further fit of Logistic Regression)
cat_vars=['person_home_ownership', "loan_intent", "loan_grade", "cb_person_default_on_file"]
for var in cat_vars:
    cat_list='var'+ '_' + var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]
data_final=data[to_keep]
data_final.columns.values
# Over-sampling with SMOTE (Synthetic Minority Oversampling Technique)
# 1. Take the existing training dataset
# 2. Add more default observations using SMOTE:
    # a. Create a synthetic samples from 'defaults' 
    # b. Randomly choose one of the k-nearest-neighbors and using it to create a similar, but randomly tweaked, new observations
X = data_final.loc[:, data_final.columns != 'loan_status']
y = data_final.loc[:, data_final.columns == 'loan_status']
os = SMOTE(random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
columns = X_train.columns
os_data_X,os_data_y = os.fit_resample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y = pd.DataFrame(data=os_data_y,columns=['loan_status'])
# We can check the numbers of our data
print('\033[1m length of oversampled data is ', len(os_data_X))
print('\033[1m Number of no default in oversampled data', len(os_data_y[os_data_y['loan_status'] == 0]))
print('\033[1m Number of default', len(os_data_y[os_data_y['loan_status'] == 1]))
print('\033[1m Proportion of no default data in oversampled data is ', len(os_data_y[os_data_y['loan_status'] == 0]) / len(os_data_X))
print('\033[1m Proportion of default data in oversampled data is ', len(os_data_y[os_data_y['loan_status'] == 1]) / len(os_data_X))
# Recursive Feature Elimination (RFE):
    # 1. Construct a model in a repetetive fashion
    # 2. Choose either the best or worst performing feature, setting the feature aside 
    # 3. Repeate the process with the rest of the features. 
    # 4. Apply it until all features in the dataset are exhausted. 
# The goal of RFE is to select features by recursively considering declining sets of features
data_final_vars = data_final.columns.values.tolist()
y = ['loan_status']
X = [i for i in data_final_vars if i not in y]
# Define a Logistic model
logreg = LogisticRegression()
# Run RFE algorithm - right now select 15 'best' variables in terms of Importance (number 10 select by trial and error method, checking the p-values of variables in the regression output)
rfe = RFE(logreg, n_features_to_select = 15)
# Fit the model
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
# print out support and ranking of each variables in X set
print(rfe.support_)
print(rfe.ranking_)
os_data_X.columns
data_X1 = pd.DataFrame({
 'Feature': os_data_X.columns,
 'Importance': rfe.ranking_},)
# Print out variables sorted by Importance measure
data_X1.sort_values(by=['Importance'])
cols=[]
for i in range (0, len(data_X1['Importance'])):
 if data_X1['Importance'][i] == 1:
     cols.append(data_X1['Feature'][i])
print(cols)
print(len(cols))
###################################################################################################
# MODELLING
###################################################################################################
X=os_data_X[cols]
y=os_data_y['loan_status']
# Fit the model
# All of the vars are significant
logit_model = sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())
# Print out the p-values for confirmation
pvalue = pd.DataFrame(result.pvalues,columns={'p_value'},)
pvalue
pvs=[]
for i in range (0, len(pvalue['p_value'])):
    if pvalue['p_value'][i] < 0.05:
        pvs.append(pvalue.index[i])
if 'const' in pvs:
    pvs.remove('const')
else:
    pvs 
print(pvs)
print(len(pvs))
# Fit the model only with the variables whose p-value < 5%
X=os_data_X[pvs]
y=os_data_y['loan_status']
logit_model=sm.Logit(y,X)
result=logit_model.fit()
# Logistic Regression Model Fitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print(result.summary())
###################################################################################################
# EVALUATION
#################################################################################################### 
# Accuracy
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(accuracy_score(y_test, y_pred)))

# Confusion Matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print('\033[1m The result is telling us that we have: ',(confusion_matrix[0,0]+confusion_matrix[1,1]),'correct predictions\033[1m')
print('\033[1m The result is telling us that we have: ',(confusion_matrix[0,1]+confusion_matrix[1,0]),'incorrect predictions\033[1m')
print('\033[1m We have a total predictions of: ',(confusion_matrix.sum()))

print(classification_report(y_test, y_pred))
# ROC Curve
# The receiver operating characteristic (ROC) curve is another common tool used with binary classifiers. 
# The dotted line represents the ROC curve of a purely random classifier; 
# a good classifier stays as far away from that line as possible (toward the top-left corner).
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r-')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc='lower right')
plt.savefig('Log_ROC')
plt.show()
###################################################################################################
# DEPLOYMENT
#################################################################################################### 
data['PD'] = logreg.predict_proba(data[X_train.columns])[:,1]
data["index"] = data.index
data[['index', 'PD']].head(10)

# The Logistic Regression model fairly accurately predicts the defaults of bank customers
# Now how do we predict the probability of default for a new loan applicant?
# Define the example values of X variables and check the probability estimated by the model
new_data = np.array([0.5,0,0,1,0,0,0,0,0,1,0,0,1]).reshape(1, -1)
new_pred=logreg.predict_proba(new_data)[:,1][0]
print("\033[1m This new loan applicant has a {:.2%}".format(new_pred), "chance of defaulting on a new debt")
