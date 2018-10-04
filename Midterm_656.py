
# coding: utf-8

# In[55]:


import pandas as pd
import numpy as np 
from Class_replace_impute_encode import ReplaceImputeEncode
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from Class_regression import logreg
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from Class_tree import DecisionTree
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.metrics import roc_curve, auc



file_path = '/Users/Pawan/Desktop/MS Folder 😇😇✈️/4th Sem/STAT 656/Midterm/'
df = pd.read_excel(file_path+'CreditCard_Defaults.xlsx')
df_1 = df.drop('Customer', axis=1)
print("Authentication Data with %i observations & %i attributes.\n" %df_1.shape, df_1.head())

attribute_map = {
'Default':[1,(0,1),[0,0]],
#'Customer':[0,(1,30000),[0,0]],
'Gender':[1,(1, 2),[0,0]],
'Education':[2,(0,1,2,3,4,5,6),[0,0]],
'Marital_Status':[2,(0,1,2,3),[0,0]],
'card_class':[2,(1,2,3),[0,0]],
'Age':[0,(20,80),[0,0]],
'Credit_Limit':[0,(100,80000),[0,0]],
'Jun_Status':[0,(-2,8),[0,0]],
'May_Status':[0,(-2,8),[0,0]],
'Apr_Status':[0,(-2,8),[0,0]],
'Mar_Status':[0,(-2,8),[0,0]],
'Feb_Status':[0,(-2,8),[0,0]],
'Jan_Status':[0,(-2,8),[0,0]],
'Jun_Bill':[0,(-12000,32000),[0,0]],
'May_Bill':[0,(-12000,32000),[0,0]],
'Apr_Bill':[0,(-12000,32000),[0,0]],
'Mar_Bill':[0,(-12000,32000),[0,0]],
'Feb_Bill':[0,(-12000,32000),[0,0]],
'Jan_Bill':[0,(-12000,32000),[0,0]],
'Jun_Payment':[0,(0,60000),[0,0]],
'May_Payment':[0,(0,60000),[0,0]],
'Apr_Payment':[0,(0,60000),[0,0]],
'Mar_Payment':[0,(0,60000),[0,0]],
'Feb_Payment':[0,(0,60000),[0,0]],
'Jan_Payment':[0,(0,60000),[0,0]],
'Jun_PayPercent':[0,(0,1),[0,0]],
'May_PayPercent':[0,(0,1),[0,0]],
'Apr_PayPercent':[0,(0,1),[0,0]],
'Mar_PayPercent':[0,(0,1),[0,0]],
'Feb_PayPercent':[0,(0,1),[0,0]],
'Jan_PayPercent':[0,(0,1),[0,0]]}

encoding = 'one-hot' # Categorical encoding: Use 'SAS', 'one-hot' or None
scale = 'std' # Interval scaling: Use 'std', 'robust' or None
scaling = 'No' # Text description for interval scaling

rie = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding= encoding,                          interval_scale = scale, drop = True,display=True)

#features_map = rie.draft_features_map(df)
cleaned_df = rie.fit_transform(df_1)

print("\n\n Replaced, Imputed and Encoded Data\n" )
print(cleaned_df.head())
Matrix = np.empty([28,8])

print ('******** Logistic Regression ********')
# Regression requires numpy arrays containing all numeric values
y = np.asarray(cleaned_df['Default'])
# Drop the target, 'object'. Axis=1 indicates the drop is for a column.
X = np.asarray(cleaned_df.drop('Default', axis=1))
lgr = LogisticRegression()
lgr.fit(X,y)
score_list = ['accuracy', 'recall', 'precision', 'f1']
scores = cross_validate(lgr, X, y, scoring=score_list,                             return_train_score=False, cv=10)
print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
for s in score_list:
    var = "test_"+s
    mean = scores[var].mean()
    std  = scores[var].std()
    print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))
Score_names = []
Var_names = []
for s in score_list:
    var1 = s+'_mean'
    var2 = s+'_std'
    var3 = "test_"+s
    Score_names.append(var1)
    Score_names.append(var2)
    Var_names.append(var3)
Matrix[0,0] = scores[Var_names[0]].mean()
Matrix[0,1] = scores[Var_names[0]].std()
Matrix[0,2] = scores[Var_names[1]].mean()
Matrix[0,3] = scores[Var_names[1]].std()
Matrix[0,4] = scores[Var_names[2]].mean()
Matrix[0,5] = scores[Var_names[2]].std()
Matrix[0,6] = scores[Var_names[3]].mean()
Matrix[0,7] = scores[Var_names[3]].std()

print ('\n\n\n**************** Decision Trees ****************')
for m in range(0,10):
    dtc = DecisionTreeClassifier(criterion='gini',max_depth = m+1                              ,min_samples_split=5, min_samples_leaf=5)
    dtc = dtc.fit(X,y)
    score_list = ['accuracy', 'recall', 'precision', 'f1']
    scores = cross_validate(dtc, X, y, scoring=score_list,                             return_train_score=False, cv=10)
    print('\n******** Decision Tree with Max_Depth = ',m+1,' ********')
    print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
    for s in score_list:
        var = "test_"+s
        mean = scores[var].mean()
        std  = scores[var].std()
        print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))
    Var_names = []
    for s in score_list:
        var3 = "test_"+s
        Var_names.append(var3)
    Matrix[m+1,0] = scores[Var_names[0]].mean()
    Matrix[m+1,1] = scores[Var_names[0]].std()
    Matrix[m+1,2] = scores[Var_names[1]].mean()
    Matrix[m+1,3] = scores[Var_names[1]].std()
    Matrix[m+1,4] = scores[Var_names[2]].mean()
    Matrix[m+1,5] = scores[Var_names[2]].std()
    Matrix[m+1,6] = scores[Var_names[3]].mean()
    Matrix[m+1,7] = scores[Var_names[3]].std()
    
print ('\n\n\n**************** Neural Networks ****************')    
np_y = np.ravel(y)
hidden_layer_sizes_List = [(3),(11),(5,4),(6,5),(7,6)] 
for m in range(0,5): 
    if m == 0: 
        print("\n******** NEURAL NETWORK; 1 hidden layer with 3 perceptrons********") 
    elif m == 1: 
        print("\n******** NEURAL NETWORK; 1 hidden layer with 11 perceptrons********")
    elif m == 2: 
        print("\n******** NEURAL NETWORK; 2 hidden layer with 5 & 4 perceptrons********")
    elif m == 3: 
        print("\n******** NEURAL NETWORK; 2 hidden layer with 6 & 5 perceptrons********")
    else: 
        print("\n******** NEURAL NETWORK; 2 hidden layer with 7 & 6 perceptrons********")
    fnn = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes_List[m],                     activation='logistic',solver='lbfgs', max_iter=1000, random_state=12345)
    fnn = fnn.fit(X,np_y)
    score_list = ['accuracy', 'recall', 'precision', 'f1']
    scores = cross_validate(fnn, X, np_y, scoring=score_list,                             return_train_score=False, cv=10)
    print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
    for s in score_list:
        var = "test_"+s
        mean = scores[var].mean()
        std  = scores[var].std()
        print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))
    Var_names = []
    for s in score_list:
        var3 = "test_"+s
        Var_names.append(var3)
    Matrix[m+11,0] = scores[Var_names[0]].mean()
    Matrix[m+11,1] = scores[Var_names[0]].std()
    Matrix[m+11,2] = scores[Var_names[1]].mean()
    Matrix[m+11,3] = scores[Var_names[1]].std()
    Matrix[m+11,4] = scores[Var_names[2]].mean()
    Matrix[m+11,5] = scores[Var_names[2]].std()
    Matrix[m+11,6] = scores[Var_names[3]].mean()
    Matrix[m+11,7] = scores[Var_names[3]].std()

print ('\n\n\n**************** Random Forests ****************')    
estimators_list   = [10, 15, 20]
max_features_list = ['auto', 0.3, 0.5, 0.7]
score_list = ['accuracy', 'recall', 'precision', 'f1']
max_f1 = 0
a=16
for e in estimators_list:
    for f in max_features_list:
        print("\n********Number of Trees: ", e, " Max_features: ", f," ********")
        rfc = RandomForestClassifier(n_estimators=e, criterion="gini",                     max_depth=None, min_samples_split=2,                     min_samples_leaf=1, max_features=f,                     n_jobs=1, bootstrap=True, random_state=12345)
        rfc= rfc.fit(X, np_y)
        scores = cross_validate(rfc, X, np_y, scoring=score_list,                                 return_train_score=False, cv=10)
        
        print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
        for s in score_list:
            var = "test_"+s
            mean = scores[var].mean()
            std  = scores[var].std()
            print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))
        Var_names = []
        for s in score_list:
            var3 = "test_"+s
            Var_names.append(var3)
        Matrix[a,0] = scores[Var_names[0]].mean()
        Matrix[a,1] = scores[Var_names[0]].std()
        Matrix[a,2] = scores[Var_names[1]].mean()
        Matrix[a,3] = scores[Var_names[1]].std()
        Matrix[a,4] = scores[Var_names[2]].mean()
        Matrix[a,5] = scores[Var_names[2]].std()
        Matrix[a,6] = scores[Var_names[3]].mean()
        Matrix[a,7] = scores[Var_names[3]].std()
        a = a+1

Index = ['Logistic regression','DT_Max_Depth_1','DT_Max_Depth_2','DT_Max_Depth_3','DT_Max_Depth_4',        'DT_Max_Depth_5','DT_Max_Depth_6','DT_Max_Depth_7','DT_Max_Depth_8','DT_Max_Depth_9',        'DT_Max_Depth_10','NN_1L_3P','NN_1L_11P','NN_2L_5P_4P','NN_2L_6P_5P','NN_2L_7P_6P',        'RF_10T_autoF','RF_10T_0.3F','RF_10T_0.5F','RF_10T_0.7F','RF_15T_autoF','RF_15T_0.3F',        'RF_15T_0.5F','RF_15T_0.7F','RF_20T_autoF','RF_20T_0.3F','RF_20T_0.5F','RF_20T_0.7F'] 
Model_Comparison = pd.DataFrame(Matrix, index = Index, columns = Score_names) 
print(Model_Comparison)



print('\n\n\n\n****Selecting Best Model****\n\n')
X_train, X_validate, y_train, y_validate =             train_test_split(X, y,test_size = 0.3, random_state=7)

rfc = RandomForestClassifier(n_estimators=15, criterion="gini",                     max_depth=None, min_samples_split=2,                     min_samples_leaf=1, max_features=0.5,                     n_jobs=1, bootstrap=True, random_state=12345)
rfc= rfc.fit(X_train, y_train)

col = ['Age','Credit_Limit','Jun_Status','May_Status','Apr_Status','Mar_Status', 'Feb_Status','Jan_Status','Jun_Bill','May_Bill','Apr_Bill','Mar_Bill','Feb_Bill', 'Jan_Bill','Jun_Payment','May_Payment','Apr_Payment','Mar_Payment','Feb_Payment', 'Jan_Payment','Jun_PayPercent','May_PayPercent','Apr_PayPercent','Mar_PayPercent', 'Feb_PayPercent','Jan_PayPercent','Gender','Education0','Education1','Education2', 'Education3','Education4','Education5','Marital_Status0','Marital_Status1','Marital_Status2', 'card_class0','card_class1']
DecisionTree.display_importance(rfc, col)
y_tpredict =  rfc.predict(X_train) 
y_vpredict =  rfc.predict(X_validate) 
print(classification_report(y_validate,y_predict))
CM_Train = pd.DataFrame(confusion_matrix(y_train,y_tpredict),index =                         ['Class 0 ', ' Class 1'], columns=['Class 0 ', ' Class 1' ])
CM_Test = pd.DataFrame(confusion_matrix(y_validate,y_vpredict),index =                        ['Class 0 ', ' Class 1'], columns=['Class 0 ', ' Class 1' ])
print('\n\nConfusion Matrix Training Set\n',CM_Train )
print('\n\nConfusion Matrix Validation Set\n',CM_Test )

