# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# Importing the dataset
dataset = pd.read_csv('data.csv')

# Data informations
dataset.info()

# Count of null values in each column 
dataset.isnull().sum()

# Drop duplicates
print("Number of duplicated recordes is : {}".format(dataset.duplicated().sum()))
dataset.drop_duplicates(inplace=True)
dataset.shape

#as we can see each no. diseases having the same percentage through bar chart
plt.figure(figsize=(10,6))
dataset['prognosis'].value_counts().plot.bar()
plt.subplots_adjust(left = 0.9, right = 2 , top = 2, bottom = 1)
plt.show();

# #checking the relationship between the variables by applying the correlation 
# corr = dataset.corr()
# mask = np.array(corr)
# mask[np.tril_indices_from(mask)] = False
# plt.subplots_adjust(left = 0.5, right = 16 , top = 20, bottom = 0.5)
# sns.heatmap(corr, mask=mask,vmax=.9, square=False,annot=True, cmap="YlGnBu")
# plt.show()

# Splitting data into x,y
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

####################### KNN ##################################################

train_accuracies = {}
test_accuracies = {}
neighbors = np.arange(1, 26)
for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train, y_train)
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)
    
plt.figure(figsize=(8, 6))
plt.title("KNN: Varying Number of Neighbors")
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")
plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy")
plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.show()    

# Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
KNN_classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
KNN_classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_KNN = KNN_classifier.predict(X_test)

# Accuracy score
from sklearn.metrics import accuracy_score
KNN_acc=accuracy_score(y_test,y_pred_KNN)
print("Accuracy of KNN is : {:.2f} ".format(KNN_acc))



####################### Decision Tree Classification #########################

# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
DT_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
DT_classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_DT = DT_classifier.predict(X_test)

# Accuracy score
from sklearn.metrics import accuracy_score
desision_tree_acc=accuracy_score(y_test,y_pred_DT)
print("Accuracy of Decision Tree is : {:.2f} ".format(desision_tree_acc))

####################### Naive Bayes #########################

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
NB_classifier = GaussianNB()
NB_classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_NB = NB_classifier.predict(X_test)

# Accuracy score
from sklearn.metrics import accuracy_score
naives_bayes_acc=accuracy_score(y_test,y_pred_NB)
print("Accuracy of Naives Bayes is : {:.2f} ".format(naives_bayes_acc))

####################### Random Forest ########################

train_accuracies = {}
test_accuracies = {}
trees = np.arange(1, 26)
for n in trees:
    RF = RandomForestClassifier(n_estimators = n, criterion = 'entropy', random_state = 0)
    RF.fit(X_train, y_train)
    train_accuracies[n] = RF.score(X_train, y_train)
    test_accuracies[n] = RF.score(X_test, y_test)
    
plt.figure(figsize=(8, 6))
plt.title("Random Forest: Varying Number of trees")
plt.plot(trees, train_accuracies.values(), label="Training Accuracy")
plt.plot(trees, test_accuracies.values(), label="Testing Accuracy")
plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.show() 

# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
RF_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
RF_classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_RF = RF_classifier.predict(X_test)

# Accuracy score
from sklearn.metrics import accuracy_score
random_forest_acc=accuracy_score(y_test,y_pred_RF)
print("Accuracy of Random Forest is : {:.2f} ".format(random_forest_acc))

####################### K-Means ###########################

# Number of clusters
k= len(dataset["prognosis"].unique())
print("Number of clusters is : {}".format(k))

# Training the KMeans model on the Training set
from sklearn.cluster import KMeans
Kmeans = KMeans(n_clusters = k)
Kmeans.fit(X)
labels = Kmeans.labels_
prognosis_col = dataset["prognosis"] 
kmeans_result = prognosis_col
kmeans_result = kmeans_result.to_frame()
kmeans_result["y"] = y
kmeans_result["clusters"] = labels

# Accuracy score
from sklearn.metrics import accuracy_score
KMeans_acc=accuracy_score(kmeans_result["y"],kmeans_result["clusters"])
print("Accuracy of K-Means is : {:.2f} ".format(KMeans_acc))

#######################################################################################

# Accuracies
Accuracies ={"KNN" : KNN_acc*100,
             "Decision Tree" : desision_tree_acc*100,
             "Naive Bayes" : naives_bayes_acc*100,
             "Random Forest" : random_forest_acc*100,
             "K-Means" : KMeans_acc*100} 

classifiers ={KNN_classifier : KNN_acc*100,
             DT_classifier : desision_tree_acc*100,
             NB_classifier : naives_bayes_acc*100,
             RF_classifier : random_forest_acc*100,
             Kmeans : KMeans_acc*100} 
best_acc = max(Accuracies.values())
label = []
for x in Accuracies:
    Accuracies[x]
    if(Accuracies.get(x)==best_acc):
      label.append(x)
if(len(label)>1):
    print("The best models are:")
    for i in range(len(label)):
        print("{}){}".format(i+1,label[i]))
    print("with accuracy : {}%".format(best_acc))    
else :    
    print("The best model is {} with accuracy : {}%".format(label,round(best_acc)))
best_classifier = list(classifiers.keys())[list(classifiers.values()).index(best_acc)]


# Visualising the accuracies 
plt.figure(figsize=(8,6))
plt.bar(range(len(Accuracies.keys())), Accuracies.values())
plt.title("Models Accuracies")
plt.ylabel("Percentage")
plt.xticks(range(len(Accuracies.keys())), Accuracies.keys())
plt.show()

a = list(range(1,133))
i_name  = (input('Enter your name :'))
i_age = (int(input('Enter your age:')))
for i in range(len(dataset.columns)-1):
    print(str(i+1) + ":", dataset.columns[i])
choices = input('Enter the Serial no.s which is your Symptoms are exist:  ')
b = [int(x) for x in choices.split()]
count = 0
while count < len(b):
    item_to_replace =  b[count]
    replacement_value = 1
    indices_to_replace = [i for i,x in enumerate(a) if x==item_to_replace]
    count += 1
    for i in indices_to_replace:
        a[i] = replacement_value
a = [0 if x !=1 else x for x in a]
y_diagnosis = best_classifier.predict([a])
y_pred_2 = best_classifier.predict_proba([a])
print(('Name of the infection = %s , confidence score of : = %s') %(y_diagnosis[0],y_pred_2.max()* 100),'%' )
print(('Name = %s , Age : = %s') %(i_name,i_age))
