import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
#Use GridSearchCV to perform hyperparameter tuning, which can help find the best parameters for the SVM
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
#Many machine learning algorithms require the data to be appropriately scaled to perform well. 
from sklearn.preprocessing import StandardScaler


# 1 - prepare your data
input_dir = '/home/adelrahuma/jupyter/Img_vid/clf-data'
# categories =['empty','not_empty']
categories =['Not_Taken','Taken']
# lables = 0 or 1 
data = []
labels = []
# reading and resizing the imagegs
print(" Reading images....")
for indx,category in enumerate (categories):
    for file in os.listdir(os.path.join(input_dir,category)):
        img_path = os.path.join(input_dir,category,file)
        img = imread(img_path)
        img = resize(img,(15,15))
        data.append(img.flatten())
        labels.append(indx)
# casting as an array
data = np.asarray(data)
labels = np.asarray(labels)

# 2 - train/test split
print(" Training....")
# y_train & y_test refrers to the class empty or not_empty "0 or 1"
# test_size= 0.2 ---> 20% from the whole data for test performance
# stratify=labels --> taking the same portion from both classes (empty  or not_empty)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size= 0.2, shuffle=True, stratify=labels )

# 3 - train your classifire
#########################################  SVM Algorithm
# SVC --> means Support Vector Classifire
# classifire = SVC()
# # the classifire is a new instasnce from SVC 
# parameters = [{'gamma':[0.01,0.001,0.0001], 'C':[1, 10, 100, 1000]}]
# #if you want to use model tuning GridSearchCV is your friend
# grid_search = GridSearchCV(classifire,parameters)
# grid_search.fit(x_train, y_train)
# best_estimator = grid_search.best_estimator_
# # best_estimator -> here in our Example it has the SVC(C=10, gamma=0.01)
# # We fed the best_estimator with x_test which the images
# y_prediction = best_estimator.predict(x_test)
# print(best_estimator)
# # 4 - Test Performance
# # comarison between the predicted class "y_prediction"  Vs true class value "y_test"
# score = accuracy_score(y_prediction,y_test)
# print(" EvaLUATION... {}% samples were corrected classified".format(str(score * 100)))
# # pickle to save our trained model "best_estimator", in which to save our time train again
# # 'wb' -> w = write , b=binary
# pickle.dump(best_estimator, open('./SVMmodel.p','wb'))


# ===============================   Naive Bayes Alogrithm    ====================================

Naive = GaussianNB()
parameters ={
    'var_smoothing':np.logspace(0,-9, num=100)
}
Naive_Tuning = GridSearchCV(Naive,parameters, verbose=1, cv=10, n_jobs=-1)
Naive_Tuning.fit(x_train, y_train)
predict_y =Naive_Tuning.predict(x_test)
print('Best EstimaTOR: ',Naive_Tuning.best_estimator_)
score = accuracy_score(predict_y,y_test)
print(" EvaLUATION... {}% samples were corrected classified".format(str(score * 100)))
pickle.dump(Naive_Tuning.best_estimator_, open('./GaussianNBmodel.p','wb'))



# ===============================   KNN    ====================================

# Scalr = StandardScaler()
# Scalr.fit(x_train)
# X1_train = Scalr.transform(x_train)
# X1_test = Scalr.transform(x_test)
# KNN_Model = KNeighborsClassifier(n_neighbors=7)
# KNN_Model.fit(x_train,y_train)
# y_predict = KNN_Model.predict(x_test)
# score = accuracy_score(y_predict,y_test)
# print(" Evaluation... {}% samples were corrected classified".format(str(score * 100)))










