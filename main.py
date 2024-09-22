
#  --->  pip install -U scikit-learn
#  --->  pip install -U scikit-image
#   instead you can run ---> pip install -r requirements.txt
import os
import pickle
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# prepare data
input_dir = r'C:\Users\user\clf-data'
categories = ['empty', 'not_empty']

data = []
labels = []
print(" 1 - Reading and Resizing the Images....!")
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        # print(img.flatten())
        # exit()
        data.append(img.flatten())
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)

# train / test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# train classifier
classifier = SVC()

parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier, parameters) 
print(" 2 - ...Training the Model...")
grid_search.fit(x_train, y_train)

# test performance
best_estimator = grid_search.best_estimator_
print(' 3 - best_estimator',best_estimator)
y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print(' 4 - {}% of samples were correctly classified'.format(str(score * 100)))

pickle.dump(best_estimator, open('./model.p', 'wb'))





# source of GaussianNB  tuning  https://medium.com/analytics-vidhya/how-to-improve-naive-bayes-9fa698e14cba

# param_grid_nb = {
#     'var_smoothing': np.logspace(0,-9, num=100)
# } 
# model = GaussianNB()
# model_grid = GridSearchCV(model, param_grid=param_grid_nb, verbose=1, cv=10, n_jobs=-1)
# model_grid.fit(x_train, y_train)
# predicted = model_grid.predict(x_test)
# print(model_grid.best_estimator_)
# score = accuracy_score(predicted, y_test)
# print('{}% of samples were correctly classified'.format(str(score * 100)))










# from sklearn.preprocessing import StandardScaler 
# from sklearn.neighbors import KNeighborsClassifier
# scaler = StandardScaler()
# scaler.fit(x_train)

# X_train = scaler.transform(x_train)
# X_test = scaler.transform(x_test)
# model = KNeighborsClassifier(n_neighbors=23)
# model.fit(X_train, y_train) 

# # make predictions on the testing data
# y_predict = model.predict(X_test)
# score = accuracy_score(y_predict, y_test)
# # check results
# print('{}% of samples were correctly classified'.format(str(score * 100)))




#git remote remove origin
# git config http.postBuffer 524288000