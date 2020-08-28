import tensorflow
import keras
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
print(data.head())

#Makes non-int vals into integers. Returns numpy arrays for each attribute.
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

#x is attributes, y is labels
x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

#tests on 0.1 (10%) of the data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

#KNN model: looks at the train data points that the test data is close to (for attribute values), and given those neighbors, it predicts the class value
#Groups are each class value(similar to colors in video).
model = KNeighborsClassifier(n_neighbors = 9)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test) #how accurate the predicted y values are with the actual y values
print(acc)

predicted = model.predict(x_test)
#classifiers (class names represented as numbers 0 to 3)
names = ["unacc", "acc", "good", "vgood"]
for x in range(len(x_test)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 9, True)
    print("n: ", n)