import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("air-traffic-landings-statistics_v2.csv")
print(data.head())

# Features:
# Operating Airline: Categorical
# Operating Airline IATA Code: Categorical
# GEO Summary: Categorical
# GEO Region: Categorical
# Landing Aircraft Type: Categorical
# Aircraft Body Type: Categorical
# Aircraft Manufacturer: Categorical
# Aircraft Model: Categorical
# Landing Count: Numerical
# Total Landed Weight: Numerical

# Need to use non-numerical data for variables:
# sklearn will covert non-numerical values into numerical values.

le = preprocessing.LabelEncoder()                   # coding the labels into integer values
# creating lists for each column:
Operating_Airline = le.fit_transform(list(data["Operating Airline"]))
Op_IATA_Code = le.fit_transform(list(data["Operating Airline IATA Code"]))
GEO_Summary = le.fit_transform(list(data["GEO Summary"]))
GEO_Region = le.fit_transform(list(data["GEO Region"]))
Landing_Aircraft_Type = le.fit_transform(list(data["Landing Aircraft Type"]))
Aircraft_Body_Type = le.fit_transform(list(data["Aircraft Body Type"]))
Aircraft_Manufacturer = le.fit_transform(list(data["Aircraft Manufacturer"]))
Aircraft_Model = le.fit_transform(list(data["Aircraft Model"]))
Landing_Count = le.fit_transform(list(data["Landing Count"]))
Total_Landed_Weight = le.fit_transform(list(data["Total Landed Weight"]))

predict = "Total Landed Weight"



# x list (features):
X = list(zip(Operating_Airline, Op_IATA_Code, GEO_Summary, GEO_Region, Landing_Aircraft_Type, Aircraft_Body_Type, Aircraft_Manufacturer, Aircraft_Model, Landing_Count, Total_Landed_Weight))
# y list (labels):
y = list(Total_Landed_Weight)

# Model training:
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)  # test size ratio
print(x_train, y_test)

model = KNeighborsClassifier(n_neighbors=5)    # choose KNN number

model.fit(x_train, y_train)   # train the model
acc = model.score(x_test, y_test)  # test model for accuracy

print("Model Accuracy is: ", acc)    # accuracy

