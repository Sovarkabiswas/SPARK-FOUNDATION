PREDICTION USING DECISION TREEE CLASSIFIER //
//IMPORTING LIBRARIES 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
UPLOADING CSV FILE
df=pd.read_csv('Iris.csv')
df.head() 
USING LABEL ENCODER 
from sklearn.preprocessing import LabelEncoder
#Initialize the LabelEncoder
label_encoder = LabelEncoder()
#Fit and transform the 'Category' column
df['Species_Category'] = label_encoder.fit_transform(df['Species'])
# Display the DataFrame with the encoded column
print("\nDataFrame with encoded 'Category' column:")
print(df)
SPLITTING THE INDEPENDENT AND DEPENDENT FEATURES //
X=df.iloc[:,1:5]
y=df['Species_Category']
## Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.tree import DecisionTreeClassifier 
# FITTING THE MODEL
treemodel = DecisionTreeClassifier()
# Fit the model
treemodel.fit(X_train, y_train)
rom sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(treemodel,filled=True)
#PREDICTION 
Prediction 
y_prediction=treemodel.predict(X_test)
y_prediction 
from sklearn.metrics import accuracy_score, classification_report 
score=accuracy_score(y_prediction,y_test)
print(100*score)
print(classification_report(y_prediction,y_test)) 
