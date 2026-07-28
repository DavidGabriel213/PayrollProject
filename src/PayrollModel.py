import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score,classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib
#loading data
df=pd.read_csv("/storage/emulated/0/Download/PayrollEngineeredData.csv")
#numerical columns
num_cols=["Age","YearsOfExperience","Publications","PerformanceScore","AbsenceDays","CoursesTaught","StudentsSupervised","ResearchProductivity","TeachingLoad"]
#categorical columns
cat_cols=["University","Department","State","Rank","Qualification","EmploymentType"]
#binary columns
bina_cols=["Gender"]
#Encoding binary and columns
le=LabelEncoder()
df["PayrollCategory"]=le.fit_transform(df["PayrollCategory"])
df["Gender"]=le.fit_transform(df["Gender"])
X=df[num_cols + cat_cols + bina_cols]
Y=df["PayrollCategory"]
# Transformers/pipeline
preprocessor=ColumnTransformer(transformers=[("Scaler", StandardScaler(), num_cols),("ohe", OneHotEncoder(drop='first',sparse_output=False,handle_unknown="ignore"), cat_cols)],remainder="passthrough")
# splitting
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=7,stratify=Y)
#preprocessing
X_train=preprocessor.fit_transform(X_train)
X_test=preprocessor.transform(X_test)
#LogisticRegression
model1=LogisticRegression(class_weight="balanced",max_iter=1000,random_state=7)
model1.fit(X_train,Y_train)
y_pred1=model1.predict(X_test) 
print(f"LogisticRegression: {accuracy_score(Y_test,y_pred1)*100:.2f}%")
print(classification_report(Y_test,y_pred1))
#DecisionTreeClassifier
model2=DecisionTreeClassifier(max_depth=9,random_state=7,class_weight="balanced")
model2.fit(X_train,Y_train)
y_pred2=model2.predict(X_test) 
print(f"DecisionTree: {accuracy_score(Y_test,y_pred2)*100:.2f}%")
print(classification_report(Y_test,y_pred2))
#RandomForestClassifier
model3=RandomForestClassifier(n_estimators=200,random_state=7,class_weight="balanced",max_depth=11)
model3.fit(X_train,Y_train)
y_pred3=model3.predict(X_test) 
print(f"RandomForest: {accuracy_score(Y_test,y_pred3)*100:.2f}%")
print(classification_report(Y_test,y_pred3))
#feature importance
feature_names =preprocessor.get_feature_names_out()
importances = model3.feature_importances_
feat_imp = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})
feat_imp = feat_imp.sort_values(by='Importance', ascending=False)
feat_imp['Original_Feature'] = feat_imp['Feature'].apply(
    lambda x: x.split('__')[-1].split('_')[0]
)
grouped = feat_imp.groupby('Original_Feature')['Importance'].sum().sort_values(ascending=False)
print(grouped)
