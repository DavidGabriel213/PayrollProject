from flask import Flask, render_template, request
import numpy as np 
import pandas as pd
import joblib
import os
#loading model and preprocessor
model=joblib.load('RandomForeModel.joblib')
preprocessor=joblib.load('Preprocessor1.joblib')
print("Model and preprocessor loaded")
app=Flask(__name__)
@app.route('/',methods=['GET','POST'])
def myfunc():
    category=None
    Category_Class=None
    if request.method=='POST':
        # numerical features
        Age=float(request.form["age"])
        YearsOfExperience=float(request.form["years_experience"])
        Publications=float(request.form["publications"])
        PerformanceScore=float(request.form["performance_score"])
        CoursesTaught=float(request.form["courses_taught"])
        StudentsSupervised=float(request.form["students_supervised"])
        ResearchProductivity=(Publications/(1+YearsOfExperience))
        TeachingLoad=CoursesTaught/(1+StudentsSupervised)
        Performance=((YearsOfExperience+Publications)/(1+PerformanceScore))
        YearsStudents=(StudentsSupervised/(1+YearsOfExperience))
        #Categorical features
        University=request.form["university"]
        Department=request.form["department"]
        State=request.form["state"]
        Rank=request.form["rank"]
        Qualification=request.form["qualification"]
        EmploymentType=request.form["employment_type"]
        #binary feature
        Gender=float(request.form["gender"])
        feature=pd.DataFrame({
        "Age":[Age],"YearsOfExperience":[YearsOfExperience],"CoursesTaught":[CoursesTaught],"Publications":[Publications],"PerformanceScore":[PerformanceScore],"StudentsSupervised":[StudentsSupervised],"ResearchProductivity":[ResearchProductivity],"TeachingLoad":[TeachingLoad],"Performance":[Performance],"University":[University],"Department":[Department],"State":[State],"Rank":[Rank],"Qualification":[Qualification],"EmploymentType":[EmploymentType],"Gender":[Gender]})
        #preprocessing and prediction
        FEATURES=preprocessor.transform(feature)
        prediction=model.predict(FEATURES)[0]
        if prediction==0:
            category="Executive"
        elif prediction==1:
            category="Senior"
        elif prediction==2:
            category="Medium"
        else:
            category="Junior"
    class_map = {
       "Junior":"result-junior",
       "Medium":"result-medium",
       "Senior":"result-senior",
       "Executive":"result-executive"}
    Category_class = class_map.get(category, "")
    return render_template("PayRoll.html", category=category, Category_class=Category_class)
if __name__==('__main__'):
    port =int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0',port=port,debug=True)   
