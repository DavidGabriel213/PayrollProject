import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("/storage/emulated/0/Download/nigerian_university_payroll_messy.csv")
df=df.drop_duplicates()
df["University"]=df["University"].astype(str)
#department
df["Department"]=df["Department"].astype(str).str.capitalize().str.strip()
def department_corrector(c):
    if c in ["Med","Medicin","Medical"]:
        return "Medicine"
    elif c in ["Engg","Enginering","Engr","Eng"]:
        return "Engineering"
    elif c in ["Econ","Econs","Economcs"]:
        return "Economics"
    elif c in ["C.s","Cs","Comp sci","Compsci","Computer sci","Comp. science"]:
        return "Computer science"
    elif c in ["Maths","Mathemaics","Mathematcs","Math"]:
        return "Mathematics"
    else:
        return c
df["Department"]=df["Department"].apply(lambda x: department_corrector(x))
#State
df["State"]=df["State"].astype(str).str.strip().str.capitalize()
#Gender
df["Gender"]=df["Gender"].astype(str).str.capitalize().str.strip()
def gender_corrector(c):
    if c in ["Man","Masculine","M","Mr","Males","1"]:
        return "Male"
    elif c in ["Woman","Feminine","F","Mrs","Ms","Females","0","Female"]:
        return "Female"
    else:
        return np.nan
df["Gender"]=df["Gender"].apply(lambda x: gender_corrector(x))
df["Gender"]=df["Gender"].fillna(df.groupby("Department")["Gender"].transform(lambda x: x.mode()[0]))
#Rank
df["Rank"]=df["Rank"].astype(str).str.strip().str.capitalize()
def rank_corrector(c):
    if c in ["Lecturer 1","Li","Lect. i","Lect i","Lect1"]:
        return "Lecturer i"
    elif c in ["Sr.lect","Sl","Sr lecturer","Sr. lecturer","S.l"]:
        return "Senior lecturer"
    elif c in ["Asst.lect","Assit. lecturer","Al","Asst lecturer","Asst. lecturer"]:
        return "Assistant lecturer"
    elif c in ["Lect. ii","Lii","Lect2","Lecturer 2","Lect ii"]:
        return "Lecturer ii"
    elif c in ["A.prof","Assoc. professor","Assoc.prof","Ap","Assoc prof"]:
        return "Associate professor"
    elif c in ["Prof","Prof.","Full professor"]:
        return "Professor"
    elif c in ["G.a","Ga","Grad asst","Grad. asst"]:
        return "Graduate assistant"
    else:
        return c
df["Rank"]=df["Rank"].apply(rank_corrector)
#qualification
df["Qualification"]=df["Qualification"].astype(str).str.capitalize().str.strip()
def qualification(c):
    if c in ["Ph.d","Phd","Ph.d.","Doctorate","Dphil","D.phil"]:
        return "Ph.D"
    elif c in ["M.sc","Msc","M.sc.","Masters science","M.sc (science)","Ms","M.s"]:
        return "M.Sc"
    elif c in ["B.sc","Bsc","B.sc.","B.s","Bachelors science"]:
        return "B.Sc"
    elif c in ["M.tech","M.tech.","Mtech"]:
        return "M.Tech"
    elif c in ["B.ed","B.tech","M.phil","M.ed"]:
        k=int(c.index("."))
        return c[:k+1]+c[k+1].capitalize()+c[k+2:]
    elif c=="Nan":
        return np.nan
    else:
        return c.upper()
df["Qualification"]=df["Qualification"].apply(qualification)
df["Qualification"]=df["Qualification"].fillna(df.groupby("Rank")["Qualification"].transform(lambda x: x.mode()[0]))
#EmploymentType
df["EmploymentType"]=df["EmploymentType"].astype(str).str.strip().str.capitalize()
def employment_corrector(c):
    if c in ["Full-time","Fulltime","Full time","Permanent","F/t","Ft"]:
        return "Full-time staff"
    elif c in ["Contract","On contract","Cas","Contractual"]:
        return "Contract staff"
    elif c in ["Parttime","Part time","P/t","Pt","Part-time"]:
        return "Part-time staff"
    elif c in ["Adjunct","Adjunct lecturer"]:
        return "Adjunct staff"
    elif c in ["Visiting","Visitor","V/s"]:
        return "Visiting staff"
    elif c=="Nan":
        return np.nan
    else:
        return c
df["EmploymentType"]=df["EmploymentType"].apply(employment_corrector)
df["EmploymentType"]=df["EmploymentType"].fillna(df.groupby(["Rank","Qualification"])["EmploymentType"].transform(lambda x: x.mode()[0]))
#Age
df["Age"]=df["Age"].astype(str).str.strip()
def age_corrector(c):
    for k in ["yrs","years","yr"]:
        if k in c:
            return c.replace(k,"")
    else:
        return c
df["Age"]=df["Age"].apply(age_corrector)
df["Age"]=pd.to_numeric(df["Age"],errors="coerce")
max1=df["Age"].quantile(0.75)+1.5*(df["Age"].quantile(0.75)-df["Age"].quantile(0.25))
min1=df["Age"].quantile(0.25)-1.5*(df["Age"].quantile(0.75)-df["Age"].quantile(0.25))
df["Age"]=df["Age"].apply(lambda x: np.nan if x>max1 or x<18 else x)
df["Age"]=df["Age"].fillna(df.groupby(["Rank","Department"])["Age"].transform("mean"))
df["Age"]=df["Age"].astype(int)
#YearsOfExperience
df["YearsOfExperience"]=df["YearsOfExperience"].astype(str)
def yr_of_exprnce(c):
    if c in ["new staff","fresh"]:
        return 0
    else:
        for k in ["approx","yrs","yr(s)","Years Experience","years"]:
            if k in c:
                return c.replace(k,"")
        else:
            return c
df["YearsOfExperience"]=df["YearsOfExperience"].apply(yr_of_exprnce)
df["YearsOfExperience"]=pd.to_numeric(df["YearsOfExperience"],errors="coerce")
df["YearsOfExperience"]=df["YearsOfExperience"].fillna(df.groupby("Rank")["YearsOfExperience"].transform("mean"))
df["YearsOfExperience"]=df["YearsOfExperience"].astype(int)
#Base salary, net salary and gross salary''

def corrector(a):
        if a=="0" or a=="0.0":
            return np.nan
        if "k" in a:
            return str(float(a.replace("k",""))*1000)
        else:
            for b in ["₦","naira","NGN","#",'"',"-","N","per","pm","/","month","ly","Gross:"]:
                if b in a:
                    return a.replace(b,"")                
            else:
                return a
for c in ["BaseSalary","NetSalary","GrossSalary"]:
    df[c]=df[c].astype(str).str.strip().str.replace(",","")
    df[c]=df[c].apply(corrector)
    df[c]=pd.to_numeric(df[c],errors="coerce")
    max2=df[c].quantile(0.75)+1.5*(df[c].quantile(0.75)-df[c].quantile(0.25))
    min2=df[c].quantile(0.25)-1.5*(df[c].quantile(0.75)-df[c].quantile(0.25))
    df[c]=df[c].clip(min2,max2)
    df[c]=df[c].fillna(df.groupby(["Rank","EmploymentType"])[c].transform("mean")).round(1)
#HousingAllowance,TransportAllowance,ResearchGrant,MedicalAllowance,PensionDeduction,and NFHDeduction
for c in ["HousingAllowance","TransportAllowance","MedicalAllowance","ResearchGrant","PensionDeduction","NHFDeduction"]:
    df[c]=np.abs(df[c])
    max2=df[c].quantile(0.75)+1.5*(df[c].quantile(0.75)-df[c].quantile(0.25))
    min2=df[c].quantile(0.25)-1.5*(df[c].quantile(0.75)-df[c].quantile(0.25))
    df[c]=df[c].clip(min2,max2)
    df[c]=df[c].fillna(df.groupby(["Rank","EmploymentType"])[c].transform("mean")).round(1)
#TaxDeduction
df["TaxDeduction"]=df["TaxDeduction"].astype(str).str.strip()
df["TaxDeduction"]=df["TaxDeduction"].apply(lambda x: float(x.replace("%",""))/100 if "%" in x else x)
df["TaxDeduction"]=pd.to_numeric(df["TaxDeduction"],errors="coerce")
df["TaxDeduction"]=np.where(df["TaxDeduction"]<1, df["TaxDeduction"]*(df[["BaseSalary","HousingAllowance","MedicalAllowance","TransportAllowance"]].sum(axis=1)),df["TaxDeduction"])
max3=df["TaxDeduction"].quantile(0.75)+1.5*(df["TaxDeduction"].quantile(0.75)-df["TaxDeduction"].quantile(0.25))
min3=df["TaxDeduction"].quantile(0.25)-1.5*(df["TaxDeduction"].quantile(0.75)-df["TaxDeduction"].quantile(0.25))
df["TaxDeduction"]=df["TaxDeduction"].clip(min3,max3) 
df["TaxDeduction"]=df["TaxDeduction"].fillna(df["TaxDeduction"].median())
df["TaxDeduction"]=df["TaxDeduction"].round(1)
#Publications
df["Publications"]=df["Publications"].astype(str)
def publication_corrector(c):
    if c=="nil" or c=="none":
        return '0'
    for k in ["journals","papers","publications","articles"]:
        if k in c:
            return c.replace(k,"")
    else:
        return c
df["Publications"]=df["Publications"].apply(publication_corrector)
df["Publications"]=pd.to_numeric(df["Publications"], errors="coerce")
df["Publications"]=df["Publications"].fillna(df.groupby(["Rank","Qualification"])["Publications"].transform("median"))
df["Publications"]=df["Publications"].astype(int)
#PerformanceScore
df["PerformanceScore"]=df["PerformanceScore"].astype(str)
def performance_corrector(c):
    if "/" in c:
        k=c.index("/")
        return c[:k]
    for k in ["out of 10","Score:","points"]:
        if k in c:
            return c.replace(k,"")
    else:
        return c
df["PerformanceScore"]=df["PerformanceScore"].apply(performance_corrector)
df["PerformanceScore"]=pd.to_numeric(df["PerformanceScore"],errors="coerce")
df["PerformanceScore"]=df["PerformanceScore"].apply(lambda x: x/10 if x>10 else x)
df["PerformanceScore"]=df["PerformanceScore"].fillna(df.groupby(["Rank","Department"])["PerformanceScore"].transform("mean")).round(1)
#Absencedays
df["AbsenceDays"]=df["AbsenceDays"].astype(str).str.strip()
def absence_corrector(c):
    if c=="nil" or c=="none":
        return "0"
    for k in ["days","-"]:
        if k in c:
            return c.replace(k,"")
    else:
        return c
df["AbsenceDays"]=df["AbsenceDays"].apply(absence_corrector)
df["AbsenceDays"]=pd.to_numeric(df["AbsenceDays"],errors="coerce")
df["AbsenceDays"]=df["AbsenceDays"].clip(0,261)
df["AbsenceDays"]=(df["AbsenceDays"].fillna(df["AbsenceDays"].mean())).astype(int)
#CoursesTaught
df["CoursesTaught"]=df["CoursesTaught"].astype(str).str.strip()
def num_courses_corrector(c):
    for k in ["courses","-"]:
        if k in c:
            return c.replace(k,"")
    else:
        return c
df["CoursesTaught"]=df["CoursesTaught"].apply(num_courses_corrector)
df["CoursesTaught"]=pd.to_numeric(df["CoursesTaught"],errors="coerce")
max_courses=df["CoursesTaught"].quantile(0.75)+1.5*(df["CoursesTaught"].quantile(0.75)-df["CoursesTaught"].quantile(0.25))
df["CoursesTaught"]=df["CoursesTaught"].clip(0,max_courses)
df["CoursesTaught"]=(df["CoursesTaught"].fillna(df["CoursesTaught"].median())).astype(int)
#StudentsSupervised
df["StudentsSupervised"]=pd.to_numeric(df["StudentsSupervised"],errors="coerce")
df["StudentsSupervised"]=(df["StudentsSupervised"].fillna(df["StudentsSupervised"].median())).astype(int)
#PayrollCategory
df["PayrollCategory"] = df["PayrollCategory"].astype(str).str.strip()

def payroll_corrector(c):
    c = c.strip()
    low_values = ["Low","low","LOW","L","L1","1","Lower","lower","Lower Band","lower band","low income","low salary","low pay","low earner","low band","Band 1"]
    medium_values = ["Medium","medium","MEDIUM","Med","med","M","M1","2","mid","Mid","Middle","middle","Average","average","mid band","mid salary","Band 2"]
    high_values = ["High","high","HIGH","H","H1","H pay","3","Higher","higher","Upper","upper","high income","high salary","high band","upper band","Band 3"]
    executive_values = ["Executive","executive","EXECUTIVE","Exec","exec","E","E1","4","Senior","SENIOR","Senior Exec","Top","TOP","top","Premium","premium","Exec Pay","top band","Band 4"]
    if c in low_values:
        return "Low"
    elif c in medium_values:
        return "Medium"
    elif c in high_values:
        return "High"
    elif c in executive_values:
        return "Executive"
    elif c == "Nan":
        return np.nan
    else:
        return np.nan 
df["PayrollCategory"] = df["PayrollCategory"].apply(payroll_corrector)
df["PayrollCategory"] = df["PayrollCategory"].fillna(df.groupby("Rank")["PayrollCategory"].transform(lambda x: x.mode()[0]))
df.to_csv("/storage/emulated/0/Download/PayrollCleaned.csv",index="false")
#feature engineering
for c in ["BaseSalary","HousingAllowance","TransportAllowance","MedicalAllowance","ResearchGrant","TaxDeduction","PensionDeduction","NHFDeduction","GrossSalary","NetSalary"]:
    df[c+"_log"]=(np.log1p(df[c])).round(4)
df["SalaryEfficiency"]=(df["NetSalary"]/(1+df["YearsOfExperience"]).round(4)
df["AllowanceRatio"]= ((df["HousingAllowance"]+df["TransportAllowance"])/df["BaseSalary"]).round(4)
df["DeductionBurden"]=((df["TaxDeduction"] + df["PensionDeduction"] + df["NHFDeduction"])/(1+df["GrossSalary"]).round(4)
df["ResearchProductivity"]= (df["Publications"]/(1+df["YearsOfExperience"]).round(4)
df["TeachingLoad"]=(df["CoursesTaught"]/(1+df["StudentsSupervised"])).round(4)
df=df.drop(columns=["StaffID","BaseSalary","HousingAllowance","TransportAllowance","MedicalAllowance","ResearchGrant","TaxDeduction","PensionDeduction","NHFDeduction","GrossSalary","NetSalary"])
df.to_csv("/storage/emulated/0/Download/PayrollEngineeredData.csv",index="false")
print(df.info())
