from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)

 #placement function  for  individual student   
def knn_student_prediction_placement(f):
    p=pd.read_csv(f, sep = ',', header = None, engine = 'python', encoding = 'latin-1')
    print(p)
    keam_min_rank=1
    keam_max_rank=50000
    p[[2]]=((p[[2]]-keam_min_rank)/(keam_max_rank-keam_min_rank))*100
    print(p[[2]])
    p[[2]]=100-(p[[2]].round(2))
    p[[3]]=p[[3]]*10
    print(p)
    test_predict1 = modelp.predict(p)
    return test_predict1
#function of uploading CSV file for placement prediction
def knn_dataset_prediction(data):
    # p=pd.read_csv(data, sep = ',', header = None, engine = 'python', encoding = 'latin-1')
    p=data
    p=p.drop([0],axis=1)
    l = p[[3]].values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,100))
    x_scaled = min_max_scaler.fit_transform(l)
    df=pd.DataFrame(x_scaled)
    p[[3]]=df[[0]]
    p[[3]]=100-(p[[3]].round(2))
    p[[4]]=p[[4]]*10
    test_predict1 = modelp.predict(p)
    test_predict2=test_predict1*100
    print("Prediction")
    print(test_predict2)
    return test_predict2
#create pickle for placement
modelp=pickle.load(open('model1.pkl','rb'))

#function of backout for individual student
def knn_student_prediction_backout(n):
    p=pd.read_csv(n, sep = ',', header = None, engine = 'python', encoding = 'latin-1')
    keam_min_rank=1
    keam_max_rank=50000
    p[[2]]=((p[[2]]-keam_min_rank)/(keam_max_rank-keam_min_rank))*100
    print(p[[2]])
    p[[2]]=100-(p[[2]].round(2))
    
    test_predict1 = modelb.predict(p)
    return test_predict1
#function of backout for CSV file upload
def knn_dataset_prediction_backout(data):
    p=data
    l = p[[2]].values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,100))
    x_scaled = min_max_scaler.fit_transform(l)
    df=pd.DataFrame(x_scaled)
    p[[2]]=df[[0]]
    p[[2]]=100-(p[[2]].round(2))
    test_predict1 = modelb.predict(p)
    test_predict2=test_predict1*100
    print("PREDICT###################################")
    print(test_predict2)
    return test_predict2
  
#create model using pickle for backout
modelb=pickle.load(open('modelb.pkl','rb'))
#this is for index page
@app.route('/')
def hello_world():
    return render_template("index.html")
database={'anjali':'123','shijin':'aaa','nidhi':'asdsf'}

#TO SHOW INDEX PAGE
@app.route('/index1/',methods=['POST','GET'])
def index1():
     return render_template("index.html")   
 
#signin buttton clicks
@app.route('/login1/',methods=['POST','GET'])
def login1():
     return render_template("login.html")
 
 #login process for staff
@app.route('/login',methods=['POST','GET'])
def login():
    if request.method == "POST":
        name1=request.form['username']
        pwd=request.form['password']
        if name1 not in database:
            return render_template('login.html',info='Invalid User')
        else:
            if database[name1]!=pwd:
                return render_template('login.html',info='Invalid Password')
            else:
                return render_template('staffpage.html',name=name1)

            
            
#to link upload page for backout dataset
@app.route('/backed1/',methods=['POST','GET'])
def backed1():
     return render_template("backed.html")

#to link upload page for placement
@app.route('/placed2/',methods=['POST','GET'])
def placed2():
     return render_template("placementall.html")
 
#function to upload dataset of backout
@app.route('/backupload' ,methods=['POST','GET'])
def backupload():
    if request.method == 'POST':
        f=request.files['fileToUpload']
        data = pd.read_csv(f,header=None)
        # print(data)

        # data2 = pd.read_csv('placed.csv', sep = ',', header = None, engine = 'python', encoding = 'latin-1')
        # print(data2)
        output= knn_dataset_prediction_backout(data).tolist()
        fail_student = output.count(0)
        pass_student = output.count(100)
        print(fail_student)
        print(pass_student)
        out = "Out of "+ str(len(output)) +", number of students that will not backout  is "+str(pass_student)+" and number of students who will backout the exam is "+str(fail_student) +"."
        out2 = "Pass percentage is "+ str(round(pass_student*100.0/len(output),2))+"%"

        return render_template('backed1.html', pred=out,percent = out2,p=pass_student,f=fail_student)
        # print("OUTPUT###################################")
        # print(output)
        # return "DONE!"

    return "NOT A POST CALL!!!"


#FUNCTION TO UPLOAD AND SHOW THE GRAPH FOR PLACEMENT
@app.route('/upload' ,methods=['POST','GET'])
def upload():
    if request.method == 'POST':
        f=request.files['fileToUpload']
        data = pd.read_csv(f,header=None)
        # print(data)
        # data2 = pd.read_csv('placed.csv', sep = ',', header = None, engine = 'python', encoding = 'latin-1')
        # print(data2)
        output= knn_dataset_prediction(data).tolist()
        fail_student = output.count(0)
        pass_student = output.count(100)
        print(fail_student)
        print(pass_student)
        out = "Out of "+ str(len(output)) +", number of students that will placed is "+str(pass_student)+" and number of students who less possible to placed  is "+str(fail_student) +"."
        out2 = "Pass percentage is "+ str(round(pass_student*100.0/len(output),2))+"%"

        return render_template('placementall2.html', pred=out,percent = out2,p=pass_student,f=fail_student)
        # print("OUTPUT###################################")
       
      

    return "NOT A POST CALL!!!"


#to show the page placement for single student prediction
@app.route('/career1/' ,methods=['POST','GET'])
def career1():
    
    return render_template("placement.html")
#single user prediction
@app.route('/career' ,methods=['POST','GET'])
def career():
    int_features=[float(x) for x in request.form.values()]
    import csv
    with open('inputs.csv', 'w') as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL) 
        w.writerow(int_features)
        f= "inputs.csv"
    
    print(request.form.values())
    output= knn_student_prediction_placement(f)
    print(output)
    
    if output==1:
            return render_template('placement.html', pred='....you will placed....')
    else:
          return  render_template('placement.html', pred='student is less likely to be placed...boost yourself')
    
   
#TO SHOW THE BACKOUT PAGE

@app.route('/backout1/' ,methods=['POST','GET'])
def backout1():
    
    return render_template("backout.html")


#FUNCTION TO USE SINGLE BACKOUT PREDICTION
@app.route('/backout' ,methods=['POST','GET'])
def backout():
    
   
    int_features_backout=[int(x) for x in request.form.values()]
    import csv
    with open('inputs.csv', 'w') as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL) 
        w.writerow(int_features_backout)
        n= "inputs.csv"

    output= knn_student_prediction_backout(n)
    print(output)
    if output==1:
            return render_template('backout.html', pred='Student not may  backout....')
    else:
          return  render_template('backout.html', pred='student   may backout')

    
    

if __name__ == '__main__':
        app.run(debug=True)