import numpy as np
import pandas as pd
import re
from flask import Flask,request,jsonify
import pickle

dataset=pd.read_csv('Symptoms_dataset.csv')

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dataset['Disease']=le.fit_transform(dataset['Disease'])

X=dataset.iloc[:,1:].values
y=dataset.iloc[:,0].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

print(len(X_train))
print(len(y_train))

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(X_train,y_train)



predict=model.predict(X_test)
print(np.concatenate((predict.reshape(len(predict),1),y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import accuracy_score
acc=accuracy_score(predict, y_test)
print(acc)
colums_names=list(dataset.columns[1:])
corpus=[]
for i in colums_names:
    review=re.sub('[^a-zA-Z]',' ',i)
    review=review.lower()
    review=review.replace(" ","")
    corpus.append(review)
dataset2=pd.read_csv('symptom_Description.csv')
x1=dataset2.iloc[:,0].values
corpus1=[]
for i in x1:
    review=re.sub('[^a-zA-Z]',' ',i)
    review=review.lower()
    review=review.replace(" ","")
    corpus1.append(review)
print(corpus1)

y2=dataset2.iloc[:,-1].values

dataset3=pd.read_csv('symptom_precaution.csv')
x3=dataset3.iloc[:,0].values

corpus3=[]
for i in x3:
    review=re.sub('[^a-zA-Z]',' ',i)
    review=review.lower()
    review=review.replace(" ","")
    corpus3.append(review)
print(corpus3)
y3=dataset3.iloc[:,1:].values

with open('sheet.pkl','wb') as mod_file:
    pickle.dump(model,mod_file)
    
with open('encode.pkl','wb') as encod_file:
    pickle.dump(le,encod_file)

app=Flask(__name__)
@app.route('/predict',methods=['POST'])

def pred():
    predict_disease=np.zeros(131,dtype=int)
    data=request.json
    check=data.get('Symptoms')
    for t in check:
        answer=re.sub('[^a-zA-Z]',' ',t)
        answer=answer.lower()
        answer=answer.replace(" ","")
        ind=corpus.index(answer)
        predict_disease[ind]=1
    predict_disease=np.reshape(predict_disease,(1,131))
    with open('sheet.pkl','rb') as mod_file:
        classifier=pickle.load(mod_file)
    with open('encode.pkl','rb') as encod_file:
        rev=pickle.load(encod_file)
    final_ans=classifier.predict(predict_disease)
    final_ans=rev.inverse_transform(final_ans)
    final_ans=str(final_ans)
    final_ans=re.sub('[^a-zA-Z]',' ',final_ans)
    final_ans=final_ans.lower()
    final_ans=final_ans.replace(" ","")
    des_ind=corpus1.index(final_ans)
    description=y2[des_ind]
    prec_ind=corpus3.index(final_ans)
    precautions=y3[prec_ind]
    return jsonify({
        "Disease":final_ans,
        "Description":description,
        "Precautions":list(precautions)})
if(__name__=='__main__'):
    app.run(debug=True)
    
    
    
        
    
        
        
    

    





