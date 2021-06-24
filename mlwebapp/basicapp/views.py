from django.shortcuts import render

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

reloadModel=joblib.load('./mlmodels/Loan_model.pkl')

def index(request):
    
    return render(request,'index.html')

def prediction(request):
    re=0
    if(request.method == 'POST'):
        re=1
        val={}
        val['age']=request.POST.get('age')
        val['salary']=int(request.POST.get('sal'))//1000

    
    testDtaa=pd.DataFrame({'x':val}).transpose()
    
   
 
    scoreval=reloadModel.predict(testDtaa)[0]
    context={'scoreval':scoreval, 'sal1':request.POST.get('sal'), 'age1':request.POST.get('age'),'re':re}
    if request.method=='POST' and 'reset' in request.POST:
        re=0
        return render(request,'index.html')
        

    return render(request,'index.html',context)
    


    


# Create your views here.
