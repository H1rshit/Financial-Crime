import pandas as pd
import numpy as np
import pickle


class Patrol:

    # initializing of class attributes on creation of object
    def __init__(self):   
        
        self.__model_parameters = {}
        self.__load_model_parameters()
    
    # loading the artifacts
    def __load_model_parameters(self):
        try:
            with open("../artifacts/model_parameters.pickle","rb") as f:
                model_parameters = pickle.load(f)
            self.__model_parameters = model_parameters
        except FileNotFoundError as fnf:
            print(fnf)
        
    # new features creation for the coming transaction    
    def __data_preparation(self,transaction_id):
        try:
            trs_data = pd.read_csv('../data/transactions.csv', parse_dates = ['CREATED_DATE'])
            users_data=pd.read_csv('../data/users.csv',parse_dates = ['CREATED_DATE','BIRTH_DATE'])
            trs_data = trs_data.query(" ID == '{}'".format(transaction_id))
            user_id=trs_data['USER_ID'].iloc[0]
            users_data = users_data.query(" ID == '{}'".format(user_id))
            transactional_data = pd.merge(trs_data, users_data, left_on='USER_ID', right_on='ID')
            transactional_data.drop('ID_y',axis=1,inplace=True)
            transactional_data.rename(columns={'CREATED_DATE_x': 'TRANSACTION_DATE', 'CREATED_DATE_y': 'SIGNUP_DATE','ID_x':'ID'},inplace=True)
            transactional_data['AGE']=transactional_data['SIGNUP_DATE'] - transactional_data['BIRTH_DATE']
            transactional_data['AGE']=transactional_data['AGE']/np.timedelta64(1,'Y')
            transactional_data['trx_done_after'] = transactional_data['TRANSACTION_DATE'] - transactional_data['SIGNUP_DATE']
            transactional_data['first_trx_done_after_ndays']=transactional_data['trx_done_after']/np.timedelta64(1,'D')
            transactional_data=transactional_data[['ID','TYPE','STATE','AMOUNT_GBP','CURRENCY','COUNTRY','AGE','first_trx_done_after_ndays']].set_index('ID')
            return transactional_data
        except Exception as e:
            print(e)
    
    

    
    # applying one-hot and frequency encoding with standard scaling on new transaction data
    def __preprocessing(self,trs_data):
        try:
            predict_data=self.__model_parameters['Onehot Encoding'].transform(trs_data)
            predict_data['CURRENCY_LEVEL']=predict_data['CURRENCY'].map(self.__model_parameters['Currency Encoding'])
            predict_data['COUNTRY_LEVEL']=predict_data['COUNTRY'].map(self.__model_parameters['Country Encoding'])
            predict_data.fillna(0)
            predict_data.drop(['CURRENCY','COUNTRY'],axis=1,inplace=True)
            predict_data=self.__model_parameters['Scaling'].transform(predict_data)
            return predict_data
        except Exception as e:
            print(e)
    
    # prediction on the new transaction
    def __predictor(self,predict_data):
        try:
            model=self.__model_parameters['clf']
            threshold=self.__model_parameters['opt_threshold']
            prob=model.predict_proba(predict_data)[:, 1]
            result=prob>threshold
            return result
        except Exception as e:
            print(e)
    
    # consuming transaction id and then PASS or LOCK the transaction after calling the subordinate functions
    def check_transaction(self,transaction_id):
        try:
            trx_data = self.__data_preparation(transaction_id)

            if(trx_data['AMOUNT_GBP'][0]==0.01 and trx_data['TYPE'][0]=='TOPUP'):
                return 'PASS'

            predict_data = self.__preprocessing(trx_data)
            prediction = self.__predictor(predict_data)
            return 'LOCK_USER' if prediction[0] else 'PASS'
        except Exception as e:
            print(e)
    
if __name__== "__main__":
    try:
        transaction_id = 'a9aa681d-451e-44c5-8df0-687661ac583d'
        patrol_object = Patrol()
        print(patrol_object.check_transaction(transaction_id))
    except Exception as e:
        print(e)