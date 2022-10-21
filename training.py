import pandas as pd
from scipy.stats import norm 
import joblib
from sklearn.metrics import roc_auc_score
import os
import random
import xgboost
import numpy as np
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
import logging
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import sys

DATA_PATH=sys.argv[1]
# Get the input data from SPI and AOI files
InputSPI=pd.DataFrame()
for item in range(4):
    InputSPI=pd.concat([InputSPI,pd.read_csv(DATA_PATH + '/SPI_training_'+str(item)+'.csv')])
    

InputAOI=pd.read_csv(DATA_PATH+'/AOI_training.csv')


spi=InputSPI.copy()
aoi=InputAOI.copy()

# For merging SPI and AOI PinNumber should be the same types
spi["ComponentID2"]=spi["ComponentID"]
spi["FigureID2"]=spi["FigureID"]

aoi['PinNumber']=aoi['PinNumber'].astype('Int64').astype(str) 

# Merge spi and aoi
spi_training=spi.merge(aoi,on=['PanelID','FigureID','ComponentID','PinNumber'],how="left")

# Remove duplicated features
spi_training=spi_training.drop_duplicates(subset=['PanelID','FigureID','ComponentID','PinNumber'])

# Continuous measured features that are used by the model
Num_features_to_cat=['Volume(%)','Area(%)', 'OffsetX(%)','OffsetY(%)'] 

#Continuous informative features that are used by the model
Num_features=['Shape(um)', 'PosX(mm)', 'PosY(mm)','SizeX', 'SizeY'] 

Cat_features=['FigureID','PinNumber','PadID', 'PadType','Result'] # Categorical features used by the model

spi_training['Target']=spi_training['MachineID'].isna()
spi_training=spi_training.dropna(subset=Num_features)

for feat in Num_features:
    spi_training[feat]=spi_training[feat].astype('float')
    
spi_training["ComponentID2"]=spi_training["ComponentID"].astype('category')#.cat.codes

Train_df=spi_training.reset_index(drop=True)
X_train=Train_df
y_train=Train_df['Target']

encoder=ce.CatBoostEncoder(a=0.3)
AE_val=encoder.fit_transform(Train_df["ComponentID2"].copy(),y_train)

# Rename the columns to encoded
AE_val.columns=[x+"_encoded" for x in AE_val.columns.tolist()]

# Merge the encoded feature
Train_df=pd.concat([Train_df,AE_val],axis=1)
features_x=[x for x in Train_df.columns if "_encoded" in x]
joblib.dump(encoder,"Cat_boost_step1")

X_train_final=Train_df[features_x].to_numpy()






ratio=np.sum(y_train==0)/np.sum(y_train==1)



def objective(trial: Trial,X,y) -> float:
    
    joblib.dump(study, 'study.pkl')

    param = {
        'objective': trial.suggest_categorical('objective',['binary:logistic']), 
        'tree_method': trial.suggest_categorical('tree_method',['exact']),  # 'gpu_hist','hist'
        'lambda': trial.suggest_loguniform('lambda',1e-3,10.0),
        'alpha': trial.suggest_loguniform('alpha',1e-3,10.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.3,1.0),
        #'subsample': trial.suggest_uniform('subsample', 0.4, 1.0),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001,0.1),
        'n_estimators': trial.suggest_categorical('n_estimators', [30,40,50,70,100,150,200]),
        'max_depth': trial.suggest_categorical('max_depth', [3,5,7,9,11,13,15,17,20]),
        #'random_state': trial.suggest_categorical('random_state', [24,48,2020]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1,300),
        'nthread' : -1,
        #0'feval': trial.suggest_categorical('feval',[f1_eval]),
        #'scale_pos_weight' : trial.suggest_categorical('scale_pos_weight', [ratio])
        
    }
    
    
    model = xgboost.XGBClassifier(**param,feval=f1_eval)
    
    return cross_val_score(model, X_train_final, y_train, cv=5,scoring=make_scorer(f1_score, pos_label=0)).mean()

from sklearn.metrics import f1_score
import numpy as np

def f1_eval(y_pred, y_true):
    #y_true = dtrain.get_label()
    err = f1_score(y_true,y_pred,pos_label=0)
    return 'f1_err', err

study = optuna.create_study(
study_name="HXGBoostTask1", 
storage="sqlite:///optuna_database.db",
direction='maximize',
load_if_exists=True
)
study.optimize(lambda trial : objective(trial,X_train_final,y_train),n_trials= 1000)




print("debug")