import pandas as pd #type:ignore
import numpy as np #type:ignore
from sklearn.model_selection import train_test_split #type:ignore
from sklearn.preprocessing import StandardScaler #type:ignore
from sklearn.decomposition import PCA #type:ignore
from sklearn.feature_selection import SelectFromModel #type:ignore
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor #type:ignore
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,mean_absolute_error,mean_squared_error,r2_score #type:ignore
from micromlgen import port #type:ignore
import json

# LOAD
df=pd.read_csv("dataset.csv")

# TARGETS
mapping={"fresh_day1":6,"fresh_day2":5,"fresh_day3":4,"degrading_day4":3,"degrading_day5":2,"spoiled_day6":1,"spoiled_day7":0}
df["days_left"]=df["LABEL_STAGE"].map(mapping)
def get_stage(x): return "fresh" if "fresh" in x else "degrading" if "degrading" in x else "spoiled"
df["stage"]=df["LABEL_STAGE"].apply(get_stage)

# FEATURE ENGINEERING
df["Gas_Total"]=df["MQ3_PCT"]+df["MQ5_PCT"]+df["MQ135_PCT"]
df["MQ3_MQ5_ratio"]=df["MQ3_PCT"]/(df["MQ5_PCT"]+1e-6)
df["Temp_Humidity"]=df["Temp_C_PCT"]*df["Hum_PCT_PCT"]

# FEATURES
drop_cols=["LABEL_STAGE","stage","days_left"]
X=df.drop(columns=drop_cols)
y_class=df["stage"]
y_reg=df["days_left"]

# SCALE
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

# PCA
pca=PCA(n_components=0.95)
X_pca=pca.fit_transform(X_scaled)
print("Original Features:",X.shape[1])
print("After PCA:",X_pca.shape[1])

# FEATURE SELECTION
selector_model=RandomForestClassifier(n_estimators=200,random_state=42)
selector_model.fit(X_pca,y_class)
selector=SelectFromModel(selector_model,prefit=True)
X_selected=selector.transform(X_pca)
selected_mask=selector.get_support()
print("Selected Features:",X_selected.shape[1])

# CLASSIFIER
X_train,X_test,y_train,y_test=train_test_split(X_selected,y_class,test_size=0.2,random_state=42,stratify=y_class)
rf_classifier=RandomForestClassifier(n_estimators=300,max_depth=12,random_state=42)
rf_classifier.fit(X_train,y_train)
pred=rf_classifier.predict(X_test)
print("\nCLASSIFIER")
print("Accuracy:",accuracy_score(y_test,pred))
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))

# REGRESSOR
X_train,X_test,y_train,y_test=train_test_split(X_selected,y_reg,test_size=0.2,random_state=42)
rf_regressor=RandomForestRegressor(n_estimators=300,max_depth=12,random_state=42)
rf_regressor.fit(X_train,y_train)
pred=rf_regressor.predict(X_test)
print("\nREGRESSOR")
print("MAE:",mean_absolute_error(y_test,pred))
print("RMSE:",np.sqrt(mean_squared_error(y_test,pred)))
print("R2:",r2_score(y_test,pred))

# EXPORT RF CLASSIFIER
with open("rf_classifier.h","w") as f: f.write(port(rf_classifier))

# EXPORT RF REGRESSOR
with open("rf_regressor.h","w") as f: f.write(port(rf_regressor))

# SAVE PREPROCESS INFO
config={"scaler_mean":scaler.mean_.tolist(),"scaler_std":scaler.scale_.tolist(),"pca_components":pca.components_.tolist(),"pca_mean":pca.mean_.tolist(),"selected_mask":selected_mask.tolist(),"feature_order":X.columns.tolist()}
with open("preprocess.json","w") as f: json.dump(config,f,indent=4)

print("\nFILES CREATED")
print("rf_classifier.h")
print("rf_regressor.h")
print("preprocess.json")