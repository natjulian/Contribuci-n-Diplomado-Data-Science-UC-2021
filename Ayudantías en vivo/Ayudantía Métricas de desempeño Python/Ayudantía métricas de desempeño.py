#!/usr/bin/env python
# coding: utf-8

# # Ayudantía Métricas de Desempeño - Python

# ## Natalie Julian

# La base de datos corazón contiene datos de un estudio realizado a residentes de la ciudad de Framingham en Massachusetts. El objetivo de la clasificación es predecir si el paciente tiene un riesgo padecer una enfermedad coronaria futura.
# 
# Los atributos registrados en cada residente se describen a continuación:
# 
# - male: indica si el sexo es masculino o no
# - age: edad del paciente (en años)
# - currentSmoker: si el paciente fuma o no
# - cigsPerDay: la cantidad de cigarrillos que la persona fumó en promedio en un día
# - BPMeds: si el paciente estaba o no tomando medicación para la presión arterial
# - prevalentStroke: si el paciente había tenido previamente un accidente cerebrovascular o no 
# - prevalentHyp: si el paciente era hipertenso o no
# - diabetes: si el paciente tenía diabetes o no
# - totChol: nivel de colesterol total 
# - sysBP: presión arterial sistólica
# - diaBP: presión arterial diastólica
# - BMI: índice de masa corporal
# - heartRate: frecuencia cardíaca
# - glucose: nivel de glucosa
# - TenYearCHD: Padeció de enfermedad coronaria dentro de los próximos 10 años del estudio (0 No, 1 sí)

# ### a) Cargue la base de datos. Determine qué variables son numéricas (continuas o discretas) y cuáles son categóricas. ¿Hay alguna variable que necesite tratamiento? Realice el cambio que estime pertinente. ¿De qué naturaleza es la variable respuesta? ¿Nos encontramos en un problema de clasificación o de regresión? Discuta.
# 

# In[1]:


import pandas as pd

df = pd.read_csv('corazon.csv', delimiter=";")

df.head(200)


# In[2]:


df.tail()


# La naturaleza de las variables se describe a continuación:
# 
# - male: indica si el sexo es masculino o no (categórica binaria)
# - age: edad del paciente (en años) (numérica, continua por definición aunque está discretizada en la base de datos)
# - currentSmoker: si el paciente fuma o no (categórica binaria)
# - cigsPerDay: la cantidad de cigarrillos que la persona fumó en promedio en un día (numérica discretas)
# - BPMeds: si el paciente estaba o no tomando medicación para la presión arterial (categórica binaria)
# - prevalentStroke: si el paciente había tenido previamente un accidente cerebrovascular o no (categórica binaria)
# - prevalentHyp: si el paciente era hipertenso o no (categórica binaria)
# - diabetes: si el paciente tenía diabetes o no (categórica binaria)
# - totChol: nivel de colesterol total (numérica, continua por definición aunque está discretizada en la base de datos)
# - sysBP: presión arterial sistólica (numérica continua)
# - diaBP: presión arterial diastólica (numérica continua)
# - BMI: índice de masa corporal (numérica continua)
# - heartRate: frecuencia cardíaca (numérica discreta)
# - glucose: nivel de glucosa (numérica, continua por definición aunque está discretizada en la base de datos)
# - TenYearCHD: Padeció de enfermedad coronaria dentro de los próximos 10 años del estudio (0 No, 1 sí) (categórica binaria)
# 

# Notar que todas las variables categóricas son binarias y además, están codificadas como 1 y 0, por lo tanto no es necesario recodificarlas como dummies.

# ### b) Determine si existen o no datos faltantes, visualice la proporción de datos faltantes en la base de datos. ¿En qué variable hay mayor cantidad de datos faltantes? ¿Qué haría en este caso con los casos con datos faltantes? Discuta.

# In[3]:


df.shape #dimensiones de la bbdd


# In[4]:


df.info() #claramente hay datos faltantes en algunas variables, cigsPerDay, BPMeds, totChol, sysBP, BMI, heartRate, glucose


# In[5]:


#para instalar missingno
#conda install -c conda-forge/label/gcc7 missingno


# In[6]:


import missingno as msngo
msngo.matrix(df)


# Vemos que en la variable glucose es donde predominan los datos faltantes. Una opción: Eliminar la columna glucosa y luego eliminar los registros con algún atributo faltante. Sin embargo, la información de glucosa la perderíamos. Otra opción: imputar los casos faltantes, pero esto no es tan trivial y se requiere conocimiento de técnicas de imputación. La opción que tomaremos: eliminar los registros con algún caso faltante.

# In[7]:


df = df.dropna()


# In[8]:


df.shape


# ### c) Defina la matriz de variables predictoras X e Y vector de la variable respuesta de manera apropiada, además estandarice X. Obtenga un set de datos de entrenamiento y prueba en proporción 1:4. Utilice una semilla 2020 para este procedimiento. ¿Cómo se distribuyen los residentes con enfermedad de corazón en los distintos set? ¿Qué caso debería ser más fácil de predecir para el modelo en base a los datos? Comente.

# In[9]:


X=df.drop('TenYearCHD', axis=1) 


# In[10]:


df.drop('TenYearCHD', axis=1) .columns


# In[11]:


from sklearn.preprocessing import StandardScaler #estandariza

sc = StandardScaler()
X_stand = sc.fit_transform(X[['cigsPerDay', 'age', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']]) #cuando se estandariza se pierde toda la información de nombre de columnas y otros

X_stand = pd.DataFrame(X_stand, columns = X[['cigsPerDay', 'age','totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']].columns)

print(X_stand)


# In[12]:


X_cat=X[['male', 'currentSmoker', 'BPMeds','diabetes', 'prevalentStroke', 'prevalentHyp']]

X_cat.reset_index(drop=True, inplace=True)

print(X_cat)


# In[13]:


X=pd.concat([X_cat,X_stand], axis=1)

print(X)


# In[14]:


X.tail()


# In[15]:


y=df['TenYearCHD']


# In[16]:


print(y)


# In[17]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2020) #set de entrenamiento y prueba


# In[19]:


print(y_test)


# In[17]:


X_train.reset_index(drop=True, inplace=True) #Drop resetea el índice al n° de fila, inplace modifica el objeto
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)


# In[18]:


print(pd.crosstab(index=y_train, columns="count") )


# In[19]:


print(pd.crosstab(index=y_test, columns="count") )


# Hay más casos de residentes que no presentaron enfermedad al corazón que los que sí, por ende, al modelo le debería costar más predecir los casos donde sí hubo enfermedad (ya que tiene menos datos con los que entrenar, aunque también va a depender de las variables y de cómo logren separar los distintos casos).

# ### d) Entrene un modelo de regresión logística, obtenga los parámetros asociados a cada variable, evalúe significancia de cada uno e interprételos. ¿Qué variables son factores de riesgo/factores protectores de padecer una enfermedad al corazón? Comente.
# 

# Si queremos obtener resúmenes estadísticos como en R, con LogisticRegression de SKLearn, no es tan sencillo. Probaremos con el paquete statsmodels:

# In[20]:


import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[21]:


explicativas = "+".join(df.drop('TenYearCHD', axis=1).columns)  #necesitamos ingresarle la formula
formula = 'TenYearCHD ~ '+explicativas 

print(formula)


# In[22]:


#esta librería requiere solo una data, por lo que debemos unir y_train y X_train

data_train=pd.concat([y_train,X_train], axis=1)
data_train.head()


# In[23]:


lr_sns = smf.glm(formula, family=sm.families.Binomial(), data=data_train)


# In[24]:


result = lr_sns.fit()
result.summary()


# Las variables significativas en el modelo son:
# 
# - Edad
# - cigsPerDay
# - sysBP
# - glucose
# 
# 
# Además, podemos reconocer que las variables que se asocian con riesgo de padecer enfermedad al corazón serían todos aquellos que tienen un parámetro $\beta$ asociado mayor a 0, entre estos, ser hombre, fumador, si el paciente tomaba medicación para presión arterial, si el paciente es hipertenso, mayor colesterol, mayor presión arterial diástolica, mayor frecuencia cardíaca, mayor glucosa.

# ### e) Obtenga la matriz de confusión del modelo utilizando como punto de corte 0.5 y calcule e interprete las siguientes métricas de desempeño: 
# 
# ### - Sensibilidad
# ### - Precisión
# ### - Accuracy
# ### - Área bajo la curva roc
# 
# ### En base a lo obtenido, ¿usted cree que el modelo posee un buen desempeño? Comente.

# In[25]:


data_test=pd.concat([y_test,X_test], axis=1)

data_test=data_test.dropna()


# In[26]:


y_pred = result.predict(X_test)

print(y_pred)

predictions = [ 0 if x < 0.5 else 1 for x in y_pred] #punto de corte 0.5

print(predictions)


# In[27]:


from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(y_test, predictions)

cm_display = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
cm_display.plot(cmap='Blues');
plt.grid(None);


# In[28]:


from sklearn.metrics import accuracy_score, precision_score, recall_score
m1_acu=accuracy_score(y_test, predictions)
m1_prec = precision_score(y_test, predictions)
m1_rec = recall_score(y_test, predictions)

print( "Accuracy: ", m1_acu, "\nPrecision: ", m1_prec, "\nRecall: ", m1_rec)


# Al modelo le cuesta mucho predecir casos positivos, de ahí que el Recall sea tan bajo, predice bien los negativos, pero la clase de mayor interés son los positivos (los que tuvieron enfermedad al corazón), por ende, no sería un modelo con buen desempeño.

# In[29]:


y_test.min()


# In[30]:


y_pred.min()


# In[31]:


from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)


# ### f) Determine el mejor punto de corte según algún criterio de interés, evalúe nuevamente las métricas anteriores. ¿Qué pasó respecto al modelo anterior? Comente.

# Vamos a probar el threshold que maximiza la tasa de verdaderos positivos * (1-tasa falsos positivos)

# In[32]:


import numpy as np

def find_best_threshold(threshould, fpr, tpr):
   t = threshould[np.argmax(tpr*(1-fpr))]
   # (tpr*(1-fpr)) will be maximum if your fpr is very low and tpr is very high
   print("the maximum value of tpr*(1-fpr)", max(tpr*(1-fpr)), "for threshold", np.round(t,3))
   return t


find_best_threshold(thresholds, fpr, tpr)


# In[33]:


predictions_2 = [ 0 if x < 0.16 else 1 for x in y_pred]


# In[34]:


cm_2 = confusion_matrix(y_test, predictions_2)

cm_display = ConfusionMatrixDisplay(cm_2, display_labels=[0, 1])
cm_display.plot(cmap='Blues');
plt.grid(None);


# In[35]:


m2_acu=accuracy_score(y_test, predictions_2)
m2_prec = precision_score(y_test, predictions_2)
m2_rec = recall_score(y_test, predictions_2)

print( "Accuracy: ", m2_acu, "\nPrecision: ", m2_prec, "\nRecall: ", m2_rec)


# El modelo mejora en cuanto a predicción de casos positivos, pero empeora en casos negativos, pues ahora el punto de corte es menor, por ende la exigencia para ser clasificado como positivo es menor.

# In[ ]:




