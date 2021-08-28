#!/usr/bin/env python
# coding: utf-8

# # Ayudantía Árboles de Decisión - Python

# ## Natalie Julian

# El Departamento de Colocación del área de posgrados en negocios en la Universidad de Jain, India está buscando determinar los factores que influyen en que sus estudiantes encuentren trabajo o no. La base de datos Placement_Data_Full_Class.csv contiene información de 215 estudiantes egresados en el MBA de Bussiness Analytics.
# 
# Las variables se describen a continuación:
# 
# - sl_no. Número de fila.
# - gender: Género del estudiante (M: Hombre, F: Mujer).
# - ssc_p: Calificación de la escuela en 1°-10° Grado (en porcentaje).
# - ssc_b: Tipo de Junta de Educación en 1°-10° Grado (Central/Other)
# - hsc_p: Calificación de la escuela en 11°-12° Grado (en porcentaje).
# - hsc_b: Tipo de Junta de Educación en 11°-12° Grado (Central/Other)
# - hsc_s: Especialización en la escuela en 11°-12° Grado.
# - degree_p: Calificación en el grado de educación superior (en porcentaje).
# - degree_t: Área del grado de educación superior.
# - workex: Experiencia laboral (Yes/No).
# - etest_p: Calificación del test de empleabilidad (en porcentaje).
# - specialisation: Categoría de especialización del MBA.
# - mba_p: Calificación del MBA (en porcentaje).
# - status: Indicador si tiene o no un ofrecimiento de trabajo (Placed/Not Placed).
# - salary: Sueldo ofrecido a los candidatos en la oferta.
# 
# Interesa determinar qué características de egresados se relacionan con una mayor probabilidad de caer en el grupo que tiene Oferta de trabajo (Placed).

# ### a) Cargue la base de datos. ¿Hay alguna columna que no sea necesaria en la base de datos? Realice el cambio que estime pertinente. 

# In[23]:


import pandas as pd

df = pd.read_csv('Placement_Data_Full_Class.csv')

df.head(200)


# In[24]:


df=df.drop('sl_no', axis=1) 


# In[25]:


print(df)


# ### b) Determine si existen o no datos faltantes, visualice la proporción de datos faltantes en la base de datos. ¿En qué variable hay mayor cantidad de datos faltantes? ¿Qué haría en este caso con los casos con datos faltantes? Discuta.

# In[26]:


df.info() #El salario tiene un 30% de datos faltantes y es la única variable con datos faltantes


# In[27]:


#para instalar missingno
#conda install -c conda-forge/label/gcc7 missingno

import missingno as msngo
msngo.matrix(df)


# Debido a la cantidad de datos faltantes, aproximadamente un 30% en la variable salary, se omitirá esta columna.

# In[28]:


df=df.drop('salary', axis=1) 


# ### c) ¿Cuántos egresados tienen una oferta de trabajo posterior al MBA? ¿Cuántos no? ¿Por qué es importante revisar esto antes o posterior a particionar la data en split de entrenamiento y test? Discuta.  

# In[29]:


df['status']


# In[30]:


print(pd.crosstab(index=df["status"], columns="count"))


# Si tenemos clases muy desbalanceadas, esto puede indicar un problema. También lo que puede ocurrir es que en el set de entrenamiento o prueba no caigan observaciones de ambas clases.

# ### d) Obtenga el set de entrenamiento y testeo en una proporción 70% y 30% respectivamente (recuerde definir apropiadamente X e y). Verifique que en cada set existen observaciones de ambas clases.

# Recordemos que antes de hacer el split, X debe ser una matriz numérica con toda la información de las variables predictoras. A la vez, y debe ser una columna numérica con la información de la variable target. 

# ##### Definir apropiadamente la matriz X de variables predictoras

# In[31]:


X=df.drop('status', axis=1) 

print(X) #Debemos realizar la transformación de todas las variables categóricas como corresponden


# In[32]:


X_num=X[['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']] #Guardamos todas las numéricas en un objeto X_num


# In[33]:


#Las variables categóricas son:

#gender, ssc_b, hsc_b, hsc_s, degree_t, workex, specialisation

X_full=pd.concat([X_num, #Variables numéricas
                 pd.get_dummies(df['gender'], drop_first = True, prefix='gender'), #Género 
                 pd.get_dummies(df['ssc_b'], drop_first = True, prefix='junta110'),   #Tipo de junta 1°-10° Grado 
                 pd.get_dummies(df['hsc_b'], drop_first = True, prefix='junta1112'), #Tipo de junta 11°-12° Grado 
                 pd.get_dummies(df['degree_t'], drop_first = True, prefix='areagr'),   #Área del grado ed superior
                 pd.get_dummies(df['workex'], drop_first = True, prefix='exp'),   #Experiencia laboral
                 pd.get_dummies(df['specialisation'], drop_first = True, prefix='spec'),   #Especialización MBA
                 ], axis=1 )


X_full.head()


# ##### Definir apropiadamente la matriz Y de la variable target

# In[34]:


print(df['status']) #Debemos recodificar Y en unos y ceros


# In[35]:


Y=pd.get_dummies(df['status'], drop_first = True).Placed  #get dummies crea una matrix, extraigo solo la columna Placed

print(Y)


# ##### Obtener el split de entrenamiento y prueba

# In[36]:


from sklearn.model_selection import train_test_split

#set de entrenamiento y prueba

X_train, X_test, y_train, y_test = train_test_split(X_full, Y, test_size=0.3, random_state=2020)


# In[37]:


print(pd.crosstab(index=y_train, columns="count")) 


# In[38]:


print(pd.crosstab(index=y_test, columns="count")) 


# ### e) Entrene el árbol de decisión fijando una profundidad máxima, grafique y describa los distintos perfiles que se asocian a una mayor probabilidad de tener una oferta de trabajo una vez egresados del MBA.

# In[40]:


from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
from matplotlib import pyplot as plt

model = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(12, 12))
plot_tree(model, 
          filled=True, 
          feature_names=X_full.columns,
          class_names=['Not Placed', 'Placed']) #Me gustaría más detalle en las reglas de decisión


# In[41]:


#conda install -c anaconda python-graphviz

import graphviz

treegr = tree.export_graphviz(model, out_file=None, 
                                 feature_names=X_full.columns,
                                 class_names=['Not Placed', 'Placed'],
                                 filled=True)

# Draw graph
graph =  graphviz.Source(treegr, format="png") 
graph


# ### f) Evalúe el poder predictivo del árbol anterior. ¿Confiaría usted en este árbol como una herramienta para predecir si un egresado del MBA encontrará trabajo? ¿Y para predecir si NO encontrará trabajo? Discuta. 

# In[42]:


model.predict(X_test) #Entrega la predicción de cada observación


# In[43]:


from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)

cm=confusion_matrix(y_test, y_pred)

from sklearn.metrics import ConfusionMatrixDisplay

cm_display = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
cm_display.plot(cmap='Blues');
plt.grid(None);


# In[44]:


print(classification_report(y_test, y_pred))


# A simple vista, pareciera que el modelo predice bastante bien los casos positivos (Placed). Sin embargo, notar que de 65 observaciones, 40 casos son Verdaderos positivos y 16 casos son Falsos positivos, lo que puede estar pasando es que el modelo predice demasiados positivos. A priori no podría tomar una decisión, pues hay muy pocos datos, volvería a recopilar más data y probar si el modelo realmente tiene capacidad predictiva.

# ### g) Pruebe alguna variante relacionada con los árboles de decisión y evalúe nuevamente el poder predictivo. ¿Cuál es el pero de esta opción? Comente.

# Usaré dos variantes, la primera: Buscar los mejores parámetros con validación cruzada. La segunda, un bosque de árboles.

# ##### Grilla de parámetros en Validación Cruzada

# In[46]:


import numpy as np
param_grid = { 'criterion':['gini','entropy'],'max_depth': np.arange(2, 10)}   #Grilla de valores a probar


# In[47]:


model_2=DecisionTreeClassifier()


# In[48]:


from sklearn.model_selection import GridSearchCV

nfolds=10

dtree_gscv = GridSearchCV(model_2, param_grid, cv=nfolds)

dtree_gscv.fit(X_train, y_train)

dtree_gscv.best_params_


# Obtenemos que los mejores parámetros son: Criterio de separación de Entropía o Ganancia de Información y una profundidad máxima de 10.

# In[49]:


model_cv = DecisionTreeClassifier(max_depth=4, criterion='entropy').fit(X_train, y_train)


# In[50]:


treegr = tree.export_graphviz(model_cv, out_file=None, 
                                 feature_names=X_full.columns,
                                 class_names=['Not Placed', 'Placed'],
                                 filled=True)

# Draw graph
graph =  graphviz.Source(treegr, format="png") 
graph


# In[51]:


y_pred_cv = model_cv.predict(X_test)
print(classification_report(y_test, y_pred_cv))


# In[52]:


cm=confusion_matrix(y_test, y_pred_cv)
cm_display = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
cm_display.plot(cmap='Oranges');
plt.grid(None);


# Mejoró el Recall para la clase Not Placed (Negativa) y se niveló un poco más la precisión en ambos grupos. Mejoró el modelo, pero de todas formas considero que faltan más datos para realmente establecer un quórum de la predicción del modelo. 

# El PERO que tiene esta opción es que en un caso real, no siempre podemos probar distintos valores de parámetros como hicimos recién, el costo puede ser muy grande en una base de datos de millones de registros y además, necesitamos acotar la grilla de parámetros, lo cuál significa que pudiéramos no necesariamente llegar al mejor valor del parámetro.

# ##### Bosque de Árboles

# In[53]:


from sklearn.ensemble import RandomForestClassifier

bosque = RandomForestClassifier(n_estimators=100, random_state=2020, criterion='entropy').fit(X_train, y_train)

y_pred_bosque=bosque.predict(X_test)


# In[54]:


print(classification_report(y_test, y_pred_bosque))


# In[55]:


cm=confusion_matrix(y_test, y_pred_bosque)
cm_display = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
cm_display.plot(cmap='Greens');
plt.grid(None);


# Mejoran bastante las métricas! De 65 observaciones, existen 12 observaciones erróneamente clasificadas, siendo 2 verdaderas Placed y 10 falsas Not Placed. El PERO de esta variante es que al ser 100 árboles, no tenemos un modelo solo para interpretar y seguir los caminos (como si fuera un solo árbol de decisión), lo cual puede caer en el concepto de 'caja negra'. Sin embargo, mejora el poder predictivo, todo va a depender de lo que necesitemos! (También de los recursos disponibles).

# Otro PRO de un bosque de árboles es que podemos medir la importancia de los features a la hora de predecir cuándo un egresado del MBA tendrá una oferta de trabajo:

# In[56]:


feature_imp = pd.Series(bosque.feature_importances_,index=X_full.columns).sort_values(ascending=False)
feature_imp


# In[57]:


import seaborn as sns

fig, ax = plt.subplots(figsize=(12, 8))

sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()


# ### Gracias!! Y mucho ánimo en el semestre :D
