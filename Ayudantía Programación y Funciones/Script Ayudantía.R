### AYUDANTÍA 12 DE JUNIO:: PROGRAMACIÓN Y FUNCIONES


#Paquetes necesarios:
library(randomNames)
library(dplyr)
library(ggplot2)
library(readr)
library(purrr)
library(skimr)

#PREGUNTA 1)

#Considere los siguientes datos ficticios en el cual se posee por 
#alumno el registro de calificaciones de las interrogaciones y promedio 
#de laboratorio para un curso de la universidad:

#--------------------------------------------------------------
set.seed(2021) #Fija semilla 2021
nombres<-randomNames(50, ethnicity = 4) #Generador de nombres aleatorios hispánicos 

set.seed(2021) #Fija semilla 2021
interrogaciones<-matrix(round(runif(150, min=1, max=7),1), ncol=3) #Crea una matriz con calificaciones aleatorias

set.seed(2021) #Fija semilla 2021
laboratorio<-round(runif(nrow(interrogaciones), min=4.5, max=7),1)

data<-data.frame(nombres, interrogaciones, laboratorio)
names(data)<-c("Nombre", "I1", "I2", "I3", "prom_laboratorio")
#--------------------------------------------------------------

#La nota de presentación al examen se calcula como sigue:
  
# NP=I×70%+L×30%
# Donde I corresponde al promedio de las tres interrogaciones y 
# L al promedio de laboratorio.


## Pregunta 1) :: Item a)


#En base a la nota de presentación se tienen los siguientes casos:
  
# - Si NP≥5,0 y todas las interrogaciones son mayores o iguales a 4.0 el 
#alumno aprueba el curso con nota igual a la nota de presentación a examen.

# - Si NP<3,95 el alumno reprueba con opción para rendir examen y subir la 
# nota final del curso.
# Todos los otros casos rinden examen.

# En este contexto, sería de mucha utilidad tener una función que reciba 
#un data.frame y entregue (en objetos separados) las calificaciones y 
#el resultado de presentación al examen (Aprueba sin examen, Reprueba con opción a 
#rendir examen, Rinde examen) incluyendo el nombre del alumno. 

#Explore distintas maneras de crear la función deseada con ayuda de las funciones 
#vistas en clase y compare el tiempo computacional requerido de cada una usando 
#la función system.time().


resultado_examen1<-function(df){
  
  df$NP<-round(((df$I1+df$I2+df$I3)/3)*0.7+df$prom_laboratorio*0.3, 2)
  df$resultado<-NA
  
  for(i in 1: nrow(df)){
    
    if((df$NP[i]>=5)&(df$I1[i]>=4)&(df$I2[i]>=4)&(df$I3[i]>=4)){
      
      df$resultado[i]<-"Aprueba sin examen"
    }
    
    else if(df$NP[i]<3.95){
      df$resultado[i]<-"Reprueba con opción a rendir examen"
    }
    
    else{
      df$resultado[i]<-"Rinde examen"
    }
  }

  return(list("Notas"=df[,1:5], "Status"=df[, c(1, 7)]))
  
}


resultado_examen1(data)$Notas

resultado_examen1(data)$Status

#### FORMA 2: UTILIZANDO APPLY

resultado_examen2<-function(df){
  
    df$NP<-apply(df[,-1], MARGIN=1, FUN=function(x){
    ((x[1]+x[2]+x[3])/3)*0.7+x[4]*0.3} )
  
  df$resultado<-NA
  
  for(i in 1: nrow(df)){
    
    if((df$NP[i]>=5)&(df$I1[i]>=4)&(df$I2[i]>=4)&(df$I3[i]>=4)){
      
      df$resultado[i]<-"Aprueba sin examen"
    }
    
    else if(df$NP[i]<3.95){
      df$resultado[i]<-"Reprueba con opción a rendir examen"
    }
    
    else{
      df$resultado[i]<-"Rinde examen"
    }
  }
  
  return(list("Notas"=df[,1:5], "Status"=df[, c(1, 7)]))
  
}

resultado_examen2(data)$Notas

resultado_examen2(data)$Status

###FORMA 3 CON DPLYR: 

resultado_examen3<-function(df){
  
  df<-df%>%
      mutate(NP=round(((I1+I2+I3)/3)*0.7+prom_laboratorio*0.3, 2),
             resultado=ifelse((NP>=5)&(I1>=4)&(I2>=4)&(I3>=4), 
             "Aprueba sin examen", 
             ifelse(NP<3.95, "Reprueba con opción de rendir examen", 
                                          "Rinde examen")))
  
  return(list("Notas"=df[,1:5], "Status"=df[, c(1, 7)]))
  
}


resultado_examen3(data)$Notas

resultado_examen3(data)$Status


system.time(resultado_examen1(data))

system.time(resultado_examen2(data))

system.time(resultado_examen3(data))



## Pregunta 1) :: Item b)

#Evalúe la data.frame antes creada en la función creada, 
#¿cuántos alumnos aprobaron sin examen? Visualice esta información en un gráfico.



resultado_examen3(data)$Status


table(resultado_examen3(data)$Status[,2])


ggplot(resultado_examen3(data)$Status, aes(resultado))+geom_bar()


ggplot(resultado_examen3(data)$Status, aes(resultado))+
  geom_bar(aes(y=(..count..)/sum(..count..)))+ylim(0,1)



#PREGUNTA 2)

#La secuencia de Fibonacci es una sucesión definida por recurrencia. 
#Esto significa que para calcular un término de la sucesión se necesitan 
#los términos que le preceden.

#0,1,1,2,3,5,8,13,21,34,55,89,144,233,377,...


#Pregunta 2):: Item a)

#Cree dos funciones distintas que retornen el n-ésimo valor de 
#la serie de Fibonacci. Sugerencia: crear una función con if/ else if 
#que sea auto-recursiva y otra con un proceso condicional “while”.

###FUNCIÓN AUTORECURSIVA EXPLÍCITA

fib<-function(n){
  
  if(n==1){
    return(0)
  }
  
  else if (n==2){
    return(1)
  }
  
  else {
    return(fib(n-1)+fib(n-2))
  }

}

fib(4)

fib(10)

fib(100)


### CON WHILE


fib2<-function(n){
  valores<-c()
  valores[1]<-0
  valores[2]<-1
  
  i<-2
  while(i<=n){
    next_val<-sum(tail(valores, 2))
    valores<-c(valores, next_val)
    i<-i+1
  }
  return(valores[n])
}

fib2(4)

fib2(10)


#Pregunta 2):: Item b)
#Compare los tiempos de demora de ejecución de ambas funciones utilizando 
#la función system.time(). Explique por qué cree que se generan los 
#resultados obtenidos.

system.time(fib(30))

system.time(fib2(30))



#PREGUNTA 3)

#En clases, utilizamos la base de datos paises de la librería datos 
#para generar archivos .csv según cada continente (spoiler para los que
#no alcanzaron a verlo en clases).

#Supongamos que nos entregan estas bases de datos y que esta es nuestra 
#única fuente de información. Ud al ver esta cantidad de archivos 
#piensa en lo engorroso que puede ser tener que importar cada uno de ellos, 
#por lo que se le ocurre la brillante idea de crear un único data frame con todos los archivos 
#sin tener que importarlos uno por uno. 

#Para esto deberá seguir los siguientes pasos:
  
#Construir un vector con el directorio de los archivos.
#Seleccionar la función adecuada para leer las bases de datos.
#utilizar purrr::map convenientemente para alcanzar lo deseado.

dir_archivos<-list.files("Bases_Continentes", full.names = TRUE)

df_paises<-map_df(dir_archivos, read_csv)

list_paises<-map(dir_archivos, read_csv)

names(list_paises)<-substr(dir_archivos, 19, nchar(dir_archivos)-4)

list_paises$Africa

list_paises$Asia


#PREGUNTA 4)


#Instacart es una empresa que ofrece servicios de delivery de alimentos 
#en los Estados Unidos y Canadá. Los usuarios seleccionan los productos 
#del despacho a través de su sitio web o de la aplicación móvil. 
#La información de los pedidos se encuentra en las bases de datos: 

#departments: Indica el departamento que provee el producto (ejemplo: pets, frozen, bakery, etcétera)
#order products train: Contiene por orden (pedido) los productos despachados                                   |
#products: Contiene información de los productos, nombre de los productos e ID de los productos  


#Pregunta 4):: Item a)


#Cargue las bases de datos y realice los cruces necesarios para tener para cada pedido 
#la información completa (características del producto y departamento a cargo de proveerlo).

nombres<-list.files("Base_Instacart", full.names=TRUE)

Datas<-map(nombres, read_csv)

names(Datas)<-substr(nombres, 16, nchar(nombres)-4)

Datas$departments

Datas$order_products__train

Datas$products



#Pregunta 4):: Item b)

#Para el departamento frozen obtenga por pedido, la cantidad de productos del pedido 
#(cada fila corresponde a un producto, por ende se debe realizar un conteo de la 
#cantidad de filas por id del pedido) y realice un histograma de la cantidad de 
#productos por pedido.







#Pregunta 4):: Item c)

#Interesa tener la misma información anterior para los departamentos pets, breakfast, 
#frozen, produce, de modo que cada departamento pueda recibir y accionar sus propios resultados. 
#Utilice la función map para replicar lo anterior a cada departamento.








#Pregunta 4):: Item d)

#Con la función ggsave() es posible guardar cada gráfico creado en el apartado anterior en 
#formato png. Utilice la función map() para exportar los gráficos.



  