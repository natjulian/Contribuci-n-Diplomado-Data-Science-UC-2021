library(shiny)          # App web
library(shinydashboard) # Para formato dashboard
library(shinyjs)        # Para usar entorno javascript
library(highcharter)    # Para graficos interactivos
library(DT)             # Para tablas
library(dplyr)          # Para manipulacion de bases de datos
library(dashboardthemes) #Para modificar el theme de un shinydashboard

### Base de datos a utilizar
library(readr)
Pokemon <- read_csv("Pokemon.csv")

## Barra superior del dashboard:
header <- dashboardHeader(
  title="Pokemon Analytics",  # Titulo del dashboard
  titleWidth=300,              # Tamanio del dashboard
  #Anadiendo notificaciones en el dashboard
  dropdownMenu(type="message",                            # Menu emergente del tipo 'mensaje'
               messageItem(
                 from = "Las ayudantes dicen:", #'emisor del mensaje'
                 message = HTML("Dudas? No dudes en consultar :)"), # Mensaje
                 icon = icon("question"), #icono del mensaje
                 time = substr(Sys.time(), start=12, stop=16)        # Hora que queremos que aparezca
               )),
  dropdownMenu(type = "notifications",
               notificationItem(
                 text = "Ultima ayudantia!!",
                 icon("users")
               )),
  dropdownMenu(type = "tasks",
               badgeStatus = "success",
               taskItem(value = 90, color = "green",
                        "Status Diplomado"
               )))

## Menu de navegacion del dashboard:
sidebar <- dashboardSidebar(
           width=250,
           sidebarMenu(
             id='sidebar',                           # Nombre identificador del sidebar
             menuItem('PokeTabla',# Nombre de la pestana 1 en el dash
                      tabName = 'menu1')
           ))

## Cuerpo de cada vineta del menu
body <- dashboardBody(   
        shinyDashboardThemes(theme = "onenote")
)

ui <- dashboardPage(header, sidebar, body)

server <- function(input, output) {}

shinyApp(ui = ui, server = server)


