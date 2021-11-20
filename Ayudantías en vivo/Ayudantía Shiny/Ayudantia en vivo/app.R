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
  width = 250, #Tamanio del sidebar
  sidebarMenu(
    id='sidebar',                            # Nombre identificador del sidebar
    menuItem('Tabla con datos de pokemones', # Nombre de la pestana 1 en el dash
             tabName = 'menu1'),
    menuItem('Graficos por tipo de pokemon', # Nombre de la pestana 2 en el dash
             tabName = 'menu2', startExpanded = T,
             div(id = "sidebar1",
                 conditionalPanel("input.sidebar === 'menu21'",
                                  selectizeInput("select_tipo1",
                                                 "Seleccione Tipo",
                                                 choices = unique(Pokemon$`Type 1`),
                                                 selected = "", width = "300px",
                                                 multiple = F))),
             menuItem('Grafico de dispersion', tabName="menu21", 
                      icon = icon("zoom-in",lib = "glyphicon")))
  )
)


## Cuerpo de cada vineta del menu
body <- dashboardBody(   
        shinyDashboardThemes(theme = "onenote"),
        tabItems(
          tabItem(tabName = "menu21" ,
                  h1("Grafica de dispersion por tipo de pokemon")
                  )
        )
)

ui <- dashboardPage(header, sidebar, body)

server <- function(input, output) {}

shinyApp(ui = ui, server = server)


