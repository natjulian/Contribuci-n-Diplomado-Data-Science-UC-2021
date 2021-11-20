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
header <- dashboardHeader( )

## Menu de navegacion del dashboard:
sidebar <- dashboardSidebar( )

## Cuerpo de cada vineta del menu
body <- dashboardBody(   
        shinyDashboardThemes(theme = "flat_red")
)

ui <- dashboardPage(header, sidebar, body)

server <- function(input, output) {}

shinyApp(ui = ui, server = server)
