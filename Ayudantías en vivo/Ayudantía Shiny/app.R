library(shiny) #App web
library(shinydashboard) #Para formato dashboard
library(shinyjs) #Para usar entorno javascript
library(highcharter) #Para graficos interactivos
library(DT) #Para tablas
library(dplyr) #Para manipulacion de bases de datos
library(dashboardthemes) #Para modificar el theme de un shinydashboard
###Base de datos a utilizar
library(readr)
Pokemon <- read_csv("Pokemon.csv")
#Barra superior del dashboard:
header <- dashboardHeader(
    title= a(href='https://www.pokemon.com/el/',
             img(src='https://upload.wikimedia.org/wikipedia/commons/9/98/International_Pok%C3%A9mon_logo.svg',
                 width='200px',height='50px')), # Titulo del dashboard con logo
    titleWidth=300, #Tamano del dashboard
    #Anadiendo notificaciones en el dashboard
    dropdownMenu(
        type="message", # Menu emergente del tipo 'mensaje'
        messageItem(
            from = "Las ayudantes dicen:", #'emisor del mensaje'
            message = HTML("Dudas? No dudes en consultar :)"), # Mensaje
            icon = icon("question"), #icono del mensaje
            time = substr(Sys.time(), start=12, stop=16) # Hora que queremos que aparezca
        )),
    dropdownMenu(
        type = "notifications", #Menu emergente del tipo 'notificacion'
        notificationItem(
            text = "Ultima ayudantia!!",
            icon("users")
        )),
    dropdownMenu(
        type = "tasks", badgeStatus = "success", #Menu emergente del tipo task
        taskItem(value = 90, color = "green",
                 "Status Diplomado"
        ))
)
#Menu de navegacion del dashboard:
sidebar <- dashboardSidebar(
    width = 250, #Tamanio del sidebar
    sidebarMenu(
        id='sidebar', # Nombre identificador del sidebar
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


#Cuerpo de cada vinieta del menu
body <- dashboardBody(
    shinyDashboardThemes(
        theme = "onenote"),
    tabItems(
        tabItem(tabName = "menu1",
                h1("Tabla de datos"),
                fluidRow(dataTableOutput("table1"))
        ),
        tabItem(tabName = "menu21",
                h1("Graficas de dispersion por tipo"),
                fluidRow(highchartOutput("graf1")))
    )
)
ui <- dashboardPage(header, sidebar, body)
server <- function(input, output) {
    output$table1 <- renderDataTable({
        datatable(na.omit(Pokemon[,-1]), # Datos a mostrar
                  filter = list(position = "top"), # Posicion del buscador
                  options = list(dom="t", # Elimina un search grande de arriba
                                 #autoWidth = TRUE , #esto hace que se ajuste el ancho
                                 pageLength = 8, #Se muestran 8 registros por pagina
                                 scrollX = TRUE)) # Se avanza con una barra deslizante horizontal
    })
    output$graf1 <- renderHighchart({
        hchart(Pokemon %>%
                   filter(`Type 1`==input$select_tipo1), # Filtra por el tipo de pokemon seleccionado
               "scatter", hcaes(x = Speed, y = Attack)) %>% # Grafico de dispersion y variables x e y
            hc_yAxis(title = list(text = "Attack"))%>% #titulo eje y
            hc_title(text=paste("Velocidad y Ataque de Pokemones de tipo",
                                input$select_tipo1), # Titulo del grafico
                     align = "center")%>%
            hc_tooltip(pointFormat= "Attack: {point.y} <br>
Speed:{point.x}" ) %>% # tooltip desplegable al posicionar el raton encima de cada punto
            hc_add_theme(hc_theme_google()) #tema a usar
    })
}
shinyApp(ui = ui, server = server)

