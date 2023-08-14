# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import numpy as np
import dash_bootstrap_components as dbc

info = pd.DataFrame(np.load('/Users/lorenzotarricone/Desktop/Pasteur/App/Data_embeddings/Ctranspath_neig15_3d_info-2.npy').T)

embeddings = pd.DataFrame(np.load('/Users/lorenzotarricone/Desktop/Pasteur/App/Data_embeddings/Ctranspath_neig15_3d-2.npy'))
Ctranspath = pd.concat([embeddings, info], axis=1)
Ctranspath.columns = ['x', 'y', 'z', 'survival', 'slide']


embeddings = pd.DataFrame(np.load('/Users/lorenzotarricone/Desktop/Pasteur/App/Data_embeddings/Resnet50_neig15_3d-2.npy'))
Resnet50 = pd.concat([embeddings, info], axis=1)
Resnet50.columns = ['x', 'y', 'z', 'survival', 'slide']

embeddings = pd.DataFrame(np.load('//Users/lorenzotarricone/Desktop/Pasteur/App/Data_embeddings/Resnet50_early_neig15_3d.npy'))
Resnet50_early = pd.concat([embeddings, info], axis=1)
Resnet50_early.columns = ['x', 'y', 'z', 'survival', 'slide']

embeddings = pd.DataFrame(np.load('/Users/lorenzotarricone/Desktop/Pasteur/App/Data_embeddings/SimCLR_neig15_3d-2.npy'))
SimCLR = pd.concat([embeddings, info], axis=1)
SimCLR.columns = ['x', 'y', 'z', 'survival', 'slide']

Data_dict = {}
Data_dict["Ctranspath"] = Ctranspath
Data_dict["Resnet50"] = Resnet50
Data_dict["Resnet50_early"] = Resnet50_early
Data_dict["SimCLR"] = SimCLR

# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.LUX]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# App layout
app.layout = dbc.Container([
    
    dbc.Row([
        html.Div('UMAP Embedding Visualizer', className="text-primary text-center fs-3")
    ]),

    html.Br(),
    
    dbc.Row([
        dbc.Label("Feature Extractor selector", align="center"),
        dcc.Dropdown(
                    id="Feature-extractor-selector",
                    options=['Ctranspath', 'Resnet50', 'Resnet50_early','SimCLR' ],
                    value='Ctranspath',
                ),
    ]),

    html.Br(),
    
    dbc.Row([
        dbc.Label("Stainer selector", align="center"),
        dcc.Dropdown(
                    id="Staining-selector",
                    options=['A', 'B', 'C','D'],
                    value='A',
                ),
    ]),

    html.Br(),
    
    dbc.Row([
        dbc.RadioItems(options=['survival', 'slide'],
                       className="btn-group",
                       inputClassName="btn-check",
                       labelClassName="btn btn-outline-primary",
                       labelCheckedClassName="active",
                       value='slide',
                       inline=True,
                       id='Visualiz-selector')
    ]),

    html.Br(),

    dbc.Row([
            dcc.Graph(figure={}, id='controls-and-graph')
        ]),


], fluid=True)



@callback(
    Output(component_id='controls-and-graph', component_property='figure'),
    Input(component_id='Feature-extractor-selector', component_property='value'),
    Input(component_id='Staining-selector', component_property='value'),
    Input(component_id='Visualiz-selector', component_property='value')
)
def update_graph(feature_extractor, staining, visualiz):
    df = Data_dict[feature_extractor]
    lett_df = df[df.slide.str.contains(staining)]

    fig = px.scatter_3d(lett_df, x='x', y='y', z='z',
              color = visualiz,
              width=500, height=500
    )

    fig.update_traces(marker=dict(size=3))
    fig.update_layout(legend=dict(title_font_family="Times New Roman",
                              font=dict(size= 10)
                    ))

    return fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
