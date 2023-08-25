# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import numpy as np
import dash_bootstrap_components as dbc

info = pd.DataFrame(np.load('Ctranspath_neig15_3d_info-2.npy').T)

embeddings = pd.DataFrame(np.load('Ctranspath_neig15_3d-2.npy'))
Ctranspath = pd.concat([embeddings, info], axis=1)
Ctranspath.columns = ['x', 'y', 'z', 'survival', 'slide']


embeddings = pd.DataFrame(np.load('Resnet50_neig15_3d-2.npy'))
Resnet50 = pd.concat([embeddings, info], axis=1)
Resnet50.columns = ['x', 'y', 'z', 'survival', 'slide']

embeddings = pd.DataFrame(np.load('Resnet50_early_neig15_3d.npy'))
Resnet50_early = pd.concat([embeddings, info], axis=1)
Resnet50_early.columns = ['x', 'y', 'z', 'survival', 'slide']

embeddings = pd.DataFrame(np.load('SimCLR_neig15_3d-2.npy'))
SimCLR = pd.concat([embeddings, info], axis=1)
SimCLR.columns = ['x', 'y', 'z', 'survival', 'slide']

Data_dict = {}
Data_dict["Ctranspath"] = Ctranspath
Data_dict["Resnet50"] = Resnet50
Data_dict["Resnet50_early"] = Resnet50_early
Data_dict["SimCLR"] = SimCLR


feature_extractor_descriptions = {
    'Ctranspath': "The paper from which this extractor is taken introduces a new technique for classifying histopathological images without relying on labeled data. It uses a **Transformer-based unsupervised contrastive learning approach**, which enables the model to learn meaningful features from images by capturing both similarities and differences. The method uses a self-attention mechanism to process image patches and learns to distinguish between patches from the same image and different images. Experimental results show that the proposed approach performs well compared to traditional supervised methods, even with limited labeled data. This suggests its potential for improving histopathological image classification without the need for extensive labeled datasets. Learn more [here](https://www.sciencedirect.com/science/article/pii/S1361841522002043).",
    'Resnet50': "The ResNet (Residual Network) architecture can be employed to create a feature encoding by utilizing its deep layers to capture intricate patterns within data. To use ResNet for feature encoding, you can take a pre-trained ResNet model (e.g., ResNet-50) and remove its final classification layers. The remaining layers act as a feature extractor, transforming input images into high-level features. Pass your input images through these layers to obtain a feature representation, also known as feature encoding. The standard dimension for this encoding is 2048. Learn more [here](https://medium.com/analytics-vidhya/cnn-transfer-learning-with-vgg-16-and-resnet-50-feature-extraction-for-image-retrieval-with-keras-53320c580853)",
    'Resnet50_early': "The procedure is very similar to the one adopted with the ••Resnet50•• feature extractor. The only difference is that we take the output of two layers of Resnet50, we global average pool the resulting partial encoding and we concatenate them. This is done in order to try to produce more low level feature encodings. Learn more [here](https://medium.com/the-owl/extracting-features-from-an-intermediate-layer-of-a-pretrained-model-in-pytorch-c00589bda32b) ",
    'SimCLR': "The SimCLR (Contrastive Learning of Visual Representations) architecture is utilized to create a feature encoding by training a neural network to understand the relationships between data samples. To employ SimCLR for feature encoding, you form pairs of augmented images from your dataset and feed them into the network. The network then projects these images into a shared embedding space, ensuring that the representations of similar images are closer while dissimilar ones are farther apart. The embeddings produced by the network serve as a feature encoding, capturing the underlying structures in the data. Learn more [here](https://arxiv.org/abs/2002.05709)."
}

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
    
    dbc.Row([
        dcc.Markdown(id="feature-extractor-description")
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
            dcc.Graph(figure={}, id='controls-and-graph', style={'width': '100vh', 'height': '100vh'})
        ]),


], fluid=True)



@callback(
    Output(component_id='controls-and-graph', component_property='figure'),
    Output(component_id='feature-extractor-description', component_property='children'),
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

    description_html = feature_extractor_descriptions[feature_extractor]

    fig.update_traces(marker=dict(size=3))
    fig.update_layout(legend=dict(title_font_family="Times New Roman",
                              font=dict(size= 10), 
                              itemwidth=30, 
                              y=1.02
                    ))

    return fig, description_html

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
