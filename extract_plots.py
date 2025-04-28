import json
import nbformat
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from nbconvert import PythonExporter
import re

def extract_plotly_data(notebook_path):
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    plots_data = {
        'publication_trends': None,
        'citation_analysis': None,
        'topic_modeling': None,
        'collaboration_networks': None
    }
    
    # Load the data
    df = pd.read_csv('Combined_ResearchOutputs_Final.csv')
    
    # Publication Trends
    pub_trends = df.groupby('OutputYear').size().reset_index(name='count')
    plots_data['publication_trends'] = {
        'x': pub_trends['OutputYear'].tolist(),
        'y': pub_trends['count'].tolist()
    }
    
    # Citation Analysis
    plots_data['citation_analysis'] = {
        'x': df['OutputCitationCount'].dropna().tolist()
    }
    
    # Topic Modeling (using Keywords)
    # Extract all keywords and count their frequencies
    keywords = df['Keywords'].dropna().str.split(',').explode().str.strip()
    top_keywords = keywords.value_counts().head(5)
    plots_data['topic_modeling'] = {
        'labels': top_keywords.index.tolist(),
        'values': top_keywords.values.tolist()
    }
    
    # Collaboration Networks (using Authors)
    # Split authors and count their frequencies
    authors = df['Authors'].dropna().str.split(';').explode().str.strip()
    top_authors = authors.value_counts().head(10)
    plots_data['collaboration_networks'] = {
        'nodes': [{'name': author} for author in top_authors.index],
        'links': [{'source': 0, 'target': i, 'value': int(count)} 
                 for i, count in enumerate(top_authors.values) if i > 0]
    }
    
    # Save the extracted data
    with open('docs/js/plot_data.json', 'w') as f:
        json.dump(plots_data, f, indent=2)
    
    return plots_data

if __name__ == '__main__':
    notebook_path = 'Group1_Project3_P1Q2(EDA)+P2(DataMining).ipynb'
    plot_data = extract_plotly_data(notebook_path) 