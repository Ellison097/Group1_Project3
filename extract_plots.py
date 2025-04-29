import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import nbformat
from nbconvert import PythonExporter
import re

def extract_markdown_content(notebook_path):
    """Extract markdown content from the notebook"""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    markdown_content = []
    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            markdown_content.append(cell.source)
    
    return markdown_content

def extract_plotly_data(notebook_path):
    # Load the data
    df = pd.read_csv('Combined_ResearchOutputs_Final.csv')
    
    # Extract markdown content
    markdown_content = extract_markdown_content(notebook_path)
    
    plots_data = {
        'publication_trends': None,
        'citation_analysis': None,
        'topic_modeling': None,
        'collaboration_networks': None,
        'rdc_productivity': None,
        'author_productivity': None,
        'venue_impact': None,
        'citation_velocity': None,
        'topic_evolution': None,
        'markdown_content': markdown_content,
        'yearly_publications': None,
        'rdc_collaboration': None,
        'author_network': None,
        'keyword_cloud': None,
        'citation_network': None,
        'venue_network': None,
        'author_citation_impact': None,
        'rdc_citation_impact': None,
        'publication_types': None,
        'citation_geography': None,
        'author_geography': None,
        'rdc_geography': None,
        'publication_language': None,
        'citation_language': None,
        'author_language': None,
        'rdc_language': None,
        'publication_funding': None,
        'citation_funding': None,
        'author_funding': None,
        'rdc_funding': None,
        'publication_collaboration': None,
        'citation_collaboration': None,
        'author_collaboration': None,
        'rdc_collaboration': None,
        'publication_impact': None,
        'citation_impact': None,
        'author_impact': None,
        'rdc_impact': None
    }
    
    # Publication Trends
    pub_trends = df.groupby('OutputYear').size().reset_index(name='count')
    plots_data['publication_trends'] = {
        'x': pub_trends['OutputYear'].tolist(),
        'y': pub_trends['count'].tolist()
    }
    
    # RDC Productivity
    rdc_counts = df.groupby('ProjectRDC').size().sort_values(ascending=False).head(10)
    plots_data['rdc_productivity'] = {
        'x': rdc_counts.index.tolist(),
        'y': rdc_counts.values.tolist()
    }
    
    # Citation Analysis
    plots_data['citation_analysis'] = {
        'x': df['OutputCitationCount'].dropna().tolist()
    }
    
    # Author Productivity
    authors = df['Authors'].dropna().str.split(';').explode().str.strip()
    top_authors = authors.value_counts().head(15)
    plots_data['author_productivity'] = {
        'x': top_authors.index.tolist(),
        'y': top_authors.values.tolist()
    }
    
    # Venue Impact
    venue_citations = df.groupby('OutputVenue')['OutputCitationCount'].mean().sort_values(ascending=False).head(10)
    plots_data['venue_impact'] = {
        'x': venue_citations.index.tolist(),
        'y': venue_citations.values.tolist()
    }
    
    # Citation Velocity
    df['YearsSincePublication'] = 2024 - df['OutputYear']
    df['CitationVelocity'] = df['OutputCitationCount'] / df['YearsSincePublication']
    top_velocity = df.sort_values('CitationVelocity', ascending=False).head(10)
    plots_data['citation_velocity'] = {
        'titles': top_velocity['OutputTitle'].tolist(),
        'velocity': top_velocity['CitationVelocity'].tolist()
    }
    
    # Topic Modeling
    keywords = df['Keywords'].dropna().str.split(',').explode().str.strip()
    top_keywords = keywords.value_counts().head(10)
    plots_data['topic_modeling'] = {
        'labels': top_keywords.index.tolist(),
        'values': top_keywords.values.tolist()
    }
    
    # Topic Evolution
    yearly_keywords = df.groupby('OutputYear')['Keywords'].apply(
        lambda x: x.dropna().str.split(',').explode().str.strip().value_counts().head(5)
    )
    plots_data['topic_evolution'] = {
        'years': yearly_keywords.index.get_level_values(0).unique().tolist(),
        'keywords': yearly_keywords.index.get_level_values(1).unique().tolist(),
        'counts': yearly_keywords.values.tolist()
    }
    
    # Collaboration Networks
    authors = df['Authors'].dropna().str.split(';').explode().str.strip()
    top_authors = authors.value_counts().head(15)
    plots_data['collaboration_networks'] = {
        'nodes': [{'name': author} for author in top_authors.index],
        'links': [{'source': 0, 'target': i, 'value': int(count)} 
                 for i, count in enumerate(top_authors.values) if i > 0]
    }
    
    # Yearly Publications by Type
    yearly_pubs = df.groupby(['OutputYear', 'OutputType']).size().unstack().fillna(0)
    plots_data['yearly_publications'] = {
        'years': yearly_pubs.index.tolist(),
        'types': yearly_pubs.columns.tolist(),
        'counts': yearly_pubs.values.tolist()
    }
    
    # RDC Collaboration Network
    rdc_collab = df.groupby('ProjectRDC').size().sort_values(ascending=False).head(10)
    plots_data['rdc_collaboration'] = {
        'nodes': [{'name': rdc} for rdc in rdc_collab.index],
        'links': [{'source': 0, 'target': i, 'value': int(count)} 
                 for i, count in enumerate(rdc_collab.values) if i > 0]
    }
    
    # Author Network
    author_pairs = df['Authors'].dropna().apply(lambda x: x.split(';')).explode().str.strip()
    author_network = author_pairs.value_counts().head(20)
    plots_data['author_network'] = {
        'nodes': [{'name': author} for author in author_network.index],
        'links': [{'source': 0, 'target': i, 'value': int(count)} 
                 for i, count in enumerate(author_network.values) if i > 0]
    }
    
    # Keyword Cloud
    keyword_counts = df['Keywords'].dropna().str.split(',').explode().str.strip().value_counts().head(50)
    plots_data['keyword_cloud'] = {
        'words': keyword_counts.index.tolist(),
        'counts': keyword_counts.values.tolist()
    }
    
    # Citation Network
    citation_network = df.groupby('OutputVenue')['OutputCitationCount'].sum().sort_values(ascending=False).head(20)
    plots_data['citation_network'] = {
        'nodes': [{'name': venue} for venue in citation_network.index],
        'links': [{'source': 0, 'target': i, 'value': int(count)} 
                 for i, count in enumerate(citation_network.values) if i > 0]
    }
    
    # Save the extracted data
    with open('docs/js/plot_data.json', 'w') as f:
        json.dump(plots_data, f, indent=2)
    
    return plots_data

if __name__ == '__main__':
    notebook_path = 'Group1_Project3_P1Q2(EDA)+P2(DataMining).ipynb'
    plot_data = extract_plotly_data(notebook_path) 