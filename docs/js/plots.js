// Load the plot data
fetch('js/plot_data.json')
    .then(response => response.json())
    .then(data => {
        // Publication Trends Plot
        const publicationTrendsData = {
            x: data.publication_trends.x,
            y: data.publication_trends.y,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Publications',
            line: {
                color: '#1f77b4',
                width: 2
            },
            marker: {
                size: 6
            }
        };

        const publicationTrendsLayout = {
            title: 'Publication Trends Over Time',
            xaxis: {
                title: 'Year',
                tickmode: 'linear',
                dtick: 2
            },
            yaxis: {
                title: 'Number of Publications'
            },
            hovermode: 'closest',
            showlegend: true
        };

        Plotly.newPlot('publication-trends', [publicationTrendsData], publicationTrendsLayout);

        // Citation Analysis Plot
        const citationAnalysisData = {
            x: data.citation_analysis.x,
            type: 'histogram',
            name: 'Citations',
            nbinsx: 50,
            marker: {
                color: '#2ca02c'
            }
        };

        const citationAnalysisLayout = {
            title: 'Citation Distribution',
            xaxis: {
                title: 'Number of Citations',
                type: 'log'
            },
            yaxis: {
                title: 'Frequency'
            },
            bargap: 0.1
        };

        Plotly.newPlot('citation-analysis', [citationAnalysisData], citationAnalysisLayout);

        // Topic Modeling Plot
        const topicModelingData = {
            labels: data.topic_modeling.labels,
            values: data.topic_modeling.values,
            type: 'pie',
            hole: 0.4,
            marker: {
                colors: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            }
        };

        const topicModelingLayout = {
            title: 'Research Topics Distribution',
            showlegend: true,
            legend: {
                orientation: 'h',
                y: -0.1
            }
        };

        Plotly.newPlot('topic-modeling', [topicModelingData], topicModelingLayout);

        // Collaboration Networks Plot
        const collaborationNetworksData = {
            type: 'sankey',
            node: {
                pad: 15,
                thickness: 20,
                line: {
                    color: 'black',
                    width: 0.5
                },
                label: data.collaboration_networks.nodes.map(n => n.name)
            },
            link: {
                source: data.collaboration_networks.links.map(l => l.source),
                target: data.collaboration_networks.links.map(l => l.target),
                value: data.collaboration_networks.links.map(l => l.value)
            }
        };

        const collaborationNetworksLayout = {
            title: 'Top Author Collaboration Network',
            font: {
                size: 10
            }
        };

        Plotly.newPlot('collaboration-networks', [collaborationNetworksData], collaborationNetworksLayout);
    })
    .catch(error => console.error('Error loading plot data:', error)); 