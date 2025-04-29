// Load the plot data
fetch('js/plot_data.json')
    .then(response => response.json())
    .then(data => {
        // Update markdown content
        document.getElementById('overview-content').innerHTML = marked.parse(data.markdown_content[0]);
        document.getElementById('publication-analysis').innerHTML = marked.parse(data.markdown_content[1]);
        document.getElementById('citation-analysis-content').innerHTML = marked.parse(data.markdown_content[2]);
        document.getElementById('topic-analysis-content').innerHTML = marked.parse(data.markdown_content[3]);
        document.getElementById('collaboration-analysis-content').innerHTML = marked.parse(data.markdown_content[4]);

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

        // Yearly Publications by Type
        const yearlyPublicationsData = data.yearly_publications.types.map((type, i) => ({
            x: data.yearly_publications.years,
            y: data.yearly_publications.counts.map(row => row[i]),
            type: 'bar',
            name: type
        }));

        const yearlyPublicationsLayout = {
            title: 'Yearly Publications by Type',
            xaxis: {
                title: 'Year'
            },
            yaxis: {
                title: 'Number of Publications'
            },
            barmode: 'stack'
        };

        Plotly.newPlot('yearly-publications', yearlyPublicationsData, yearlyPublicationsLayout);

        // RDC Productivity Plot
        const rdcProductivityData = {
            x: data.rdc_productivity.x,
            y: data.rdc_productivity.y,
            type: 'bar',
            marker: {
                color: '#ff7f0e'
            }
        };

        const rdcProductivityLayout = {
            title: 'Top 10 RDCs by Publication Count',
            xaxis: {
                title: 'RDC',
                tickangle: 45
            },
            yaxis: {
                title: 'Number of Publications'
            }
        };

        Plotly.newPlot('rdc-productivity', [rdcProductivityData], rdcProductivityLayout);

        // RDC Collaboration Network
        const rdcCollaborationData = {
            type: 'sankey',
            node: {
                pad: 15,
                thickness: 20,
                line: {
                    color: 'black',
                    width: 0.5
                },
                label: data.rdc_collaboration.nodes.map(n => n.name)
            },
            link: {
                source: data.rdc_collaboration.links.map(l => l.source),
                target: data.rdc_collaboration.links.map(l => l.target),
                value: data.rdc_collaboration.links.map(l => l.value)
            }
        };

        const rdcCollaborationLayout = {
            title: 'RDC Collaboration Network',
            font: {
                size: 10
            }
        };

        Plotly.newPlot('rdc-collaboration', [rdcCollaborationData], rdcCollaborationLayout);

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

        // Venue Impact Plot
        const venueImpactData = {
            x: data.venue_impact.x,
            y: data.venue_impact.y,
            type: 'bar',
            marker: {
                color: '#d62728'
            }
        };

        const venueImpactLayout = {
            title: 'Top 10 Venues by Average Citations',
            xaxis: {
                title: 'Venue',
                tickangle: 45
            },
            yaxis: {
                title: 'Average Citations'
            }
        };

        Plotly.newPlot('venue-impact', [venueImpactData], venueImpactLayout);

        // Citation Velocity Plot
        const citationVelocityData = {
            x: data.citation_velocity.titles,
            y: data.citation_velocity.velocity,
            type: 'bar',
            marker: {
                color: '#9467bd'
            }
        };

        const citationVelocityLayout = {
            title: 'Top 10 Papers by Citation Velocity',
            xaxis: {
                title: 'Paper Title',
                tickangle: 45
            },
            yaxis: {
                title: 'Citations per Year'
            }
        };

        Plotly.newPlot('citation-velocity', [citationVelocityData], citationVelocityLayout);

        // Topic Modeling Plot
        const topicModelingData = {
            labels: data.topic_modeling.labels,
            values: data.topic_modeling.values,
            type: 'pie',
            hole: 0.4,
            marker: {
                colors: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            }
        };

        const topicModelingLayout = {
            title: 'Top 10 Research Topics',
            showlegend: true,
            legend: {
                orientation: 'h',
                y: -0.1
            }
        };

        Plotly.newPlot('topic-modeling', [topicModelingData], topicModelingLayout);

        // Topic Evolution Plot
        const topicEvolutionData = {
            x: data.topic_evolution.years,
            y: data.topic_evolution.keywords,
            z: data.topic_evolution.counts,
            type: 'heatmap',
            colorscale: 'Viridis'
        };

        const topicEvolutionLayout = {
            title: 'Topic Evolution Over Time',
            xaxis: {
                title: 'Year'
            },
            yaxis: {
                title: 'Topic'
            }
        };

        Plotly.newPlot('topic-evolution', [topicEvolutionData], topicEvolutionLayout);

        // Author Productivity Plot
        const authorProductivityData = {
            x: data.author_productivity.x,
            y: data.author_productivity.y,
            type: 'bar',
            marker: {
                color: '#8c564b'
            }
        };

        const authorProductivityLayout = {
            title: 'Top 15 Authors by Publication Count',
            xaxis: {
                title: 'Author',
                tickangle: 45
            },
            yaxis: {
                title: 'Number of Publications'
            }
        };

        Plotly.newPlot('author-productivity', [authorProductivityData], authorProductivityLayout);

        // Author Network
        const authorNetworkData = {
            type: 'sankey',
            node: {
                pad: 15,
                thickness: 20,
                line: {
                    color: 'black',
                    width: 0.5
                },
                label: data.author_network.nodes.map(n => n.name)
            },
            link: {
                source: data.author_network.links.map(l => l.source),
                target: data.author_network.links.map(l => l.target),
                value: data.author_network.links.map(l => l.value)
            }
        };

        const authorNetworkLayout = {
            title: 'Author Collaboration Network',
            font: {
                size: 10
            }
        };

        Plotly.newPlot('author-network', [authorNetworkData], authorNetworkLayout);

        // Keyword Cloud
        const keywordCloudData = {
            type: 'scatter',
            mode: 'text',
            text: data.keyword_cloud.words,
            textposition: 'middle center',
            textfont: {
                size: data.keyword_cloud.counts.map(count => Math.sqrt(count) * 10)
            },
            x: Array(data.keyword_cloud.words.length).fill(0),
            y: Array(data.keyword_cloud.words.length).fill(0)
        };

        const keywordCloudLayout = {
            title: 'Keyword Cloud',
            showlegend: false,
            xaxis: {
                showgrid: false,
                zeroline: false,
                showticklabels: false
            },
            yaxis: {
                showgrid: false,
                zeroline: false,
                showticklabels: false
            }
        };

        Plotly.newPlot('keyword-cloud', [keywordCloudData], keywordCloudLayout);

        // Citation Network
        const citationNetworkData = {
            type: 'sankey',
            node: {
                pad: 15,
                thickness: 20,
                line: {
                    color: 'black',
                    width: 0.5
                },
                label: data.citation_network.nodes.map(n => n.name)
            },
            link: {
                source: data.citation_network.links.map(l => l.source),
                target: data.citation_network.links.map(l => l.target),
                value: data.citation_network.links.map(l => l.value)
            }
        };

        const citationNetworkLayout = {
            title: 'Citation Network',
            font: {
                size: 10
            }
        };

        Plotly.newPlot('citation-network', [citationNetworkData], citationNetworkLayout);
    })
    .catch(error => console.error('Error loading plot data:', error)); 