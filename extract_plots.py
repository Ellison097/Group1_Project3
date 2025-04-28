import json
import nbformat
from nbconvert import PythonExporter
import re

def extract_plotly_data(notebook_path):
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    plot_data = {}
    
    # Extract Plotly figures
    for cell in nb.cells:
        if cell.cell_type == 'code':
            # Look for Plotly figure creation
            if 'go.Figure' in cell.source or 'px.' in cell.source:
                # Extract the figure data
                match = re.search(r'fig\.to_json\(\)', cell.source)
                if match:
                    # Execute the cell to get the figure data
                    # Note: This is a simplified version. In practice, you'd need to
                    # properly execute the cell in a Python environment
                    plot_data[cell.execution_count] = {
                        'source': cell.source,
                        'outputs': cell.outputs
                    }
    
    return plot_data

def main():
    notebook_path = 'Group1_Project3_P1Q2(EDA)+P2(DataMining).ipynb'
    plot_data = extract_plotly_data(notebook_path)
    
    # Save the extracted data
    with open('docs/js/plot_data.json', 'w') as f:
        json.dump(plot_data, f, indent=2)

if __name__ == '__main__':
    main() 