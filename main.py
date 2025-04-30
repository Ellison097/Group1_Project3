import os
import subprocess
import time
import json
from datetime import datetime
import argparse
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

class PipelineProgress:
    def __init__(self, progress_file='pipeline_progress.json'):
        self.progress_file = progress_file
        self.progress = self._load_progress()
    
    def _load_progress(self):
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def save_progress(self):
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f)
    
    def mark_complete(self, script_name):
        self.progress[script_name] = {
            'completed': True,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.save_progress()
    
    def is_completed(self, script_name):
        return self.progress.get(script_name, {}).get('completed', False)

def check_required_files(script_name):
    """Check if required input files exist for the script"""
    required_files = {
        'Code/clean_data.py': ['Project3_Data/group1.csv', 'Project3_Data/group2.csv', 'Project3_Data/group3.csv', 
                         'Project3_Data/group4.csv', 'Project3_Data/group5.csv', 'Project3_Data/group6.csv',
                         'Project3_Data/group7.csv', 'Project3_Data/group8.csv', 'ResearchOutputs.xlsx']
    }
    
    output_files = {
        'Code/clean_data.py': ['Project3_Data_Clean_step1/group1_clean2.csv'],
        'Code/join_and_map_data.py': ['Project3_Data_Clean_step2/combined_mapped_data_raw.csv'],
        'Code/enrich_all_combined_data.py': ['Project_Data_Enriched_Combined/combined_mapped_data_raw_enriched_final.csv'],
        'Code/update_venue_column.py': ['Project_Data_Enriched_Combined/combined_mapped_data_raw_enriched_final_final.csv'],
        'Code/enrich_all_combined_data_metadata.py': ['Project_Data_Enriched_Combined/meta_data_enriched_all.csv'],
        'Code/enrich_all_combined_data_metadata2.py': ['Project_Data_Enriched_Combined/meta_data_enriched_all_v2.csv'],
        'Code/enrich_all_combined_data_metadata3.py': ['Project_Data_Enriched_Combined/meta_data_enriched_all_v3.csv'],
        'Code/fetch_citations_by_title.py': ['EDA_2698_with_citations.csv']
    }
    
    # If output file exists, consider the step as completed
    if script_name in output_files:
        output_exists = all(os.path.exists(f) for f in output_files[script_name])
        if output_exists:
            return []
    
    missing_files = []
    for file in required_files.get(script_name, []):
        if not os.path.exists(file):
            missing_files.append(file)
    
    return missing_files

def run_script(script_name, description, progress_tracker, force=False):
    """Execute the specified Python script and print progress information"""
    print(f"\n{'='*80}")
    print(f"Starting: {description}")
    print(f"Running script: {script_name}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('='*80)
    
    # Check if already completed and not forcing rerun
    if not force and (progress_tracker.is_completed(script_name) or check_required_files(script_name) == []):
        print(f"\nScript {script_name} has already been completed. Skipping...")
        if script_name in progress_tracker.progress:
            print(f"Previous completion time: {progress_tracker.progress[script_name]['timestamp']}")
        progress_tracker.mark_complete(script_name)  # Mark as complete if output exists
        return True
    
    # Check required files
    missing_files = check_required_files(script_name)
    if missing_files:
        print(f"\nError: Required files missing for {script_name}:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    try:
        result = subprocess.run(['python', script_name], 
                              capture_output=True, 
                              text=True)
        
        if result.returncode == 0:
            print(f"\nScript {script_name} executed successfully!")
            print("Output information:")
            print(result.stdout)
            progress_tracker.mark_complete(script_name)
            return True
        else:
            print(f"\nScript {script_name} execution failed!")
            print("Error information:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"Error occurred while executing script {script_name}: {str(e)}")
        return False
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('='*80)
    time.sleep(2)  # Add a brief delay for clearer output

def execute_notebook(notebook_path, output_path=None):
    """Execute an IPython notebook via nbconvert and collect output"""
    print(f"\n{'='*80}")
    print(f"Executing notebook: {notebook_path}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('='*80)

    try:
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)
        
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)
        
        print(f"\nNotebook {notebook_path} executed successfully!")
        return True
    except Exception as e:
        print(f"Error executing notebook {notebook_path}: {str(e)}")
        return False

def main():
    """Main function to execute all processing steps in sequence"""
    parser = argparse.ArgumentParser(description='Run the data enrichment pipeline')
    parser.add_argument('--force', action='store_true', help='Force run all steps, ignoring previous progress')
    parser.add_argument('--start-from', type=str, help='Start from a specific script (e.g., Code/clean_data.py)')
    parser.add_argument('--skip-notebook', action='store_true', help='Skip executing the Jupyter notebook')
    args = parser.parse_args()
    
    print("Starting data enrichment pipeline...")
    progress_tracker = PipelineProgress()
    
    # Define processing steps
    steps = [
        ('Code/clean_data.py', 'Data cleaning and deduplication'),
        ('Code/join_and_map_data.py', 'Data mapping and consolidation'),
        ('Code/enrich_all_combined_data.py', 'Output-level enrichment'),
        ('Code/update_venue_column.py', 'Post-enrichment filtering and venue refinement'),
        ('Code/enrich_all_combined_data_metadata.py', 'Metadata enrichment using project PI matching'),
        ('Code/enrich_all_combined_data_metadata2.py', 'Metadata enrichment using output title matching'),
        ('Code/enrich_all_combined_data_metadata3.py', 'Metadata enrichment using researcher/PI matching'),
        ('Code/fetch_citations_by_title.py', 'Citation data enrichment')
    ]
    
    # Determine starting point
    start_index = 0
    if args.start_from:
        for i, (script, _) in enumerate(steps):
            if script == args.start_from:
                start_index = i
                break
    
    # Execute steps
    for i, (script, description) in enumerate(steps[start_index:], start=start_index):
        print(f"\nProcessing step {i+1}/{len(steps)}")
        success = run_script(script, description, progress_tracker, args.force)
        if not success:
            print(f"\nPipeline stopped due to failure in {script}")
            print("You can resume from this point using: --start-from " + script)
            return
    
    # Execute the Jupyter notebook if not skipped
    if not args.skip_notebook:
        notebook_path = "Group1_Project3_P1artQ2(EDA)+P2(DataMining).ipynb"
        output_path = "Group1_Project3_P1artQ2(EDA)+P2(DataMining)_executed.ipynb"
        
        if not execute_notebook(notebook_path, output_path):
            print("\nPipeline stopped due to failure in notebook execution")
            return
    
    print("\nAll processing steps completed successfully!")

if __name__ == "__main__":
    main() 