# Group1_Project3 Data Analysis and Processing Project

## Project Overview
This project implements a comprehensive data processing and analysis pipeline for research output data, including data cleaning, data mapping, data enrichment, and data analysis steps.

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installing Dependencies
1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes:
- Core data processing libraries (pandas, numpy, openpyxl)
- Web request libraries (requests, aiohttp)
- Data matching tools (fuzzywuzzy, rapidfuzz)
- Progress tracking (tqdm)
- Jupyter notebook processing (nbformat, nbconvert)
- And other essential dependencies

### Optional Dependencies
For better performance and additional features, the following optional packages are included:
- numexpr and bottleneck for improved pandas performance
- matplotlib and seaborn for data visualization

## Project Structure

### Data Files
- `Combined_ResearchOutputs_Final.csv` (5.0MB) - Final merged research output data(Part2 Answer)
- `EDA_2698.csv` (4.4MB) - Dataset containing 2,698 records
- `EDA_2698_with_citations.csv` (4.4MB) - Dataset with citation information
- `EDA_2698_with_citations_processed.xlsx` (1.3MB) - Processed dataset with citations
- `ResearchOutputs_Group1.xlsx` (1.3MB) - Research outputs from Group 1(Part1 Answer)
- `ResearchOutputs.xlsx` (401KB) - Original research output data
- `ProjectsAllMetadata.xlsx` (1.5MB) - Metadata for all projects

### Code Files
- `main.py` - Main control script managing the entire data processing pipeline
- `Group1_Project3_P1artQ2(EDA)+P2(DataMining).ipynb` - Jupyter notebook for data analysis

### Directory Structure

#### 1. Code Directory (`Code/`)
Contains all processing scripts:
- `clean_data.py` - Data cleaning and deduplication script
  - Input: Raw data files from `Project3_Data/`
  - Output: Cleaned data in `Project3_Data_Clean_step1/`
  - Functionality: Removes duplicates, standardizes formats, handles missing values

- `join_and_map_data.py` - Data mapping and consolidation script
  - Input: Cleaned data from `Project3_Data_Clean_step1/`
  - Output: Mapped data in `Project3_Data_Clean_step2/`
  - Functionality: Joins related data, creates unified identifiers

- `enrich_all_combined_data.py` - Data enrichment script
  - Input: Mapped data from `Project3_Data_Clean_step2/`
  - Output: Enriched data in `Project_Data_Enriched_Combined/`
  - Functionality: Adds additional information to records

- `update_venue_column.py` - Venue information update script
  - Input: Enriched data from previous step
  - Output: Updated venue information
  - Functionality: Standardizes and updates venue names

- `enrich_all_combined_data_metadata.py` - Metadata enrichment script (Version 1)
  - Input: Combined data
  - Output: Enriched metadata
  - Functionality: Adds project PI matching information

- `enrich_all_combined_data_metadata2.py` - Metadata enrichment script (Version 2)
  - Input: Combined data
  - Output: Enriched metadata
  - Functionality: Adds output title matching information

- `enrich_all_combined_data_metadata3.py` - Metadata enrichment script (Version 3)
  - Input: Combined data
  - Output: Enriched metadata
  - Functionality: Adds researcher/PI matching information

- `fetch_citations_by_title.py` - Citation data retrieval script
  - Input: Research output titles
  - Output: Citation information
  - Functionality: Fetches citation data for research outputs

#### 2. Data Directories

##### 2.1 Raw Data (`Project3_Data/`)
Contains original data files:
- `group1.csv` to `group8.csv` - Grouped data files
  - Format: CSV
  - Content: Original research output records
  - Size: Varies by group

##### 2.2 First Cleaning Stage (`Project3_Data_Clean_step1/`)
Contains first-stage cleaned data:
- `group1_clean2.csv` - Cleaned data for Group 1
  - Format: CSV
  - Content: Deduplicated and standardized records
  - Size: ~4MB

##### 2.3 Second Cleaning Stage (`Project3_Data_Clean_step2/`)
Contains second-stage processed data:
- `combined_mapped_data_raw.csv` - Combined and mapped data
  - Format: CSV
  - Content: Unified and mapped research outputs
  - Size: ~5MB

##### 2.4 Enriched Data (`Project_Data_Enriched_Combined/`)
Contains final enriched data:
- `combined_mapped_data_raw_enriched_final.csv` - Final enriched data
  - Format: CSV
  - Content: Complete enriched research outputs
  - Size: ~6MB

- `combined_mapped_data_raw_enriched_final_final.csv` - Final processed data
  - Format: CSV
  - Content: Final version with all enrichments
  - Size: ~6MB

- `meta_data_enriched_all.csv` - Enriched metadata (Version 1)
  - Format: CSV
  - Content: Metadata with PI matching
  - Size: ~2MB

- `meta_data_enriched_all_v2.csv` - Enriched metadata (Version 2)
  - Format: CSV
  - Content: Metadata with title matching
  - Size: ~2MB

- `meta_data_enriched_all_v3.csv` - Enriched metadata (Version 3)
  - Format: CSV
  - Content: Metadata with researcher matching
  - Size: ~2MB

## Data Processing Pipeline

1. **Data Cleaning Stage**
   - Script: `clean_data.py`
   - Input: Raw data from `Project3_Data/`
   - Output: Cleaned data in `Project3_Data_Clean_step1/`
   - Processes: Deduplication, standardization, missing value handling

2. **Data Mapping and Consolidation Stage**
   - Script: `join_and_map_data.py`
   - Input: Cleaned data from `Project3_Data_Clean_step1/`
   - Output: Mapped data in `Project3_Data_Clean_step2/`
   - Processes: Data joining, identifier creation

3. **Data Enrichment Stage**
   - Scripts: `enrich_all_combined_data.py`, `update_venue_column.py`
   - Input: Mapped data from `Project3_Data_Clean_step2/`
   - Output: Enriched data in `Project_Data_Enriched_Combined/`
   - Processes: Information enrichment, venue standardization

4. **Metadata Enrichment Stage**
   - Scripts: Three versions of metadata enrichment scripts
   - Input: Combined data
   - Output: Enriched metadata files
   - Processes: PI matching, title matching, researcher matching

5. **Citation Data Retrieval**
   - Script: `fetch_citations_by_title.py`
   - Input: Research output titles
   - Output: `EDA_2698_with_citations.csv`
   - Processes: Citation data fetching and integration

6. **Data Analysis Stage**
   - Tool: Jupyter notebook
   - Input: All processed data
   - Output: Analysis results and visualizations
   - Processes: Statistical analysis, visualization, insights generation

## Usage Instructions

### Run Complete Pipeline(Highly Recommended)
```bash
python main.py
```

### Skip Notebook Execution
```bash
python main.py --skip-notebook
```

### Start from Specific Step
```bash
python main.py --start-from Code/clean_data.py
```

### Force Rerun All Steps
```bash
python main.py --force
```

## Dependencies
The project requires the following Python packages (all specified in `requirements.txt`):

### Core Libraries
- pandas>=1.5.0 - Data manipulation and analysis
- numpy>=1.21.0 - Numerical computing
- openpyxl>=3.0.0 - Excel file handling

### Web and Async
- requests>=2.28.0 - HTTP requests
- aiohttp>=3.8.0 - Async HTTP requests

### Data Processing
- fuzzywuzzy>=0.18.0 - String matching
- python-Levenshtein>=0.12.0 - String distance calculation
- rapidfuzz>=2.13.0 - Fast string matching

### Machine Learning and Data Science
- scikit-learn>=1.0.0 - Machine learning algorithms
- tensorflow>=2.8.0 - Deep learning framework
- xgboost>=1.5.0 - Gradient boosting framework
- sentence-transformers>=2.2.0 - Sentence embeddings
- bertopic>=0.12.0 - Topic modeling
- umap-learn>=0.5.0 - Dimensionality reduction
- lifelines>=0.26.0 - Survival analysis

### Network Analysis
- networkx>=2.6.0 - Network analysis
- python-louvain>=0.15 - Community detection

### Visualization
- matplotlib>=3.5.0 - Basic plotting
- seaborn>=0.12.0 - Statistical visualization
- plotly>=5.10.0 - Interactive visualization

### Text Processing
- nltk>=3.7.0 - Natural language processing
- spacy>=3.3.0 - Advanced NLP

### Progress and Logging
- tqdm>=4.65.0 - Progress bars
- logging>=0.5.1.2 - Logging functionality

### Notebook Processing
- nbformat>=5.7.0 - Jupyter notebook format
- nbconvert>=7.0.0 - Notebook conversion

### Performance Optimization
- numexpr>=2.8.0 - Fast numerical expression evaluation
- bottleneck>=1.3.0 - Fast array operations

## Important Notes
1. Ensure all data files are in their correct directories
2. Check the status of `pipeline_progress.json` before running
3. Backup important data files before execution
4. Ensure sufficient disk space for processing results
5. Monitor memory usage during large data processing steps
6. Check log files for any processing errors
7. Verify data integrity after each processing stage 