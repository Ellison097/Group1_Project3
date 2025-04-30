import os
from datetime import datetime
from typing import Dict, List, Optional, Union
import logging

import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataProcessingError(Exception):
    """Custom exception for data processing errors"""
    pass

def validate_input_file(file_path: str) -> None:
    """
    Validate input file existence and format
    
    Args:
        file_path: Path to the input file
        
    Raises:
        FileNotFoundError: If file doesn't exist
        DataProcessingError: If file format is invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    if not file_path.endswith('.csv'):
        raise DataProcessingError(f"Invalid file format. Expected CSV file, got: {file_path}")

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> None:
    """
    Validate DataFrame structure and content
    
    Args:
        df: Input DataFrame
        required_columns: List of required columns
        
    Raises:
        DataProcessingError: If validation fails
    """
    if df.empty:
        raise DataProcessingError("Input DataFrame is empty")
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise DataProcessingError(f"Missing required columns: {', '.join(missing_columns)}")

def safe_int_convert(value: Union[str, float, int]) -> Optional[int]:
    """
    Safely convert value to integer
    
    Args:
        value: Value to convert
        
    Returns:
        Optional[int]: Converted value or None if conversion fails
    """
    if pd.isna(value):
        return np.nan
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return np.nan

def safe_date_parse(date_str: str) -> tuple[Optional[int], Optional[int]]:
    """
    Safely parse date string to year and month
    
    Args:
        date_str: Date string to parse
        
    Returns:
        tuple: (year, month) or (None, None) if parsing fails
    """
    if pd.isna(date_str):
        return np.nan, np.nan
    try:
        date_obj = datetime.strptime(str(date_str), "%Y-%m-%d")
        return date_obj.year, date_obj.month
    except (ValueError, TypeError):
        return np.nan, np.nan

# Define the output columns for the final dataset
OUTPUT_COLUMNS = [
    "ProjID",
    "ProjectStatus",
    "ProjectTitle",
    "ProjectRDC",
    "ProjectYearStarted",
    "ProjectYearEnded",
    "ProjectPI",
    "OutputTitle",
    "OutputBiblio",
    "OutputType",
    "OutputStatus",
    "OutputVenue",
    "OutputYear",
    "OutputMonth",
    "OutputVolume",
    "OutputNumber",
    "OutputPages",
    "SourceFile",  # Added SourceFile to track origin
]

# Type mapping dictionaries
TYPE_CROSSREF_MAPPING = {
    "article": "JA",
    "preprint": "WP",
    "report": "RE",
    "dataset": "DS",
    "review": "JA",
    "other": "MI",
    "book-chapter": "BC",
    "dissertation": "DI",
    "editorial": "JA",
    "book": "BC",
    "paratext": "MI",
    "letter": "JA",
    "journal-article": "JA",
    "reference-entry": "MI",
}


def process_group1(file_path: str) -> pd.DataFrame:
    """
    Process group1_clean2.csv according to the mapping rules.
    
    Args:
        file_path: Path to the input file
        
    Returns:
        pd.DataFrame: Processed DataFrame
        
    Raises:
        DataProcessingError: If processing fails
    """
    logger.info(f"Processing {file_path}...")
    
    try:
        # Validate input file
        validate_input_file(file_path)
        
        # Read and validate DataFrame
        df = pd.read_csv(file_path)
        validate_dataframe(df, ['title'])  # Add minimum required columns
        
        # Create a new DataFrame with the standardized columns
        result_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
        
        # Map columns according to the rules
        result_df["ProjectStatus"] = df.get("project_status", np.nan)
        result_df["ProjectRDC"] = df.get("Agency", np.nan)
        result_df["ProjectPI"] = df.get("project_pi", np.nan)
        result_df["OutputTitle"] = df.get("title", np.nan)
        result_df["OutputVenue"] = df.get("source", np.nan)
        
        # Convert year to int if possible
        if "year" in df.columns:
            result_df["OutputYear"] = df["year"].apply(safe_int_convert)
        
        # Derive OutputStatus based on DOI presence
        if "doi" in df.columns:
            result_df["OutputStatus"] = df["doi"].apply(lambda x: "PB" if pd.notna(x) else "UP")
        
        # Add source file information
        result_df["SourceFile"] = "group1_clean2.csv"
        
        return result_df
        
    except Exception as e:
        raise DataProcessingError(f"Error processing group1 file: {str(e)}")


def process_group2(file_path: str) -> pd.DataFrame:
    """
    Process group2_clean2.csv according to the mapping rules.
    
    Args:
        file_path: Path to the input file
        
    Returns:
        pd.DataFrame: Processed DataFrame
        
    Raises:
        DataProcessingError: If processing fails
    """
    logger.info(f"Processing {file_path}...")
    
    try:
        # Validate input file
        validate_input_file(file_path)
        
        # Read and validate DataFrame
        df = pd.read_csv(file_path)
        validate_dataframe(df, ['title'])  # Add minimum required columns
        
        # Create a new DataFrame with the standardized columns
        result_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
        
        # Map columns according to the rules
        result_df["ProjectRDC"] = df.get("location", np.nan)
        result_df["ProjectPI"] = df.get("researcher", np.nan)
        result_df["OutputTitle"] = df.get("title", np.nan)
        result_df["OutputVenue"] = df.get("source_display_name", np.nan)
        
        # Map type_crossref to standardized OutputType
        if "type_crossref" in df.columns:
            result_df["OutputType"] = df["type_crossref"].map(TYPE_CROSSREF_MAPPING)
        
        # Derive OutputStatus based on DOI presence
        if "doi" in df.columns:
            result_df["OutputStatus"] = df["doi"].apply(lambda x: "PB" if pd.notna(x) else "UP")
        
        # Extract year and month from publication_date
        if "publication_date" in df.columns:
            years_months = df["publication_date"].apply(safe_date_parse)
            result_df["OutputYear"] = [ym[0] for ym in years_months]
            result_df["OutputMonth"] = [ym[1] for ym in years_months]
        
        # Add source file information
        result_df["SourceFile"] = "group2_clean2.csv"
        
        return result_df
        
    except Exception as e:
        raise DataProcessingError(f"Error processing group2 file: {str(e)}")


def process_group3(file_path):
    """Process group3_clean2.csv according to the mapping rules."""
    print(f"Processing {file_path}...")
    df = pd.read_csv(file_path)

    # Create a new DataFrame with the standardized columns
    result_df = pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Map columns according to the rules
    result_df["ProjectRDC"] = df.get("RDC", np.nan)
    result_df["ProjectPI"] = df.get("PI", np.nan)
    result_df["OutputTitle"] = df.get("Title", np.nan)  # Note: it's all lowercase in the original
    result_df["OutputVenue"] = df.get("host_organization_name", np.nan)

    # Map type_crossref to standardized OutputType
    if "type_crossref" in df.columns:
        # For group3, only dataset=DS and journal-article=JA are present
        result_df["OutputType"] = df["type_crossref"].map({"dataset": "DS", "journal-article": "JA"})

    # Derive OutputStatus based on is_published
    if "is_published" in df.columns:
        result_df["OutputStatus"] = df["is_published"].apply(
            lambda x: "PB" if pd.notna(x) and str(x).upper() == "TRUE" else "UP"
        )

    # Convert publication_year to int if possible
    if "publication_year" in df.columns:
        result_df["OutputYear"] = df["publication_year"].apply(lambda x: int(x) if pd.notna(x) else np.nan)

    # Add source file information
    result_df["SourceFile"] = "group3_clean2.csv"

    return result_df


def process_group4(file_path):
    """Process group4_clean2.csv according to the mapping rules."""
    print(f"Processing {file_path}...")
    df = pd.read_csv(file_path)

    # Create a new DataFrame with the standardized columns
    result_df = pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Map columns according to the rules
    result_df["ProjectPI"] = df.get("researcher", np.nan)
    result_df["OutputTitle"] = df.get("title", np.nan)

    # Convert year to int if possible
    if "year" in df.columns:
        result_df["OutputYear"] = df["year"].apply(lambda x: int(x) if pd.notna(x) else np.nan)

    # Add source file information
    result_df["SourceFile"] = "group4_clean2.csv"

    return result_df


def process_group5(file_path):
    """Process group5_clean2.csv according to the mapping rules."""
    print(f"Processing {file_path}...")
    df = pd.read_csv(file_path)

    # Create a new DataFrame with the standardized columns
    result_df = pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Map columns according to the rules
    result_df["ProjectPI"] = df.get("pi", np.nan)
    result_df["OutputTitle"] = df.get("title", np.nan)

    # All rows have DOIs, so all are published
    result_df["OutputStatus"] = "PB"

    # Year is already in correct format
    result_df["OutputYear"] = df.get("year", np.nan)

    # Add source file information
    result_df["SourceFile"] = "group5_clean2.csv"

    return result_df


def process_group6(file_path):
    """Process group6_clean2.csv according to the mapping rules."""
    print(f"Processing {file_path}...")
    df = pd.read_csv(file_path)

    # Create a new DataFrame with the standardized columns
    result_df = pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Map columns according to the rules
    result_df["ProjectRDC"] = df.get("RDC", np.nan)
    result_df["OutputTitle"] = df.get("Title", np.nan)

    # Derive OutputStatus based on DOI presence
    if "DOI" in df.columns:
        result_df["OutputStatus"] = df["DOI"].apply(lambda x: "PB" if pd.notna(x) else "UP")

    # Convert Year to int if possible
    if "Year" in df.columns:
        result_df["OutputYear"] = df["Year"].apply(lambda x: int(x) if pd.notna(x) else np.nan)

    # Add source file information
    result_df["SourceFile"] = "group6_clean2.csv"

    return result_df


def process_group7(file_path):
    """Process group7_clean2.csv according to the mapping rules."""
    print(f"Processing {file_path}...")
    df = pd.read_csv(file_path)

    # Create a new DataFrame with the standardized columns
    result_df = pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Map columns according to the rules
    result_df["OutputTitle"] = df.get("title", np.nan)

    # Add source file information
    result_df["SourceFile"] = "group7_clean2.csv"

    return result_df


def process_group8(file_path):
    """Process group8_clean2.csv according to the mapping rules."""
    print(f"Processing {file_path}...")
    df = pd.read_csv(file_path)

    # Create a new DataFrame with the standardized columns
    result_df = pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Map columns according to the rules
    result_df["OutputTitle"] = df.get("OutputTitle", np.nan)

    # Map OutputType if available
    if "OutputType" in df.columns:
        result_df["OutputType"] = df["OutputType"].map(TYPE_CROSSREF_MAPPING)

    # Convert OutputYear to int if possible
    if "OutputYear" in df.columns:
        result_df["OutputYear"] = df["OutputYear"].apply(lambda x: int(x) if pd.notna(x) else np.nan)

    # Map other available columns
    result_df["OutputVolume"] = df.get("OutputVolume", np.nan)
    result_df["OutputNumber"] = df.get("OutputNumber", np.nan)
    result_df["OutputPages"] = df.get("OutputPages", np.nan)

    # Add source file information
    result_df["SourceFile"] = "group8_clean2.csv"

    return result_df


def process_all_files(data_dir: str = "Project3_Data_Clean_step1") -> pd.DataFrame:
    """
    Process all group files and combine them.
    
    Args:
        data_dir: Directory containing the input files
        
    Returns:
        pd.DataFrame: Combined processed DataFrame
        
    Raises:
        DataProcessingError: If processing fails
    """
    try:
        # Validate input directory
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Initialize list to store processed DataFrames
        processed_dfs = []
        
        # Process each group file
        processing_functions = {
            "group1": process_group1,
            "group2": process_group2,
            "group3": process_group3,
            "group4": process_group4,
            "group5": process_group5,
            "group6": process_group6,
            "group7": process_group7,
            "group8": process_group8,
        }
        
        for group, process_func in processing_functions.items():
            file_path = os.path.join(data_dir, f"{group}_clean2.csv")
            try:
                if os.path.exists(file_path):
                    df = process_func(file_path)
                    processed_dfs.append(df)
                else:
                    logger.warning(f"File not found: {file_path}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                continue
        
        if not processed_dfs:
            raise DataProcessingError("No files were successfully processed")
        
        # Combine all processed DataFrames
        combined_df = pd.concat(processed_dfs, ignore_index=True)
        
        # Validate final DataFrame
        if combined_df.empty:
            raise DataProcessingError("Combined DataFrame is empty")
        
        return combined_df
        
    except Exception as e:
        raise DataProcessingError(f"Error in process_all_files: {str(e)}")

def main():
    """Main function to process and combine all data files."""
    try:
        # Process all files
        combined_df = process_all_files()
        
        # Save the combined DataFrame
        output_file = "combined_data.csv"
        logger.info(f"Saving combined data to {output_file}...")
        
        try:
            combined_df.to_csv(output_file, index=False)
            logger.info("Successfully saved combined data")
        except Exception as e:
            raise DataProcessingError(f"Error saving combined data: {str(e)}")
        
        # Print summary statistics
        logger.info("\nSummary Statistics:")
        logger.info(f"Total rows: {len(combined_df)}")
        logger.info(f"Columns: {', '.join(combined_df.columns)}")
        logger.info(f"Non-null counts:\n{combined_df.count()}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
