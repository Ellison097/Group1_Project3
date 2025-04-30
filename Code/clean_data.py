import glob
import os
import sys
import logging
import pandas as pd
from fuzzywuzzy import fuzz
from tqdm import tqdm
from typing import Optional, List, Tuple, Set

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataCleaningError(Exception):
    """Custom exception for data cleaning errors"""
    pass

def validate_input_parameters(title1: str, title2: str, threshold: int) -> None:
    """
    Validate input parameters for fuzzy matching
    
    Args:
        title1: First title
        title2: Second title
        threshold: Similarity threshold
        
    Raises:
        ValueError: If parameters are invalid
    """
    if not isinstance(threshold, (int, float)):
        raise ValueError("Threshold must be a number")
    if threshold < 0 or threshold > 100:
        raise ValueError("Threshold must be between 0 and 100")
    if not isinstance(title1, str) and not pd.isna(title1):
        raise ValueError("title1 must be a string or NA")
    if not isinstance(title2, str) and not pd.isna(title2):
        raise ValueError("title2 must be a string or NA")

def fuzzy_match(title1, title2, threshold=85):
    """
    Compare two titles using fuzzy matching

    Args:
        title1: First title
        title2: Second title
        threshold: Similarity threshold, above which titles are considered duplicates

    Returns:
        bool: True if similarity exceeds threshold, False otherwise
        
    Raises:
        ValueError: If input parameters are invalid
    """
    try:
        validate_input_parameters(title1, title2, threshold)
    except ValueError as e:
        logger.error(f"Parameter validation failed: {str(e)}")
        raise

    if pd.isna(title1) or pd.isna(title2):
        return False

    try:
        # Convert to lowercase and calculate similarity
        similarity = fuzz.ratio(str(title1).lower(), str(title2).lower())
        return similarity >= threshold
    except Exception as e:
        logger.error(f"Error in fuzzy matching: {str(e)}")
        raise DataCleaningError(f"Failed to perform fuzzy matching: {str(e)}")

def validate_dataframe(df: pd.DataFrame, title_column: str) -> None:
    """
    Validate DataFrame and its title column
    
    Args:
        df: Input DataFrame
        title_column: Name of title column
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    if not isinstance(title_column, str):
        raise ValueError("Title column name must be a string")
    if title_column not in df.columns:
        raise ValueError(f"Title column '{title_column}' not found in DataFrame")
    if df.empty:
        raise ValueError("DataFrame is empty")

def clean_against_research_outputs(csv_file: str, title_column: str, research_outputs: pd.DataFrame, threshold: int = 85) -> Optional[pd.DataFrame]:
    """
    Deduplicate CSV file against ResearchOutputs.xlsx using exact then fuzzy matching

    Args:
        csv_file: Path to CSV file
        title_column: Name of title column in CSV file
        research_outputs: ResearchOutputs DataFrame
        threshold: Similarity threshold

    Returns:
        DataFrame: Deduplicated DataFrame
        
    Raises:
        DataCleaningError: If cleaning process fails
    """
    logger.info(f"Processing file: {csv_file}")

    # Validate input file
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    # Read CSV file
    try:
        df = pd.read_csv(csv_file)
        logger.info(f"Original file {csv_file} contains {len(df)} rows")
    except Exception as e:
        logger.error(f"Error reading file {csv_file}: {e}")
        raise DataCleaningError(f"Failed to read CSV file: {str(e)}")

    try:
        # Validate input DataFrames
        validate_dataframe(df, title_column)
        validate_dataframe(research_outputs, "OutputTitle")

        if title_column not in df.columns:
            print(f"Warning: Column '{title_column}' not found in {csv_file}")
            return df

        # Step 1: Exact match deduplication
        # Convert all titles to lowercase for comparison
        df["title_lower"] = df[title_column].astype(str).str.lower()
        research_outputs["title_lower"] = research_outputs["OutputTitle"].astype(str).str.lower()

        # Find exact matches
        exact_matches = df[df["title_lower"].isin(research_outputs["title_lower"])].index
        df_after_exact = df.drop(exact_matches)

        exact_removed = len(df) - len(df_after_exact)
        print(f"Exact match: Removed {exact_removed} duplicate rows from {csv_file}")

        # Step 2: Fuzzy match remaining data
        # Mark rows to delete
        to_drop = []
        for i, row in tqdm(df_after_exact.iterrows(), total=len(df_after_exact), desc="Fuzzy matching"):
            title = row[title_column]
            # Check for matches with any title in ResearchOutputs
            for output_title in research_outputs["OutputTitle"]:
                if fuzzy_match(title, output_title, threshold):
                    to_drop.append(i)
                    break

        # Remove duplicate rows
        df_clean = df_after_exact.drop(to_drop)

        # Remove temporary column
        df_clean = df_clean.drop(columns=["title_lower"])

        # Output results
        fuzzy_removed = len(df_after_exact) - len(df_clean)
        total_removed = exact_removed + fuzzy_removed
        print(f"Fuzzy match: Removed {fuzzy_removed} duplicate rows from {csv_file}")
        print(f"Total: Removed {total_removed} duplicate rows from {csv_file}, {len(df_clean)} rows remaining")

        return df_clean
    except Exception as e:
        logger.error(f"Error in clean_against_research_outputs: {str(e)}")
        raise DataCleaningError(f"Failed to clean CSV file: {str(e)}")


def remove_duplicates_between_files(clean_files: List[str], title_columns: List[str], threshold: int = 85) -> List[pd.DataFrame]:
    """
    Deduplicate between multiple CSV files using exact then fuzzy matching

    Args:
        clean_files: List of cleaned CSV files
        title_columns: Title column names for each file
        threshold: Similarity threshold

    Returns:
        list: List of deduplicated DataFrames
        
    Raises:
        DataCleaningError: If deduplication process fails
    """
    # Validate input parameters
    if not isinstance(clean_files, list) or not isinstance(title_columns, list):
        raise ValueError("clean_files and title_columns must be lists")
    if len(clean_files) != len(title_columns):
        raise ValueError("Number of files and title columns must match")
    if not all(os.path.exists(f) for f in clean_files):
        raise FileNotFoundError("One or more input files not found")

    # Initialize result list
    result_dfs = []

    # Initialize accumulator (stores lowercase titles)
    accumulator_titles_lower = set()
    # Initialize fuzzy match accumulator (stores original titles)
    fuzzy_accumulator_titles = []

    for i, (file, title_col) in enumerate(zip(clean_files, title_columns)):
        logger.info(f"Processing inter-file deduplication: {file}")

        # Read current file
        df = pd.read_csv(file)
        initial_count = len(df)

        # Step 1: Exact match deduplication
        # Convert titles to lowercase
        df["title_lower"] = df[title_col].astype(str).str.lower()

        # Find exact matches
        exact_matches = df[df["title_lower"].isin(accumulator_titles_lower)].index
        df_after_exact = df.drop(exact_matches)

        exact_removed = initial_count - len(df_after_exact)
        print(f"Exact match: Removed {exact_removed} duplicate rows from {file}")

        # Step 2: Fuzzy match remaining data
        # Mark rows to delete
        to_drop = []
        for j, row in tqdm(df_after_exact.iterrows(), total=len(df_after_exact), desc="Fuzzy matching"):
            title = row[title_col]
            # Check for matches with any title in fuzzy accumulator
            for acc_title in fuzzy_accumulator_titles:
                if fuzzy_match(title, acc_title, threshold):
                    to_drop.append(j)
                    break

        # Remove duplicate rows
        df_clean = df_after_exact.drop(to_drop)

        # Remove temporary column
        df_clean = df_clean.drop(columns=["title_lower"])

        # Update accumulators
        accumulator_titles_lower.update(df_clean[title_col].astype(str).str.lower())
        fuzzy_accumulator_titles.extend(df_clean[title_col].tolist())

        # Save cleaned DataFrame
        result_dfs.append(df_clean)

        # Output results
        fuzzy_removed = len(df_after_exact) - len(df_clean)
        total_removed = exact_removed + fuzzy_removed
        print(f"Fuzzy match: Removed {fuzzy_removed} duplicate rows from {file}")
        print(f"Total: Removed {total_removed} duplicate rows from {file}, {len(df_clean)} rows remaining")

    return result_dfs


def main():
    try:
        # Define file paths and title column mappings
        data_dir = "Project3_Data"
        output_dir = "Project3_Data_Clean_step1"

        # Validate directories
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # File groups and corresponding title columns - using ordered list instead of dictionary
        group_info = [
            ("group1", "title"),
            ("group2", "title"),
            ("group3", "Title"),
            ("group4", "title"),
            ("group5", "title"),
            ("group6", "Title"),
            ("group7", "title"),
            ("group8", "OutputTitle"),
        ]

        # Read ResearchOutputs.xlsx
        try:
            research_outputs = pd.read_excel("ResearchOutputs.xlsx")
            logger.info(f"Successfully read ResearchOutputs.xlsx, containing {len(research_outputs)} rows")
        except Exception as e:
            logger.error(f"Error reading ResearchOutputs.xlsx: {e}")
            raise DataCleaningError(f"Failed to read ResearchOutputs.xlsx: {str(e)}")

        # Step 1: Deduplicate each CSV file against ResearchOutputs.xlsx
        clean_files = []
        title_columns = []

        for group, title_col in group_info:
            # Find corresponding CSV files
            csv_files = glob.glob(os.path.join(data_dir, f"{group}*.csv"))

            if not csv_files:
                print(f"Warning: No CSV files found for {group}")
                continue

            csv_file = csv_files[0]  # Take the first matching file
            # Clean against ResearchOutputs
            df_clean = clean_against_research_outputs(csv_file, title_col, research_outputs)

            if df_clean is not None:
                # Save cleaned file
                clean_file = os.path.join(output_dir, f"{group}_clean1.csv")
                df_clean.to_csv(clean_file, index=False)
                print(f"Saved cleaned file to {clean_file}")

                # Add to lists for inter-file deduplication
                clean_files.append(clean_file)
                title_columns.append(title_col)

        # Step 2: Deduplicate between cleaned files
        if clean_files:
            logger.info("\nStarting inter-file deduplication...")
            clean_dfs = remove_duplicates_between_files(clean_files, title_columns)

            # Save final cleaned files
            for (group, _), df in zip(group_info[:len(clean_dfs)], clean_dfs):
                output_file = os.path.join(output_dir, f"{group}_clean2.csv")
                df.to_csv(output_file, index=False)
                print(f"Saved final cleaned file to {output_file}")
        else:
            print("No files to process for inter-file deduplication")

    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
