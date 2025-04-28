import logging
import sys
import time
from typing import List, Dict, Any, Optional, Set

import pandas as pd
from rapidfuzz import fuzz, process
from tqdm import tqdm

# --- Configuration ---
FINAL_DF_PATH = "Project_Data_Enriched_Combined/combined_mapped_data_raw_enriched_final_final.csv"
METADATA_PATH = "ProjectsAllMetadata.xlsx"
METADATA_SHEET = "All Metadata"
OUTPUT_PATH = "Project_Data_Enriched_Combined/meta_data_enriched_all.csv"
FUZZY_SCORE_CUTOFF = 70
LOG_LEVEL = logging.INFO

# --- Setup Logging ---
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

class MetadataEnrichmentError(Exception):
    """Custom exception for metadata enrichment errors"""
    pass

def validate_config() -> None:
    """
    Validate configuration parameters
    
    Raises:
        ValueError: If configuration parameters are invalid
    """
    if not isinstance(FUZZY_SCORE_CUTOFF, (int, float)) or FUZZY_SCORE_CUTOFF < 0 or FUZZY_SCORE_CUTOFF > 100:
        raise ValueError("FUZZY_SCORE_CUTOFF must be between 0 and 100")
    if not isinstance(FINAL_DF_PATH, str) or not FINAL_DF_PATH.endswith('.csv'):
        raise ValueError("FINAL_DF_PATH must be a CSV file path")
    if not isinstance(METADATA_PATH, str) or not METADATA_PATH.endswith('.xlsx'):
        raise ValueError("METADATA_PATH must be an Excel file path")
    if not isinstance(METADATA_SHEET, str):
        raise ValueError("METADATA_SHEET must be a string")
    if not isinstance(OUTPUT_PATH, str) or not OUTPUT_PATH.endswith('.csv'):
        raise ValueError("OUTPUT_PATH must be a CSV file path")

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> None:
    """
    Validate DataFrame structure and required columns
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Raises:
        MetadataEnrichmentError: If validation fails
    """
    if not isinstance(df, pd.DataFrame):
        raise MetadataEnrichmentError("Input must be a pandas DataFrame")
    if df.empty:
        raise MetadataEnrichmentError("DataFrame is empty")
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise MetadataEnrichmentError(f"Missing required columns: {missing_columns}")

def get_unique_authors(row: pd.Series) -> List[str]:
    """
    Combines authors from 'Authors' and 'RawAuthorNames',
    cleans them, and returns a unique list.
    Handles potential NaN values.
    
    Args:
        row: DataFrame row containing author information
        
    Returns:
        List[str]: List of unique author names
        
    Raises:
        MetadataEnrichmentError: If author processing fails
    """
    authors: Set[str] = set()
    try:
        # Process 'Authors' column
        if pd.notna(row["Authors"]):
            authors.update([name.strip() for name in str(row["Authors"]).split(";") if name.strip()])
        
        # Process 'RawAuthorNames' column
        if pd.notna(row["RawAuthorNames"]):
            authors.update([name.strip() for name in str(row["RawAuthorNames"]).split(";") if name.strip()])
            
        return list(authors)
    except Exception as e:
        raise MetadataEnrichmentError(f"Error processing authors for row {row.name}: {str(e)}")

def validate_year_columns(df: pd.DataFrame, year_columns: List[str]) -> None:
    """
    Validate year columns in DataFrame
    
    Args:
        df: DataFrame containing year columns
        year_columns: List of year column names to validate
        
    Raises:
        MetadataEnrichmentError: If validation fails
    """
    for col in year_columns:
        if col not in df.columns:
            raise MetadataEnrichmentError(f"Year column '{col}' not found in DataFrame")
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        except Exception as e:
            raise MetadataEnrichmentError(f"Error converting year column '{col}': {str(e)}")

def main():
    """Main function to run the enrichment process."""
    try:
        # Validate configuration
        validate_config()
        
        # --- Load Data ---
        logging.info(f"Loading final_df from {FINAL_DF_PATH}...")
        try:
            final_df = pd.read_csv(FINAL_DF_PATH)
            validate_dataframe(final_df, ["Authors", "RawAuthorNames", "OutputYear"])
            logging.info(f"Successfully loaded {len(final_df)} rows from final_df.")
        except FileNotFoundError:
            raise MetadataEnrichmentError(f"File not found at {FINAL_DF_PATH}")
        except Exception as e:
            raise MetadataEnrichmentError(f"Error loading {FINAL_DF_PATH}: {str(e)}")

        logging.info(f"Loading metadata from {METADATA_PATH}, sheet '{METADATA_SHEET}'...")
        try:
            metadata = pd.read_excel(METADATA_PATH, sheet_name=METADATA_SHEET)
            validate_dataframe(metadata, ["Proj ID", "Status", "Title", "RDC", "Start Year", "End Year", "PI"])
            logging.info(f"Successfully loaded {len(metadata)} rows from metadata.")
        except FileNotFoundError:
            raise MetadataEnrichmentError(f"File not found at {METADATA_PATH}")
        except Exception as e:
            raise MetadataEnrichmentError(f"Error loading {METADATA_PATH}: {str(e)}")

        # --- Prepare final_df ---
        logging.info("Preparing final_df...")
        final_df["IsEnrichedByMetadata1"] = False
        validate_year_columns(final_df, ["OutputYear"])

        # --- Prepare metadata ---
        logging.info("Preparing metadata...")
        metadata.rename(
            columns={
                "Proj ID": "Meta_ProjID",
                "Status": "Meta_Status",
                "Title": "Meta_Title",
                "RDC": "Meta_RDC",
                "Start Year": "Meta_StartYear",
                "End Year": "Meta_EndYear",
                "PI": "Meta_PI",
            },
            inplace=True,
        )

        metadata["Meta_PI"] = metadata["Meta_PI"].astype(str).fillna("")
        unique_pi_list = [pi for pi in metadata["Meta_PI"].unique().tolist() if pi]
        logging.info(f"Found {len(unique_pi_list)} unique PIs in metadata.")

        validate_year_columns(metadata, ["Meta_StartYear", "Meta_EndYear"])

        # Pre-group metadata by PI for faster lookup
        logging.info("Grouping metadata by PI...")
        metadata_grouped = metadata.groupby("Meta_PI")

        # --- Matching and Enrichment ---
        logging.info("Starting matching and enrichment process...")
        start_time = time.time()
        enrichment_count = 0

        for index, row in tqdm(final_df.iterrows(), total=final_df.shape[0], desc="Enriching Rows"):
            try:
                best_match_score = -1
                best_match_project_info = None

                row_authors = get_unique_authors(row)
                output_year = row["OutputYear"]

                if pd.isna(output_year) or not row_authors:
                    continue

                for author in row_authors:
                    if not author:
                        continue

                    match_result = process.extractOne(
                        author, unique_pi_list, scorer=fuzz.WRatio, score_cutoff=FUZZY_SCORE_CUTOFF
                    )

                    if match_result:
                        matched_pi_name, current_score, _ = match_result

                        try:
                            candidate_projects = metadata_grouped.get_group(matched_pi_name)
                        except KeyError:
                            continue

                        for _, project_row in candidate_projects.iterrows():
                            project_start_year = project_row["Meta_StartYear"]
                            project_end_year = project_row["Meta_EndYear"]

                            time_valid_start = pd.notna(project_start_year) and project_start_year <= output_year
                            time_valid_end = pd.isna(project_end_year) or (
                                pd.notna(project_end_year) and output_year <= project_end_year
                            )

                            if time_valid_start and time_valid_end and current_score > best_match_score:
                                best_match_score = current_score
                                best_match_project_info = project_row

                if best_match_project_info is not None:
                    enrichment_count += 1
                    final_df.loc[index, "ProjID"] = best_match_project_info["Meta_ProjID"]
                    final_df.loc[index, "ProjectStatus"] = best_match_project_info["Meta_Status"]
                    final_df.loc[index, "ProjectTitle"] = best_match_project_info["Meta_Title"]
                    final_df.loc[index, "ProjectRDC"] = best_match_project_info["Meta_RDC"]
                    final_df.loc[index, "ProjectYearStarted"] = best_match_project_info["Meta_StartYear"]
                    final_df.loc[index, "ProjectYearEnded"] = best_match_project_info["Meta_EndYear"]
                    final_df.loc[index, "ProjectPI"] = best_match_project_info["Meta_PI"]
                    final_df.loc[index, "IsEnrichedByMetadata1"] = True

            except Exception as e:
                logging.error(f"Error processing row {index}: {str(e)}")
                continue

        # --- Final Steps ---
        end_time = time.time()
        logging.info(f"Enrichment process completed in {end_time - start_time:.2f} seconds.")
        logging.info(f"Total rows enriched: {enrichment_count} out of {len(final_df)}")

        # Convert enriched columns to appropriate types
        final_df["ProjID"] = final_df["ProjID"].astype("Int64")
        final_df["ProjectYearStarted"] = final_df["ProjectYearStarted"].astype("Int64")
        final_df["ProjectYearEnded"] = final_df["ProjectYearEnded"].astype("Int64")

        logging.info(f"Saving enriched data to {OUTPUT_PATH}...")
        try:
            final_df.to_csv(OUTPUT_PATH, index=False)
            logging.info("Enriched data saved successfully.")
        except PermissionError:
            raise MetadataEnrichmentError(f"Permission denied when saving to {OUTPUT_PATH}")
        except Exception as e:
            raise MetadataEnrichmentError(f"Error saving output file: {str(e)}")

    except MetadataEnrichmentError as e:
        logging.error(str(e))
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
