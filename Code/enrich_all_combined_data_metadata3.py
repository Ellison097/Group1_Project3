import logging
import sys
import time
from typing import List, Dict, Any, Optional, Set

import pandas as pd
from rapidfuzz import fuzz, process
from tqdm import tqdm

# --- Configuration ---
INPUT_DF_PATH = "Project_Data_Enriched_Combined/meta_data_enriched_all_v2.csv"  # Output from script 2
METADATA3_PATH = "ProjectsAllMetadata.xlsx"
METADATA3_SHEET = "Researchers"  # Use the Researchers sheet
OUTPUT_PATH = "Project_Data_Enriched_Combined/meta_data_enriched_all_v3.csv"  # Final output path
FUZZY_SCORE_CUTOFF = 70  # Same cutoff as script 1
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
    if not isinstance(INPUT_DF_PATH, str) or not INPUT_DF_PATH.endswith('.csv'):
        raise ValueError("INPUT_DF_PATH must be a CSV file path")
    if not isinstance(METADATA3_PATH, str) or not METADATA3_PATH.endswith('.xlsx'):
        raise ValueError("METADATA3_PATH must be an Excel file path")
    if not isinstance(OUTPUT_PATH, str) or not OUTPUT_PATH.endswith('.csv'):
        raise ValueError("OUTPUT_PATH must be a CSV file path")
    if not isinstance(METADATA3_SHEET, str):
        raise ValueError("METADATA3_SHEET must be a string")

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
    try:
        authors = set()
        # Process 'Authors' column
        if pd.notna(row["Authors"]):
            authors.update([name.strip() for name in str(row["Authors"]).split(";") if name.strip()])
        # Process 'RawAuthorNames' column
        if pd.notna(row["RawAuthorNames"]):
            authors.update([name.strip() for name in str(row["RawAuthorNames"]).split(";") if name.strip()])
        return list(authors)
    except Exception as e:
        raise MetadataEnrichmentError(f"Error processing authors for row {row.name}: {str(e)}")

def main():
    """Main function to run the third enrichment process."""
    try:
        # Validate configuration
        validate_config()
        
        logging.info("Script 3 started.")

        # --- Load Data ---
        logging.info(f"Loading data from {INPUT_DF_PATH}...")
        try:
            final_df = pd.read_csv(INPUT_DF_PATH)
            validate_dataframe(final_df, ["Authors", "RawAuthorNames", "OutputYear", "IsEnrichedByMetadata1", 
                                        "IsEnrichedByMetadata2", "IsEnriched"])
            validate_year_columns(final_df, ["OutputYear", "ProjID", "ProjectYearStarted", "ProjectYearEnded"])
            final_df["IsEnrichedByMetadata1"] = final_df["IsEnrichedByMetadata1"].astype(bool)
            final_df["IsEnrichedByMetadata2"] = final_df["IsEnrichedByMetadata2"].astype(bool)
            final_df["IsEnriched"] = final_df["IsEnriched"].astype(bool)
            logging.info("Data loaded successfully.")
        except FileNotFoundError:
            raise MetadataEnrichmentError(f"File not found at {INPUT_DF_PATH}")
        except Exception as e:
            raise MetadataEnrichmentError(f"Error loading {INPUT_DF_PATH}: {str(e)}")

        logging.info(f"Loading metadata3 from {METADATA3_PATH}, sheet '{METADATA3_SHEET}'...")
        try:
            metadata3 = pd.read_excel(METADATA3_PATH, sheet_name=METADATA3_SHEET)
            validate_dataframe(metadata3, ["Proj ID", "Status", "Title", "RDC", "Start Year", 
                                         "End Year", "PI", "Researcher"])
            logging.info("Metadata3 loaded successfully.")
        except FileNotFoundError:
            raise MetadataEnrichmentError(f"File not found at {METADATA3_PATH}")
        except Exception as e:
            raise MetadataEnrichmentError(f"Error loading {METADATA3_PATH} sheet {METADATA3_SHEET}: {str(e)}")

        # --- Prepare final_df ---
        logging.info("Preparing final_df...")
        final_df["IsEnrichedByMetadata3"] = False

        # --- Prepare metadata3 ---
        logging.info("Preparing metadata3...")
        metadata3.rename(
            columns={
                "Proj ID": "Meta3_ProjID",
                "Status": "Meta3_Status",
                "Title": "Meta3_Title",
                "RDC": "Meta3_RDC",
                "Start Year": "Meta3_StartYear",
                "End Year": "Meta3_EndYear",
                "PI": "Meta3_PI",
                "Researcher": "Meta3_Researcher",
            },
            inplace=True,
        )

        metadata3["Meta3_Researcher"] = metadata3["Meta3_Researcher"].astype(str).fillna("")
        metadata3["Meta3_PI"] = metadata3["Meta3_PI"].astype(str).fillna("")

        validate_year_columns(metadata3, ["Meta3_StartYear", "Meta3_EndYear"])

        # Create lists of unique Researchers and PIs
        unique_researcher_list = [r for r in metadata3["Meta3_Researcher"].unique().tolist() if r]
        unique_pi_list_meta3 = [pi for pi in metadata3["Meta3_PI"].unique().tolist() if pi]
        logging.info(f"Found {len(unique_researcher_list)} unique Researchers in metadata3.")
        logging.info(f"Found {len(unique_pi_list_meta3)} unique PIs in metadata3.")

        # Pre-group metadata3 for faster lookup
        logging.info("Grouping metadata3 by Researcher and PI...")
        try:
            metadata3_grouped_by_researcher = metadata3.groupby("Meta3_Researcher")
            metadata3_grouped_by_pi = metadata3.groupby("Meta3_PI")
            logging.info("Metadata3 grouped successfully.")
        except Exception as e:
            raise MetadataEnrichmentError(f"Error grouping metadata3: {str(e)}")

        # --- Filter Rows to Enrich ---
        indices_to_enrich = final_df[~final_df["IsEnriched"]].index
        logging.info(f"Found {len(indices_to_enrich)} rows not enriched previously. Attempting enrichment with metadata3...")

        # --- Matching and Enrichment ---
        start_time = time.time()
        enrichment_count_3 = 0

        for index in tqdm(indices_to_enrich, desc="Enriching Rows (Metadata3)"):
            try:
                row = final_df.loc[index]
                row_authors = get_unique_authors(row)
                output_year = row["OutputYear"]

                if pd.isna(output_year) or not row_authors:
                    continue

                best_match_score = -1
                best_match_project_info = None
                match_found_this_row = False

                # --- Attempt 1: Match Researchers ---
                for author in row_authors:
                    if not author:
                        continue
                    match_result = process.extractOne(
                        author, unique_researcher_list, scorer=fuzz.WRatio, score_cutoff=FUZZY_SCORE_CUTOFF
                    )
                    if match_result:
                        matched_name, current_score, _ = match_result
                        try:
                            candidate_projects = metadata3_grouped_by_researcher.get_group(matched_name)
                        except KeyError:
                            continue

                        for _, project_row in candidate_projects.iterrows():
                            project_start_year = project_row["Meta3_StartYear"]
                            project_end_year = project_row["Meta3_EndYear"]
                            time_valid_start = pd.notna(project_start_year) and project_start_year <= output_year
                            time_valid_end = pd.isna(project_end_year) or (
                                pd.notna(project_end_year) and output_year <= project_end_year
                            )

                            if time_valid_start and time_valid_end:
                                if current_score > best_match_score:
                                    best_match_score = current_score
                                    best_match_project_info = project_row
                                    match_found_this_row = True

                # --- Attempt 2: Match PIs ---
                if not match_found_this_row:
                    for author in row_authors:
                        if not author:
                            continue
                        match_result = process.extractOne(
                            author, unique_pi_list_meta3, scorer=fuzz.WRatio, score_cutoff=FUZZY_SCORE_CUTOFF
                        )
                        if match_result:
                            matched_name, current_score, _ = match_result
                            try:
                                candidate_projects = metadata3_grouped_by_pi.get_group(matched_name)
                            except KeyError:
                                continue

                            for _, project_row in candidate_projects.iterrows():
                                project_start_year = project_row["Meta3_StartYear"]
                                project_end_year = project_row["Meta3_EndYear"]
                                time_valid_start = pd.notna(project_start_year) and project_start_year <= output_year
                                time_valid_end = pd.isna(project_end_year) or (
                                    pd.notna(project_end_year) and output_year <= project_end_year
                                )

                                if time_valid_start and time_valid_end:
                                    if current_score > best_match_score:
                                        best_match_score = current_score
                                        best_match_project_info = project_row

                # Apply Enrichment if a best match was found
                if best_match_project_info is not None:
                    enrichment_count_3 += 1
                    final_df.loc[index, "ProjID"] = best_match_project_info["Meta3_ProjID"]
                    final_df.loc[index, "ProjectStatus"] = best_match_project_info["Meta3_Status"]
                    final_df.loc[index, "ProjectTitle"] = best_match_project_info["Meta3_Title"]
                    final_df.loc[index, "ProjectRDC"] = best_match_project_info["Meta3_RDC"]
                    final_df.loc[index, "ProjectYearStarted"] = best_match_project_info["Meta3_StartYear"]
                    final_df.loc[index, "ProjectYearEnded"] = best_match_project_info["Meta3_EndYear"]
                    final_df.loc[index, "ProjectPI"] = best_match_project_info["Meta3_PI"]
                    final_df.loc[index, "IsEnrichedByMetadata3"] = True

            except Exception as e:
                logging.error(f"Error processing row {index}: {str(e)}")
                continue

        # --- Final Steps ---
        end_time = time.time()
        logging.info(f"Metadata3 enrichment process completed in {end_time - start_time:.2f} seconds.")
        logging.info(f"Total rows enriched in this step: {enrichment_count_3}")

        # Calculate the final overall enrichment flag
        final_df["IsEnriched"] = (
            final_df["IsEnrichedByMetadata1"] | final_df["IsEnrichedByMetadata2"] | final_df["IsEnrichedByMetadata3"]
        )
        total_enriched_count = final_df["IsEnriched"].sum()
        logging.info(
            f"Total rows enriched overall (Metadata1 + Metadata2 + Metadata3): {total_enriched_count} out of {len(final_df)}"
        )

        # Ensure correct types for newly populated columns
        final_df["ProjID"] = final_df["ProjID"].astype("Int64")
        final_df["ProjectYearStarted"] = final_df["ProjectYearStarted"].astype("Int64")
        final_df["ProjectYearEnded"] = final_df["ProjectYearEnded"].astype("Int64")

        # Clean up unnamed columns
        final_df = final_df.loc[:, ~final_df.columns.str.contains('^Unnamed')]
        final_df = final_df.iloc[:, 1:]

        logging.info(f"Saving final enriched data to {OUTPUT_PATH}...")
        try:
            final_df.to_csv(OUTPUT_PATH, index=False)
            logging.info("Final enriched data saved successfully.")
        except PermissionError:
            raise MetadataEnrichmentError(f"Permission denied when saving to {OUTPUT_PATH}")
        except Exception as e:
            raise MetadataEnrichmentError(f"Error saving output file: {str(e)}")

        logging.info("Script 3 finished.")

    except MetadataEnrichmentError as e:
        logging.error(str(e))
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
