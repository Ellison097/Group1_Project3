import logging
import sys
import time
from typing import List, Dict, Any, Optional, Set

import pandas as pd
from rapidfuzz import fuzz, process
from tqdm import tqdm

# --- Configuration ---
INPUT_DF_PATH = "Project_Data_Enriched_Combined/meta_data_enriched_all.csv"  # Output from script 1
METADATA2_PATH = "ResearchOutputs.xlsx"
OUTPUT_PATH = "Project_Data_Enriched_Combined/meta_data_enriched_all_v2.csv"  # Final output path
TITLE_SCORE_CUTOFF = 90  # Score for matching OutputTitle (adjust as needed)
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
    if not isinstance(TITLE_SCORE_CUTOFF, (int, float)) or TITLE_SCORE_CUTOFF < 0 or TITLE_SCORE_CUTOFF > 100:
        raise ValueError("TITLE_SCORE_CUTOFF must be between 0 and 100")
    if not isinstance(INPUT_DF_PATH, str) or not INPUT_DF_PATH.endswith('.csv'):
        raise ValueError("INPUT_DF_PATH must be a CSV file path")
    if not isinstance(METADATA2_PATH, str) or not METADATA2_PATH.endswith('.xlsx'):
        raise ValueError("METADATA2_PATH must be an Excel file path")
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

def create_title_index(metadata2: pd.DataFrame) -> Dict[str, List[int]]:
    """
    Create an index mapping from titles to their indices in metadata2
    
    Args:
        metadata2: DataFrame containing titles to index
        
    Returns:
        Dict[str, List[int]]: Mapping from title to list of indices
        
    Raises:
        MetadataEnrichmentError: If indexing fails
    """
    try:
        title_to_indices = {}
        for idx, title in enumerate(metadata2["Meta2_OutputTitle"]):
            if title and pd.notna(title):  # Don't map empty or NaN titles
                if title not in title_to_indices:
                    title_to_indices[title] = []
                title_to_indices[title].append(idx)
        return title_to_indices
    except Exception as e:
        raise MetadataEnrichmentError(f"Error creating title index: {str(e)}")

# --- Main Execution Block ---
def main():
    """Main function to run the second enrichment process."""
    try:
        # Validate configuration
        validate_config()
        
        print("Script 2 started.")

        # --- Load Data ---
        print(f"Loading previously enriched data from {INPUT_DF_PATH}...")
        try:
            final_df = pd.read_csv(INPUT_DF_PATH)
            validate_dataframe(final_df, ["OutputTitle", "OutputYear", "IsEnrichedByMetadata1"])
            validate_year_columns(final_df, ["OutputYear", "ProjID", "ProjectYearStarted", "ProjectYearEnded"])
            final_df["IsEnrichedByMetadata1"] = final_df["IsEnrichedByMetadata1"].astype(bool)
            print("Previously enriched data loaded successfully.")
        except FileNotFoundError:
            print(f"Error: File not found at {INPUT_DF_PATH}")
            exit()
        except Exception as e:
            print(f"Error loading {INPUT_DF_PATH}: {e}")
            exit()

        print(f"Loading metadata2 from {METADATA2_PATH}...")
        try:
            metadata2 = pd.read_excel(METADATA2_PATH)
            validate_dataframe(metadata2, ["ProjectID", "ProjectStatus", "ProjectTitle", "ProjectRDC", 
                                         "ProjectStartYear", "ProjectEndYear", "ProjectPI", "OutputTitle", "OutputYear"])
            print("Metadata2 loaded successfully.")
        except FileNotFoundError:
            print(f"Error: File not found at {METADATA2_PATH}")
            exit()
        except Exception as e:
            print(f"Error loading {METADATA2_PATH}: {e}")
            exit()

        # --- Prepare final_df ---
        print("Preparing final_df...")
        final_df["IsEnrichedByMetadata2"] = False
        final_df["OutputTitle"] = final_df["OutputTitle"].astype(str).fillna("")

        # --- Prepare metadata2 ---
        print("Preparing metadata2...")
        metadata2.rename(
            columns={
                "ProjectID": "Meta2_ProjectID",
                "ProjectStatus": "Meta2_ProjectStatus",
                "ProjectTitle": "Meta2_ProjectTitle",
                "ProjectRDC": "Meta2_ProjectRDC",
                "ProjectStartYear": "Meta2_ProjectStartYear",
                "ProjectEndYear": "Meta2_ProjectEndYear",
                "ProjectPI": "Meta2_ProjectPI",
                "OutputTitle": "Meta2_OutputTitle",
                "OutputYear": "Meta2_OutputYear",
            },
            inplace=True,
        )

        validate_year_columns(metadata2, ["Meta2_OutputYear", "Meta2_ProjectStartYear", "Meta2_ProjectEndYear"])
        metadata2["Meta2_OutputTitle"] = metadata2["Meta2_OutputTitle"].astype(str).fillna("")

        # Create title index for efficient matching
        print("Creating title index for metadata2...")
        title_to_indices = create_title_index(metadata2)
        metadata2_titles = metadata2["Meta2_OutputTitle"].tolist()

        # --- Filter Rows to Enrich ---
        indices_to_enrich = final_df[~final_df["IsEnrichedByMetadata1"]].index
        print(f"Found {len(indices_to_enrich)} rows not enriched by metadata1. Attempting enrichment with metadata2...")

        # --- Matching and Enrichment ---
        start_time = time.time()
        enrichment_count_2 = 0

        for index in tqdm(indices_to_enrich, desc="Enriching Rows (Metadata2)"):
            try:
                row = final_df.loc[index]
                output_title = row["OutputTitle"]
                output_year = row["OutputYear"]

                if not output_title or pd.isna(output_title):
                    continue

                match_result = process.extractOne(
                    output_title, metadata2_titles, scorer=fuzz.WRatio, score_cutoff=TITLE_SCORE_CUTOFF
                )

                if match_result:
                    matched_title, score, _ = match_result

                    if matched_title in title_to_indices:
                        possible_match_indices = title_to_indices[matched_title]
                        best_candidate_row = None

                        for meta_idx in possible_match_indices:
                            candidate_row = metadata2.iloc[meta_idx]
                            match_year = candidate_row["Meta2_OutputYear"]

                            years_match = False
                            if pd.isna(output_year) and pd.isna(match_year):
                                years_match = True
                            elif pd.notna(output_year) and pd.notna(match_year) and output_year == match_year:
                                years_match = True

                            if years_match:
                                best_candidate_row = candidate_row
                                break

                        if best_candidate_row is not None:
                            enrichment_count_2 += 1
                            final_df.loc[index, "ProjID"] = best_candidate_row["Meta2_ProjectID"]
                            final_df.loc[index, "ProjectStatus"] = best_candidate_row["Meta2_ProjectStatus"]
                            final_df.loc[index, "ProjectTitle"] = best_candidate_row["Meta2_ProjectTitle"]
                            final_df.loc[index, "ProjectRDC"] = best_candidate_row["Meta2_ProjectRDC"]
                            final_df.loc[index, "ProjectYearStarted"] = best_candidate_row["Meta2_ProjectStartYear"]
                            final_df.loc[index, "ProjectYearEnded"] = best_candidate_row["Meta2_ProjectEndYear"]
                            final_df.loc[index, "ProjectPI"] = best_candidate_row["Meta2_ProjectPI"]
                            final_df.loc[index, "IsEnrichedByMetadata2"] = True

            except Exception as e:
                print(f"Error processing row {index}: {str(e)}")
                continue

        # --- Final Steps ---
        end_time = time.time()
        print(f"\nMetadata2 enrichment process completed in {end_time - start_time:.2f} seconds.")
        print(f"Total rows enriched in this step: {enrichment_count_2}")

        # Calculate the overall enrichment flag
        final_df["IsEnriched"] = final_df["IsEnrichedByMetadata1"] | final_df["IsEnrichedByMetadata2"]
        total_enriched_count = final_df["IsEnriched"].sum()
        print(f"Total rows enriched overall (Metadata1 + Metadata2): {total_enriched_count} out of {len(final_df)}")

        # Ensure correct types for newly populated columns
        final_df["ProjID"] = final_df["ProjID"].astype("Int64")
        final_df["ProjectYearStarted"] = final_df["ProjectYearStarted"].astype("Int64")
        final_df["ProjectYearEnded"] = final_df["ProjectYearEnded"].astype("Int64")

        print(f"Saving final enriched data to {OUTPUT_PATH}...")
        try:
            final_df.to_csv(OUTPUT_PATH, index=False)
            print("Final enriched data saved successfully.")
        except PermissionError:
            print(f"Permission denied when saving to {OUTPUT_PATH}")
            exit()
        except Exception as e:
            print(f"Error saving output file: {e}")
            exit()

        print("Script 2 finished.")

    except MetadataEnrichmentError as e:
        print(str(e))
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
