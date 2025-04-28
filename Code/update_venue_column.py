import argparse
import logging
import os
import sys
import time
from typing import Optional, Dict, Any, List

import pandas as pd
import requests
from tqdm import tqdm

# --- Configuration ---

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="update_venue.log",
    filemode="w",  # Overwrite log file each run
)
logger = logging.getLogger(__name__)

# API Configuration
OPENALEX_API_URL = "https://api.openalex.org/works"
REQUEST_DELAY = 0.12  # Seconds delay between requests (~6-7 req/sec)
REQUEST_TIMEOUT = 30  # Seconds timeout for requests
USER_AGENT = "VenueUpdateScript/1.1 (mailto:juzhan@upenn.edu)"  # Update email if needed, version bump

class VenueUpdateError(Exception):
    """Custom exception for venue update errors"""
    pass

def validate_config() -> None:
    """
    Validate configuration parameters
    
    Raises:
        ValueError: If configuration parameters are invalid
    """
    if not isinstance(REQUEST_DELAY, (int, float)) or REQUEST_DELAY < 0:
        raise ValueError("REQUEST_DELAY must be a non-negative number")
    if not isinstance(REQUEST_TIMEOUT, (int, float)) or REQUEST_TIMEOUT <= 0:
        raise ValueError("REQUEST_TIMEOUT must be a positive number")
    if not isinstance(USER_AGENT, str) or not USER_AGENT:
        raise ValueError("USER_AGENT must be a non-empty string")
    if not isinstance(OPENALEX_API_URL, str) or not OPENALEX_API_URL.startswith(('http://', 'https://')):
        raise ValueError("OPENALEX_API_URL must be a valid HTTP/HTTPS URL")

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> None:
    """
    Validate DataFrame structure and required columns
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Raises:
        VenueUpdateError: If validation fails
    """
    if not isinstance(df, pd.DataFrame):
        raise VenueUpdateError("Input must be a pandas DataFrame")
    if df.empty:
        raise VenueUpdateError("DataFrame is empty")
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise VenueUpdateError(f"Missing required columns: {missing_columns}")

def get_openalex_venue(title: str, session: requests.Session) -> Optional[str]:
    """
    Queries OpenAlex for a title and extracts the venue name using prioritized logic.
    
    Args:
        title: The title to search for
        session: Requests session for making HTTP requests
        
    Returns:
        Optional[str]: The venue name string or None if not found or error occurs
        
    Raises:
        VenueUpdateError: If there is an error processing the request
    """
    if pd.isna(title) or not title:
        logger.warning("Skipping row due to missing title.")
        return None

    params = {"search": title, "per-page": 1}  # Only need the first result
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}

    try:
        response = session.get(OPENALEX_API_URL, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        data = response.json()

        if data and "results" in data and data["results"]:
            item = data["results"][0]  # Get the first result

            # Extract venue using prioritized logic
            primary_location = item.get("primary_location", {})
            source = primary_location.get("source", {}) if primary_location else {}
            venue_from_source = source.get("display_name") if source else None

            if venue_from_source:
                logger.debug(f"Title '{title[:50]}...': Found venue via primary_location: '{venue_from_source}'")
                return venue_from_source
            else:
                # Fallback to host_venue
                host_venue = item.get("host_venue", {})
                venue_from_host = host_venue.get("display_name") if host_venue else None
                if venue_from_host:
                    logger.debug(f"Title '{title[:50]}...': Found venue via host_venue: '{venue_from_host}'")
                    return venue_from_host
                else:
                    logger.info(f"Title '{title[:50]}...': No venue found in primary_location or host_venue.")
                    return None
        else:
            logger.info(f"Title '{title[:50]}...': No results found in OpenAlex.")
            return None

    except requests.exceptions.Timeout:
        logger.error(f"Timeout error querying OpenAlex for title: {title[:50]}...")
        return None
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error querying OpenAlex for title '{title[:50]}...': {e.response.status_code} - {e}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error querying OpenAlex for title '{title[:50]}...': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error processing title '{title[:50]}...': {e}")
        return None

# --- Main Script ---

def main():
    """Main function to run the venue update process."""
    try:
        # Validate configuration
        validate_config()
        
        parser = argparse.ArgumentParser(
            description="Filter FSRDC-related records with DOIs and update their 'OutputVenue' column using OpenAlex."
        )
        parser.add_argument(
            "--input",
            required=True,
            help="Path to the input CSV file (output from enrich_all_combined_data.py, e.g., combined_mapped_data_raw_enriched_final.csv)",
        )
        parser.add_argument("--output", required=True, help="Path to save the filtered and venue-updated output CSV file.")
        parser.add_argument(
            "--title-column", default="OutputTitle", help="Name of the column containing the title to search for."
        )
        parser.add_argument("--venue-column", default="OutputVenue", help="Name of the venue column to update.")
        parser.add_argument(
            "--fsrdc-column", default="IsFSRDCRelated", help="Name of the boolean column indicating FSRDC relevance."
        )
        parser.add_argument("--doi-column", default="DOI", help="Name of the column containing the DOI.")

        args = parser.parse_args()

        input_file = args.input
        output_file = args.output
        title_col = args.title_column
        venue_col = args.venue_column
        fsrdc_col = args.fsrdc_column
        doi_col = args.doi_column

        logger.info("Starting filtering and venue update process.")
        logger.info(f"Input file: {input_file}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Title column: {title_col}")
        logger.info(f"Venue column: {venue_col}")
        logger.info(f"FSRDC column: {fsrdc_col}")
        logger.info(f"DOI column: {doi_col}")

        # Load data
        try:
            df_full = pd.read_csv(input_file, low_memory=False)
            logger.info(f"Loaded {len(df_full)} total records from {input_file}")
        except FileNotFoundError:
            raise VenueUpdateError(f"Input file not found: {input_file}")
        except Exception as e:
            raise VenueUpdateError(f"Error loading input file: {str(e)}")

        # Check for required columns before filtering
        required_cols = [title_col, venue_col, fsrdc_col, doi_col]
        validate_dataframe(df_full, required_cols)

        # --- Filtering Step ---
        logger.info("Filtering records: IsFSRDCRelated == True and DOI is not null...")
        df_filtered = df_full[
            (df_full[fsrdc_col] == True) & (df_full[doi_col].notna())
        ].copy()  # Use .copy() to avoid SettingWithCopyWarning later
        logger.info(f"Filtered down to {len(df_filtered)} records for venue update.")

        if df_filtered.empty:
            logger.warning("No records remaining after filtering. Exiting.")
            print("No records matched the filtering criteria (FSRDC-related with DOI). Output file not created.")
            sys.exit(0)

        # --- Venue Update Step ---
        updated_venues = []
        updated_count = 0
        failed_count = 0

        # Use a session for connection pooling
        with requests.Session() as session:
            # Iterate over the filtered DataFrame
            for index, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Updating venues"):
                try:
                    title = row[title_col]
                    original_venue = row[venue_col]

                    # Get new venue from OpenAlex
                    new_venue = get_openalex_venue(title, session)

                    # Decide which venue value to keep
                    if new_venue is not None:
                        updated_venues.append(new_venue)
                        if new_venue != original_venue:
                            updated_count += 1
                            logger.debug(f"Index {index}: Updated venue from '{original_venue}' to '{new_venue}'")
                        else:
                            logger.debug(f"Index {index}: Fetched venue '{new_venue}' matches original.")
                    else:
                        # Keep the original venue if fetching failed or returned None
                        updated_venues.append(original_venue)
                        failed_count += 1
                        logger.debug(f"Index {index}: Failed to fetch new venue, keeping original: '{original_venue}'")

                    # Apply rate limiting delay
                    time.sleep(REQUEST_DELAY)
                except Exception as e:
                    logger.error(f"Error processing row {index}: {str(e)}")
                    updated_venues.append(original_venue)
                    failed_count += 1
                    continue

        # Update the filtered DataFrame column directly
        if len(updated_venues) == len(df_filtered):
            df_filtered.loc[:, venue_col] = updated_venues
            logger.info(
                f"Finished processing. Updated {updated_count} venues. Failed to fetch/find new venue for {failed_count} records."
            )
        else:
            raise VenueUpdateError(
                f"Mismatch between number of updated venues ({len(updated_venues)}) and filtered DataFrame rows ({len(df_filtered)})"
            )

        # Save the updated and filtered DataFrame
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logger.info(f"Created output directory: {output_dir}")

            df_filtered.to_csv(output_file, index=False, encoding="utf-8")
            logger.info(f"Filtered and updated data saved to {output_file}")
            print(f"\nFiltering and venue update complete. Results saved to: {output_file}")
            print(f" - Records processed (after filtering): {len(df_filtered)}")
            print(f" - Venues potentially changed: {updated_count}")
            print(f" - Records where new venue fetch failed/not found: {failed_count}")
            print("See 'update_venue.log' for details.")

        except PermissionError:
            raise VenueUpdateError(f"Permission denied when saving to {output_file}")
        except Exception as e:
            raise VenueUpdateError(f"Error saving output file: {str(e)}")

    except VenueUpdateError as e:
        logger.error(str(e))
        print(f"\nError: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        print(f"\nAn unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
