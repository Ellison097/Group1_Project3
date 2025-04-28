import logging
import sys
import time
import urllib.parse
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm  # Use standard tqdm for console progress

# --- Configuration ---
INPUT_PATH = "EDA_2698.csv"
OUTPUT_PATH = "EDA_2698_with_citations.csv"
OPENALEX_EMAIL = "YOUR_EMAIL@example.com"  # Replace with your email for the polite pool
REQUEST_DELAY = 1 / 10  # Seconds between requests (~9 requests/sec)
LOG_LEVEL = logging.INFO  # Set to logging.DEBUG for more detailed API call info
# --- End Configuration ---

# --- Setup Logging ---
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,  # Log to standard output (console)
)
# --- End Logging Setup ---

class CitationFetchError(Exception):
    """Custom exception for citation fetching errors"""
    pass

class RateLimitError(CitationFetchError):
    """Custom exception for rate limit errors"""
    pass

def validate_config() -> None:
    """
    Validate configuration parameters
    
    Raises:
        ValueError: If configuration parameters are invalid
    """
    if not isinstance(REQUEST_DELAY, (int, float)) or REQUEST_DELAY <= 0:
        raise ValueError("REQUEST_DELAY must be a positive number")
    if not isinstance(OPENALEX_EMAIL, str) or '@' not in OPENALEX_EMAIL:
        raise ValueError("OPENALEX_EMAIL must be a valid email address")
    if not isinstance(INPUT_PATH, str) or not INPUT_PATH.endswith('.csv'):
        raise ValueError("INPUT_PATH must be a CSV file path")
    if not isinstance(OUTPUT_PATH, str) or not OUTPUT_PATH.endswith('.csv'):
        raise ValueError("OUTPUT_PATH must be a CSV file path")

def validate_response(response: requests.Response, title: str) -> Dict[str, Any]:
    """
    Validate and parse API response
    
    Args:
        response: API response object
        title: Title being searched
        
    Returns:
        dict: Parsed response data
        
    Raises:
        CitationFetchError: If response is invalid
    """
    try:
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise CitationFetchError(f"Invalid response format for title '{title[:50]}...'")
        return data
    except requests.exceptions.JSONDecodeError:
        raise CitationFetchError(f"Invalid JSON response for title '{title[:50]}...'")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            raise RateLimitError("Rate limit exceeded")
        raise CitationFetchError(f"HTTP error {e.response.status_code}")

def process_search_results(data: Dict[str, Any], title: str) -> Optional[int]:
    """
    Process search results and extract citation count
    
    Args:
        data: API response data
        title: Original search title
        
    Returns:
        Optional[int]: Citation count if found, None otherwise
    """
    results = data.get("results", [])
    if not results:
        return None
        
    first_result = results[0]
    if not isinstance(first_result, dict):
        raise CitationFetchError(f"Invalid result format for title '{title[:50]}...'")
        
    citation_count = first_result.get("cited_by_count")
    if citation_count is not None and not isinstance(citation_count, (int, float)):
        raise CitationFetchError(f"Invalid citation count format for title '{title[:50]}...'")
        
    return citation_count

def fetch_citations(df: pd.DataFrame) -> List[float]:
    """
    Fetches citation counts from OpenAlex using title search.
    
    Args:
        df: Input DataFrame
        
    Returns:
        List[float]: List of citation counts
        
    Raises:
        CitationFetchError: If citation fetching fails
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    citation_counts = []
    total_rows = len(df)
    logging.info(f"Starting citation fetch for {total_rows} records using title search.")

    for index, row in tqdm(df.iterrows(), total=total_rows, desc="Fetching Citations"):
        citation_count = np.nan
        title = row.get("OutputTitle")
        year = row.get("OutputYear")

        # Validate title
        if not (pd.notna(title) and isinstance(title, str) and title.strip()):
            logging.debug(f"Row {index}: Skipping due to missing or invalid title.")
            citation_counts.append(np.nan)
            time.sleep(REQUEST_DELAY)
            continue

        try:
            # Construct and validate search URL
            search_query = urllib.parse.quote_plus(title.strip())
            api_url = f"https://api.openalex.org/works?search={search_query}"

            # Add year filter if available
            if pd.notna(year):
                try:
                    year_int = int(year)
                    api_url += f"&filter=publication_year:{year_int}"
                except (ValueError, TypeError):
                    logging.warning(f"Row {index}: Invalid year format '{year}'. Proceeding without year filter.")

            # Make API request
            response = requests.get(
                api_url,
                params={"mailto": OPENALEX_EMAIL},
                timeout=15
            )
            
            # Validate and process response
            data = validate_response(response, title)
            citation_count = process_search_results(data, title) or np.nan
            
            logging.debug(
                f"Row {index}: Title '{title[:50]}...' - Found count {citation_count}"
            )

        except RateLimitError:
            logging.error(f"Row {index}: Rate limit exceeded (429). Increase REQUEST_DELAY. Stopping.")
            raise
        except CitationFetchError as e:
            logging.error(f"Row {index}: Title '{title[:50]}...' - {str(e)}")
        except requests.exceptions.Timeout:
            logging.error(f"Row {index}: Title '{title[:50]}...' - Request timed out.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Row {index}: Title '{title[:50]}...' - Request Error: {e}")
        except Exception as e:
            logging.error(f"Row {index}: Title '{title[:50]}...' - Unexpected error: {e}")

        citation_counts.append(citation_count)
        time.sleep(REQUEST_DELAY)

    return citation_counts

def validate_citations(citations: List[float], total_rows: int) -> None:
    """
    Validate citation results
    
    Args:
        citations: List of citation counts
        total_rows: Expected number of rows
        
    Raises:
        ValueError: If validation fails
    """
    if len(citations) != total_rows:
        raise ValueError(f"Citation count mismatch: got {len(citations)}, expected {total_rows}")
    if not all(isinstance(x, (float, int)) or pd.isna(x) for x in citations):
        raise ValueError("Invalid citation count types found")

def main():
    """Main function to load data, fetch citations, and save."""
    try:
        # Validate configuration
        validate_config()
        
        logging.info(f"Loading data from {INPUT_PATH}...")
        try:
            df = pd.read_csv(INPUT_PATH)
            if df.empty:
                raise ValueError("Input file is empty")
            logging.info(f"Successfully loaded {len(df)} rows.")
        except FileNotFoundError:
            raise CitationFetchError(f"Input file not found at {INPUT_PATH}")
        except pd.errors.EmptyDataError:
            raise CitationFetchError("Input file is empty")
        except Exception as e:
            raise CitationFetchError(f"Error loading input file: {str(e)}")

        # Fetch citations
        citations = fetch_citations(df)
        
        # Validate results
        validate_citations(citations, len(df))

        # Add column and save
        df["OutputCitationCount"] = citations
        logging.info("Finished fetching citations.")

        found_count = df["OutputCitationCount"].notna().sum()
        logging.info(f"Obtained {found_count} non-NaN citation counts out of {len(df)} records.")

        # Save results
        logging.info(f"Saving updated data to {OUTPUT_PATH}...")
        try:
            df.to_csv(OUTPUT_PATH, index=False)
            logging.info("Successfully saved updated data.")
        except PermissionError:
            raise CitationFetchError(f"Permission denied when saving to {OUTPUT_PATH}")
        except Exception as e:
            raise CitationFetchError(f"Error saving output file: {str(e)}")

    except CitationFetchError as e:
        logging.error(str(e))
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
