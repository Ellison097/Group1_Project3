import argparse
import asyncio
import csv
import logging
import os
import re
import time  # Import time for dynamic delays
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional, Union
import numpy as np

import aiohttp
import pandas as pd
from fuzzywuzzy import fuzz
from tqdm.asyncio import tqdm_asyncio  # Use tqdm's async version

# --- Configuration ---

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="enrich_combined_data.log",
    filemode="a",  # Append to the log file
)
logger = logging.getLogger(__name__)

# API Endpoints
OPENALEX_API_URL = "https://api.openalex.org/works"
CROSSREF_API_URL = "https://api.crossref.org/works"

# Rate Limiting Delays (seconds) - Slightly increased OpenAlex delay
OPENALEX_DELAY = 0.16  # ~6 req/sec average target
CROSSREF_DELAY = 0.05
RETRY_DELAY = 2.0  # Increased retry delay

# Concurrency Limiter
# Limit the max number of concurrent requests to OpenAlex/Crossref
# Adjust based on testing and API behavior (start lower, e.g., 15-20)
API_CONCURRENCY_LIMIT = 10

# Other Constants
MAX_RETRIES = 3
MAX_RESULTS_PER_API = 5  # How many results to fetch per API call
MIN_MATCH_SCORE = 70  # Minimum fuzzy match score to consider a result
HIGH_CONFIDENCE_SCORE = 80  # Score above which we trust the first API's result
REQUEST_TIMEOUT = 45  # Increased from 30
USER_AGENT = "FSRDC_Enrichment/1.2 (mailto:juzhan@upenn.edu)"  # Version bump

# Define the mapping from API types to our schema types
TYPE_MAPPING = {
    # Crossref types
    "journal-article": "JA",
    "book-chapter": "BC",
    "book": "BC",
    "monograph": "BC",
    "posted-content": "WP",  # Often preprints/working papers
    "proceedings-article": "JA",  # Often similar to journal articles
    "dissertation": "DI",
    "report": "RE",
    "dataset": "DS",
    "peer-review": "MI",  # Map peer reviews to Micsellaneous for now
    "reference-entry": "BC",  # Often part of a book
    "journal-issue": "MI",  # Whole issue, map to Misc
    "edited-book": "BC",
    "book-set": "BC",
    "book-track": "BC",
    "standard": "RE",
    "book-part": "BC",
    "book-section": "BC",
    "component": "SW",  # Software component
    "other": "MI",
    # OpenAlex types (some overlap, some unique)
    "journal": "MI",  # Represents the journal itself, not an article
    "proceedings": "MI",  # Represents the proceedings itself
    "reference-book": "BC",
    "repository": "DS",  # Often datasets or code
    "report-series": "RE",
    "book-series": "BC",
    "proceedings-series": "MI",
    "journal-volume": "MI",
    "proceedings-part": "JA",  # Often like an article in proceedings
    "editorial": "JA",  # Often published in journals
    "erratum": "JA",  # Often published in journals
    "grant": "MI",
    "letter": "JA",  # Often published in journals
    "parity": "MI",  # Unclear type
    # Add mappings found during testing if needed
    "reference-entry": "BC",
    "journal-issue": "MI",
    "book-part": "BC",
    "book-track": "BC",
    "dataset": "DS",
    "report": "RE",
    "standard": "RE",
}


# FSRDC Related Keywords (lowercase)
FSRDC_KEYWORDS = [
    "census bureau",
    "cecus",
    "bureau of the census",
    "fsrdc",
    "federal statistical research data center",
    "research data center",
    "rdc",
    "bea",
    "bureau of economic analysis",
    "restricted microdata",
    "confidential data",
    "annual survey of manufactures",
    "asm",
    "census of construction industries",
    "census of agriculture",
    "census of retail trade",
    "census of manufacturing",
    "com",
    "census of transportation",
    "census of population",
    "restricted data",
    "microdata",
    "confidential microdata",
    "restricted access",
    "irs",
    "internal revenue service",
    "federal reserve",
    "nber",
    "national bureau of economic research",
    "cepr",
    "center for economic studies",
    "ces",
    "longitudinal business database",
    "lbd",
    "longitudinal employer-household dynamics",
    "lehd",
    "survey of income and program participation",
    "sipp",
    "national health interview survey",
    "nhis",
    "national health and nutrition examination survey",
    "nhanes",
    "american community survey",
    "acs",
    "current population survey",
    "cps",
]

# --- Helper Functions ---


def clean_title(title):
    """Clean and normalize a title for better matching."""
    if pd.isna(title) or title is None:
        return ""
    title = str(title).lower()
    # Remove punctuation but keep spaces and alphanumeric characters
    title = re.sub(r"[^\w\s]", " ", title)
    # Replace multiple spaces with a single space
    title = re.sub(r"\s+", " ", title).strip()
    return title


def get_title_variations(title, source_file):
    """Generate title variations for searching."""
    variations = set()
    if pd.isna(title) or not title:
        return list(variations)

    original_title = str(title).strip()
    # Ensure original title is always included if not empty
    if original_title:
        variations.add(original_title)

    # Specific handling for group3 which might be all lowercase
    if source_file == "group3_clean2.csv":
        # Try title case
        title_case = original_title.title()
        if title_case != original_title:
            variations.add(title_case)
        # Try capitalizing only the first letter
        if original_title:
            first_upper = original_title[0].upper() + original_title[1:]
            if first_upper != original_title:
                variations.add(first_upper)
    else:
        # For other groups, assume title might already have casing
        # Add lowercase version for broader matching
        lower_case = original_title.lower()
        if lower_case != original_title:
            variations.add(lower_case)

    # Add cleaned version only if it's different from others already added
    cleaned = clean_title(original_title)
    if cleaned and cleaned not in variations:
        variations.add(cleaned)

    # Remove empty strings if any were generated
    variations.discard("")

    # Limit the number of variations to avoid excessive API calls for noisy titles
    return list(variations)[:5]  # Limit to max 5 variations


def format_authors(authors_list):
    """Formats a list of author dictionaries into standard strings."""
    display_names = []
    raw_names = []
    if not authors_list:
        return "", ""

    for author in authors_list:
        display = author.get("display_name", "").strip()
        raw = author.get("raw_name", "").strip()
        if display:
            display_names.append(display)
        if raw:
            raw_names.append(raw)
        elif display and not raw:  # Use display name if raw is missing
            raw_names.append(display)

    return "; ".join(display_names), "; ".join(raw_names)


def format_affiliations(affiliations_list):
    """Formats a list of affiliation dictionaries/strings."""
    display_names = set()
    raw_strings = set()
    if not affiliations_list:
        return "", ""

    for aff in affiliations_list:
        if isinstance(aff, dict):
            display = aff.get("display_name", "").strip()
            raw = aff.get("raw_string", "").strip()
            if display:
                display_names.add(display)
            if raw:
                raw_strings.add(raw)
            elif display and not raw:  # Use display name if raw is missing
                raw_strings.add(display)
        elif isinstance(aff, str) and aff.strip():  # Handle list of raw strings
            raw_strings.add(aff.strip())
            # Attempt to use the raw string as display name if no dicts provided
            display_names.add(aff.strip())

    # Prioritize display names if available, otherwise use raw strings for the 'Affiliations' column
    final_display = "; ".join(sorted(list(display_names))) if display_names else "; ".join(sorted(list(raw_strings)))
    final_raw = "; ".join(sorted(list(raw_strings)))

    return final_display, final_raw


def extract_crossref_data(item):
    """Extracts and standardizes data from a Crossref API item."""
    data = {}
    try:
        # Title
        title_list = item.get("title", [])
        data["std_title"] = (
            title_list[0].strip()
            if title_list and isinstance(title_list, list)
            else (str(item.get("title", "")).strip() if item.get("title") else None)
        )

        # Type
        data["std_type"] = item.get("type", "other")

        # Date (prefer published-print, then published-online, then created)
        date_parts = None
        for date_field in ["published-print", "published-online", "created"]:
            if item.get(date_field) and item[date_field].get("date-parts"):
                date_parts = item[date_field]["date-parts"][0]
                break
        data["std_year"] = int(date_parts[0]) if date_parts and len(date_parts) >= 1 else None
        data["std_month"] = int(date_parts[1]) if date_parts and len(date_parts) >= 2 else None

        # Authors
        authors = []
        for author in item.get("author", []):
            family = author.get("family", "")
            given = author.get("given", "")
            name = f"{given} {family}".strip()
            # Crossref sometimes lacks raw names, use constructed name
            if name:  # Only add if name is not empty
                authors.append({"display_name": name, "raw_name": name})
        data["std_authors"] = authors

        # Affiliations (Crossref often less structured here)
        affiliations = []
        for author in item.get("author", []):
            for aff in author.get("affiliation", []):
                aff_name = aff.get("name", "").strip()
                if aff_name:
                    # Use name as both display and raw string
                    affiliations.append({"display_name": aff_name, "raw_string": aff_name})
        data["std_affiliations"] = affiliations

        # DOI
        data["std_doi"] = item.get("DOI", None)

        # Venue (Journal/Book Title)
        container_list = item.get("container-title", [])
        data["std_venue"] = (
            container_list[0].strip()
            if container_list and isinstance(container_list, list) and container_list[0]
            else (str(item.get("container-title", "")).strip() if item.get("container-title") else None)
        )

        # Volume, Issue, Pages
        data["std_volume"] = str(item.get("volume", "")).strip() or None
        data["std_issue"] = str(item.get("issue", "")).strip() or None
        data["std_pages"] = str(item.get("page", "")).strip() or None

        # Keywords (Subject) - often less common/standardized in Crossref
        subjects = item.get("subject", [])
        data["std_keywords"] = "; ".join([str(s).strip() for s in subjects if str(s).strip()]) or None

        # Abstract - Crossref often doesn't provide full abstracts via this endpoint
        # Attempt to get abstract, might be inside <jats:p> tags
        abstract_text = item.get("abstract", None)
        if abstract_text and isinstance(abstract_text, str):
            # Basic regex to strip potential XML tags (may need refinement)
            abstract_text = re.sub("<[^<]+?>", "", abstract_text).strip()
        data["std_abstract"] = abstract_text or None

    except Exception as e:
        logger.error(f"Error parsing Crossref item: {e} - Item: {item}")
        return {}  # Return empty dict on parsing error

    # Filter out None values before returning
    return {k: v for k, v in data.items() if v is not None and v != ""}


def extract_openalex_data(item):
    """Extracts standardized data fields from an OpenAlex API item."""
    standardized_data = {
        "title": None,
        "year": None,
        "month": None,
        "doi": None,
        "authors": [],
        "affiliations": [],
        "keywords": [],
        "abstract": None,
        "venue": None,
        "volume": None,
        "issue": None,
        "pages": None,
        "type": None,
        "status": None,
    }

    # Basic fields
    standardized_data["title"] = item.get("title")
    standardized_data["year"] = item.get("publication_year")
    standardized_data["month"] = (
        item.get("publication_date", "")[5:7]
        if item.get("publication_date") and len(item.get("publication_date")) >= 7
        else None
    )  # Extract month MM
    standardized_data["doi"] = item.get("doi")

    # Type
    standardized_data["type"] = item.get("type")

    # Venue - **NEW LOGIC: Prioritize primary_location.source.display_name**
    primary_location = item.get("primary_location", {})
    source = primary_location.get("source", {}) if primary_location else {}
    venue_from_source = source.get("display_name") if source else None

    if venue_from_source:
        standardized_data["venue"] = venue_from_source
        logger.debug(f"Extracted venue from primary_location.source: {venue_from_source}")
    else:
        # Fallback to host_venue
        host_venue = item.get("host_venue", {})
        venue_from_host = host_venue.get("display_name") if host_venue else None
        if venue_from_host:
            standardized_data["venue"] = venue_from_host
            logger.debug(f"Extracted venue from host_venue: {venue_from_host}")
        else:
            logger.debug("Could not extract venue from primary_location or host_venue.")

    # Volume, Issue, Pages (often available in biblio object)
    biblio = item.get("biblio", {})
    standardized_data["volume"] = biblio.get("volume")
    standardized_data["issue"] = biblio.get("issue")
    first_page = biblio.get("first_page")
    last_page = biblio.get("last_page")
    if first_page and last_page:
        standardized_data["pages"] = f"{first_page}-{last_page}"
    elif first_page:
        standardized_data["pages"] = str(first_page)

    # Authors and Affiliations
    authors_list = []
    affiliations_list = set()  # Use set to store unique raw strings
    for authorship in item.get("authorships", []):
        author_info = authorship.get("author", {})
        inst_info = authorship.get("institutions", [])
        raw_aff_strings = authorship.get("raw_affiliation_strings", [])

        authors_list.append(
            {
                "display_name": author_info.get("display_name"),
                "raw_name": authorship.get("raw_author_name"),  # Get raw name from authorship
            }
        )

        # Collect affiliations - prioritize display_name, fallback to raw strings
        author_affs = set()
        for inst in inst_info:
            if inst.get("display_name"):
                author_affs.add(inst["display_name"].strip())
        for raw_aff in raw_aff_strings:
            if raw_aff.strip():
                author_affs.add(raw_aff.strip())  # Add raw strings as well

        affiliations_list.update(author_affs)  # Add unique affiliations for this author

    standardized_data["authors"] = authors_list
    # Convert set of affiliations back to a list for consistency, though format_affiliations handles it
    standardized_data["affiliations"] = list(affiliations_list)

    # Keywords (Concepts in OpenAlex)
    keywords_list = [
        concept.get("display_name") for concept in item.get("concepts", []) if concept.get("display_name")
    ]
    standardized_data["keywords"] = keywords_list

    # Abstract (Reconstruct from inverted index)
    abstract_inverted_index = item.get("abstract_inverted_index")
    if abstract_inverted_index:
        try:
            max_index = -1
            for positions in abstract_inverted_index.values():
                max_index = max(max_index, max(positions))

            words = [""] * (max_index + 1)
            for word, positions in abstract_inverted_index.items():
                for pos in positions:
                    if 0 <= pos < len(words):
                        words[pos] = word
                    else:
                        logger.warning(f"Abstract index {pos} out of bounds (max: {max_index}) for word '{word}'")

            standardized_data["abstract"] = " ".join(filter(None, words))
        except Exception as e:
            logger.error(f"Error reconstructing abstract: {e}")
            standardized_data["abstract"] = None
    else:
        standardized_data["abstract"] = None

    # Status (less common in OpenAlex, maybe use is_oa or type?)
    # For now, leave as None unless a better field is identified
    standardized_data["status"] = None

    return standardized_data


def create_biblio(std_data, output_type):
    """Create a formatted bibliographic entry based on standardized metadata and type."""
    authors_display, _ = format_authors(std_data.get("std_authors", []))
    year = std_data.get("std_year", "n.d.")
    title = std_data.get("std_title", "[Title missing]")

    # Basic author formatting for biblio (Last, F. M.)
    author_bib_list = []
    for author in std_data.get("std_authors", []):
        name = author.get("display_name", "").strip()
        if not name:
            continue
        parts = name.split()
        if len(parts) > 1:
            family = parts[-1]
            given = "".join([p[0] + "." for p in parts[:-1] if p])
            author_bib_list.append(f"{family}, {given}")
        else:
            author_bib_list.append(name)  # Single name

    if len(author_bib_list) == 0:
        authors_bib = "[Authors missing]"
    elif len(author_bib_list) == 1:
        authors_bib = author_bib_list[0]
    elif len(author_bib_list) == 2:
        authors_bib = f"{author_bib_list[0]} & {author_bib_list[1]}"
    elif len(author_bib_list) <= 6:  # List all if 6 or fewer
        authors_bib = ", ".join(author_bib_list[:-1]) + f", & {author_bib_list[-1]}"
    else:  # Use et al. for more than 6
        authors_bib = f"{author_bib_list[0]} et al."

    # Format based on type
    biblio = f"{authors_bib} ({year}). {title}."

    venue = std_data.get("std_venue")
    volume = std_data.get("std_volume")
    issue = std_data.get("std_issue")
    pages = std_data.get("std_pages")
    doi = std_data.get("std_doi")

    if output_type == "JA":  # Journal article
        if venue:
            biblio += f" *{venue}*"  # Italicize journal title
        if volume:
            biblio += f", *{volume}*"  # Italicize volume
        if issue:
            biblio += f"({issue})"
        if pages:
            biblio += f", {pages}"
        biblio += "."
    elif output_type == "BC":  # Book chapter / Book
        if venue:
            biblio += f" In *{venue}*"  # Book title
        if pages:
            biblio += f" (pp. {pages})"
        # Publisher info might be in venue for books, or missing. Add if available?
        biblio += "."
    elif output_type == "DI":  # Dissertation
        biblio += " [Dissertation]"
        if venue:
            biblio += f", {venue}"  # Institution
        biblio += "."
    elif output_type == "RE":  # Report
        biblio += " [Report]"
        if venue:
            biblio += f". {venue}"  # Publisher/Institution
        if issue:
            biblio += f" (No. {issue})"  # Report number if available in issue field
        biblio += "."
    elif output_type == "WP":  # Working Paper
        biblio += " [Working Paper]"
        if venue:
            biblio += f". {venue}"  # Series or Institution
        if issue:
            biblio += f" (No. {issue})"  # Paper number
        biblio += "."
    else:  # Default/MI/Other
        if venue:
            biblio += f" {venue}."  # Add venue if available

    if doi:
        biblio += f" https://doi.org/{doi}"

    return biblio.replace("..", ".").strip()  # Clean up double periods


def check_fsrdc_relevance(std_data, abstract_text):
    """Check if the paper is related to FSRDC based on keywords."""
    matching_keywords = set()
    if not std_data:
        return False, ""

    text_fields_to_check = [
        std_data.get("std_title", ""),
        abstract_text or "",  # Use the passed abstract
        std_data.get("std_keywords", ""),
        std_data.get("std_venue", ""),  # Check venue name too
    ]

    # Add affiliations for checking
    _, raw_affiliations_str = format_affiliations(std_data.get("std_affiliations", []))
    text_fields_to_check.append(raw_affiliations_str)

    full_text = " ".join(text_fields_to_check).lower()

    for keyword in FSRDC_KEYWORDS:
        # Use regex to match whole words/phrases to avoid partial matches like 'ces' in 'processes'
        # Handle keywords with spaces correctly
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if re.search(pattern, full_text):
            matching_keywords.add(keyword)

    is_related = len(matching_keywords) > 0
    return is_related, ", ".join(sorted(list(matching_keywords)))


# --- API Interaction ---


async def fetch_api(url, session, params, api_name, base_delay, semaphore, record_id="N/A"):
    """Fetches data from an API with rate limiting, retries, and semaphore."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    last_exception = None

    for attempt in range(MAX_RETRIES):
        # Acquire semaphore before making the request
        async with semaphore:
            # Apply base delay before the request
            await asyncio.sleep(base_delay)
            try:
                logger.debug(
                    f"Record {record_id}: Attempt {attempt+1}/{MAX_RETRIES} - Querying {api_name}: {url} with params: {params}"
                )
                async with session.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT) as response:
                    # Log status immediately
                    logger.debug(
                        f"Record {record_id}: Received status {response.status} from {api_name} for params {params}"
                    )

                    if response.status == 200:
                        try:
                            data = await response.json()
                            logger.debug(f"Record {record_id}: Successfully fetched data from {api_name}.")
                            return data
                        except aiohttp.ContentTypeError:
                            logger.error(
                                f"Record {record_id}: Invalid JSON response from {api_name} (status {response.status}). Response: {await response.text()[:200]}"
                            )
                            last_exception = ValueError("Invalid JSON response")
                            # Consider breaking if JSON is invalid, retrying might not help
                            break  # Don't retry on bad JSON
                        except Exception as json_e:
                            logger.error(
                                f"Record {record_id}: Error decoding JSON from {api_name} (status {response.status}): {json_e}"
                            )
                            last_exception = json_e
                            break  # Don't retry on JSON decoding error

                    elif response.status == 429:  # Rate limit hit
                        retry_after_header = response.headers.get("Retry-After")
                        # Calculate default backoff first
                        wait_time = RETRY_DELAY * (attempt + 1)

                        if retry_after_header:
                            try:
                                # Try parsing as integer seconds first
                                wait_time = float(retry_after_header)
                                logger.warning(
                                    f"Record {record_id}: Rate limit hit for {api_name}. Retrying after {wait_time:.2f} seconds (from header). Attempt {attempt+1}/{MAX_RETRIES}"
                                )
                            except ValueError:
                                # If float conversion fails, try parsing as HTTP-date
                                try:
                                    retry_dt = parsedate_to_datetime(retry_after_header)
                                    now_dt = datetime.now(timezone.utc)  # Use timezone-aware datetime
                                    delta = retry_dt - now_dt
                                    wait_time = max(delta.total_seconds(), 1.0)  # Ensure at least 1s wait
                                    logger.warning(
                                        f"Record {record_id}: Rate limit hit for {api_name}. Retrying after {wait_time:.2f} seconds (calculated from date header: {retry_after_header}). Attempt {attempt+1}/{MAX_RETRIES}"
                                    )
                                except Exception as date_e:
                                    # If date parsing also fails, use default backoff
                                    logger.warning(
                                        f"Record {record_id}: Rate limit hit for {api_name}. Could not parse Retry-After header '{retry_after_header}' as seconds or date. Using default backoff {wait_time:.2f}s. Error: {date_e}. Attempt {attempt+1}/{MAX_RETRIES}"
                                    )
                        else:
                            logger.warning(
                                f"Record {record_id}: Rate limit hit for {api_name}. No Retry-After header. Retrying after {wait_time:.2f} seconds (default backoff). Attempt {attempt+1}/{MAX_RETRIES}"
                            )

                        # Apply wait time and continue to next attempt
                        await asyncio.sleep(wait_time)
                        continue  # Go to next attempt

                    elif response.status == 404:
                        logger.info(
                            f"Record {record_id}: Received 404 Not Found from {api_name} for params {params}. Likely no match."
                        )
                        return None  # Treat 404 as no results found

                    else:
                        # Handle other HTTP errors
                        error_text = await response.text()
                        logger.warning(
                            f"Record {record_id}: HTTP error {response.status} from {api_name} for params {params}. Attempt {attempt+1}/{MAX_RETRIES}. Response: {error_text[:200]}"
                        )
                        last_exception = aiohttp.ClientResponseError(
                            response.request_info,
                            response.history,
                            status=response.status,
                            message=error_text[:200],
                            headers=response.headers,
                        )
                        # Apply a shorter delay before retrying non-429 errors
                        await asyncio.sleep(RETRY_DELAY / 2 * (attempt + 1))
                        # Continue retrying for other server errors (e.g., 5xx)

            except asyncio.TimeoutError:
                logger.warning(
                    f"Record {record_id}: Timeout error querying {api_name} (Attempt {attempt+1}/{MAX_RETRIES}). Retrying..."
                )
                last_exception = asyncio.TimeoutError(f"Timeout after {REQUEST_TIMEOUT}s")
                # Apply retry delay before next attempt
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            except aiohttp.ClientConnectorError as e:
                logger.warning(
                    f"Record {record_id}: Connection error querying {api_name} (Attempt {attempt+1}/{MAX_RETRIES}): {e}. Retrying..."
                )
                last_exception = e
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            except Exception as e:
                logger.error(
                    f"Record {record_id}: Unexpected error during {api_name} fetch (Attempt {attempt+1}/{MAX_RETRIES}): {e}"
                )
                last_exception = e
                # Break on unexpected errors? Or retry? Let's retry with delay.
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))

    # If loop finishes without returning data
    logger.error(
        f"Record {record_id}: Failed to fetch from {api_name} after {MAX_RETRIES} attempts. Last exception: {last_exception}. URL: {url}. Params: {params}"
    )
    return None


async def get_metadata_async(title, source_file, session, semaphore, record_id="N/A"):
    """Search APIs for metadata, preferring OpenAlex, using semaphore."""
    clean_search_title = clean_title(title)
    title_variations = get_title_variations(title, source_file)

    if not title_variations:
        logger.warning(f"Record {record_id}: No valid title variations generated for '{title}'.")
        return None, None, 0

    best_match_overall = None
    best_score_overall = 0
    source_api_overall = None

    # --- Try OpenAlex First ---
    logger.debug(f"Record {record_id}: Trying OpenAlex for title '{title}'")
    openalex_match = None
    openalex_score = 0
    for i, title_variant in enumerate(title_variations):
        params = {"search": title_variant, "per-page": MAX_RESULTS_PER_API}
        logger.debug(f"Record {record_id}: OpenAlex variation {i+1}/{len(title_variations)}: '{title_variant}'")
        # Pass semaphore to fetch_api
        data = await fetch_api(OPENALEX_API_URL, session, params, "OpenAlex", OPENALEX_DELAY, semaphore, record_id)

        if data and "results" in data and data["results"]:
            items = data["results"]
            for item in items:
                item_title = item.get("title")
                if item_title:
                    clean_item_title = clean_title(item_title)
                    score = fuzz.token_sort_ratio(clean_search_title, clean_item_title)
                    logger.debug(f"Record {record_id}: OpenAlex Candidate: '{item_title}' -> Score: {score}")
                    if score > openalex_score and score >= MIN_MATCH_SCORE:
                        openalex_score = score
                        openalex_match = item
                        openalex_match["_source_api"] = "openalex"  # Tag the source

            # If we found a high-confidence match, stop trying variations for OpenAlex
            if openalex_score >= HIGH_CONFIDENCE_SCORE:
                logger.debug(f"Record {record_id}: High confidence match found in OpenAlex (Score: {openalex_score}).")
                break
        elif data is None:  # Handle fetch_api returning None on error/404
            logger.debug(f"Record {record_id}: No results or error from OpenAlex for variation '{title_variant}'.")
        # Small delay between variations if needed, but semaphore + base delay should handle it
        # await asyncio.sleep(0.02)

    if openalex_match:
        best_match_overall = openalex_match
        best_score_overall = openalex_score
        source_api_overall = "openalex"
        logger.info(f"Record {record_id}: Best OpenAlex match for '{title}': Score {openalex_score}")

    # --- Try Crossref if OpenAlex wasn't good enough ---
    if best_score_overall < HIGH_CONFIDENCE_SCORE:
        logger.debug(
            f"Record {record_id}: OpenAlex score ({best_score_overall}) below threshold {HIGH_CONFIDENCE_SCORE}. Trying Crossref."
        )
        crossref_match = None
        crossref_score = 0
        for i, title_variant in enumerate(title_variations):
            # Crossref uses query.bibliographic or query.title
            params = {"query.title": title_variant, "rows": MAX_RESULTS_PER_API}
            logger.debug(f"Record {record_id}: Crossref variation {i+1}/{len(title_variations)}: '{title_variant}'")
            # Pass semaphore to fetch_api
            data = await fetch_api(CROSSREF_API_URL, session, params, "Crossref", CROSSREF_DELAY, semaphore, record_id)

            if data and "message" in data and "items" in data["message"] and data["message"]["items"]:
                items = data["message"]["items"]
                for item in items:
                    item_title_list = item.get("title", [])
                    item_title = item_title_list[0] if item_title_list else None
                    if item_title:
                        clean_item_title = clean_title(item_title)
                        score = fuzz.token_sort_ratio(clean_search_title, clean_item_title)
                        logger.debug(f"Record {record_id}: Crossref Candidate: '{item_title}' -> Score: {score}")
                        if score > crossref_score and score >= MIN_MATCH_SCORE:
                            crossref_score = score
                            crossref_match = item
                            crossref_match["_source_api"] = "crossref"  # Tag the source

                # If we found a high-confidence match, stop trying variations for Crossref
                if crossref_score >= HIGH_CONFIDENCE_SCORE:
                    logger.debug(
                        f"Record {record_id}: High confidence match found in Crossref (Score: {crossref_score})."
                    )
                    break
            elif data is None:
                logger.debug(f"Record {record_id}: No results or error from Crossref for variation '{title_variant}'.")
            # await asyncio.sleep(0.02)

        # Compare Crossref result with OpenAlex result (if any)
        if crossref_match and crossref_score > best_score_overall:
            best_match_overall = crossref_match
            best_score_overall = crossref_score
            source_api_overall = "crossref"
            logger.info(f"Record {record_id}: Found better match in Crossref for '{title}': Score {crossref_score}")
        elif crossref_match:
            logger.info(
                f"Record {record_id}: Crossref match found (Score: {crossref_score}), but OpenAlex match was better (Score: {best_score_overall})."
            )

    if best_match_overall:
        logger.info(
            f"Record {record_id}: Final best match for '{title}' from {source_api_overall} with score {best_score_overall}."
        )
        return best_match_overall, source_api_overall, best_score_overall
    else:
        logger.info(f"Record {record_id}: No suitable match found for '{title}' in OpenAlex or Crossref.")
        return None, None, 0


async def process_single_record(record_dict, session, semaphore):
    """Processes a single record dictionary, fetches metadata, enriches, and returns."""
    # Use a unique identifier if available (like ProjID or index), otherwise use title
    record_id = record_dict.get("ProjID", record_dict.get("index", "N/A"))
    title = record_dict.get("OutputTitle", "")
    source_file = record_dict.get("SourceFile", "")
    result = record_dict.copy()  # Start with original data
    status = "unknown"
    error_message = None

    # Define enrichment fields to ensure they exist even if enrichment fails
    enrichment_fields = [
        "OutputBiblio",
        "OutputType",
        "OutputStatus",
        "OutputVenue",
        "OutputYear",
        "OutputMonth",
        "OutputVolume",
        "OutputNumber",
        "OutputPages",
        "DOI",
        "Keywords",
        "Authors",
        "RawAuthorNames",
        "Affiliations",
        "RawAffiliations",
        "IsFSRDCRelated",
        "MatchingFSRDCKeywords",
        "_match_score",
        "_metadata_source",
        "_error",
    ]
    for field in enrichment_fields:
        result[field] = ""  # Initialize enrichment fields

    # --- Step 1: Check Title ---
    if pd.isna(title) or not str(title).strip():
        logger.warning(f"Record {record_id}: Skipped - Missing or invalid title.")
        result["_error"] = "Missing or invalid title"
        status = "skipped_no_title"
        return result, status  # Return early

    try:
        # --- Step 2: Get Metadata ---
        # Pass semaphore here
        metadata, source_api, score = await get_metadata_async(title, source_file, session, semaphore, record_id)

        # --- Step 3: Enrich Record ---
        if metadata and source_api and score >= MIN_MATCH_SCORE:
            result["_metadata_source"] = source_api
            result["_match_score"] = score
            standardized_data = None

            # --- Step 3a: Standardize Data ---
            try:
                if source_api == "openalex":
                    standardized_data = extract_openalex_data(metadata)
                elif source_api == "crossref":
                    standardized_data = extract_crossref_data(metadata)

                if not standardized_data:
                    raise ValueError("Standardized data extraction returned None")

                logger.debug(f"Record {record_id}: Standardized data extracted from {source_api}.")

            except Exception as e:
                status = "error_parsing"
                error_message = f"Failed to parse {source_api} metadata: {e}"
                logger.error(
                    f"Record {record_id}: {error_message} for title '{title}'. Metadata: {str(metadata)[:500]}"
                )  # Log part of metadata on error
                # Continue without enrichment, but log error

            # --- Step 3b: Populate Result Fields ---
            if standardized_data:
                result["OutputTitle"] = standardized_data.get("std_title", title)  # Update title if better one found
                result["OutputType"] = TYPE_MAPPING.get(standardized_data.get("std_type", "other"), "MI")
                result["OutputVenue"] = standardized_data.get("std_venue", "")
                result["OutputYear"] = standardized_data.get("std_year", "")
                result["OutputMonth"] = standardized_data.get("std_month", "")
                result["OutputVolume"] = standardized_data.get("std_volume", "")
                result["OutputNumber"] = standardized_data.get("std_issue", "")
                result["OutputPages"] = standardized_data.get("std_pages", "")
                result["DOI"] = standardized_data.get("std_doi", "")
                result["Keywords"] = standardized_data.get("std_keywords", "")

                authors_list = standardized_data.get("std_authors", [])
                result["Authors"], result["RawAuthorNames"] = format_authors(authors_list)

                affiliations_list = standardized_data.get("std_affiliations", [])
                result["Affiliations"], result["RawAffiliations"] = format_affiliations(affiliations_list)

                # Determine status (simple version: published if year exists)
                result["OutputStatus"] = "PB" if result["OutputYear"] else "UP"

                # FSRDC Check (requires abstract)
                abstract = standardized_data.get("std_abstract")
                is_related, matching_keywords = check_fsrdc_relevance(standardized_data, abstract)
                result["IsFSRDCRelated"] = is_related
                result["MatchingFSRDCKeywords"] = matching_keywords

                # Create Bibliography (after all fields are populated)
                result["OutputBiblio"] = create_biblio(standardized_data, result["OutputType"])

                status = "enriched"
                logger.info(
                    f"Record {record_id}: Successfully enriched '{title}' from {source_api} (Score: {score}). FSRDC Related: {is_related}"
                )

            # If parsing failed, status is already set to error_parsing

        else:
            # No match found or score too low
            status = "no_match"
            result["_metadata_source"] = "none"
            result["_match_score"] = 0
            logger.info(f"Record {record_id}: No suitable metadata match found for title '{title}'")
            # Keep original title, leave enrichment fields empty/default

    except Exception as e:
        status = "error_processing"
        error_message = f"Error processing record '{title}': {e}"
        logger.exception(f"Record {record_id}: {error_message}")  # Log full traceback

    if error_message:
        result["_error"] = error_message

    # Ensure all expected columns are present before returning
    final_result = {}
    # Define expected columns based on input + enrichment
    all_expected_columns = list(record_dict.keys()) + enrichment_fields
    # Make unique and preserve order somewhat
    ordered_unique_columns = list(dict.fromkeys(all_expected_columns).keys())

    for col in ordered_unique_columns:
        final_result[col] = result.get(col, "")  # Default to empty string if somehow missing

    return final_result, status


async def process_dataframe_async(df, output_file):
    """Processes the entire dataframe asynchronously and saves incrementally."""
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")

    # Define headers based on input columns + enrichment fields
    enrichment_fields = [
        "OutputBiblio",
        "OutputType",
        "OutputStatus",
        "OutputVenue",
        "OutputYear",
        "OutputMonth",
        "OutputVolume",
        "OutputNumber",
        "OutputPages",
        "DOI",
        "Keywords",
        "Authors",
        "RawAuthorNames",
        "Affiliations",
        "RawAffiliations",
        "IsFSRDCRelated",
        "MatchingFSRDCKeywords",
        "_match_score",
        "_metadata_source",
        "_error",
    ]
    original_cols = list(df.columns)
    final_headers = original_cols + [h for h in enrichment_fields if h not in original_cols]
    final_headers = list(dict.fromkeys(final_headers).keys())

    # Create the semaphore to limit concurrency
    semaphore = asyncio.Semaphore(API_CONCURRENCY_LIMIT)
    logger.info(f"Using Semaphore with concurrency limit: {API_CONCURRENCY_LIMIT}")

    # Adjust connector limit based on system resources and API politeness
    connector = aiohttp.TCPConnector(
        limit=API_CONCURRENCY_LIMIT + 10,  # Allow slightly more connections than semaphore limit
        ssl=False,  # Disable SSL verification if needed for local issues
    )
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT + 5)  # Client timeout slightly > request timeout
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Prepare tasks, passing the semaphore to each worker
        tasks = [process_single_record(row.to_dict(), session, semaphore) for _, row in df.iterrows()]

        # Counters for progress reporting
        processed_count = 0
        enriched_count = 0
        skipped_count = 0
        no_match_count = 0
        error_count = 0

        file_exists = os.path.exists(output_file)
        with open(output_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=final_headers, extrasaction="ignore")
            if not file_exists or os.path.getsize(output_file) == 0:
                writer.writeheader()
                logger.info(f"Writing header to new file: {output_file}")

            logger.info(f"Starting processing {len(tasks)} records...")
            pbar = tqdm_asyncio(total=len(tasks), desc="Enriching records", unit="record")
            start_time = time.time()

            for task_future in asyncio.as_completed(tasks):
                try:
                    result_dict, status = await task_future
                    writer.writerow(result_dict)
                    processed_count += 1

                    # Update counters based on status
                    if status == "enriched":
                        enriched_count += 1
                    elif status == "skipped_no_title":
                        skipped_count += 1
                    elif status == "no_match":
                        no_match_count += 1
                    elif status.startswith("error"):
                        error_count += 1

                    # Update tqdm postfix every N records
                    if processed_count % 20 == 0:
                        elapsed_time = time.time() - start_time
                        rate = processed_count / elapsed_time if elapsed_time > 0 else 0
                        pbar.set_postfix(
                            enriched=enriched_count,
                            skipped=skipped_count,
                            no_match=no_match_count,
                            errors=error_count,
                            rate=f"{rate:.1f} rec/s",
                            refresh=False,  # Set refresh=False for potentially better performance
                        )
                        # Log progress less frequently
                        if processed_count % 200 == 0:
                            logger.info(
                                f"Progress: {processed_count}/{len(tasks)} records. "
                                f"Enriched: {enriched_count}, Skipped: {skipped_count}, "
                                f"No Match: {no_match_count}, Errors: {error_count}. Rate: {rate:.1f} rec/s"
                            )

                except Exception as e:
                    error_count += 1
                    processed_count += 1
                    logger.error(f"Error processing task in main loop: {e}")
                finally:
                    pbar.update(1)

            pbar.close()

    logger.info(f"Finished processing. Total records processed: {processed_count}")
    return processed_count, enriched_count, skipped_count, no_match_count, error_count


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich combined publication data using OpenAlex and Crossref APIs.")
    parser.add_argument(
        "--input", type=str, default="combined_mapped_data_raw.csv", help="Input CSV file path (combined data)."
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file path. If not provided, defaults to [input_filename]_enriched_[timestamp].csv",
    )
    parser.add_argument(
        "--title-column",
        type=str,
        default="OutputTitle",  # Assuming the mapping step created this column
        help="Name of the column containing the title to search for.",
    )

    args = parser.parse_args()

    input_file = args.input
    output_file = args.output
    title_col = args.title_column

    # Determine output filename if not provided
    if not output_file:
        base_filename = os.path.splitext(os.path.basename(input_file))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{base_filename}_enriched_{timestamp}.csv"
        # Ensure output is saved in a dedicated directory
        output_dir = "Project_Data_Enriched_Combined"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, os.path.basename(output_file))

    logger.info("=" * 50)
    logger.info(f"Starting enrichment process at {datetime.now()}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Title column: {title_col}")
    logger.info(f"OpenAlex Delay: {OPENALEX_DELAY}s, Crossref Delay: {CROSSREF_DELAY}s")
    logger.info(f"Min Match Score: {MIN_MATCH_SCORE}, High Confidence Score: {HIGH_CONFIDENCE_SCORE}")
    logger.info(f"Max Retries: {MAX_RETRIES}, Retry Delay: {RETRY_DELAY}s")
    logger.info(f"Request Timeout: {REQUEST_TIMEOUT}s")
    logger.info(f"API Concurrency Limit (Semaphore): {API_CONCURRENCY_LIMIT}")

    # Load data
    try:
        # Address DtypeWarning by using low_memory=False
        df = pd.read_csv(input_file, low_memory=False)
        # Ensure the title column exists
        if title_col not in df.columns:
            logger.error(
                f"Title column '{title_col}' not found in input file '{input_file}'. Available columns: {df.columns.tolist()}"
            )
            exit(1)
        # Rename title column to 'OutputTitle' internally if it's different, for consistency
        if title_col != "OutputTitle":
            logger.info(f"Renaming column '{title_col}' to 'OutputTitle' for processing.")
            df = df.rename(columns={title_col: "OutputTitle"})

        # Ensure 'SourceFile' column exists for group-specific logic
        if "SourceFile" not in df.columns:
            logger.warning(
                "'SourceFile' column not found. Group-specific logic (e.g., for group3 titles) will not be applied."
            )
            df["SourceFile"] = ""  # Add dummy column if missing

        # Add a unique index if not present, useful for logging
        if "index" not in df.columns:
            df.reset_index(inplace=True)

        logger.info(f"Loaded {len(df)} records from {input_file}")
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}")
        exit(1)
    except Exception as e:
        logger.error(f"Error loading input file: {e}")
        exit(1)

    # Run asynchronous processing
    main_start_time = datetime.now()
    processed_count, enriched_count, skipped_count, no_match_count, error_count = asyncio.run(
        process_dataframe_async(df, output_file)
    )
    main_end_time = datetime.now()
    duration = main_end_time - main_start_time

    logger.info(f"Enrichment process finished at {main_end_time}")
    logger.info(f"Total time taken: {duration}")
    logger.info(f"Total records processed: {processed_count}")
    logger.info(f"Records enriched: {enriched_count}")
    logger.info(f"Records skipped (no title): {skipped_count}")
    logger.info(f"Records with no match found: {no_match_count}")
    logger.info(f"Records with errors: {error_count}")

    # Final summary (using counts from processing)
    print("\n--- Enrichment Summary ---")
    print(f"Output saved to: {output_file}")
    print(f"Total records processed: {processed_count}")
    if processed_count > 0:
        print(f"Successfully enriched: {enriched_count} ({enriched_count/processed_count*100:.2f}%)")
        print(f"Skipped (no title): {skipped_count} ({skipped_count/processed_count*100:.2f}%)")
        print(f"No match found: {no_match_count} ({no_match_count/processed_count*100:.2f}%)")
        print(f"Errors during processing: {error_count} ({error_count/processed_count*100:.2f}%)")

        # Try reading output for FSRDC stats if file exists
        try:
            enriched_df_final = pd.read_csv(output_file, low_memory=False)
            fsrdc_related_count = enriched_df_final[enriched_df_final["IsFSRDCRelated"] == True].shape[0]
            print(
                f"FSRDC related (in output): {fsrdc_related_count} ({fsrdc_related_count/len(enriched_df_final)*100:.2f}%)"
            )
            source_counts = enriched_df_final["_metadata_source"].value_counts()
            print(f"Metadata sources: {source_counts.to_dict()}")
        except Exception as e:
            logger.error(f"Could not read output file for final FSRDC/Source stats: {e}")
            print("Could not read output file for final FSRDC/Source stats.")

    print(f"Total time: {duration}")
    print("See 'enrich_combined_data.log' for detailed logs.")
