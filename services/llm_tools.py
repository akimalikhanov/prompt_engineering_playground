import importlib
import os
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET  # nosemgrep: use-defused-xml
from collections.abc import Iterable
from datetime import datetime
from typing import Any

import requests

from utils.load_configs import _load_tools_registry


def get_custom_movie_details(title: str, api_key: str | None = None, plot: str = "full"):
    import os

    if api_key is None:
        from dotenv import load_dotenv

        load_dotenv()
        api_key = os.getenv("OMDB_API_KEY", "")
    url = f"http://www.omdbapi.com/?t={title}&apikey={api_key}&plot={plot}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if data.get("Response") == "True":
            custom_json = {
                "Title": data.get("Title"),
                "Year": data.get("Year"),
                "Rated": data.get("Rated"),
                "Release Date": data.get("Released"),
                "Runtime": data.get("Runtime"),
                "Genre": data.get("Genre"),
                "Director": data.get("Director"),
                "Actors": data.get("Actors"),
                "Plot": data.get("Plot"),
                "Country": data.get("Country"),
                "Awards": data.get("Awards"),
                "Ratings": {r["Source"]: r["Value"] for r in data.get("Ratings", [])},
                "imdb_link": f"https://www.imdb.com/title/{data.get('imdbID')}/"
                if data.get("imdbID")
                else None,
                "Box office (USA)": data.get("BoxOffice"),
            }
            return custom_json
        else:
            return {"error": data.get("Error", "Unknown error")}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {e}"}


def search_arxiv(query: str, max_results: int = 10, start: int = 0) -> list[dict]:
    """
    Search arXiv API and return results in JSON format.

    Args:
        query (str): Search query string
        max_results (int): Maximum number of results to return (default: 10)
        start (int): Starting index for pagination (default: 0)

    Returns:
        List[Dict]: List of article dictionaries, each containing:
            - id: arXiv ID
            - title: Article title
            - links: Dictionary of links (e.g., {'html': '...', 'pdf': '...'})
            - summary: Article summary/abstract
            - published_date: Published date
            - comment: Comment field (if available)
            - authors: List of author names
            - categories: List of all category terms
            - primary_category: Primary category term
    """
    # Construct the API URL with proper encoding
    base_url = "http://export.arxiv.org/api/query"
    params = {"search_query": query, "start": start, "max_results": max_results}
    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    try:
        # Fetch the XML response
        with urllib.request.urlopen(url) as response:  # nosemgrep: dynamic-urllib-use-detected
            xml_data = response.read().decode("utf-8")

        # Parse the XML
        root = ET.fromstring(xml_data)

        # Define namespaces
        namespaces = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

        articles = []

        # Find all entry elements
        entries = root.findall("atom:entry", namespaces)

        for entry in entries:
            # Extract ID (e.g., "http://arxiv.org/abs/2502.13957v2" -> "2502.13957v2")
            id_elem = entry.find("atom:id", namespaces)
            arxiv_id = None
            if id_elem is not None and id_elem.text:
                # Extract the arXiv ID from the URL
                id_text = id_elem.text
                if "/abs/" in id_text:
                    arxiv_id = id_text.split("/abs/")[-1]
                elif "/pdf/" in id_text:
                    arxiv_id = id_text.split("/pdf/")[-1]
                else:
                    arxiv_id = id_text.split("/")[-1]

            # Extract title
            title_elem = entry.find("atom:title", namespaces)
            title = title_elem.text.strip() if title_elem is not None and title_elem.text else None

            # Extract all links (HTML, PDF, etc.)
            links = {}
            for link_elem in entry.findall("atom:link", namespaces):
                rel = link_elem.get("rel", "")
                href = link_elem.get("href")
                link_type = link_elem.get("type", "")
                link_title = link_elem.get("title", "")

                # Determine link type
                if rel == "alternate" and "text/html" in link_type:
                    links["html"] = href
                elif rel == "related" and "application/pdf" in link_type:
                    links["pdf"] = href
                elif link_title:
                    # Use title as key if available (e.g., 'pdf')
                    links[link_title.lower()] = href
                elif rel:
                    # Use rel as key
                    links[rel] = href

            # Extract summary
            summary_elem = entry.find("atom:summary", namespaces)
            summary = (
                summary_elem.text.strip()
                if summary_elem is not None and summary_elem.text
                else None
            )

            # Extract published date
            published_elem = entry.find("atom:published", namespaces)
            published_date = None
            if published_elem is not None and published_elem.text:
                try:
                    # Parse ISO 8601 format and convert to readable format
                    dt = datetime.fromisoformat(published_elem.text.replace("Z", "+00:00"))
                    published_date = dt.strftime("%Y-%m-%d")
                except Exception:
                    published_date = published_elem.text

            # Extract comment (arxiv:comment)
            comment_elem = entry.find("arxiv:comment", namespaces)
            comment = (
                comment_elem.text.strip()
                if comment_elem is not None and comment_elem.text
                else None
            )

            # Extract authors
            authors = []
            for author_elem in entry.findall("atom:author", namespaces):
                name_elem = author_elem.find("atom:name", namespaces)
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text.strip())

            # Extract categories (these are in the Atom namespace, not arxiv namespace)
            categories = []
            for category_elem in entry.findall("atom:category", namespaces):
                term = category_elem.get("term")
                if term:
                    categories.append(term)

            # Extract primary category
            primary_category_elem = entry.find("arxiv:primary_category", namespaces)
            primary_category = None
            if primary_category_elem is not None:
                primary_category = primary_category_elem.get("term")

            # Create article dictionary
            article = {
                "id": arxiv_id,
                "title": title,
                "links": links,
                "authors": authors,
                "summary": summary,
                "published_date": published_date,
                "comment": comment,
                "categories": categories,
                "primary_category": primary_category,
            }

            articles.append(article)

        return articles

    except urllib.error.URLError as e:
        return [{"error": f"Failed to fetch from arXiv API: {e}"}]
    except ET.ParseError as e:
        return [{"error": f"Failed to parse XML response: {e}"}]
    except Exception as e:
        return [{"error": f"Unexpected error: {e}"}]


def get_fmp_company_data(
    symbol: str,
    api_key: str | None = None,
    statements: Iterable[str] | None = None,
) -> dict[str, Any]:
    """
    Fetch company data from FinancialModelingPrep API and return formatted output with derived metrics.

    Always returns:
      - profile (filtered to specific fields)

    Optionally returns:
      - income_statement (if "income" in `statements`, filtered to specific fields)
      - balance_sheet   (if "balance" in `statements`, filtered to specific fields)
      - derived_metrics (calculated from income statement and balance sheet)

    Returns a dictionary with the structure:
    {
        "profile": {...},
        "income_statement": {...},  # if requested
        "balance_sheet": {...},      # if requested
        "derived_metrics": {...}     # if income or balance sheet requested
    }
    """

    def _get_jsonparsed_data(url: str) -> Any:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()

    def _get_fmp_api_key(api_key: str | None = None) -> str:
        """Resolve the FMP API key from argument or `FMP_API_KEY` env var."""
        if api_key:
            return api_key
        else:
            from dotenv import load_dotenv

            load_dotenv()
            env_key = os.getenv("FMP_API_KEY")
        if not env_key:
            raise RuntimeError(
                "Missing FinancialModelingPrep API key. Set FMP_API_KEY or pass api_key explicitly."
            )
        return env_key

    def _extract_profile_fields(profile_data: list) -> dict[str, Any]:
        """Extract only the specified fields from profile data."""
        if not profile_data or len(profile_data) == 0:
            return {}

        profile = profile_data[0]  # API returns a list, take first item

        return {
            "symbol": profile.get("symbol"),
            "companyName": profile.get("companyName"),
            "exchange": profile.get("exchange"),
            "sector": profile.get("sector"),
            "description": profile.get("description"),
            "industry": profile.get("industry"),
            "currency": profile.get("currency"),
            "price": profile.get("price"),
            "marketCap": profile.get("marketCap"),
            "beta": profile.get("beta"),
            "range52w": profile.get("range52w"),
            "volume": profile.get("volume"),
            "averageVolume": profile.get("averageVolume"),
            "ipoDate": profile.get("ipoDate"),
        }

    def _extract_income_statement_fields(income_data: list) -> dict[str, Any]:
        """Extract only the specified fields from income statement data."""
        if not income_data or len(income_data) == 0:
            return {}

        income = income_data[0]  # Get most recent (first item)

        return {
            "date": income.get("date"),
            "fiscalYear": income.get("fiscalYear"),
            "period": income.get("period"),
            "reportedCurrency": income.get("reportedCurrency"),
            "revenue": income.get("revenue"),
            "grossProfit": income.get("grossProfit"),
            "operatingIncome": income.get("operatingIncome"),
            "ebitda": income.get("ebitda"),
            "researchAndDevelopmentExpenses": income.get("researchAndDevelopmentExpenses"),
            "sellingGeneralAndAdministrativeExpenses": income.get(
                "sellingGeneralAndAdministrativeExpenses"
            ),
            "netIncome": income.get("netIncome"),
            "epsDiluted": income.get("epsDiluted"),
        }

    def _extract_balance_sheet_fields(balance_data: list) -> dict[str, Any]:
        """Extract only the specified fields from balance sheet data."""
        if not balance_data or len(balance_data) == 0:
            return {}

        balance = balance_data[0]  # Get most recent (first item)

        return {
            "date": balance.get("date"),
            "fiscalYear": balance.get("fiscalYear"),
            "period": balance.get("period"),
            "reportedCurrency": balance.get("reportedCurrency"),
            "cashAndCashEquivalents": balance.get("cashAndCashEquivalents"),
            "shortTermInvestments": balance.get("shortTermInvestments"),
            "totalCurrentAssets": balance.get("totalCurrentAssets"),
            "totalAssets": balance.get("totalAssets"),
            "totalCurrentLiabilities": balance.get("totalCurrentLiabilities"),
            "shortTermDebt": balance.get("shortTermDebt"),
            "longTermDebt": balance.get("longTermDebt"),
            "totalDebt": balance.get("totalDebt"),
            "totalStockholdersEquity": balance.get("totalStockholdersEquity"),
            "netDebt": balance.get("netDebt"),
        }

    def _calculate_derived_metrics(
        income: dict[str, Any] | None = None,
        balance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Calculate derived financial metrics from income statement and balance sheet."""
        metrics = {}

        # Margin calculations (from income statement)
        if income:
            revenue = income.get("revenue")
            gross_profit = income.get("grossProfit")
            operating_income = income.get("operatingIncome")
            net_income = income.get("netIncome")

            if revenue and revenue != 0:
                if gross_profit is not None:
                    metrics["grossMargin"] = round(gross_profit / revenue, 3)
                if operating_income is not None:
                    metrics["operatingMargin"] = round(operating_income / revenue, 3)
                if net_income is not None:
                    metrics["netMargin"] = round(net_income / revenue, 3)

        # Ratio calculations (from balance sheet)
        if balance:
            total_current_assets = balance.get("totalCurrentAssets")
            total_current_liabilities = balance.get("totalCurrentLiabilities")
            total_debt = balance.get("totalDebt")
            total_stockholders_equity = balance.get("totalStockholdersEquity")

            if (
                total_current_assets
                and total_current_liabilities
                and total_current_liabilities != 0
            ):
                metrics["currentRatio"] = round(total_current_assets / total_current_liabilities, 3)

            if (
                total_debt is not None
                and total_stockholders_equity
                and total_stockholders_equity != 0
            ):
                metrics["debtToEquity"] = round(total_debt / total_stockholders_equity, 3)

        # netDebtToEBITDA (requires both balance sheet and income statement)
        if balance and income:
            net_debt = balance.get("netDebt")
            ebitda = income.get("ebitda")

            if net_debt is not None and ebitda and ebitda != 0:
                metrics["netDebtToEBITDA"] = round(net_debt / ebitda, 3)

        return metrics

    def get_fmp_profile(symbol: str, api_key: str | None = None) -> Any:
        """Return the company profile for the given symbol using FinancialModelingPrep."""
        fmp_base_url = os.getenv("FMP_BASE_URL")
        key = _get_fmp_api_key(api_key)
        url = f"{fmp_base_url}/profile?symbol={symbol}&apikey={key}"
        return _get_jsonparsed_data(url)

    def get_fmp_income_statement(symbol: str, api_key: str | None = None) -> Any:
        """Return income statement data for the given symbol."""
        key = _get_fmp_api_key(api_key)
        fmp_base_url = os.getenv("FMP_BASE_URL")
        url = f"{fmp_base_url}/income-statement?symbol={symbol}&limit=1&apikey={key}"
        return _get_jsonparsed_data(url)

    def get_fmp_balance_sheet(symbol: str, api_key: str | None = None) -> Any:
        """Return balance sheet data for the given symbol."""
        key = _get_fmp_api_key(api_key)
        fmp_base_url = os.getenv("FMP_BASE_URL")
        url = f"{fmp_base_url}/balance-sheet-statement?symbol={symbol}&limit=1&apikey={key}"
        return _get_jsonparsed_data(url)

    key = _get_fmp_api_key(api_key)

    # Fetch raw data
    raw_profile = get_fmp_profile(symbol, key)
    profile = _extract_profile_fields(raw_profile)

    result: dict[str, Any] = {
        "profile": profile,
    }

    # Default: only profile if statements is None or empty
    statements = list(statements or [])

    income = None
    balance = None

    if "income" in statements:
        raw_income = get_fmp_income_statement(symbol, key)
        income = _extract_income_statement_fields(raw_income)
        result["income_statement"] = income

    if "balance" in statements:
        raw_balance = get_fmp_balance_sheet(symbol, key)
        balance = _extract_balance_sheet_fields(raw_balance)
        result["balance_sheet"] = balance

    # Calculate derived metrics if we have income or balance sheet data
    if income or balance:
        derived_metrics = _calculate_derived_metrics(income, balance)
        if derived_metrics:  # Only add if we calculated any metrics
            result["derived_metrics"] = derived_metrics

    return result


# -----------------------------------------------------------------------------
# Tool registry helpers
# -----------------------------------------------------------------------------


def get_tool_schema(name: str) -> dict[str, Any]:
    """Return the OpenAI-style tool schema (without python metadata) for a tool."""
    registry = _load_tools_registry()
    tool = registry.get(name)
    if not tool:
        raise KeyError(f"Tool '{name}' not found in tools registry")

    # Strip out python-specific metadata before giving to the LLM API.
    return {k: v for k, v in tool.items() if k != "python"}


def get_all_tool_schemas() -> list[dict[str, Any]]:
    """Return all tool schemas defined in the YAML registry (without python metadata)."""
    registry = _load_tools_registry()
    return [get_tool_schema(name) for name in registry]


def get_tool_callable(name: str):
    """Resolve the Python callable for a given tool name using the YAML registry."""
    registry = _load_tools_registry()
    tool = registry.get(name)
    if not tool:
        raise KeyError(f"Tool '{name}' not found in tools registry")

    python_meta = tool.get("python", {}) or {}
    module_name = python_meta.get("module", "services.llm_tools")
    func_name = python_meta.get("callable", name)

    module = importlib.import_module(module_name)
    try:
        return getattr(module, func_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Callable '{func_name}' not found in module '{module_name}' for tool '{name}'"
        ) from exc
