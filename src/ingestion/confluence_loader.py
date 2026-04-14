"""
Confluence REST API v2 client.
Fetches all pages from a Confluence space with pagination support.
"""

from typing import Optional
import requests
from requests.auth import HTTPBasicAuth
from langchain_core.documents import Document

from config.settings import settings
from src.utils.logger import logger
from src.ingestion.html_parser import parse_confluence_html


class ConfluenceLoader:
    """Loads pages from Atlassian Confluence via REST API v2."""

    def __init__(
        self,
        url: Optional[str] = None,
        email: Optional[str] = None,
        api_token: Optional[str] = None,
        space_key: Optional[str] = None,
    ):
        self.base_url = (url or settings.confluence_url).rstrip("/")
        self.email = email or settings.confluence_email
        self.api_token = api_token or settings.confluence_api_token
        self.space_key = space_key or settings.confluence_space_key
        self.auth = HTTPBasicAuth(self.email, self.api_token)
        self.headers = {"Accept": "application/json"}
        self.wiki_url = f"{self.base_url}/wiki"
        self._space_id = None  # resolved lazily

    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Make authenticated GET request to Confluence API."""
        url = f"{self.wiki_url}{endpoint}"
        logger.debug(f"GET {url} params={params}")

        response = requests.get(
            url,
            auth=self.auth,
            headers=self.headers,
            params=params or {},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def _resolve_space_id(self) -> str:
        """
        Resolve space key to numeric space ID.
        Confluence v2 API requires numeric IDs for space endpoints.
        """
        if self._space_id:
            return self._space_id

        data = self._get("/api/v2/spaces")
        for space in data.get("results", []):
            if space.get("key") == self.space_key:
                self._space_id = str(space["id"])
                logger.info(
                    f"Resolved space key '{self.space_key}' → ID '{self._space_id}'"
                )
                return self._space_id

        # If space_key is already a numeric ID, use it directly
        self._space_id = self.space_key
        logger.info(f"Using space key as ID: '{self._space_id}'")
        return self._space_id

    def get_all_page_ids(self) -> list[dict]:
        """
        Fetch all page IDs and titles in the space using pagination.
        Returns list of {id, title, status}.
        """
        space_id = self._resolve_space_id()
        pages = []
        cursor = None

        while True:
            params = {"limit": 50}
            if cursor:
                params["cursor"] = cursor

            # Confluence v2: list pages in a space (uses numeric space ID)
            endpoint = f"/api/v2/spaces/{space_id}/pages"
            data = self._get(endpoint, params)

            results = data.get("results", [])
            for page in results:
                pages.append(
                    {
                        "id": page["id"],
                        "title": page.get("title", "Untitled"),
                        "status": page.get("status", "current"),
                    }
                )

            # Handle pagination via _links.next
            next_link = data.get("_links", {}).get("next")
            if not next_link:
                break

            # Extract cursor from next link
            if "cursor=" in next_link:
                cursor = next_link.split("cursor=")[-1].split("&")[0]
            else:
                break

        logger.info(f"Found {len(pages)} pages in space '{self.space_key}'")
        return pages

    def get_page_content(self, page_id: str) -> dict:
        """
        Fetch full content of a single page.
        Returns dict with title, body_html, url, metadata.
        """
        endpoint = f"/api/v2/pages/{page_id}"
        params = {"body-format": "storage"}

        data = self._get(endpoint, params)

        # Build the full page URL
        web_url = data.get("_links", {}).get("webui", "")
        full_url = f"{self.wiki_url}{web_url}" if web_url else ""

        return {
            "page_id": str(data["id"]),
            "title": data.get("title", "Untitled"),
            "body_html": data.get("body", {}).get("storage", {}).get("value", ""),
            "url": full_url,
            "space_key": self.space_key,
            "author": data.get("authorId", "unknown"),
            "last_modified": data.get("version", {}).get("createdAt", ""),
        }

    def load(self) -> list[Document]:
        """
        Full pipeline: fetch all pages → parse HTML → return LangChain Documents.
        """
        pages_meta = self.get_all_page_ids()
        documents = []

        for page_meta in pages_meta:
            try:
                page = self.get_page_content(page_meta["id"])
                body_html = page["body_html"]

                if not body_html or not body_html.strip():
                    logger.warning(f"Skipping empty page: {page['title']}")
                    continue

                # Parse Confluence XHTML → clean text
                clean_text = parse_confluence_html(body_html)

                if not clean_text or len(clean_text.strip()) < 10:
                    logger.warning(
                        f"Skipping page with insufficient content: {page['title']}"
                    )
                    continue

                doc = Document(
                    page_content=clean_text,
                    metadata={
                        "page_id": page["page_id"],
                        "title": page["title"],
                        "url": page["url"],
                        "space_key": page["space_key"],
                        "author": page["author"],
                        "last_modified": page["last_modified"],
                        "source": "confluence",
                    },
                )
                documents.append(doc)
                logger.info(
                    f"Loaded page: '{page['title']}' ({len(clean_text)} chars)"
                )

            except requests.exceptions.HTTPError as e:
                logger.error(
                    f"HTTP error loading page {page_meta['id']}: {e}"
                )
            except Exception as e:
                logger.error(
                    f"Error loading page {page_meta['id']}: {e}"
                )

        logger.info(
            f"Successfully loaded {len(documents)} documents from Confluence"
        )
        return documents
