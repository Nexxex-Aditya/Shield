"""
Shield SDK Client — thin wrapper around the Shield REST API.
"""
import requests
from typing import Optional, List, Dict, Any


class ShieldError(Exception):
    """Raised when the Shield API returns an error."""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Shield API Error {status_code}: {message}")


class Client:
    """
    Shield API Client.

    Usage:
        client = shield.Client(api_key="sk-your-key")
        result = client.retrieve("What is our refund policy?")
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.shield.nexxex.com",
        timeout: int = 30,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "shield-sdk/0.1.0",
        })

    def _request(self, method: str, path: str, **kwargs) -> Any:
        url = f"{self.base_url}/api/v1{path}"
        resp = self._session.request(method, url, timeout=self.timeout, **kwargs)
        if not resp.ok:
            raise ShieldError(resp.status_code, resp.text)
        return resp.json()

    # ── Health ──
    def health(self) -> Dict:
        """Check API health."""
        return self._request("GET", "/health")

    # ── Agents ──
    def create_agent(self, description: str, model: str = "gpt-4") -> Dict:
        """Create an agent from an English description."""
        return self._request("POST", "/templates/generate", json={
            "description": description,
            "model": model,
        })

    def list_agents(self) -> List[Dict]:
        """List all agent templates."""
        return self._request("GET", "/templates")

    # ── RAG Retrieval ──
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        strategy: Optional[str] = None,
    ) -> List[Dict]:
        """Retrieve documents using adaptive RAG."""
        payload: Dict[str, Any] = {"query": query, "top_k": top_k}
        if strategy:
            payload["strategy"] = strategy
        return self._request("POST", "/retrieve", json=payload)

    # ── Documents ──
    def ingest(self, text: str, source: str = "upload") -> Dict:
        """Ingest and chunk a document for retrieval."""
        return self._request("POST", "/documents/ingest", json={
            "text": text,
            "source_name": source,
        })

    # ── Embeddings ──
    def embed(self, text: str) -> Dict:
        """Generate embeddings for text."""
        return self._request("POST", "/embeddings/embed", json={"text": text})

    # ── Vector Search ──
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Direct vector similarity search."""
        return self._request("POST", "/vectors/search", json={
            "query": query,
            "top_k": top_k,
        })

    # ── Deploy ──
    def deploy(self, agent_id: str, version: str = "1.0.0") -> Dict:
        """Deploy an agent with safety pipeline."""
        return self._request("POST", "/deploy", json={
            "agent_id": agent_id,
            "version": version,
        })

    def list_deployments(self) -> List[Dict]:
        """List all deployments."""
        return self._request("GET", "/deploy")

    # ── Cache ──
    def cache_stats(self) -> Dict:
        """Get semantic cache statistics."""
        return self._request("GET", "/cache/stats")

    # ── Events ──
    def events(self, channel: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Get recent events from the Neural Bus."""
        params = {"limit": limit}
        if channel:
            params["channel"] = channel
        return self._request("GET", "/bus/events", params=params)

    # ── Stats ──
    def stats(self) -> Dict:
        """Get unified system statistics."""
        return self._request("GET", "/next-gen/stats")
