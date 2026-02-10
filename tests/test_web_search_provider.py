import pytest

from nanobot.agent.tools.web import WebSearchTool


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


class _FakeClient:
    def __init__(self, get_response: _FakeResponse | None = None, post_response: _FakeResponse | None = None):
        self.get_response = get_response
        self.post_response = post_response
        self.last_get_url = None
        self.last_post_url = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url, **kwargs):
        self.last_get_url = url
        if self.get_response is None:
            raise RuntimeError("missing get_response")
        return self.get_response

    async def post(self, url, **kwargs):
        self.last_post_url = url
        if self.post_response is None:
            raise RuntimeError("missing post_response")
        return self.post_response


@pytest.mark.asyncio
async def test_web_search_brave_provider(monkeypatch):
    fake_client = _FakeClient(
        get_response=_FakeResponse(
            {
                "web": {
                    "results": [
                        {
                            "title": "Example",
                            "url": "https://example.com",
                            "description": "Sample result",
                        }
                    ]
                }
            }
        )
    )

    monkeypatch.setattr("nanobot.agent.tools.web.httpx.AsyncClient", lambda *args, **kwargs: fake_client)

    tool = WebSearchTool(provider="brave", api_key="brave-test")
    output = await tool.execute(query="nanobot", count=1)

    assert "Results for: nanobot" in output
    assert "https://example.com" in output


@pytest.mark.asyncio
async def test_web_search_grok_provider(monkeypatch):
    fake_client = _FakeClient(
        post_response=_FakeResponse(
            {
                "output_text": "Searched summary",
                "citations": ["https://example.com/a", "https://example.com/b"],
            }
        )
    )

    monkeypatch.setattr("nanobot.agent.tools.web.httpx.AsyncClient", lambda *args, **kwargs: fake_client)

    tool = WebSearchTool(provider="grok", grok_api_key="xai-test")
    output = await tool.execute(query="nanobot")

    assert "Searched summary" in output
    assert "Sources:" in output
    assert "https://example.com/a" in output


@pytest.mark.asyncio
async def test_web_search_grok_custom_base_url(monkeypatch):
    fake_client = _FakeClient(
        post_response=_FakeResponse(
            {
                "output_text": "Searched summary",
                "citations": ["https://example.com/a"],
            }
        )
    )

    monkeypatch.setattr("nanobot.agent.tools.web.httpx.AsyncClient", lambda *args, **kwargs: fake_client)

    custom_url = "https://xai-proxy.example.com/v1/responses"
    tool = WebSearchTool(provider="grok", grok_api_key="xai-test", grok_base_url=custom_url)
    output = await tool.execute(query="nanobot")

    assert "Searched summary" in output
    assert fake_client.last_post_url == custom_url


@pytest.mark.asyncio
async def test_web_search_grok_missing_key(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setenv("BRAVE_API_KEY", "")

    tool = WebSearchTool(provider="grok", grok_api_key="")
    output = await tool.execute(query="nanobot")

    assert "XAI_API_KEY" in output


@pytest.mark.asyncio
async def test_web_search_brave_missing_key(monkeypatch):
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.setenv("XAI_API_KEY", "")

    tool = WebSearchTool(provider="brave", api_key="")
    output = await tool.execute(query="nanobot")

    assert "BRAVE_API_KEY" in output
