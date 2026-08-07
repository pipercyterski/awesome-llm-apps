"""Dystopic entrypoint for beifong's search agent.

Topology: `proxy` — the platform imports this file and calls
`run(task_input, *, proxy_url, run_token)`. This is the documented normal shape
for a Python agent with an importable seam, which beifong has; `in_sandbox` is
for CLI/coding harnesses and non-Python runtimes.

The agent's eight search tools are split across two execution modes:

  Simulated (answered by the world engine, called through the Odyssey proxy):
    google_news_discovery_run, duckduckgo_search, wikipedia_search,
    jikan_search, embedding_search
  Simulated + declared projection (`ledger_read`, zero LLM):
    social_media_trending_search
  Executed (the customer's REAL code runs; its SQL hits the /data plane):
    search_articles, social_media_search

Executed tools are NOT wrapped here — the agent calls them directly, and their
data operations route through `data_call` inside the tool modules themselves.
Wrapping one with proxy_call would be refused with `tool_executes_in_sandbox`.

`run_browser_search` (tools/web_search.py) remains excluded: real Chromium over
a persistent logged-in profile, gpt-4o, up to 75 actions at the model's
discretion. See BEIFONG_EGRESS_AUDIT.md.
"""

import json
import os
import sys
import types
import urllib.error
import urllib.request

BEIFONG_ROOT = os.path.dirname(os.path.abspath(__file__))
if BEIFONG_ROOT not in sys.path:
    sys.path.insert(0, BEIFONG_ROOT)

# Populated from the entrypoint's keyword args; the env vars are the fallback
# the platform also injects, which is what data_call reads inside the tools.
_CTX = {
    "proxy_url": (os.environ.get("DYSTOPIC_ODYSSEY_PROXY_URL") or "").rstrip("/"),
    "run_token": os.environ.get("DYSTOPIC_RUN_TOKEN") or "",
}
DEBUG = os.environ.get("DYSTOPIC_AGENT_SDK_DEBUG") == "1"


def proxy_call(tool_name: str, args: dict) -> dict:
    """POST one simulated tool call to the Odyssey proxy, return its `response`.

    Deliberately dependency-free (urllib): if the SDK import ever fails in the
    sandbox, only the Executed tools degrade, not the whole simulated surface.
    """
    body = json.dumps({k: v for k, v in args.items() if v is not None}).encode()
    req = urllib.request.Request(
        f"{_CTX['proxy_url']}/tools/{tool_name}",
        data=body,
        headers={
            "Authorization": f"Bearer {_CTX['run_token']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            envelope = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:400]
        print(f"[proxy] {tool_name} HTTP {e.code}: {detail}", file=sys.stderr)
        return {"error": f"tool call failed: HTTP {e.code}", "results": []}
    except Exception as e:  # network / timeout — never crash the run
        print(f"[proxy] {tool_name} failed: {e}", file=sys.stderr)
        return {"error": f"tool call failed: {e}", "results": []}
    if DEBUG:
        print(f"[proxy] {tool_name} -> source={envelope.get('source')}", file=sys.stderr)
    return envelope.get("response") or {}


def _results(payload: dict) -> list:
    """Defensive extraction — a simulated payload may not honour output_schema."""
    if not isinstance(payload, dict):
        return []
    items = payload.get("results")
    return items if isinstance(items, list) else []


def _failed(payload: dict, label: str):
    """Surface a proxy failure as an error string, the way the real tools do.

    Without this a failed tool call is indistinguishable from an empty world,
    which is how a broken port gets misdiagnosed as a suite that legitimately
    found nothing.
    """
    if isinstance(payload, dict) and payload.get("error"):
        return f"Error in {label}: {payload['error']}"
    return None


# --- Stub only the tool modules we replace with proxied shims ---------------
# search_articles and social_media_search are deliberately NOT stubbed: those
# are the Executed tools, and their real (data_call-instrumented) code is what
# must run. tools.web_search and tools.embedding_search stay stubbed because
# they drag in browser_use/playwright and faiss respectively.
for _mod, _attrs in {
    "tools.wikipedia_search": ["wikipedia_search"],
    "tools.google_news_discovery": ["google_news_discovery_run"],
    "tools.jikan_search": ["jikan_search"],
    "tools.embedding_search": ["embedding_search"],
    "tools.web_search": ["run_browser_search"],
}.items():
    _stub = types.ModuleType(_mod)
    for _attr in _attrs:
        setattr(_stub, _attr, None)
    sys.modules[_mod] = _stub

# agno's DuckDuckGo wrapper imports the real client at import time. We route
# DuckDuckGo through the proxy, so stub it rather than install the live client.
_ddg_stub = types.ModuleType("agno.tools.duckduckgo")


class _DuckDuckGoToolsStub:  # never registered as a tool; see PROXIED_TOOLS
    def __init__(self, *a, **kw):
        pass


_ddg_stub.DuckDuckGoTools = _DuckDuckGoToolsStub
sys.modules["agno.tools.duckduckgo"] = _ddg_stub

from agents.search_agent import (  # noqa: E402
    SEARCH_AGENT_DESCRIPTION,
    SEARCH_AGENT_INSTRUCTIONS,
    SearchResults,
)
from agno.agent import Agent  # noqa: E402
from agno.models.openai import OpenAIChat  # noqa: E402

# The Executed tools — the customer's real code, instrumented onto /data.
from tools.search_articles import search_articles  # noqa: E402
from tools.social_media_search import social_media_search  # noqa: E402


# --- Proxied (simulated) tools ----------------------------------------------
# Signatures and return shapes mirror the customer's originals exactly: every
# beifong tool returns a *string* (prose prefix + embedded JSON), never a dict,
# because that string is what lands in the LLM's context. Docstrings are kept
# because agno derives the tool descriptions the model selects on from them.


def google_news_discovery_run(keyword: str = None, max_results: int = 5, top_news: bool = False) -> str:
    """
    This is a wrapper function for the google news.

    Args:
        keyword: The search query for specific news
        top_news: Whether to get top news instead of keyword search (default: False)
        max_results: The maximum number of results to return (default: 20)

    Returns:
        List of news results
    """
    payload = proxy_call(
        "google_news_discovery_run",
        {"keyword": keyword, "max_results": max_results, "top_news": top_news},
    )
    failed = _failed(payload, "Google News discovery")
    if failed:
        return failed
    results = _results(payload)
    return f"for all results is_scrapping_required: True, results: {json.dumps(results)}"


def duckduckgo_search(agent: Agent, query: str, max_results: int = 5) -> str:
    """
    Search the general web via DuckDuckGo for a query.

    Args:
        agent: The agent instance
        query: The search query
        max_results: Maximum number of results to return

    Returns:
        Web search results
    """
    payload = proxy_call("duckduckgo_search", {"query": query, "max_results": max_results})
    failed = _failed(payload, "web search")
    if failed:
        return failed
    results = _results(payload)
    if not results:
        return "No web results found."
    return f"for all results is_scrapping_required: True, results: {json.dumps(results)}"


def wikipedia_search(agent: Agent, query: str, srlimit: int = 5) -> str:
    """
    Search Wikipedia for articles using Wikipedia API.
    Returns only links and short summaries without fetching full content.

    Args:
        agent: The agent instance
        query: The search query
        srlimit: The maximum number of results to return

    Returns:
        Wikipedia search results
    """
    payload = proxy_call("wikipedia_search", {"query": query, "srlimit": srlimit})
    failed = _failed(payload, "Wikipedia search")
    if failed:
        return failed
    results = _results(payload)
    if not results:
        return "No relevant Wikipedia articles found for this topic."
    return f"for all results is_scrapping_required: True, results: {json.dumps(results, ensure_ascii=False, indent=2)}"


def jikan_search(agent: Agent, query: str) -> str:
    """
    Search for anime information using the Jikan API (MyAnimeList API).
    This provides anime data, reviews, and recommendations to enhance podcast content.

    Args:
        agent: The agent instance
        query: The anime title or topic to search for

    Returns:
        Anime information results
    """
    payload = proxy_call("jikan_search", {"query": query})
    failed = _failed(payload, "anime search")
    if failed:
        return failed
    results = _results(payload)
    if not results:
        return "No relevant anime found for this topic. Continuing with other search methods."
    return f"Found {len(results)} anime titles related to your topic. results {json.dumps(results, indent=2)}."


def embedding_search(agent: Agent, prompt: str) -> str:
    """
    Semantic search over the locally tracked article corpus using embeddings.

    Args:
        agent: The agent instance
        prompt: The semantic search query

    Returns:
        Semantically matched articles
    """
    payload = proxy_call("embedding_search", {"prompt": prompt})
    failed = _failed(payload, "semantic search")
    if failed:
        return failed
    results = _results(payload)
    if not results:
        return "No high-quality semantic matches found (threshold: 85%). Continuing with other search methods."
    return f"Found {len(results)}, results: {json.dumps(results, indent=2)}"


def social_media_trending_search(agent: Agent, limit: int = 10) -> str:
    """
    Get trending positive news posts from the social media database, highest engagement first.

    Args:
        agent: The agent instance
        limit: Maximum number of posts to return

    Returns:
        Trending social media posts
    """
    payload = proxy_call("social_media_trending_search", {"limit": limit})
    failed = _failed(payload, "trending search")
    if failed:
        return failed
    results = _results(payload)
    if not results:
        return "No trending positive news found in the last 7 days."
    return f"Found {len(results)} trending positive news posts. {json.dumps({'results': results}, indent=2)}"


TOOLS = [
    google_news_discovery_run,
    duckduckgo_search,
    wikipedia_search,
    jikan_search,
    embedding_search,
    social_media_trending_search,
    search_articles,        # Executed — real code, /data plane
    social_media_search,    # Executed — real code, /data plane
]


def build_agent() -> Agent:
    """Rebuild the customer's search agent over the ported tool set."""
    return Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions=SEARCH_AGENT_INSTRUCTIONS,
        description=SEARCH_AGENT_DESCRIPTION,
        use_json_mode=True,
        response_model=SearchResults,
        tools=TOOLS,
        session_id="dystopic-run",
        # agno defaults telemetry=True and posts to api.agno.com once per run.
        # The first run's trace caught 15 escaped connects to it. It is not in
        # beifong's code at all — it is a framework default — so no grep of the
        # customer's repo would ever have found it.
        telemetry=False,
    )


def _extract_items(response_dict: dict) -> list:
    """Pull the result items out of an agno response, defensively.

    `content` is normally the parsed SearchResults model, but agno hands back a
    raw JSON *string* when structured-output parsing degrades. Indexing that
    string with ["items"] raises "string indices must be integers" and takes the
    whole run down with it — which is exactly what happened on run 127334, where
    a scenario reported an agent crash rather than the sources it had already
    retrieved. Never let a response-shape wobble destroy a completed search.
    """
    content = (response_dict or {}).get("content")
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except Exception:
            return []
    if hasattr(content, "model_dump"):
        content = content.model_dump()
    if isinstance(content, dict):
        items = content.get("items")
        return items if isinstance(items, list) else []
    if isinstance(content, list):
        return content
    return []


def run(task_input: dict, *, proxy_url: str, run_token: str) -> dict:
    """Platform entrypoint. Returns a dict whose final_response must be non-empty."""
    _CTX["proxy_url"] = (proxy_url or "").rstrip("/")
    _CTX["run_token"] = run_token or ""
    # data_call inside the Executed tool modules reads these off the environment
    # when no dispatch ContextVar is bound, so mirror them for the tools' sake.
    os.environ["DYSTOPIC_ODYSSEY_PROXY_URL"] = proxy_url or ""
    os.environ["DYSTOPIC_RUN_TOKEN"] = run_token or ""

    task_input = task_input or {}
    query = ""
    for key in ("user_instruction", "query", "instruction", "task"):
        if task_input.get(key):
            query = str(task_input[key])
            break
    if not query:
        return {"final_response": "No task input was provided to the agent."}

    try:
        response_dict = build_agent().run(query, session_id="dystopic-run").to_dict()
        items = _extract_items(response_dict)
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"final_response": f"Search agent failed: {e}"}

    summary = f"Found {len(items)} sources about {query}."
    if items:
        lines = [
            f"- {it.get('title')} ({it.get('source_name')}, via {it.get('tool_used')}): {it.get('url')}"
            for it in items
        ]
        summary += "\n" + "\n".join(lines)
    return {"final_response": summary, "metadata": {"item_count": len(items), "items": items}}


if __name__ == "__main__":
    # Local smoke test: python dystopic_entry.py "some query"
    print(run(
        {"user_instruction": sys.argv[1] if len(sys.argv) > 1 else "EU AI Act"},
        proxy_url=os.environ.get("DYSTOPIC_ODYSSEY_PROXY_URL", ""),
        run_token=os.environ.get("DYSTOPIC_RUN_TOKEN", ""),
    )["final_response"])
