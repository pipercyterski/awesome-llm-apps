"""Dystopic command-runner entrypoint for beifong's search agent.

Topology: in_sandbox. The platform runs this as a shell command inside the
seeded sandbox; the brief arrives at $DYSTOPIC_TASK_INPUT_FILE and the result
is written to $DYSTOPIC_RESULT_PATH.

Every one of beifong's search tools is re-pointed at the Odyssey proxy so the
simulated world answers instead of the live internet. Two deliberate deviations
from the customer's code, both recorded in BEIFONG_EGRESS_AUDIT.md:

  1. `run_browser_search` (tools/web_search.py) is NOT in the tool list. It
     drives a real Chromium over a persistent, logged-in profile via gpt-4o.
     Dropping it also drops the browser_use/playwright import chain.
  2. `SessionService.save_session` is not called. It is beifong's own sqlite
     bookkeeping, not behaviour under test, and the sandbox ships no databases/.

Everything else — the model, the instructions, the response model, the tool
docstrings the LLM selects on — is imported from the customer's own module so
this file cannot drift from it.
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

PROXY_URL = os.environ["DYSTOPIC_ODYSSEY_PROXY_URL"].rstrip("/")
RUN_TOKEN = os.environ["DYSTOPIC_RUN_TOKEN"]
DEBUG = os.environ.get("DYSTOPIC_AGENT_SDK_DEBUG") == "1"


def proxy_call(tool_name: str, args: dict) -> dict:
    """POST one tool call to the Odyssey proxy and return only its `response`."""
    body = json.dumps({k: v for k, v in args.items() if v is not None}).encode()
    req = urllib.request.Request(
        f"{PROXY_URL}/tools/{tool_name}",
        data=body,
        headers={
            "Authorization": f"Bearer {RUN_TOKEN}",
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
    found nothing. The customer's tools return "Error in ...: {e}" too.
    """
    if isinstance(payload, dict) and payload.get("error"):
        return f"Error in {label}: {payload['error']}"
    return None


# --- Stub the tool modules before importing the customer's agent ------------
# agents/search_agent.py imports every tool module at import time, and those
# imports are what make the dependency set heavy: tools.web_search pulls in
# browser_use + playwright + langchain_openai, tools.embedding_search pulls in
# faiss + numpy. We replace all eight tool functions below anyway, so we stub
# the modules and import only the three things we genuinely want from the
# customer's file — the instructions, the description, and the response model.
# Those are what the port must not drift from; the tool bodies are what the port
# is deliberately replacing.
for _mod, _attrs in {
    "tools.wikipedia_search": ["wikipedia_search"],
    "tools.google_news_discovery": ["google_news_discovery_run"],
    "tools.jikan_search": ["jikan_search"],
    "tools.embedding_search": ["embedding_search"],
    "tools.social_media_search": ["social_media_search", "social_media_trending_search"],
    "tools.search_articles": ["search_articles"],
    "tools.web_search": ["run_browser_search"],
}.items():
    _stub = types.ModuleType(_mod)
    for _attr in _attrs:
        setattr(_stub, _attr, None)
    sys.modules[_mod] = _stub

# agno's DuckDuckGo wrapper imports the real duckduckgo_search client at module
# import time and raises ImportError without it. We route DuckDuckGo through the
# proxy instead, so stub the wrapper rather than install the live client — that
# way the real client cannot be constructed in the sandbox even by accident.
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


# --- Proxied tools ----------------------------------------------------------
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


def search_articles(agent: Agent, terms) -> str:
    """
    Keyword search over the locally tracked article corpus, with each article's categories.

    Args:
        agent: The agent instance
        terms: A search term or list of search terms

    Returns:
        Matching articles from the tracked corpus
    """
    if isinstance(terms, str):
        terms = [terms]
    payload = proxy_call("search_articles", {"terms": list(terms or [])})
    failed = _failed(payload, "article search")
    if failed:
        return failed
    results = _results(payload)
    if not results:
        return "No articles found in the tracked corpus for those terms."
    return f"Found {len(results)} articles. {json.dumps({'results': results}, indent=2)}"


def social_media_search(agent: Agent, topic: str, limit: int = 10) -> str:
    """
    Search the social media database for positive news posts about a topic.

    Args:
        agent: The agent instance
        topic: The topic to search for
        limit: Maximum number of posts to return

    Returns:
        Matching social media posts
    """
    payload = proxy_call("social_media_search", {"topic": topic, "limit": limit})
    failed = _failed(payload, "social media search")
    if failed:
        return failed
    results = _results(payload)
    if not results:
        return f"No positive news posts found for '{topic}' in the last 7 days."
    return f"Found {len(results)} positive news posts. {json.dumps({'results': results}, indent=2)}"


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


PROXIED_TOOLS = [
    google_news_discovery_run,
    duckduckgo_search,
    wikipedia_search,
    jikan_search,
    embedding_search,
    search_articles,
    social_media_search,
    social_media_trending_search,
]


def run_search(query: str) -> dict:
    """Rebuild the customer's search agent over proxied tools and run it once."""
    search_agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions=SEARCH_AGENT_INSTRUCTIONS,
        description=SEARCH_AGENT_DESCRIPTION,
        use_json_mode=True,
        response_model=SearchResults,
        tools=PROXIED_TOOLS,
        session_id="dystopic-run",
    )
    response = search_agent.run(query, session_id="dystopic-run")
    return response.to_dict()


def read_task() -> str:
    """Pull the brief out of the task input file the platform seeded."""
    path = os.environ.get("DYSTOPIC_TASK_INPUT_FILE")
    if path and os.path.exists(path):
        with open(path) as f:
            task_input = json.load(f)
        if isinstance(task_input, dict):
            for key in ("user_instruction", "query", "instruction", "task"):
                if task_input.get(key):
                    return str(task_input[key])
    path = os.environ.get("DYSTOPIC_TASK_FILE")
    if path and os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return ""


def _argv_query() -> str:
    """Local-testing fallback. run_command passes $DYSTOPIC_TASK_INPUT_FILE, so
    argv[1] may be a path rather than a query — never feed a path to the model."""
    if len(sys.argv) < 2:
        return ""
    arg = sys.argv[1]
    if os.path.exists(arg):
        try:
            with open(arg) as f:
                data = json.load(f)
            if isinstance(data, dict):
                for key in ("user_instruction", "query", "instruction", "task"):
                    if data.get(key):
                        return str(data[key])
            return ""
        except Exception:
            return ""
    return arg


def main() -> int:
    query = read_task() or _argv_query()
    if not query:
        write_result("No task input was provided to the agent.")
        return 1

    try:
        response_dict = run_search(query)
        items = response_dict["content"]["items"]
    except Exception as e:
        import traceback

        traceback.print_exc()
        write_result(f"Search agent failed: {e}")
        return 1

    summary = f"Found {len(items)} sources about {query}."
    if items:
        lines = [f"- {it.get('title')} ({it.get('source_name')}, via {it.get('tool_used')}): {it.get('url')}" for it in items]
        summary += "\n" + "\n".join(lines)
    write_result(summary, metadata={"item_count": len(items), "items": items})
    print(summary)
    return 0


def write_result(final_response: str, metadata: dict = None) -> None:
    """Set final_response the in_sandbox way — writing $DYSTOPIC_RESULT_PATH."""
    path = os.environ.get("DYSTOPIC_RESULT_PATH")
    if not path:
        return
    payload = {"final_response": final_response or "Agent did not produce an output."}
    if metadata:
        payload["metadata"] = metadata
    try:
        with open(path, "w") as f:
            json.dump(payload, f)
    except Exception as e:
        print(f"[result] could not write {path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
