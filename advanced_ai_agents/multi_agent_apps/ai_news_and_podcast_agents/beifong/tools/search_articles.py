"""Article keyword search.

Ported for Dystopic: this tool is declared `executed`, so its real logic runs in
the sandbox while its data operations execute against the run's ledger-backed
simulated world instead of a local SQLite file. Only the *transport* changed —
the SQL, the LIKE-per-term structure, the DISTINCT, the ordering, the 3-row cap
and the response formatting are all the customer's, unchanged. That unchanged
logic is the whole point of running this tool as Executed rather than simulating
its output.

The values are interpolated into the operation text rather than bound as `?`
params because the /data plane canonicalizes each operation and strips literals
into named parameters server-side; the plan is cached per query *shape*, so
repeat calls with different terms reuse the same translated plan.
"""

import json
from typing import List, Union

from agno.agent import Agent
from dystopic.odyssey import data_call

TOOL = "search_articles"


def _sql_quote(value: str) -> str:
    """Escape a value for inlining into the operation text."""
    return str(value).replace("'", "''")


def search_articles(agent: Agent, terms: Union[str, List[str]]) -> str:
    """
    Search for articles related to a podcast topic using direct SQL queries.
    The agent can pass either a string topic or a list of search terms.

    Args:
        agent: The agent instance
        terms: Either a single topic string or a list of search terms

    Returns:
        A formatted string response with the search results
    """
    print(f"Search Internal Articles terms: {terms}")
    search_terms = terms if isinstance(terms, list) else [terms]
    limit = 3
    try:
        results = execute_simple_search(search_terms, limit)
        if not results:
            return "No relevant articles found in our database. Would you like to try a different topic or provide specific URLs?"
        for article in results:
            article["categories"] = get_article_categories(article.get("id"))
            article["source_name"] = article.get("source_id", "Unknown Source")
        return f"is_scrapping_required: False, Found {len(results)}, {json.dumps(results, indent=2)} potential sources that might be relevant to your topic careful my search is text bassed do quality check and ignore invalid resutls."
    except Exception as e:
        print(f"Error searching articles: {e}")
        return "I encountered a database error while searching. Would you like to try a different approach?"


def execute_simple_search(terms, limit):
    # COALESCE moved out of the SELECT. The plan grammar rejects it:
    #   "SELECT DISTINCT includes COALESCE(ca.summary, ca.content) AS content,
    #    which is an expression — DISTINCT over an expression cannot be
    #    represented."
    # Both columns are selected bare and the fallback is applied in Python
    # below, which produces an identical `content` value per row.
    base_query = """
        SELECT DISTINCT ca.id, ca.title, ca.url, ca.published_date,
               ca.summary, ca.content,
               ca.source_id, ca.feed_id
        FROM crawled_articles ca
        WHERE ca.processed = 1
          AND (
    """
    # ILIKE, not LIKE. The /data plane's plan grammar rejects LIKE outright:
    #   "The query uses case-sensitive LIKE which cannot be represented;
    #    only case-insensitive ILIKE is supported."
    # That reads LIKE with Postgres semantics. In SQLite — which is what beifong
    # actually runs — LIKE is already case-insensitive for ASCII, so ILIKE is a
    # semantics-preserving translation of this tool's real behaviour, not a
    # change to the logic under test.
    clauses = []
    for term in terms:
        like_term = f"'%{_sql_quote(term)}%'"
        clauses.append(f"(ca.title ILIKE {like_term} OR ca.content ILIKE {like_term} OR ca.summary ILIKE {like_term})")

    query = base_query + " OR ".join(clauses) + f") ORDER BY ca.published_date DESC LIMIT {int(limit)}"
    result = data_call(TOOL, kind="sql", intent="read", operation=query, client="sqlite")
    rows = list(result.get("rows") or [])
    for row in rows:
        # The COALESCE the SELECT can no longer carry.
        row["content"] = row.get("summary") or row.get("content")
        row.pop("summary", None)
    return rows


def get_article_categories(article_id):
    try:
        result = data_call(
            TOOL,
            kind="sql",
            intent="read",
            operation=f"SELECT category_name FROM article_categories WHERE article_id = '{_sql_quote(article_id)}'",
            client="sqlite",
        )
        return [row["category_name"] for row in (result.get("rows") or []) if row.get("category_name")]
    except Exception as e:
        print(f"Error fetching article categories: {e}")
        return []
