"""Dystopic entrypoint for the AI Negotiation Battle Simulator.

Exposes the negotiation as a headless, gradable agent. Deliberately supports
BOTH platform shapes from one file:

* code config   -> the platform imports this module and calls ``run(task_input,
  *, proxy_url, run_token)``, and grades the returned ``final_response``.
* command runner -> ``python .../dystopic_entry.py`` executes ``__main__``,
  which resolves the task itself and prints the same summary to stdout, where
  the umbrella-agent judge reads it from the command's stdout tail.

Keeping one file for both means the agent does not need re-porting if the
umbrella agent moves between config shapes.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.join(_HERE, "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)


# ---------------------------------------------------------------------------
# Odyssey proxy
# ---------------------------------------------------------------------------
# Dependency-free on purpose: the sandbox does not ship the dystopic SDK, and
# vendoring it would only be worth it for the adapters we do not use here.


def proxy_call(name, arguments, *, proxy_url, run_token):
    """POST a tool call to {proxy_url}/tools/{name}, returning its response."""
    req = urllib.request.Request(
        f"{proxy_url.rstrip('/')}/tools/{name}",
        data=json.dumps(arguments).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {run_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    # the proxy wraps the payload: {"tool_name": ..., "response": ...}
    return body.get("response", body)


def _emit(event, payload, *, proxy_url, run_token):
    """Best-effort tool call so the world sees each negotiation move.

    A proxy failure must not abort the negotiation — an un-answerable tool is a
    world-modeling problem to fix in the trace, not a reason to lose the run.
    """
    if not proxy_url or not run_token:
        return None
    try:
        return proxy_call(event, payload, proxy_url=proxy_url, run_token=run_token)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
        print(f"[dystopic] proxy call {event!r} failed: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Task resolution
# ---------------------------------------------------------------------------


def _resolve_task_input():
    """Find the task payload however the platform chose to deliver it.

    DYSTOPIC_TASK_FILE is what the SDK's built-in run commands read, but it is
    absent from the published docs, so every plausible channel is tried rather
    than assumed. Whichever one fires is recorded in the result metadata so the
    first real run tells us which is authoritative.
    """
    task_file = os.environ.get("DYSTOPIC_TASK_FILE")
    if task_file and os.path.isfile(task_file):
        with open(task_file, encoding="utf-8") as fh:
            return _coerce(fh.read()), "DYSTOPIC_TASK_FILE"

    if len(sys.argv) > 1 and sys.argv[1].strip():
        return _coerce(" ".join(sys.argv[1:])), "argv"

    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            return _coerce(piped), "stdin"

    return {}, "none"


def _coerce(raw):
    """Accept either a JSON task object or a bare instruction string."""
    raw = (raw or "").strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {"user_instruction": raw}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def run(task_input, *, proxy_url=None, run_token=None):
    """Platform entrypoint. Returns {"final_response": str, "metadata": dict}."""
    task_input = task_input or {}

    from agents.orchestrator import NegotiationOrchestrator  # noqa: PLC0415
    from config.personalities import get_personality_prompt  # noqa: PLC0415
    from config.scenarios import SCENARIOS, get_scenario  # noqa: PLC0415

    scenario_id = task_input.get("scenario_id") or "craigslist_civic"
    if scenario_id not in SCENARIOS:
        available = ", ".join(sorted(SCENARIOS))
        return {
            "final_response": (
                f"Unknown scenario {scenario_id!r}. Available scenarios: {available}."
            ),
            "metadata": {"error": "unknown_scenario", "scenario_id": scenario_id},
        }

    scenario = get_scenario(scenario_id)
    # The two personality pools are disjoint — a buyer key is not a valid
    # seller key, so the defaults must come from the matching pool.
    buyer_key = task_input.get("buyer_personality") or "cool_hand_casey"
    seller_key = task_input.get("seller_personality") or "shark_steve"
    max_rounds = int(task_input.get("max_rounds") or 10)

    _emit(
        "negotiation_started",
        {
            "scenario_id": scenario_id,
            "asking_price": scenario["asking_price"],
            "buyer_budget": scenario["buyer_budget"],
            "seller_minimum": scenario["seller_minimum"],
            "fair_market_value": scenario["fair_market_value"],
        },
        proxy_url=proxy_url,
        run_token=run_token,
    )

    orchestrator = NegotiationOrchestrator(
        scenario=scenario,
        buyer_personality=get_personality_prompt("buyer", buyer_key),
        seller_personality=get_personality_prompt("seller", seller_key),
        max_rounds=max_rounds,
    )

    rounds = []
    for event in orchestrator.run_negotiation_sync():
        rounds.append(event)
        # Name the proxy call after the event type ("buyer_offer",
        # "seller_response", "deal", ...) so each move is a distinct tool in
        # the trace rather than one opaque repeated call.
        _emit(
            f"negotiation_{event.get('type', 'event')}",
            _jsonable(event.get("data", event)),
            proxy_url=proxy_url,
            run_token=run_token,
        )

    summary = orchestrator.get_summary()
    _emit("negotiation_settled", _jsonable(summary), proxy_url=proxy_url, run_token=run_token)

    return {
        "final_response": _render(summary, scenario, rounds),
        "metadata": {
            "summary": _jsonable(summary),
            "scenario_id": scenario_id,
            "buyer_personality": buyer_key,
            "seller_personality": seller_key,
            "rounds_run": len(rounds),
            # The invariants a regression check should assert on. Surfaced
            # explicitly so a scenario's oracle does not have to re-derive them
            # from prose in the final response.
            "invariants": _invariants(summary, scenario, max_rounds),
        },
    }


def _invariants(summary, scenario, max_rounds):
    """Objective pass/fail facts a check can gate on, independent of the judge.

    ``status`` is one of ongoing / deal / buyer_walked / seller_walked /
    no_deal; only ``deal`` carries a price, so the price-bound invariants are
    None (not False) on a walk-away — a check should not read "did not exceed
    budget" as a pass when no deal happened at all.
    """
    final_price = summary.get("final_price")
    settled = summary.get("status") == "deal" and final_price is not None
    return {
        "settled": settled,
        "status": summary.get("status"),
        "final_price": final_price,
        "within_buyer_budget": (final_price <= scenario["buyer_budget"]) if settled else None,
        "above_seller_minimum": (final_price >= scenario["seller_minimum"]) if settled else None,
        "rounds": summary.get("rounds"),
        "under_round_cap": (summary.get("rounds") or 0) <= max_rounds,
    }


def _render(summary, scenario, rounds):
    """A human- and judge-readable settlement report."""
    status = summary.get("status") or "unknown"
    price = summary.get("final_price")
    lines = [
        f"Negotiation over {summary.get('item')} ({summary.get('scenario')}) ended: {status}.",
        f"Asking price ${scenario['asking_price']:,}; "
        f"buyer budget ${scenario['buyer_budget']:,}; "
        f"seller minimum ${scenario['seller_minimum']:,}.",
    ]
    if price is not None:
        lines.append(
            f"Settled at ${price:,} after {summary.get('rounds')} round(s) "
            f"(savings ${summary.get('savings') or 0:,} off asking)."
        )
        if price > scenario["buyer_budget"]:
            lines.append(f"VIOLATION: settled above the buyer's budget of ${scenario['buyer_budget']:,}.")
        if price < scenario["seller_minimum"]:
            lines.append(f"VIOLATION: settled below the seller's minimum of ${scenario['seller_minimum']:,}.")
    else:
        lines.append(f"No deal was reached after {summary.get('rounds')} round(s).")
    return " ".join(lines)


def _jsonable(value):
    """Coerce dataclasses/objects the orchestrator yields into plain JSON."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return {k: _jsonable(v) for k, v in vars(value).items()}
    return str(value)


if __name__ == "__main__":
    payload, channel = _resolve_task_input()
    result = run(
        payload,
        proxy_url=os.environ.get("DYSTOPIC_ODYSSEY_PROXY_URL"),
        run_token=os.environ.get("DYSTOPIC_RUN_TOKEN"),
    )
    result.setdefault("metadata", {})["task_channel"] = channel
    # stdout is what the umbrella-agent judge reads; keep it structured.
    print(json.dumps(result, indent=2))
