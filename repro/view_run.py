#!/usr/bin/env python3
"""Turn one run's database into readable, customer-by-customer conversations.

Usage (from the repository root):
  .venv/bin/python repro/view_run.py                      # newest run
  .venv/bin/python repro/view_run.py path/to/run.db       # a specific run
  .venv/bin/python repro/view_run.py --customer customer_0001

Prints each customer's goal, what they searched, every message they sent and
received (text, order proposals, payments), and how the run ended for them,
scored the same way as the upstream analytics (exact menu-name matching).
Also writes the same text to <run>.transcript.md next to the database.

Needs only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def newest_db() -> Path | None:
    dbs = sorted(
        (p for p in (REPO_ROOT / "paper_experiments").rglob("*.db")),
        key=lambda p: p.stat().st_mtime,
    )
    return dbs[-1] if dbs else None


def load(db_path: Path):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    customers, businesses = {}, {}
    for (raw,) in con.execute("SELECT data FROM agents"):
        profile = json.loads(raw)
        if "customer" in profile:
            customers[profile["id"]] = profile["customer"]
        elif "business" in profile:
            businesses[profile["id"]] = profile["business"]
    actions = [json.loads(raw) for (raw,) in con.execute("SELECT data FROM actions ORDER BY rowid")]
    con.close()
    return customers, businesses, actions


def money(x: float) -> str:
    return f"${x:,.2f}"


def describe_message(msg: dict, names: dict[str, str]) -> list[str]:
    kind = msg.get("type")
    if kind == "text":
        return [f'"{msg.get("content", "").strip()}"']
    if kind == "order_proposal":
        lines = [f"ORDER PROPOSAL {msg.get('id')}  total {money(msg.get('total_price', 0))}"]
        for item in msg.get("items", []):
            lines.append(
                f"    - {item.get('quantity')} x {item.get('item_name')} @ {money(item.get('unit_price', 0))}"
            )
        if msg.get("special_instructions"):
            lines.append(f"    note: {msg['special_instructions']}")
        return lines
    if kind == "payment":
        line = f"PAYMENT for proposal {msg.get('proposal_message_id')}"
        if msg.get("payment_message"):
            line += f'  "{msg["payment_message"]}"'
        return [line]
    return [json.dumps(msg)]


def build_report(db_path: Path, only_customer: str | None) -> str:
    customers, businesses, actions = load(db_path)
    names = {**{k: v["name"] for k, v in businesses.items()}, **{k: v["name"] for k, v in customers.items()}}

    timeline: dict[str, list[list[str]]] = defaultdict(list)
    proposals: dict[str, tuple[str, dict]] = {}  # proposal id -> (business id, proposal)
    payments: dict[str, list[str]] = defaultdict(list)  # customer -> proposal ids paid
    failed_sends = defaultdict(int)

    for row in actions:
        agent = row.get("agent_id")
        req = row.get("request", {})
        res = row.get("result", {})
        params = req.get("parameters", {})
        kind = params.get("type")

        if kind == "search" and agent in customers:
            if res.get("is_error"):
                timeline[agent].append([f"SEARCH \"{params.get('query')}\"  -> error"])
                continue
            found = [b.get("business", {}).get("name", b.get("id")) for b in res.get("content", {}).get("businesses", [])]
            timeline[agent].append([f"SEARCH \"{params.get('query')}\"", f"    results: {', '.join(found) or 'none'}"])

        elif kind == "send_message":
            src, dst = params.get("from_agent_id"), params.get("to_agent_id")
            msg = params.get("message", {})
            if res.get("is_error"):
                failed_sends[src] += 1
                continue
            if msg.get("type") == "order_proposal" and src in businesses:
                proposals[msg.get("id")] = (src, msg)
            if msg.get("type") == "payment" and src in customers:
                payments[src].append(msg.get("proposal_message_id"))
            customer = src if src in customers else dst
            if customer not in customers:
                continue
            arrow = (
                f"{names.get(src, src)}  ->  {names.get(dst, dst)}"
            )
            body = describe_message(msg, names)
            timeline[customer].append([arrow, *("    " + b for b in body)])

    out: list[str] = [f"# Run: {db_path.name}", ""]
    total_welfare, met_count = 0.0, 0
    ids = [only_customer] if only_customer else sorted(customers)
    for cid in ids:
        if cid not in customers:
            sys.exit(f"Unknown customer {cid}. Known: {', '.join(sorted(customers))}")
        c = customers[cid]
        out.append(f"## {cid} ({c['name']})")
        wants = ", ".join(f"{k} (values it at {money(v)})" for k, v in c["menu_features"].items())
        out.append(f"Wants: {wants}")
        out.append(f"Needs amenities: {', '.join(c['amenity_features']) or 'none'}")
        out.append("")
        for i, event in enumerate(timeline[cid], 1):
            out.append(f"{i:>3}. {event[0]}")
            out.extend(f"     {line}" for line in event[1:])
        out.append("")

        # Outcome, mirroring upstream analytics (exact item names, amenities must all be available)
        paid_total, needs_met, notes = 0.0, False, []
        requested = set(c["menu_features"])
        for pid in payments[cid]:
            if pid not in proposals:
                notes.append(f"paid for unknown proposal {pid}")
                continue
            bid, prop = proposals[pid]
            paid_total += prop.get("total_price", 0)
            offered = {i.get("item_name") for i in prop.get("items", [])}
            amenities = businesses.get(bid, {}).get("amenity_features", {})
            missing_items = sorted(requested - offered)
            missing_amen = [a for a in c["amenity_features"] if not amenities.get(a, False)]
            if not missing_items and not missing_amen:
                needs_met = True
                notes.append(f"paid {money(prop.get('total_price', 0))} to {names.get(bid, bid)}: got everything it needed")
            else:
                why = []
                if missing_items:
                    why.append("missing items: " + ", ".join(missing_items))
                if missing_amen:
                    why.append("restaurant lacks: " + ", ".join(missing_amen))
                notes.append(f"paid {money(prop.get('total_price', 0))} to {names.get(bid, bid)}, but " + "; ".join(why))
        if not payments[cid]:
            notes.append("never paid for anything")
        value = 2 * sum(c["menu_features"].values()) if needs_met else 0.0
        utility = round(value - paid_total, 2)
        total_welfare += utility
        met_count += needs_met
        out.append("Outcome:")
        out.extend(f"  - {n}" for n in notes)
        out.append(f"  - needs met: {'yes' if needs_met else 'no'}   welfare: {utility:+.2f}")
        out.append("")

    if not only_customer:
        out.append(f"## Total: welfare {total_welfare:+.2f}, needs met {met_count}/{len(customers)}")
        rejected = sum(failed_sends.values())
        if rejected:
            out.append(f"({rejected} messages were rejected by the marketplace and are not shown.)")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("db", nargs="?", help="Path to a run's .db file (default: newest run)")
    p.add_argument("--customer", help="Only show this customer, e.g. customer_0001")
    args = p.parse_args(argv)

    db = Path(args.db) if args.db else newest_db()
    if db is None:
        sys.exit("No run database found under paper_experiments/. Run an experiment first, or pass the .db path.")
    if not db.exists():
        sys.exit(f"File not found: {db}")
    report = build_report(db, args.customer)
    print(report)
    if not args.customer:
        target = db.with_suffix(".transcript.md")
        target.write_text(report)
        print(f"Saved to {target.relative_to(REPO_ROOT) if target.is_relative_to(REPO_ROOT) else target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
