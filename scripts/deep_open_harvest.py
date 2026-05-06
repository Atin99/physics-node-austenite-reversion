"""Open scholarly lead harvester for medium-Mn retained-austenite data.

This script does not bypass paywalls. It queries OpenAlex metadata, fetches
open landing pages when available, and extracts sentences containing
austenite/retained-austenite terms plus numeric percentages.

Outputs:
  - deep_openalex_leads.csv
  - deep_candidate_ra_sentences.csv
"""

from __future__ import annotations

import csv
import html
import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
LEADS_CSV = ROOT / "deep_openalex_leads.csv"
SENTENCES_CSV = ROOT / "deep_candidate_ra_sentences.csv"

QUERIES = [
    '"medium Mn" "retained austenite"',
    '"medium-Mn" "retained austenite"',
    '"medium manganese" "retained austenite"',
    '"medium Mn steel" "austenite fraction"',
    '"medium-Mn steel" "austenite fraction"',
    '"intercritical annealing" "retained austenite" "Mn"',
    '"reverted austenite" "medium Mn"',
    '"reverse transformation annealing" "medium manganese"',
    '"ART annealing" "medium Mn"',
    '"austenite reversion" "medium Mn"',
    '"Fe-Mn-C" "retained austenite" "intercritical"',
    '"Fe-Mn-Al-C" "retained austenite"',
    '"medium manganese steel" "XRD" "austenite"',
]

UA = "Mozilla/5.0 (compatible; austenite-data-harvester/0.1; +local research)"


@dataclass
class Lead:
    query: str
    title: str
    year: str
    doi: str
    open_url: str
    landing_url: str
    host: str
    is_oa: str
    cited_by_count: str


def fetch_json(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def fetch_text(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
        enc = response.headers.get_content_charset() or "utf-8"
    text = raw.decode(enc, errors="replace")
    text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def host_of(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        return ""


def openalex_search(query: str, per_page: int = 50) -> Iterable[Lead]:
    params = urllib.parse.urlencode(
        {
            "search": query,
            "per-page": per_page,
            "filter": "from_publication_date:2000-01-01",
            "mailto": "research@example.com",
        }
    )
    url = f"https://api.openalex.org/works?{params}"
    data = fetch_json(url)
    for item in data.get("results", []):
        title = item.get("title") or ""
        doi = (item.get("doi") or "").replace("https://doi.org/", "")
        year = str(item.get("publication_year") or "")
        primary = item.get("primary_location") or {}
        landing_url = primary.get("landing_page_url") or ""
        oa = item.get("open_access") or {}
        open_url = oa.get("oa_url") or landing_url
        yield Lead(
            query=query,
            title=title,
            year=year,
            doi=doi,
            open_url=open_url or "",
            landing_url=landing_url or "",
            host=host_of(open_url or landing_url),
            is_oa=str(oa.get("is_oa", "")),
            cited_by_count=str(item.get("cited_by_count") or 0),
        )


def unique_leads(leads: Iterable[Lead]) -> list[Lead]:
    seen = set()
    out = []
    for lead in leads:
        key = lead.doi.lower() or lead.open_url.lower() or lead.title.lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(lead)
    return out


RA_SENTENCE_RE = re.compile(
    r"([^.!?;]{0,180}(?:retained austenite|austenite fraction|austenite volume fraction|RA fraction|RA content|reverted austenite|"
    r"残余奥氏体|残留奥氏体|奥氏体体积分数)[^.!?;]{0,220}(?:\d+(?:\.\d+)?\s*(?:%|pct|vol\.?\s*%|vol%|percent|％))[^.!?;]{0,180})",
    re.I,
)


def candidate_sentences(text: str) -> list[str]:
    found = []
    for match in RA_SENTENCE_RE.finditer(text):
        sent = match.group(1)
        sent = re.sub(r"\s+", " ", sent).strip()
        if sent and sent not in found:
            found.append(sent)
    return found[:20]


def write_leads(leads: list[Lead]) -> None:
    with LEADS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(Lead.__annotations__.keys()))
        w.writeheader()
        for lead in leads:
            w.writerow(lead.__dict__)


def write_sentences(rows: list[dict]) -> None:
    fieldnames = ["title", "year", "doi", "url", "host", "sentence"]
    with SENTENCES_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    all_leads = []
    for q in QUERIES:
        try:
            all_leads.extend(openalex_search(q))
            time.sleep(0.4)
        except Exception as exc:
            print(f"WARN query failed: {q}: {exc}")
    leads = unique_leads(all_leads)
    write_leads(leads)
    print(f"Wrote {len(leads)} leads to {LEADS_CSV.name}")

    sentence_rows = []
    for i, lead in enumerate(leads[:250], 1):
        url = lead.open_url or lead.landing_url
        if not url or url.lower().endswith(".pdf"):
            continue
        if any(blocked in lead.host for blocked in ["sciencedirect.com", "tandfonline.com"]):
            continue
        try:
            text = fetch_text(url, timeout=20)
        except Exception:
            continue
        for sent in candidate_sentences(text):
            sentence_rows.append(
                {
                    "title": lead.title,
                    "year": lead.year,
                    "doi": lead.doi,
                    "url": url,
                    "host": lead.host,
                    "sentence": sent,
                }
            )
        if i % 25 == 0:
            print(f"Scanned {i}/{len(leads)} leads; sentences={len(sentence_rows)}")
        time.sleep(0.2)
    write_sentences(sentence_rows)
    print(f"Wrote {len(sentence_rows)} candidate sentences to {SENTENCES_CSV.name}")


if __name__ == "__main__":
    main()
