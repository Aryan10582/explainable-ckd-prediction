"""Look up every DOI cited in the report on Crossref (or DataCite) and save what the registrar says.

    python scripts/verify_references.py

Writes docs/references/doi_metadata.json. Used to check the synopsis references
and the new references before they were placed in the IEEE paper.
"""
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "references" / "doi_metadata.json"

DOIS = {
    # synopsis references (S1-S18)
    "S1": "10.1016/S0140-6736(25)01853-7", "S2": "10.1111/j.1523-1755.2005.00365.x",
    "S4": "10.1145/2939672.2939778", "S5": "10.1145/2939672.2939785", "S6": "10.1023/A:1010933404324",
    "S7": "10.1613/jair.953", "S8": "10.1016/j.cmpb.2022.107161", "S9": "10.3390/bioengineering12040356",
    "S10": "10.1016/j.iswa.2024.200397", "S11": "10.1038/s41598-024-54375-4", "S12": "10.7717/peerj-cs.2291",
    "S13": "10.1371/journal.pone.0343205", "S14": "10.1177/11779322251356563",
    "S15": "10.1109/ACCESS.2024.3499355", "S16": "10.1109/MITICON.2016.8025242",
    "S17": "10.1155/2023/9266889", "S18": "10.3389/frai.2023.1339988",
    # new references
    "N_uci": "10.24432/C5G020", "N_treeshap": "10.1038/s42256-019-0138-9",
    "N_cortes": "10.1007/BF00994018", "N_cox": "10.1111/j.2517-6161.1958.tb00292.x",
    "N_kdigo2024": "10.1016/j.kint.2023.10.018", "N_webster": "10.1016/S0140-6736(16)32064-5",
    "N_ckdepi": "10.7326/0003-4819-150-9-200905050-00006", "N_anaemia": "10.1371/journal.pone.0084943",
    "N_rudin": "10.1038/s42256-019-0048-x", "N_leakage": "10.1145/2382577.2382579",
    "N_varma": "10.1186/1471-2105-7-91", "N_tripod": "10.1136/bmj-2023-078378",
    "N_imbalance_harm": "10.1093/jamia/ocac093", "N_informative_missing": "10.1186/s41512-020-00077-0",
}


def crossref(doi):
    r = requests.get(f"https://api.crossref.org/works/{doi}", timeout=30)
    if r.status_code != 200:
        return None
    m = r.json()["message"]
    year = (m.get("published-print") or m.get("published-online") or m.get("issued"))["date-parts"][0]
    return {
        "registrar": "Crossref",
        "title": (m.get("title") or [""])[0],
        "authors": [f"{a.get('given', '')} {a.get('family', a.get('name', ''))}".strip() for a in m.get("author", [])],
        "container": (m.get("container-title") or [""])[0],
        "volume": m.get("volume"), "issue": m.get("issue"), "page": m.get("page"),
        "article_number": m.get("article-number"), "date": year, "type": m.get("type"),
    }


def datacite(doi):
    r = requests.get(f"https://api.datacite.org/dois/{doi}", timeout=30)
    if r.status_code != 200:
        return None
    a = r.json()["data"]["attributes"]
    return {"registrar": "DataCite", "title": a["titles"][0]["title"],
            "authors": [c.get("name") for c in a.get("creators", [])], "container": a.get("publisher"),
            "date": [a.get("publicationYear")], "type": a.get("types", {}).get("resourceTypeGeneral")}


def main():
    out = {}
    for key, doi in DOIS.items():
        meta = crossref(doi) or datacite(doi)
        out[key] = {"doi": doi, "found": meta is not None, **(meta or {})}
        status = "OK " if meta else "NOT FOUND"
        print(f"{status} {key:22s} {doi:45s} {(meta or {}).get('title', '')[:70]}")
        time.sleep(0.3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
