#!/usr/bin/env python3
"""
Generates eval/queries.json from the actual seeded catalog, rather than
hand-typing expected product IDs — ground truth is a title/category/price
filter over the real Product table, so it stays correct as long as the
deterministic catalog generator (catalog/products_data.py, seed=42) is
unchanged, and is trivially regenerable if the catalog ever does change.

Run from the repo root with the backend's venv:
    backend/.venv/bin/python eval/generate_queries.py

Requires the catalog to be seeded (`manage.py seed_catalog`) but NOT
ingested — this only reads title/category/price, no embeddings needed.
"""

import json
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django  # noqa: E402

django.setup()

from catalog.models import Product  # noqa: E402

# (query text, category, title substrings ALL must contain, price_max or None)
# Deliberately mixes two kinds of query: most are the bare product-type
# phrase (e.g. "running shoes") that appears verbatim in every matching
# title regardless of which adjective the generator picked; a handful
# add an adjective ("waterproof") or a price phrase ("under 3000") that
# only Phase 4's LLM query understanding (or Phase 2's embeddings, for
# the adjective case) can actually resolve — see eval/results.md for why
# that split matters for a keyword-only baseline.
SPECS = [
    ("waterproof hiking boots", "Footwear", ["Hiking Boots"], None),
    ("waterproof hiking boots under 3000", "Footwear", ["Hiking Boots"], 3000),
    ("running shoes", "Footwear", ["Running Shoes"], None),
    ("formal shoes", "Footwear", ["Formal Shoes"], None),
    ("waterproof sandals under 2000", "Footwear", ["Sandals"], 2000),
    ("rain boots", "Footwear", ["Rain Boots"], None),
    ("wireless earbuds", "Electronics", ["Wireless Earbuds"], None),
    ("bluetooth speaker", "Electronics", ["Bluetooth Speaker"], None),
    ("mechanical keyboard", "Electronics", ["Mechanical Keyboard"], None),
    ("power bank", "Electronics", ["Power Bank"], None),
    ("laptop stand under 7000", "Electronics", ["Laptop Stand"], 7000),
    ("webcam", "Electronics", ["Webcam"], None),
    ("denim jacket", "Apparel", ["Denim Jacket"], None),
    ("cotton t-shirt", "Apparel", ["Cotton T-Shirt"], None),
    ("wool sweater", "Apparel", ["Wool Sweater"], None),
    ("puffer jacket", "Apparel", ["Puffer Jacket"], None),
    ("chino trousers", "Apparel", ["Chino Trousers"], None),
    ("non-stick frying pan", "Home & Kitchen", ["Frying Pan"], None),
    ("electric kettle", "Home & Kitchen", ["Electric Kettle"], None),
    ("knife set", "Home & Kitchen", ["Knife Set"], None),
    ("air fryer", "Home & Kitchen", ["Air Fryer"], None),
    ("camping tent", "Sports & Outdoors", ["Camping Tent"], None),
    ("yoga mat", "Sports & Outdoors", ["Yoga Mat"], None),
    ("trekking backpack", "Sports & Outdoors", ["Trekking Backpack"], None),
    ("cycling helmet", "Sports & Outdoors", ["Cycling Helmet"], None),
    ("facial cleanser", "Beauty & Personal Care", ["Facial Cleanser"], None),
    ("sunscreen lotion", "Beauty & Personal Care", ["Sunscreen Lotion"], None),
    ("electric toothbrush", "Beauty & Personal Care", ["Electric Toothbrush"], None),
    ("mystery novel", "Books", ["Mystery Novel"], None),
    ("science fiction novel", "Books", ["Science Fiction Novel"], None),
    ("board game", "Toys & Games", ["Board Game"], None),
    ("building block set", "Toys & Games", ["Building Block Set"], None),
    ("remote control car", "Toys & Games", ["Remote Control Car"], None),
]


def main():
    queries = []
    for text, category, substrings, price_max in SPECS:
        qs = Product.objects.filter(category=category)
        for substring in substrings:
            qs = qs.filter(title__icontains=substring)
        if price_max is not None:
            qs = qs.filter(price__lte=price_max)
        ids = sorted(qs.values_list("external_id", flat=True))
        queries.append({"query": text, "relevant_external_ids": ids})
        print(f"{text!r}: {len(ids)} relevant")

    out_path = Path(__file__).resolve().parent / "queries.json"
    out_path.write_text(json.dumps({"queries": queries}, indent=2) + "\n")
    print(f"wrote {len(queries)} queries to {out_path}")


if __name__ == "__main__":
    main()
