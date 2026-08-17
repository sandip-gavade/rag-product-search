"""
Reciprocal Rank Fusion (RRF) for merging the two retrieval signals.

Why RRF over a weighted sum of raw scores: pgvector's cosine distance and
Postgres's ts_rank live on completely different, non-comparable scales (one
is bounded [0, 2], the other is an unbounded, corpus-dependent weight sum),
and neither is normalized against the other's distribution. A weighted sum
would need that normalization tuned by hand and re-tuned as the catalog
changes. RRF sidesteps the problem by discarding raw scores and fusing on
*rank position* instead — a product ranked #1 by vector search contributes
the same regardless of how "confident" that similarity score was. This is
the same technique Elasticsearch/OpenSearch ship as their default
hybrid-search fusion.

k=60 is the constant from the original RRF paper (Cormack, Clarke &
Buettcher, 2009): it dampens the difference between low ranks (#1 vs #2
matters a lot more than #49 vs #50) and isn't sensitive to tune for a
catalog this size.
"""

from collections import defaultdict
from typing import Hashable, Sequence


def reciprocal_rank_fusion(
    *ranked_lists: Sequence[Hashable], k: int = 60
) -> list[tuple[Hashable, float]]:
    """Fuse any number of ranked ID lists into one, scored by RRF.

    Each input is a sequence of item IDs in rank order (best first). An ID
    missing from a given list simply doesn't earn points from that source —
    it isn't penalized beyond not getting that list's contribution.

    Returns (id, fused_score) pairs sorted by fused_score descending.
    """
    scores: dict[Hashable, float] = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, item_id in enumerate(ranked_list, start=1):
            scores[item_id] += 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
