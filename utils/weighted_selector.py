"""
Weighted Proportional Top-K Chunk Selector

Selects top-k chunks using a role-aware proportional algorithm.

Algorithm:
  - effective_count(role) = chunk_count(role) * role_weight(role)
  - total_effective       = sum of all effective_counts
  - quota(role)           = floor(top_k * effective_count(role) / total_effective)
  - Within each role, select by descending doc_similarity
  - If a role has fewer chunks than its quota, take only what's available
    (unused slots are NOT redistributed — top_k is a ceiling, not a guarantee)

With uniform weights (all 1.0) this reduces to the original proportional algorithm.
With biased weights (e.g. Precedent Finder) high-weight roles claim more slots.
"""
import math
import logging
from collections import defaultdict
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def weighted_topk_selection(
    chunks: List[Dict],
    top_k: int,
    similarity_key: str = "doc_similarity",
    role_weights: Optional[Dict[str, float]] = None
) -> List[int]:
    """
    Select chunk indices using the weighted proportional algorithm.

    Args:
        chunks:         List of chunk dicts, each with 'role' and similarity_key fields.
        top_k:          Target number of chunks. Actual count may be lower due to
                        floor quotas and role underflow.
        similarity_key: Field used to rank chunks within a role.
        role_weights:   Dict mapping role name -> weight multiplier.
                        Defaults to uniform (1.0) for all roles if None.

    Returns:
        List of selected indices into the original chunks list.
    """
    n = len(chunks)

    if top_k >= n:
        logger.debug(f"top_k ({top_k}) >= n ({n}), returning all chunks")
        return list(range(n))

    if role_weights is None:
        role_weights = {}

    # --- Group chunk indices by role, sorted by similarity descending ---
    role_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, chunk in enumerate(chunks):
        role = chunk.get("role", "Others")
        role_to_indices[role].append(idx)

    for role in role_to_indices:
        role_to_indices[role].sort(
            key=lambda i: chunks[i].get(similarity_key, 0.0),
            reverse=True
        )

    # --- Compute effective counts using weights ---
    total_effective = sum(
        len(indices) * role_weights.get(role, 1.0)
        for role, indices in role_to_indices.items()
    )

    if total_effective == 0:
        return list(range(min(top_k, n)))

    # --- Compute floor quota per role, take up to what's available ---
    selected_indices: List[int] = []

    for role, indices in role_to_indices.items():
        weight = role_weights.get(role, 1.0)
        effective_count = len(indices) * weight
        quota = math.floor(top_k * effective_count / total_effective)
        take = min(quota, len(indices))

        selected_indices.extend(indices[:take])

        logger.debug(
            f"Role '{role}': {len(indices)} chunks, weight={weight}, "
            f"quota={quota}, selected={take}"
        )

    logger.info(
        f"Weighted selection: top_k={top_k}, n={n}, "
        f"selected={len(selected_indices)} "
        f"({top_k - len(selected_indices)} slots unfilled)"
    )

    return selected_indices
