def hit_rate_at_k(gold, retrieved, k):
    """Fraction of queries with at least one gold item in top-k."""
    gold = set(gold)
    topk = retrieved[:k]
    return int(any(x in gold for x in topk))


def recall_at_k(gold, retrieved, k):
    """
    Recall@K = (# relevant retrieved in top-k) / (# all relevant items), averaged over queries.
    results: list of (gold_list, retrieved_list)
    """
    topk = retrieved[:k]
    return len(set(gold) & set(topk)) / len(gold)


def mrr_at_k(gold, retrieved, k):
    """
    MRR@K = mean reciprocal rank of first relevant item in top-k.
    results: list of (gold_list, retrieved_list)
    """
    gold = set(gold)
    topk = retrieved[:k]
    rr = 0.0
    for rank, item in enumerate(topk, start=1):
        if item in gold:
            rr = 1.0 / rank
            break
    return rr