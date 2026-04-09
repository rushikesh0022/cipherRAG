import json


def recall_at_k(reference_ids, test_ids, k):
    ref = list(reference_ids)[:k]
    test = list(test_ids)[:k]
    if k == 0:
        return 0.0
    return len(set(ref) & set(test)) / float(k)


def precision_at_k(reference_ids, test_ids, k):
    ref = list(reference_ids)[:k]
    test = list(test_ids)[:k]
    if k == 0:
        return 0.0
    return len(set(ref) & set(test)) / float(k)


def jaccard_topk(reference_ids, test_ids, k):
    ref = set(list(reference_ids)[:k])
    test = set(list(test_ids)[:k])
    union = ref | test
    if not union:
        return 0.0
    return len(ref & test) / float(len(union))


def mean_abs_score_error(ref_scores, test_scores, topk=None):
    if topk is not None:
        ref_items = sorted(ref_scores.items(), key=lambda x: x[1], reverse=True)[:topk]
        keys = [k for k, _ in ref_items]
    else:
        keys = sorted(set(ref_scores.keys()) & set(test_scores.keys()))

    errors = []
    for k in keys:
        if k in ref_scores and k in test_scores:
            errors.append(abs(ref_scores[k] - test_scores[k]))

    return 0.0 if not errors else sum(errors) / len(errors)


def payload_size_kb(obj):
    if obj is None:
        return 0.0
    if isinstance(obj, bytes):
        return len(obj) / 1024.0
    if isinstance(obj, str):
        return len(obj.encode("utf-8")) / 1024.0
    try:
        blob = json.dumps(obj).encode("utf-8")
        return len(blob) / 1024.0
    except Exception:
        return 0.0


def payload_size_mb(obj):
    return payload_size_kb(obj) / 1024.0
