import numpy as np


# --------------------------------------------------
# CMC + mAP computation
# --------------------------------------------------

def evaluate(dist_matrix, query_ids, gallery_ids, query_cams, gallery_cams, topk=10):
    """
    Standard Re-ID evaluation.
    - dist_matrix: shape (num_queries, num_gallery)
    - query_ids: person IDs for queries
    - gallery_ids: person IDs for gallery
    - query_cams: camera IDs for queries
    - gallery_cams: camera IDs for gallery
    """

    num_queries, num_gallery = dist_matrix.shape
    indices = np.argsort(dist_matrix, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])

    all_cmc = []
    all_AP = []

    for i in range(num_queries):
        # remove gallery images from same camera & same ID (Market-1501 rule)
        valid = ~((gallery_ids[indices[i]] == query_ids[i]) &
                  (gallery_cams[indices[i]] == query_cams[i]))

        y_true = matches[i][valid]
        if not np.any(y_true):
            continue

        # positions of true matches
        y_score = -dist_matrix[i][indices[i]][valid]

        # Compute CMC
        cmc = y_true.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:topk])

        # Compute AP
        num_rel = y_true.sum()
        tmp_cmc = y_true.cumsum()
        precision = tmp_cmc / (np.arange(len(tmp_cmc)) + 1)
        AP = (precision * y_true).sum() / num_rel
        all_AP.append(AP)

    # Final metrics
    mean_cmc = np.mean(all_cmc, axis=0)
    mAP = np.mean(all_AP)

    return {
        "Rank-1": mean_cmc[0],
        "Rank-5": mean_cmc[4] if topk >= 5 else None,
        "Rank-10": mean_cmc[9] if topk >= 10 else None,
        "mAP": mAP
    }
