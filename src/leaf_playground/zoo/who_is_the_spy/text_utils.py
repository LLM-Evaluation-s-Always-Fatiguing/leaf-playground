import sys
from typing import List


def get_most_similar_text(
    target: str,
    candidates: List[str]
) -> str:
    def levenshtein_distance(seq1: str, seq2: str):
        seq1 = seq1.strip()
        seq2 = seq2.strip()
        if seq1 == seq2:
            return 0
        num_rows = len(seq1) + 1
        num_cols = len(seq2) + 1
        dp_matrix = [[0 for _ in range(num_cols)] for _ in range(num_rows)]
        for i in range(num_cols):
            dp_matrix[0][i] = i
        for j in range(num_rows):
            dp_matrix[j][0] = j

        for i in range(1, num_rows):
            for j in range(1, num_cols):
                if seq1[i - 1] == seq2[j - 1]:
                    dp_matrix[i][j] = dp_matrix[i - 1][j - 1]
                else:
                    dp_matrix[i][j] = min([dp_matrix[i - 1][j - 1], dp_matrix[i - 1][j], dp_matrix[i][j - 1]]) + 1
        return dp_matrix[num_rows - 1][num_cols - 1]

    def get_closest_candidate(pred: str) -> str:
        min_id = sys.maxsize
        min_edit_distance = sys.maxsize
        for i, candidate in enumerate(candidates):
            edit_distance = levenshtein_distance(candidate, pred)
            if edit_distance < min_edit_distance:
                min_id = i
                min_edit_distance = edit_distance
        return candidates[min_id]

    return get_closest_candidate(target)


__all__ = ["get_most_similar_text"]
