"""
Implementation of Spearman's Rho squared as a pairwise distance metric
Base on
http://www.plhyu.com/administrator/components/com_jresearch/files/publications/Mixtures_of_weighted_distance-based_models_for_ranking_data_with_applications_in_political_studies.pdf

TODO:
- add tests
"""


def spearman_squared_distance(r_1, r_2):
    """
    Computes a weighted Spearman's Rho squared distance. Runs in O(n)

    Args:
        r_1, r_2 (list): list of weighted rankings.
                       Index corresponds to an item and the value is the weight
                       Entries should be positive and sum to 1
                       Example: r_1 = [0.1, 0.2, 0.7]
    Returns: float >= representing the distance between the rankings
    """
    # confirm r_1 and r_2 have same lengths
    if len(r_1) != len(r_2):
        raise ValueError("rankings must contain the same number of elements")
    distance = 0

    for r_1_value, r_2_value in zip(r_1, r_2):
        order_penalty = (r_1_value - r_2_value)**2
        weight = r_1_value * r_2_value * 100 * 100
        distance += weight * order_penalty

    return distance


def pairwise_spearman_distance_matrix(rankings):
    """
    Computes a matrix of pairwise distance

    Args:
        rankings (list): each element is a list of weighted rankings (see ktau_weighted_distance)

    Returns: matrix (list of lists) containing pairwise distances
    """
    D = []
    for r_1 in rankings:
        row = []
        for r_2 in rankings:
            distance = spearman_squared_distance(r_1, r_2)
            row.append(distance)
        D.append(row)
    return D
