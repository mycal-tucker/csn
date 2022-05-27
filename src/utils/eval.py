import networkx as nx


# Helper functions for evaluation
def trees_match(tree1, tree2):
    def labels_match(node1, node2):
        return node1['name'] == node2['name']
    iso = nx.is_isomorphic(tree1, tree2, node_match=labels_match)
    return iso


def graph_edit_dist(gold_tree, pred_tree):
    def labels_match(node1, node2):
        return node1['name'] == node2['name']
    return nx.graph_edit_distance(gold_tree, pred_tree, node_match=labels_match, timeout=120)
