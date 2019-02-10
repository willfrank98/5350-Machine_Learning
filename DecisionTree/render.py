from graphviz import Digraph

def render(root, name):
    graph = Digraph()
    nodes = get_nodes(root)
    edges = get_edges(root)

    for node in nodes:
        graph.node(node[0], node[1])

    for edge in edges:
        graph.edge(edge[0], edge[1], edge[2])

    graph.render(name, view=True, format="pdf")


def get_nodes(root):
    nodes = []
    if root.splitsOn == "":
        nodes.append((str(id(root)), "Label: " + root.prediction))
    else:
        nodes.append((str(id(root)), root.splitsOn))
    
    for child in root.children:
        nodes.extend(get_nodes(child))
    return nodes

def get_edges(root):
    edges = []
    for child in root.children:
        edges.append((str(id(root)), str(id(child)), child.label))
        edges.extend(get_edges(child))
    return edges