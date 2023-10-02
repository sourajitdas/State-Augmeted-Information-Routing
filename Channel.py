import networkx as nx
import numpy as np

# np.random.seed(42)  # set the random seed to a fixed value


def create_network(N):
    # Generate some random data
    n = N # number of nodes
    d = 2 # dimensions
    points = np.random.rand(n, d)

    # Define the value of k
    k = 4

    # Create a graph
    G = nx.Graph()

    # Add nodes with their positions as attributes
    for i in range(n):
        G.add_node(i, pos=tuple(points[i]))

    # Compute pairwise distances between nodes
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distances[i, j] = distances[j, i] = np.linalg.norm(points[i] - points[j])

    # Connect each node to its k nearest neighbors
    for i in range(n):
        # Sort nodes by distance to the current node
        sorted_nodes = np.argsort(distances[i])
        # Connect the current node to its k nearest neighbors
        for j in sorted_nodes[1:k+1]:
            G.add_edge(i, j, weight=distances[i, j])

    # # Visualize the graph
    # pos = nx.get_node_attributes(G, 'pos')
    # nx.draw(G, pos, with_labels=True)
    # plt.title("Geometric graph for N=%d nodes" % N)
    # plt.show()

    A = nx.to_numpy_array(G, weight='weight')
    A[A != 0] = 10
    A[A == 0] = 1e-10

    return A

# To generate a perturbed network, uncomment the code below and set the values of N0 and shift

# def create_network(N, N0=0, shift=0):
#     # Generate some random data
#     n = N # number of nodes
#     d = 2 # dimensions
#     points = np.random.rand(n, d)
#
#     if N0 > 0:
#         indices = np.random.choice(n, int(n * N0), replace=False)
#         offsets = np.random.randn(int(N0 * n), d) * shift
#         points[indices] += offsets
#
#     # Define the value of k
#     k = 4
#
#     # Create a graph
#     G = nx.Graph()
#
#     # Add nodes with their positions as attributes
#     for i in range(n):
#         G.add_node(i, pos=tuple(points[i]))
#
#     # Compute pairwise distances between nodes
#     distances = np.zeros((n, n))
#     for i in range(n):
#         for j in range(i + 1, n):
#             distances[i, j] = distances[j, i] = np.linalg.norm(points[i] - points[j])
#
#     # Connect each node to its k nearest neighbors
#     for i in range(n):
#         # Sort nodes by distance to the current node
#         sorted_nodes = np.argsort(distances[i])
#         # Connect the current node to its k nearest neighbors
#         for j in sorted_nodes[1:k+1]:
#             G.add_edge(i, j, weight=distances[i, j])
#
#     # Scale edge weights by 10
#     for (u, v, d) in G.edges(data=True):
#         d['weight'] *= np.random.randint(10,11)
#
#     A = nx.to_numpy_array(G, weight='weight')
#     A[A != 0] = 10
#     A[A == 0] = 1e-10
#     np.fill_diagonal(A, 0)
#
#     return A
