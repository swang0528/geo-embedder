import numpy as np
import gudhi


def compute_bottleneck_distance(verts1: np.ndarray, verts2: np.ndarray, max_edge_length: float = 10.0) -> float:
    """
    Computes the Bottleneck distance between the persistence diagrams of two 2D point clouds.

    This serves as a topologically robust measure of shape dissimilarity and is related
    to the Gromov-Hausdorff distance by the Stability Theorem. The function focuses on
    1-dimensional topological features (loops), which are most relevant for comparing
    single, non-intersecting polygons.

    Args:
        verts1: A numpy array of shape (n, 2) representing the first polygon's vertices.
        verts2: A numpy array of shape (m, 2) representing the second polygon's vertices.
        max_edge_length: The maximum edge length to consider when building the complex.
                         This acts as a filter to speed up computation.

    Returns:
        The Bottleneck distance between the H1 persistence diagrams of the two shapes.
    """
    # --- Step 1: Compute Persistence Diagram for the first polygon ---

    # An Alpha Complex is a good choice for point clouds in Euclidean space.
    alpha_complex1 = gudhi.AlphaComplex(points=verts1)
    # The simplex tree is the data structure that holds the topological complex.
    simplex_tree1 = alpha_complex1.create_simplex_tree(max_alpha_square=max_edge_length ** 2)
    # Compute the persistence diagram, which records the birth/death of topological features.
    diag1 = simplex_tree1.persistence()

    # --- Step 2: Compute Persistence Diagram for the second polygon ---

    alpha_complex2 = gudhi.AlphaComplex(points=verts2)
    simplex_tree2 = alpha_complex2.create_simplex_tree(max_alpha_square=max_edge_length ** 2)
    diag2 = simplex_tree2.persistence()

    # --- Step 3: Filter diagrams to keep only 1-dimensional features (loops) ---
    # The output of .persistence() is a list of (dimension, (birth, death)) tuples.
    # We only care about H1 features (loops) for comparing these polygons.
    diag1_h1 = simplex_tree1.persistence_intervals_in_dimension(1)
    diag2_h1 = simplex_tree2.persistence_intervals_in_dimension(1)

    # --- Step 4: Compute the Bottleneck distance between the two diagrams ---
    # This measures the "cost" of matching the loops from one shape to the other.
    bottleneck_dist = gudhi.bottleneck_distance(diag1_h1, diag2_h1)

    return bottleneck_dist


if __name__ == '__main__':
    """
    Example usage of the function to demonstrate its functionality.
    """
    print("--- Gudhi Bottleneck Distance Demonstration ---")

    # Example 1: Comparing two very similar shapes (a square and a noisy square)
    square = np.array([
        [0., 0.], [1., 0.], [1., 1.], [0., 1.]
    ])

    noisy_square = square + np.random.normal(0, 0.05, square.shape)

    dist1 = compute_bottleneck_distance(square, noisy_square)
    print(f"Distance between a square and a noisy square: {dist1:.4f}")
    print("Expected: A small value, as the core 'loop' feature is stable under noise.")

    # Example 2: Comparing two different shapes (a square and a line/degenerate polygon)
    line = np.array([
        [0., 0.5], [0.25, 0.5], [0.5, 0.5], [1.0, 0.5]
    ])

    dist2 = compute_bottleneck_distance(square, line)
    print(f"\nDistance between a square and a line: {dist2:.4f}")
    print("Expected: A large value, as the square has a prominent loop feature and the line has none.")

    # Example 3: Comparing two topologically similar but geometrically different shapes
    # A large circle and a small circle both have one "perfect" loop.
    theta = np.linspace(0, 2 * np.pi, 50)
    large_circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    small_circle = 0.5 * large_circle

    dist3 = compute_bottleneck_distance(large_circle, small_circle)
    print(f"\nDistance between a large circle and a small circle: {dist3:.4f}")
    print("Note: The distance is based on the 'persistence' of the loop, which is related to its size.")