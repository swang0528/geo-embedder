# dataset.py (with pathlib for robust path handling and rich features)
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import directed_hausdorff
import os
import pickle
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from pathlib import Path


def compute_gromov_hausdorff_approx(verts1: np.ndarray, verts2: np.ndarray) -> float:
    # This function remains unchanged for ground-truth calculation
    verts1 = verts1 - np.mean(verts1, axis=0)
    verts2 = verts2 - np.mean(verts2, axis=0)
    H = verts1.T @ verts2
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    verts2_rotated = verts2 @ R
    return max(directed_hausdorff(verts1, verts2_rotated)[0], directed_hausdorff(verts2_rotated, verts1)[0])


class PolygonDataset(Dataset):
    def __init__(self, num_pairs, cache_path, force_regenerate=False, n_resample=128, min_verts=4, max_verts=12):
        self.cache_path = Path(cache_path)
        self.n_resample = n_resample
        self.min_verts = min_verts
        self.max_verts = max_verts
        self.k_gh_scaling = 5.0

        if not force_regenerate and self.cache_path.exists():
            print(f"Loading dataset from cache: {self.cache_path}")
            with open(self.cache_path, 'rb') as f:
                self.data = pickle.load(f)
        else:
            print("Generating new dataset...")
            self.data = []
            for _ in tqdm(range(num_pairs), desc="Generating polygon pairs"):
                self.data.append(self._generate_pair())

            print(f"Saving dataset to cache: {self.cache_path}")
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.data, f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _generate_pair(self):
        verts1, phi1 = self._generate_polygon()
        verts2, phi2 = self._generate_polygon()

        # --- UPDATED: Use the new rich preprocessing ---
        processed_verts1 = self._preprocess_rich(verts1)
        processed_verts2 = self._preprocess_rich(verts2)

        # Ground truth calculation remains the same, using the original vertex data
        # before rich feature generation.
        d_gh_approx = compute_gromov_hausdorff_approx(self._preprocess(verts1), self._preprocess(verts2))

        rot_target = (phi1 - phi2) % (2 * np.pi)
        sim_target = np.exp(-self.k_gh_scaling * d_gh_approx)

        return {
            'g1': torch.tensor(processed_verts1, dtype=torch.float32),
            'g2': torch.tensor(processed_verts2, dtype=torch.float32),
            'rot_target': torch.tensor(rot_target, dtype=torch.float32),
            'sim_target': torch.tensor(sim_target, dtype=torch.float32),
            'phi1': torch.tensor(phi1, dtype=torch.float32),
            'phi2': torch.tensor(phi2, dtype=torch.float32)
        }

    def _generate_polygon(self):
        n_verts = np.random.randint(self.min_verts, self.max_verts + 1)
        angles = np.sort(np.random.rand(n_verts) * 2 * np.pi)
        radii = 1.0 + np.random.uniform(-0.4, 0.4, n_verts)
        x, y = radii * np.cos(angles), radii * np.sin(angles)
        phi = np.random.rand() * 2 * np.pi
        c, s = np.cos(phi), np.sin(phi)
        rot_matrix = np.array([[c, -s], [s, c]])
        verts = np.dot(np.stack([x, y], axis=1), rot_matrix.T)
        return verts, phi

    def _preprocess_rich(self, verts: np.ndarray) -> np.ndarray:
        """
        NEW: Preprocesses a polygon into a sequence of rich 8D feature vectors.
        """
        # Step 1: Resample to a fixed number of vertices (same as before)
        resampled_verts = self._preprocess(verts)  # shape: (n_resample, 2)

        if np.all(resampled_verts == 0):
            return np.zeros((self.n_resample, 8))

        # Step 2: Calculate geometric features for the sequence
        # Use np.roll to easily get previous and next vertices for circulant boundary conditions
        prev_verts = np.roll(resampled_verts, shift=1, axis=0)
        next_verts = np.roll(resampled_verts, shift=-1, axis=0)

        # Feature 2 & 3: Edge vector and length
        edge_vectors = resampled_verts - prev_verts
        edge_lengths = np.linalg.norm(edge_vectors, axis=1, keepdims=True)

        # Feature 4: Normalized edge vector
        # Add a small epsilon to avoid division by zero for zero-length edges
        norm_edge_vectors = edge_vectors / (edge_lengths + 1e-6)

        # Feature 5: Turning angles
        v_in = resampled_verts - prev_verts
        v_out = next_verts - resampled_verts

        # Use atan2 for robust angle calculation in [-pi, pi]
        angles1 = np.arctan2(v_in[:, 1], v_in[:, 0])
        angles2 = np.arctan2(v_out[:, 1], v_out[:, 0])
        turning_angles = (angles2 - angles1)
        # Normalize angles to be in [-pi, pi]
        turning_angles = (turning_angles + np.pi) % (2 * np.pi) - np.pi
        turning_angles = turning_angles.reshape(-1, 1)

        # Step 3: Concatenate all features
        # [coords (2), edge_vec (2), norm_edge_vec (2), length (1), angle (1)]
        rich_features = np.concatenate([
            resampled_verts,  # Feature 1
            edge_vectors,  # Feature 2
            norm_edge_vectors,  # Feature 4
            edge_lengths,  # Feature 3
            turning_angles  # Feature 5
        ], axis=1)

        return rich_features

    def _preprocess(self, verts):
        """Original preprocessing, returns (n_resample, 2) array. Kept for ground truth calculation."""
        centroid = np.mean(verts, axis=0)
        verts = verts - centroid
        try:
            area = ConvexHull(verts).volume
            if area > 1e-6:
                verts = verts / np.sqrt(area)
        except:
            pass  # Ignore degenerate polygons

        looped_verts = np.vstack([verts, verts[0]])
        distances = np.sqrt(np.sum(np.diff(looped_verts, axis=0) ** 2, axis=1))
        cumulative_dist = np.insert(np.cumsum(distances), 0, 0)
        total_perimeter = cumulative_dist[-1]

        if total_perimeter < 1e-6:
            return np.zeros((self.n_resample, 2))

        interp_points = np.linspace(0, total_perimeter, self.n_resample)
        interp_x = np.interp(interp_points, cumulative_dist, looped_verts[:, 0])
        interp_y = np.interp(interp_points, cumulative_dist, looped_verts[:, 1])

        return np.stack([interp_x, interp_y], axis=1)


# (The __main__ block remains the same)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate and cache a dataset of polygon pairs.")
    parser.add_argument("--num-pairs", type=int, default=1000, help="Number of polygon pairs to generate.")
    parser.add_argument("--out-file", type=str, default="data/polygons_rich.pkl",
                        help="Path to save the cached dataset.")
    parser.add_argument("--force", action="store_true", help="Force regeneration even if a cache file exists.")
    args = parser.parse_args()
    dataset = PolygonDataset(
        num_pairs=args.num_pairs,
        cache_path=args.out_file,
        force_regenerate=args.force
    )
    print(f"Dataset generation complete. {len(dataset)} pairs created and saved.")
