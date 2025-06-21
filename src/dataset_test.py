import matplotlib.pyplot as plt
import numpy as np
from dataset_gen import PolygonDataset


def run_tests():
    print("--- Running Dataset Sanity Checks ---")

    # Instantiate dataset - will generate a small one if cache doesn't exist
    test_dataset = PolygonDataset(num_pairs=10, cache_path=r"C:\Users\Siyang_Wang_work\Documents\A_IndependentResearch\GenAI\LayoutML\Geo-Embedder\dataset\test_data_synth_v1\test_dataset.pkl", force_regenerate=True)

    # 1. Test Dataset Length
    assert len(test_dataset) == 10
    print("[PASS] Dataset length is correct.")

    # 2. Get a sample and test shapes
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        g1 = sample['g1']
        assert g1.shape == (128, 2)
        print("[PASS] Polygon tensor shape is correct (128, 2).")

        # 3. Test Label Ranges
        rot_target = sample['rot_target']
        sim_target = sample['sim_target']
        assert 0 <= rot_target < 2 * np.pi
        print(f"[PASS] Rotation target is within [0, 2*pi). Value: {rot_target:.4f}")
        assert 0 < sim_target <= 1.0
        print(f"[PASS] Similarity target is within (0, 1.0]. Value: {sim_target:.4f}")

        # 4. Test Rotation Calculation Integrity
        phi1 = sample['phi1']
        phi2 = sample['phi2']
        expected_rot = (phi1 - phi2) % (2 * np.pi)
        assert np.isclose(rot_target.item(), expected_rot), "Rotation target does not match phi difference."
        print("[PASS] Rotation target correctly calculated from source angles.")

        print("\n--- All tests passed! ---")

        # 5. Visual Sanity Check
        print("\nGenerating visual sanity check plot...")
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Plot polygon 1
        g1_verts = g1.numpy()
        ax[0].plot(np.append(g1_verts[:, 0], g1_verts[0, 0]), np.append(g1_verts[:, 1], g1_verts[0, 1]), '-o')
        ax[0].set_title(f"Polygon 1 (phi ~ {phi1:.2f} rad)")
        ax[0].set_aspect('equal', 'box')
        ax[0].grid(True)

        # Plot polygon 2
        g2_verts = sample['g2'].numpy()
        ax[1].plot(np.append(g2_verts[:, 0], g2_verts[0, 0]), np.append(g2_verts[:, 1], g2_verts[0, 1]), '-o', color='r')
        ax[1].set_title(f"Polygon 2 (phi ~ {phi2:.2f} rad)")
        ax[1].set_aspect('equal', 'box')
        ax[1].grid(True)

        plt.suptitle(f"Visual Check | Relative Rot: {rot_target:.2f} | Sim: {sim_target:.2f}")
        plt.tight_layout()
        plt.savefig(f"dataset_visual_check_{i}.png")
        # print("Plot saved to dataset_visual_check.png")
        # plt.show()


if __name__ == '__main__':
    run_tests()
