# extract.py
import numpy as np
import os
from src.utils import (
    load_and_normalize_pointcloud, compute_nearest_neighbors,
    scheme2_global_score_selection, assign_points_to_keypoints,
    bits_to_binary_image
)


def qim_coordinate_z_group_extract_vote(watermarked_points, dict_key, len_wm=1024, threshold_r=10,
                                        r_index_threshold=10000, r_bit_threshold=10000000):
    r = threshold_r
    vote_dict = {i: [] for i in range(len_wm)}

    for kp_idx, point_indices in dict_key.items():
        if len(point_indices) < 2:
            continue
        z_ref = watermarked_points[kp_idx, 2]
        group_z = watermarked_points[point_indices, 2]

        D = group_z - z_ref
        Dmin = np.min(D)
        Dmax = np.max(D)
        if Dmax == Dmin:
            continue
        D_norm = (D - Dmin) / (Dmax - Dmin)

        for i in range(len(point_indices)):
            d_norm = D_norm[i]
            index_z = int((d_norm * r_index_threshold) % len_wm)
            d_scaled = d_norm * r_bit_threshold

            bit_ori = 0 if (d_scaled % r) >= r / 2 else 1
            vote_dict[index_z].append(bit_ori)

    bit_seq = np.zeros(len_wm, dtype=np.uint8)
    for idx, votes in vote_dict.items():
        if len(votes) == 0:
            bit_seq[idx] = 0
        else:
            bit_seq[idx] = 1 if sum(votes) >= len(votes) / 2 else 0
    return bit_seq


if __name__ == '__main__':
    # ================= 参数配置 =================
    input_npy_path = r" "
    extracted_img_path = r" "

    k_max_knn = 10
    k_neighbors = 10
    feature_top_ratio = 0.05
    points = load_and_normalize_pointcloud(input_npy_path)
    dists_knn, indices_knn = compute_nearest_neighbors(points=points, k_max=k_max_knn)
    keypoints_idx = scheme2_global_score_selection(
        points=points,
        indices_knn=indices_knn,
        dists_knn=dists_knn,
        k_neighbors=k_neighbors,
    )
    dict_key, _ = assign_points_to_keypoints(points, keypoints_idx, indices_knn, dists_knn, k_neighbors=10)
    extracted_bits = qim_coordinate_z_group_extract_vote(
        watermarked_points=points, dict_key=dict_key, len_wm=1024, threshold_r=10,
        r_index_threshold=10000, r_bit_threshold=1000000
    )

    bits_to_binary_image(bit_array=extracted_bits, save_path=extracted_img_path, height=32, width=32)
    print(f"[Extraction process complete] The blind watermark restoration image has been successfully saved to:{extracted_img_path}")