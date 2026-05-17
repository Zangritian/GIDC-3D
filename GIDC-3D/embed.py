# embed.py
import numpy as np
import os
from src.utils import (
    load_and_normalize_pointcloud, compute_nearest_neighbors,
    scheme2_global_score_selection, assign_points_to_keypoints,
    Watermark_preprocessing
)

def qim_coordinate_z_group(ori_points, dict_key, bit_watermark, threshold_r, r_index_threshold, r_bit_threshold):
    r = threshold_r
    points = ori_points.copy()

    for kp_idx, point_indices in dict_key.items():
        if len(point_indices) < 2:
            continue
        group_z = points[point_indices, 2]
        z_ref = points[kp_idx, 2]

        D = group_z - z_ref
        Dmin, Dmax = D.min(), D.max()
        if Dmax == Dmin:
            continue
        D_norm = (D - Dmin) / (Dmax - Dmin)

        for i, idx in enumerate(point_indices):
            d_norm = D_norm[i]
            wm_index = int((d_norm * r_index_threshold) % len(bit_watermark))
            wm_bit = bit_watermark[wm_index]

            d_scaled = d_norm * r_bit_threshold
            if wm_bit == 0 and d_scaled % r > r / 2:
                d_norm = (d_scaled - r / 2) / r_bit_threshold
            elif wm_bit == 1 and d_scaled % r <= r / 2:
                d_norm = (d_scaled + r / 2) / r_bit_threshold

            D[i] = d_norm * (Dmax - Dmin) + Dmin
        points[point_indices, 2] = z_ref + D
    return points


if __name__ == '__main__':
    # ================= 参数配置 =================
    input_las_path = r" "
    watermark_img_path = r""
    output_dir = r" "
    output_name = " "

    k_max_knn = 10
    k_neighbors = 10
    feature_top_ratio = 0.05
    points = load_and_normalize_pointcloud(input_las_path)
    bit_num = Watermark_preprocessing(watermark_img_path)

    dists_knn, indices_knn = compute_nearest_neighbors(points=points, k_max=k_max_knn)

    keypoints_idx = scheme2_global_score_selection(
        points=points,
        indices_knn=indices_knn,
        dists_knn=dists_knn,
        k_neighbors=k_neighbors,
    )
    dict_key, _ = assign_points_to_keypoints(points, keypoints_idx, indices_knn, dists_knn, k_neighbors=10)
    watermarked_points = qim_coordinate_z_group(
        points, dict_key, bit_num, threshold_r=10, r_index_threshold=10000, r_bit_threshold=1000000
    )
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, output_name + ".npy"), watermarked_points)
    print("[Embedding process complete] The watermarked point cloud has been properly saved.")