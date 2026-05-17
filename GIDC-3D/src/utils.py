import numpy as np
import laspy
import cv2
from scipy.spatial import cKDTree
from typing import Dict, List, Tuple
import os

def load_and_normalize_pointcloud(input_path: str):
    ext = os.path.splitext(input_path)[1].lower()
    if ext in ['.las', '.laz']:
        las = laspy.read(input_path)
        points = np.vstack((las.x, las.y, las.z)).T
    elif ext == '.npy':
        points = np.load(input_path)
        if points.shape[1] != 3:
            raise ValueError("The NPY file should be of shape (N,3)")
    else:
        raise ValueError(f"Unsupported file formats:{ext}")
    return points

def compute_nearest_neighbors(points: np.ndarray, k_max: int = 10, extra: int = 3):
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid

    scale = np.max(np.linalg.norm(points_centered, axis=1))
    points_norm = points_centered / scale

    kd = cKDTree(points_norm)
    query_k = k_max + extra + 1
    dists_all, inds_all = kd.query(points_norm, k=query_k)

    dists_all = dists_all[:, 1:]
    inds_all = inds_all[:, 1:]

    dists = dists_all[:, :k_max]
    inds = inds_all[:, :k_max]
    return dists, inds
def scheme2_global_score_selection(points: np.ndarray,

                                   indices_knn: np.ndarray,
                                   dists_knn: np.ndarray,
                                   k_neighbors: int = 10,
                                   alpha: float = 0.2,
                                   beta: float = 0.8,
                                   nbins: int = 10,
                                   top_ratio: float = 0.2):
    N = points.shape[0]
    if N == 0:
        return np.array([], dtype=np.int64)
    entropy_vals = np.empty(N, dtype=np.float64)
    for idx in range(N):
        neigh_idx = indices_knn[idx, :k_neighbors]
        neigh_pts = points[neigh_idx]
        diff = neigh_pts - points[idx]
        avg_dist = np.mean(dists_knn[idx, :k_neighbors])
        dist = np.linalg.norm(diff, axis=1) / (avg_dist + 1e-12)
        hist, _ = np.histogram(dist, bins=nbins, density=True)
        hist += 1e-12
        entropy_vals[idx] = -np.sum(hist * np.log(hist))

    num_select = max(1, int(N * top_ratio))
    final_keypoints_idx = np.argsort(-entropy_vals)[:num_select]
    return final_keypoints_idx

def assign_points_to_keypoints(points: np.ndarray, keypoint_idx: np.ndarray, indices_knn: np.ndarray,
                               dists_knn: np.ndarray, k_neighbors: int = 10) -> Tuple[Dict[int, List[int]], List[int]]:

    keypoint_set = set(keypoint_idx.tolist())
    assign_dict: Dict[int, List[int]] = {kp: [] for kp in keypoint_idx}
    unassigned_points: List[int] = []

    for i in range(points.shape[0]):
        neigh_idx = indices_knn[i, :k_neighbors]
        neigh_dist = dists_knn[i, :k_neighbors]

        candidate_kps = []
        for n_idx, n_dist in zip(neigh_idx, neigh_dist):
            if n_idx in keypoint_set:
                candidate_kps.append((n_idx, n_dist))

        if len(candidate_kps) == 0:
            unassigned_points.append(i)
            continue

        nearest_kp = min(candidate_kps, key=lambda x: x[1])[0]
        assign_dict[nearest_kp].append(i)

    return assign_dict, unassigned_points

def Watermark_preprocessing(input_img):

    img = cv2.imread(input_img, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {input_img}")

    _, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    bit_num = binary.flatten().astype(np.uint8)
    return bit_num

def bits_to_binary_image(bit_array, save_path, height=32, width=32, value_mode="0255", resize_scale=None):

    bit_array = np.asarray(bit_array).astype(np.uint8)
    if bit_array.size != height * width:
        raise ValueError("bit_array length does not match the height width.")

    binary_img = bit_array.reshape((height, width))
    if value_mode == "0255":
        binary_img = binary_img * 255

    if save_path is not None:
        img_to_save = binary_img
        if resize_scale is not None:
            img_to_save = cv2.resize(img_to_save, None, fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_NEAREST)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img_to_save)

    return binary_img