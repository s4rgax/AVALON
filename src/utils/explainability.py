import math
import cv2
import numpy as np

def get_pixel_attention_map(attention_maps, original_image_shape, size: int):
    combined_attention_image = np.zeros(original_image_shape)
    count_matrix = np.zeros(original_image_shape)
    offset = math.floor(size/2)
    for i in range(original_image_shape[0]):
        for j in range(original_image_shape[1]):
            top_left_i = max(i - offset, 0)
            top_left_j = max(j - offset, 0)
            bottom_right_i = min(i + offset + 1, original_image_shape[0])
            bottom_right_j = min(j + offset + 1, original_image_shape[1])
            attention_patch = attention_maps[i * original_image_shape[1] + j]
            combined_attention_image[top_left_i:bottom_right_i, top_left_j:bottom_right_j] += attention_patch[offset - (i - top_left_i):size - (bottom_right_i - i - 1),offset - (j - top_left_j):size - (bottom_right_j - j - 1),]
            count_matrix[top_left_i:bottom_right_i, top_left_j:bottom_right_j] += 1
    combined_attention_image /= count_matrix

def save_from_scaled_attention_map (normalized_arr, save_path: str, attention_map_tag = ''):
    attention_map = (normalized_arr * 255).astype(np.uint8)
    rgb_image = cv2.applyColorMap(attention_map, cv2.COLORMAP_JET)
    cv2.imwrite(save_path, rgb_image)

def save_from_scaled_attention_map_with_label (normalized_arr, save_path: str, label_attention):
    attention_map = (normalized_arr * 255).astype(np.uint8)
    rgb_image = cv2.applyColorMap(attention_map, cv2.COLORMAP_JET)
    resized_rgb = cv2.resize(rgb_image, (200,200), interpolation=cv2.INTER_NEAREST)
    cell_size = 40
    for i in range(5):
        for j in range(5):
            cell_value = 'H' if label_attention[i,j] == 0 else 'D'
            x_center = (i+.3) * cell_size
            y_center = (j +.7) * cell_size
            cv2.putText(resized_rgb, str(cell_value), (int(x_center), int(y_center)), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
    cv2.imwrite(save_path, resized_rgb)


def computeAttentionScene(attention_tensor, ground_truth_matrix, k):
    n, m = ground_truth_matrix.shape
    X = np.zeros((n + 2*k, m + 2*k))
    times = np.zeros((n + 2 * k, m + 2 * k))

    for i in range(n):
        for j in range(m):
            if ground_truth_matrix[i, j] == 1:
                X[i:i+2*k+1, j:j+2*k+1] += attention_tensor[i, j, :, :]
            else:
                X[i:i+2*k+1, j:j+2*k+1] += -1 * attention_tensor[i, j, :, :]
            times[i:i + 2 * k + 1, j:j + 2 * k + 1] += np.ones((2 * k + 1, 2 * k + 1))

    att_scene = X[k:n+k, k:m+k]
    clipped_times = times[k:n+k, k:m+k]
    att_scene = att_scene / clipped_times
    min_val, max_val = att_scene.min(), att_scene.max()
    att_scene = (att_scene - min_val) / (max_val - min_val)
    return att_scene

def computePixelBoundary(mask_matrix, k):
    n, m = mask_matrix.shape
    pos_map = np.zeros((n, m))

    padded_mask_matrix = np.pad(mask_matrix, ((k, k), (k, k)), 'constant', constant_values=0)

    for i in range(n):
        for j in range(m):
            sub_matrix = padded_mask_matrix[i:i+2*k+1, j:j+2*k+1]
            total = np.sum(sub_matrix)

            if mask_matrix[i, j] == 0:
                if total == 0:
                    pos_map[i, j] = 1
                else:
                    pos_map[i, j] = 2
            else:
                if total == (2*k+1)**2:
                    pos_map[i, j] = 3
                else:
                    pos_map[i, j] = 4
    pure_0_matrix = np.where(pos_map == 1, 1, 0)
    spure_0_matrix = np.where(pos_map == 2, 1, 0)
    pure_1_matrix = np.where(pos_map == 3, 1, 0)
    spure_1_matrix = np.where(pos_map == 4, 1, 0)

    return pure_0_matrix, spure_0_matrix, pure_1_matrix, spure_1_matrix


def extract_neigh_from_predictions(original_matrix, window_size: int):
    flattened_submatrices = []
    padded_matrix = np.pad(original_matrix,
                           ((window_size // 2, window_size // 2), (window_size // 2, window_size // 2)),
                           mode='constant')
    for i in range(original_matrix.shape[0]):
        for j in range(original_matrix.shape[1]):
            submatrix = padded_matrix[i:i + window_size, j:j + window_size]
            flattened_submatrices.append(submatrix)
    return np.array(flattened_submatrices).reshape((len(flattened_submatrices), window_size, window_size))

