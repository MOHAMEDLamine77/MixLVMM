import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional, Union
from enum import Enum
import warnings


class ColorSpace(Enum):
    BGR2LAB = "BGR2LAB"
    RGB2LAB = "RGB2LAB" 
    BGR2RGB = "BGR2RGB"
    RGB = "RGB"
    BGR = "BGR"
    LAB = "LAB"


class FASASaliencyDetector(nn.Module):   
    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        tot_bins: int = 8,
        sigma_c: float = 16.0,
        gaussian_blur_kernel: int = 10,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.image_height, self.image_width = image_size
        self.tot_bins = tot_bins
        self.sigma_c = sigma_c
        self.gaussian_blur_kernel = gaussian_blur_kernel
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.register_buffer(
            'mean_vector',
            torch.tensor([0.5555, 0.6449, 0.0002, 0.0063], dtype=torch.float32)
        )
        
        self.register_buffer(
            'covariance_matrix_inv',
            torch.tensor([
                [43.3777, 1.7633, -0.4059, 1.0997],
                [1.7633, 40.7221, -0.0165, 0.0447],
                [-0.4059, -0.0165, 87.0455, -3.2744],
                [1.0997, 0.0447, -3.2744, 125.1503]
            ], dtype=torch.float32)
        )
        
        self._reset_internal_state()
        
    def _reset_internal_state(self):
        self.histogram = None
        self.unique_pixels = None
        self.color_distance_matrix = None
        self.exponential_color_distance_matrix = None
        self.map_3d_1d = None
        self.image_quantized = None
        
    def _convert_color_space(
        self, 
        image: np.ndarray, 
        color_space: ColorSpace
    ) -> np.ndarray:

        if color_space == ColorSpace.BGR2LAB:
            return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        elif color_space == ColorSpace.RGB2LAB:
            return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        elif color_space == ColorSpace.BGR2RGB:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif color_space in [ColorSpace.RGB, ColorSpace.BGR, ColorSpace.LAB]:
            return image
        else:
            raise ValueError(f"Unsupported color space: {color_space}")
    
    def _compute_color_histogram(self, lab_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        l_min, l_max = lab_image[:, :, 0].min(), lab_image[:, :, 0].max()
        a_min, a_max = lab_image[:, :, 1].min(), lab_image[:, :, 1].max()
        b_min, b_max = lab_image[:, :, 2].min(), lab_image[:, :, 2].max()

        self.l_range = np.linspace(l_min, l_max, num=self.tot_bins, endpoint=False)
        self.a_range = np.linspace(a_min, a_max, num=self.tot_bins, endpoint=False)
        self.b_range = np.linspace(b_min, b_max, num=self.tot_bins, endpoint=False)

        l_quantized = np.digitize(lab_image[:, :, 0], self.l_range, right=False) - 1
        a_quantized = np.digitize(lab_image[:, :, 1], self.a_range, right=False) - 1
        b_quantized = np.digitize(lab_image[:, :, 2], self.b_range, right=False) - 1

        l_quantized = np.clip(l_quantized, 0, self.tot_bins - 1)
        a_quantized = np.clip(a_quantized, 0, self.tot_bins - 1)
        b_quantized = np.clip(b_quantized, 0, self.tot_bins - 1)
        
        self.image_quantized = np.dstack((l_quantized, a_quantized, b_quantized))

        histogram = cv2.calcHist(
            [lab_image], 
            channels=[0, 1, 2], 
            mask=None,
            histSize=[self.tot_bins, self.tot_bins, self.tot_bins],
            ranges=[l_min, l_max, a_min, a_max, b_min, b_max]
        ).flatten()
        
        return histogram, self.image_quantized
    
    def _precompute_color_features(self, histogram: np.ndarray) -> int:

        nonzero_indices = np.where(histogram > 0)[0]
        self.number_of_colors = len(nonzero_indices)
        
        if self.number_of_colors == 0:
            warnings.warn("No colors found in histogram")
            return 0
        
        hist_3d = histogram.reshape(self.tot_bins, self.tot_bins, self.tot_bins)
        self.index_matrix = np.column_stack(np.where(hist_3d > 0))

        self.map_3d_1d = np.zeros((self.tot_bins, self.tot_bins, self.tot_bins), dtype=np.int32)

        l_centroids, a_centroids, b_centroids = np.meshgrid(
            self.l_range, self.a_range, self.b_range, indexing='ij'
        )
        
        self.unique_pixels = np.zeros((self.number_of_colors, 3))
        self.centx_matrix = np.zeros(self.number_of_colors)
        self.centy_matrix = np.zeros(self.number_of_colors)
        self.centx2_matrix = np.zeros(self.number_of_colors)
        self.centy2_matrix = np.zeros(self.number_of_colors)

        for i, (l_idx, a_idx, b_idx) in enumerate(self.index_matrix):
            self.unique_pixels[i] = [
                l_centroids[l_idx, a_idx, b_idx],
                a_centroids[l_idx, a_idx, b_idx],
                b_centroids[l_idx, a_idx, b_idx]
            ]

            self.map_3d_1d[l_idx, a_idx, b_idx] = i

            mask = ((self.image_quantized[:, :, 0] == l_idx) & 
                   (self.image_quantized[:, :, 1] == a_idx) & 
                   (self.image_quantized[:, :, 2] == b_idx))
            
            if np.any(mask):
                y_coords, x_coords = np.where(mask)

                self.centx_matrix[i] = np.sum(x_coords)
                self.centy_matrix[i] = np.sum(y_coords)
                self.centx2_matrix[i] = np.sum(x_coords ** 2)
                self.centy2_matrix[i] = np.sum(y_coords ** 2)

        color_diff = self.unique_pixels[:, np.newaxis] - self.unique_pixels[np.newaxis, :]
        color_diff_squared = np.sum(color_diff ** 2, axis=2)
        
        self.color_distance_matrix = np.sqrt(color_diff_squared)
        self.exponential_color_distance_matrix = np.exp(
            -color_diff_squared / (2 * self.sigma_c ** 2)
        )
        
        return self.number_of_colors
    
    def _compute_bilateral_filtering(self, histogram: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        hist_values = histogram[histogram > 0]

        self.contrast = np.dot(self.color_distance_matrix, hist_values)

        normalization_array = np.dot(self.exponential_color_distance_matrix, hist_values)

        normalization_array = np.maximum(normalization_array, 1e-8)

        mx = np.dot(self.exponential_color_distance_matrix, self.centx_matrix) / normalization_array
        my = np.dot(self.exponential_color_distance_matrix, self.centy_matrix) / normalization_array
        mx2 = np.dot(self.exponential_color_distance_matrix, self.centx2_matrix) / normalization_array
        my2 = np.dot(self.exponential_color_distance_matrix, self.centy2_matrix) / normalization_array

        vx = np.maximum(mx2 - mx ** 2, 0.0)
        vy = np.maximum(my2 - my ** 2, 0.0)
        
        return mx, my, vx, vy
    
    def _compute_shape_probability(self, mx: np.ndarray, my: np.ndarray, vx: np.ndarray, vy: np.ndarray) -> np.ndarray:

        width_norm = float(self.image_width)
        height_norm = float(self.image_height)
        
        g = np.array([
            np.sqrt(12 * vx) / width_norm, 
            np.sqrt(12 * vy) / height_norm,  
            (mx - width_norm / 2.0) / width_norm,  
            (my - height_norm / 2.0) / height_norm  
        ]).T

        mean_vec = self.mean_vector.cpu().numpy()
        cov_inv = self.covariance_matrix_inv.cpu().numpy()
        
        X = g - mean_vec
        mahalanobis_dist = np.sum((X @ cov_inv) * X, axis=1)

        shape_probability = np.exp(-mahalanobis_dist / 2.0)
        
        return shape_probability
    
    def _compute_final_saliency(self, shape_probability: np.ndarray) -> np.ndarray:

        saliency = self.contrast * shape_probability
        normalization = np.sum(self.exponential_color_distance_matrix, axis=1)
        normalization = np.maximum(normalization, 1e-8)
        
        smoothed_saliency = (
            np.dot(self.exponential_color_distance_matrix, saliency) / normalization
        )

        min_sal, max_sal = smoothed_saliency.min(), smoothed_saliency.max()
        if max_sal > min_sal:
            normalized_saliency = 255.0 * (smoothed_saliency - min_sal) / (max_sal - min_sal)
        else:
            normalized_saliency = np.zeros_like(smoothed_saliency)
        
        return normalized_saliency
    
    def _create_saliency_map(self, saliency_values: np.ndarray) -> np.ndarray:

        saliency_map = np.zeros((self.image_height, self.image_width), dtype=np.float32)

        for y in range(self.image_height):
            for x in range(self.image_width):
                color_idx = self.image_quantized[y, x]
                linear_idx = self.map_3d_1d[color_idx[0], color_idx[1], color_idx[2]]
                saliency_map[y, x] = saliency_values[linear_idx]
        
        return saliency_map
    
    def _apply_gaussian_smoothing(self, saliency_map: np.ndarray) -> np.ndarray:

        kernel_size = (self.gaussian_blur_kernel, self.gaussian_blur_kernel)
        return cv2.blur(saliency_map.astype(np.float32), kernel_size)
    
    def forward(
        self, 
        image: Union[torch.Tensor, np.ndarray], 
        color_space: ColorSpace = ColorSpace.RGB2LAB
    ) -> torch.Tensor:

        if isinstance(image, torch.Tensor):
            if image.dim() == 4:  
                image = image.squeeze(0)  
            if image.dim() == 3:  
                image = image.permute(1, 2, 0)  
            image = image.cpu().numpy()

        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        lab_image = self._convert_color_space(image, color_space)

        histogram, _ = self._compute_color_histogram(lab_image)

        num_colors = self._precompute_color_features(histogram)
        if num_colors == 0:
            zero_map = torch.zeros(1, self.image_height, self.image_width, device=self.device)
            return zero_map

        mx, my, vx, vy = self._compute_bilateral_filtering(histogram)
        shape_prob = self._compute_shape_probability(mx, my, vx, vy)
        saliency_values = self._compute_final_saliency(shape_prob)
        saliency_map = self._create_saliency_map(saliency_values)
        smoothed_map = self._apply_gaussian_smoothing(saliency_map)
        saliency_tensor = torch.from_numpy(smoothed_map).float().to(self.device)
        saliency_tensor = saliency_tensor / 255.0  
        saliency_tensor = saliency_tensor.unsqueeze(0)
        
        return saliency_tensor


def create_fasa_detector(
    image_size: Tuple[int, int] = (256, 256),
    device: Optional[torch.device] = None
) -> FASASaliencyDetector:
    return FASASaliencyDetector(
        image_size=image_size,
        device=device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )