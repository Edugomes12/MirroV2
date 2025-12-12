#!/usr/bin/env python3

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.restoration import estimate_sigma
from scipy.stats import entropy, kurtosis
from PIL import Image
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass


class SceneType(Enum):
    UNKNOWN = "unknown"
    CAR = "car"
    GLASS = "glass"
    CAR_GLASS = "car_glass"
    CAR_REFLECTION = "car_reflection"
    GLASS_REFLECTION = "glass_reflection"
    CAR_GLASS_REFLECTION = "car_glass_reflection"
    CAR_SMOOTH_WALL = "car_smooth_wall"
    GLASS_WALL = "glass_wall"
    GLASS_DOCUMENT = "glass_document"
    CAR_DOCUMENT = "car_document"
    DOCUMENT = "document"
    SMOOTH_SURFACE = "smooth_surface"


@dataclass
class WeightConfig:
    weight_texture: float = 1.0
    weight_edge: float = 1.0
    weight_noise: float = 1.0
    weight_lighting: float = 1.0
    boost_reflection: float = 0.0
    boost_smooth: float = 0.0
    boost_glass: float = 0.0
    th_texture_manipulated: int = 45
    th_texture_natural: int = 75
    th_edge_manipulated: int = 35
    th_noise_manipulated: int = 35
    th_lighting_manipulated: int = 8
    th_final_suspicious: int = 50


@dataclass
class CLAHEConfig:
    texture_clahe: bool = False
    edge_clahe: bool = True
    noise_clahe: bool = True
    lighting_clahe: bool = True
    clip_limit: float = 2.0


@dataclass
class ManualConfig:
    auto_mode: bool = True
    enable_texture: bool = True
    enable_edge: bool = True
    enable_noise: bool = True
    enable_lighting: bool = True
    enable_boost_reflection: bool = True
    enable_boost_smooth: bool = True
    enable_boost_glass: bool = True
    enable_watermark_detection: bool = True
    custom_weights: Optional[WeightConfig] = None
    force_scene_type: Optional[SceneType] = None
    clahe_config: Optional[CLAHEConfig] = None
    use_car_glass_specific: bool = True


class WeightPresets:
    
    @staticmethod
    def get_preset(scene_type: SceneType) -> WeightConfig:
        presets = {
            SceneType.UNKNOWN: WeightConfig(),
            
            SceneType.CAR: WeightConfig(
                weight_texture=0.9,
                weight_edge=1.1,
                weight_noise=1.0,
                weight_lighting=1.0,
                boost_reflection=10,
                boost_smooth=5,
                th_texture_manipulated=40,
                th_texture_natural=70
            ),
            
            SceneType.GLASS: WeightConfig(
                weight_texture=0.7,
                weight_edge=0.8,
                weight_noise=0.9,
                weight_lighting=1.0,
                boost_reflection=25,
                boost_smooth=20,
                boost_glass=30,
                th_texture_manipulated=35,
                th_texture_natural=65
            ),
            
            SceneType.CAR_GLASS: WeightConfig(
                weight_texture=0.75,
                weight_edge=0.9,
                weight_noise=0.9,
                weight_lighting=1.0,
                boost_reflection=20,
                boost_smooth=15,
                boost_glass=25,
                th_texture_manipulated=38,
                th_texture_natural=68
            ),
            
            SceneType.CAR_REFLECTION: WeightConfig(
                weight_texture=0.85,
                weight_edge=1.0,
                weight_noise=0.95,
                weight_lighting=0.9,
                boost_reflection=30,
                boost_smooth=10,
                th_texture_manipulated=35,
                th_texture_natural=65
            ),
            
            SceneType.GLASS_REFLECTION: WeightConfig(
                weight_texture=0.65,
                weight_edge=0.75,
                weight_noise=0.85,
                weight_lighting=0.9,
                boost_reflection=35,
                boost_smooth=25,
                boost_glass=35,
                th_texture_manipulated=30,
                th_texture_natural=60
            ),
            
            SceneType.CAR_GLASS_REFLECTION: WeightConfig(
                weight_texture=1.35,
                weight_edge=0.8,
                weight_noise=0.85,
                weight_lighting=0.85,
                boost_reflection=35,
                boost_smooth=20,
                boost_glass=30,
                th_texture_manipulated=30,
                th_texture_natural=60
            ),
            
            SceneType.CAR_SMOOTH_WALL: WeightConfig(
                weight_texture=0.8,
                weight_edge=1.0,
                weight_noise=1.0,
                weight_lighting=1.1,
                boost_reflection=15,
                boost_smooth=25,
                th_texture_manipulated=35,
                th_texture_natural=65
            ),
            
            SceneType.GLASS_WALL: WeightConfig(
                weight_texture=0.7,
                weight_edge=0.85,
                weight_noise=0.9,
                weight_lighting=1.0,
                boost_reflection=20,
                boost_smooth=30,
                boost_glass=25,
                th_texture_manipulated=32,
                th_texture_natural=62
            ),
            
            SceneType.GLASS_DOCUMENT: WeightConfig(
                weight_texture=0.75,
                weight_edge=0.9,
                weight_noise=0.95,
                weight_lighting=1.0,
                boost_reflection=15,
                boost_smooth=15,
                boost_glass=20,
                th_texture_manipulated=40,
                th_texture_natural=70
            ),
            
            SceneType.CAR_DOCUMENT: WeightConfig(
                weight_texture=0.85,
                weight_edge=1.0,
                weight_noise=1.0,
                weight_lighting=1.0,
                boost_reflection=10,
                boost_smooth=10,
                th_texture_manipulated=42,
                th_texture_natural=72
            ),
            
            SceneType.DOCUMENT: WeightConfig(
                weight_texture=0.9,
                weight_edge=1.1,
                weight_noise=1.0,
                weight_lighting=1.0,
                boost_reflection=5,
                boost_smooth=10,
                th_texture_manipulated=45,
                th_texture_natural=75
            ),
            
            SceneType.SMOOTH_SURFACE: WeightConfig(
                weight_texture=0.7,
                weight_edge=0.8,
                weight_noise=0.9,
                weight_lighting=1.0,
                boost_reflection=20,
                boost_smooth=35,
                th_texture_manipulated=30,
                th_texture_natural=60
            )
        }
        
        return presets.get(scene_type, WeightConfig())


class AnalysisLogger:
    
    def __init__(self):
        self.logs = []
    
    def log(self, phase: str, message: str, data: Dict = None):
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "message": message,
            "data": data or {}
        })
    
    def get_logs(self) -> List[Dict]:
        return self.logs


def apply_clahe(img_gray: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    if img_gray.dtype != np.uint8:
        img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(img_gray)


class GeminiWatermarkDetector:
    
    def __init__(self, sensitivity: str = "balanced"):
        self.sensitivity = sensitivity
        self.thresholds = {
            "conservative": {"min_confidence": 0.80, "corner_ratio": 0.12},
            "balanced": {"min_confidence": 0.70, "corner_ratio": 0.15},
            "aggressive": {"min_confidence": 0.55, "corner_ratio": 0.18}
        }.get(sensitivity, {"min_confidence": 0.70, "corner_ratio": 0.15})
    
    def detectar_watermark(self, image: Union[np.ndarray, Image.Image]) -> Dict[str, Any]:
        return self.detect(image)
    
    def detect(self, image: Union[np.ndarray, Image.Image]) -> Dict[str, Any]:
        img = self._prepare_image(image)
        result = self._detect_gemini_star(img)
        
        if result["detected"]:
            return {
                "gemini_detected": True,
                "confidence": round(result["confidence"], 2),
                "reason": f"Gemini star watermark detected ({result['corner']})",
                "manipulated": True,
                "watermark_type": "gemini_star",
                "location": result.get("location"),
                "method": result.get("method", "unknown")
            }
        
        return {
            "gemini_detected": False,
            "confidence": 0.0,
            "reason": "No Gemini watermark detected",
            "manipulated": False,
            "watermark_type": "none",
            "location": None
        }
    
    def _prepare_image(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(image, Image.Image):
            img = np.array(image.convert('RGB'))
        else:
            img = image.copy()
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return img
    
    def _detect_gemini_star(self, image: np.ndarray) -> Dict[str, Any]:
        h, w = image.shape[:2]
        ratio = self.thresholds["corner_ratio"]
        
        corners = [
            ("bottom_right", image[int(h*(1-ratio)):, int(w*(1-ratio)):], (int(w*(1-ratio)), int(h*(1-ratio)))),
            ("bottom_left", image[int(h*(1-ratio)):, :int(w*ratio)], (0, int(h*(1-ratio)))),
        ]
        
        best = {"detected": False, "confidence": 0.0, "corner": None, "location": None, "method": None}
        
        for corner_name, corner_img, offset in corners:
            result = self._analyze_corner_adaptive(corner_img, corner_name)
            if result["confidence"] > best["confidence"]:
                if result.get("location"):
                    x, y, ww, hh = result["location"]
                    result["location"] = (x + offset[0], y + offset[1], ww, hh)
                best = result
        
        best["detected"] = best["confidence"] >= self.thresholds["min_confidence"]
        return best
    
    def _analyze_corner_adaptive(self, corner: np.ndarray, corner_name: str) -> Dict[str, Any]:
        if corner.size == 0 or corner.shape[0] < 20 or corner.shape[1] < 20:
            return {"detected": False, "confidence": 0.0, "corner": corner_name, "location": None}
        
        gray = cv2.cvtColor(corner, cv2.COLOR_RGB2GRAY)
        mean_val = np.mean(gray)
        
        if mean_val < 80:
            result = self._detect_light_on_dark(gray, corner_name)
        elif mean_val > 180:
            result = self._detect_dark_on_light(gray, corner_name)
        else:
            result1 = self._detect_light_on_dark(gray, corner_name)
            result2 = self._detect_dark_on_light(gray, corner_name)
            result = result1 if result1["confidence"] > result2["confidence"] else result2
        
        return result
    
    def _detect_light_on_dark(self, gray: np.ndarray, corner_name: str) -> Dict[str, Any]:
        best_result = {"detected": False, "confidence": 0.0, "corner": corner_name, "location": None, "method": "light_on_dark"}
        
        for thresh in [90, 100, 110, 120, 130, 140, 150]:
            mask = (gray > thresh).astype(np.uint8) * 255
            
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            result = self._find_star_shape(mask, gray, corner_name, "light_on_dark")
            if result["confidence"] > best_result["confidence"]:
                best_result = result
        
        return best_result
    
    def _detect_dark_on_light(self, gray: np.ndarray, corner_name: str) -> Dict[str, Any]:
        best_result = {"detected": False, "confidence": 0.0, "corner": corner_name, "location": None, "method": "dark_on_light"}
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        for thresh in [170, 180, 190, 200, 210, 220]:
            mask = (blurred > thresh).astype(np.uint8) * 255
            
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            result = self._find_star_shape(mask, gray, corner_name, "dark_on_light")
            if result["confidence"] > best_result["confidence"]:
                best_result = result
        
        return best_result
    
    def _find_star_shape(self, mask: np.ndarray, gray: np.ndarray, corner_name: str, method: str) -> Dict[str, Any]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best = {"detected": False, "confidence": 0.0, "corner": corner_name, "location": None, "method": method}
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 150 or area > 12000:
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            x, y, bw, bh = cv2.boundingRect(contour)
            aspect = bw / bh if bh > 0 else 0
            
            is_gemini_star = (
                0.30 < circularity < 0.65 and
                0.50 < solidity < 0.88 and
                0.75 < aspect < 1.35 and
                bw >= 18 and bh >= 18
            )
            
            if is_gemini_star:
                base_score = 0.45
                
                if 0.38 < circularity < 0.55:
                    base_score += 0.15
                elif 0.30 < circularity < 0.65:
                    base_score += 0.08
                
                if 0.60 < solidity < 0.78:
                    base_score += 0.15
                elif 0.50 < solidity < 0.88:
                    base_score += 0.08
                
                if 0.90 < aspect < 1.12:
                    base_score += 0.10
                
                symmetry = self._check_symmetry(gray, (x, y, bw, bh))
                if symmetry > 0.60:
                    base_score += 0.20
                elif symmetry > 0.45:
                    base_score += 0.10
                
                if self._has_four_points(contour):
                    base_score += 0.15
                
                if base_score > best["confidence"]:
                    best = {
                        "detected": False,
                        "confidence": min(1.0, base_score),
                        "corner": corner_name,
                        "location": (x, y, bw, bh),
                        "method": method
                    }
        
        return best
    
    def _check_symmetry(self, gray: np.ndarray, location: Tuple) -> float:
        if location is None:
            return 0.0
        
        x, y, w, h = location
        cx, cy = x + w // 2, y + h // 2
        half = max(w, h) // 2 + 3
        
        h_img, w_img = gray.shape
        if cx - half < 0 or cx + half >= w_img or cy - half < 0 or cy + half >= h_img:
            return 0.0
        
        roi = gray[cy - half:cy + half, cx - half:cx + half].astype(np.float32)
        if roi.size == 0 or roi.shape[0] < 8:
            return 0.0
        
        rot90 = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
        rot180 = cv2.rotate(roi, cv2.ROTATE_180)
        
        min_size = min(roi.shape[0], roi.shape[1], rot90.shape[0], rot90.shape[1])
        roi = roi[:min_size, :min_size]
        rot90 = rot90[:min_size, :min_size]
        rot180 = rot180[:min_size, :min_size]
        
        diff90 = np.mean(np.abs(roi - rot90)) / 255
        diff180 = np.mean(np.abs(roi - rot180)) / 255
        
        return max(0, 1 - (diff90 + diff180) / 2 * 2.5)
    
    def _has_four_points(self, contour: np.ndarray) -> bool:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return 6 <= len(approx) <= 12
    
    def _verify_star_pattern(self, gray: np.ndarray, location: Tuple) -> bool:
        if location is None:
            return False
        
        x, y, w, h = location
        cx, cy = x + w // 2, y + h // 2
        radius = max(w, h) // 2
        
        h_img, w_img = gray.shape
        if cx - radius < 0 or cx + radius >= w_img or cy - radius < 0 or cy + radius >= h_img:
            return False
        
        center_val = gray[cy, cx] if 0 <= cy < h_img and 0 <= cx < w_img else 0
        edge_vals = []
        
        for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
            rad = np.radians(angle)
            ex = int(cx + radius * 0.8 * np.cos(rad))
            ey = int(cy + radius * 0.8 * np.sin(rad))
            if 0 <= ey < h_img and 0 <= ex < w_img:
                edge_vals.append(gray[ey, ex])
        
        if len(edge_vals) < 4:
            return False
        
        return center_val > np.mean(edge_vals) + 10
    
    def analyze_batch(self, images: List[Union[np.ndarray, Image.Image]]) -> List[Dict[str, Any]]:
        return [self.detect(img) for img in images]


    
    def __init__(self, sensitivity: str = "balanced"):
        self.sensitivity = sensitivity
        self.thresholds = {
            "conservative": {"min_confidence": 0.85, "corner_ratio": 0.12},
            "balanced": {"min_confidence": 0.75, "corner_ratio": 0.15},
            "aggressive": {"min_confidence": 0.60, "corner_ratio": 0.18}
        }.get(sensitivity, {"min_confidence": 0.75, "corner_ratio": 0.15})
    
    def detectar_watermark(self, image: Union[np.ndarray, Image.Image]) -> Dict[str, Any]:
        return self.detect(image)
    
    def detect(self, image: Union[np.ndarray, Image.Image]) -> Dict[str, Any]:
        img = self._prepare_image(image)
        result = self._detect_gemini_star(img)
        
        if result["detected"]:
            return {
                "gemini_detected": True,
                "confidence": round(result["confidence"], 2),
                "reason": f"Gemini star watermark detected ({result['corner']})",
                "manipulated": True,
                "watermark_type": "gemini_star",
                "location": result.get("location")
            }
        
        return {
            "gemini_detected": False,
            "confidence": 0.0,
            "reason": "No Gemini watermark detected",
            "manipulated": False,
            "watermark_type": "none",
            "location": None
        }
    
    def _prepare_image(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(image, Image.Image):
            img = np.array(image.convert('RGB'))
        else:
            img = image.copy()
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return img
    
    def _detect_gemini_star(self, image: np.ndarray) -> Dict[str, Any]:
        h, w = image.shape[:2]
        ratio = self.thresholds["corner_ratio"]
        
        corners = [
            ("bottom_right", image[int(h*(1-ratio)):, int(w*(1-ratio)):], (int(w*(1-ratio)), int(h*(1-ratio)))),
            ("bottom_left", image[int(h*(1-ratio)):, :int(w*ratio)], (0, int(h*(1-ratio)))),
        ]
        
        best = {"detected": False, "confidence": 0.0, "corner": None, "location": None}
        
        for corner_name, corner_img, offset in corners:
            result = self._analyze_corner(corner_img, corner_name)
            if result["confidence"] > best["confidence"]:
                if result.get("location"):
                    x, y, ww, hh = result["location"]
                    result["location"] = (x + offset[0], y + offset[1], ww, hh)
                best = result
        
        best["detected"] = best["confidence"] >= self.thresholds["min_confidence"]
        return best
    
    def _analyze_corner(self, corner: np.ndarray, corner_name: str) -> Dict[str, Any]:
        if corner.size == 0 or corner.shape[0] < 20 or corner.shape[1] < 20:
            return {"detected": False, "confidence": 0.0, "corner": corner_name, "location": None}
        
        gray = cv2.cvtColor(corner, cv2.COLOR_RGB2GRAY)
        star_match = self._find_gemini_star_shape(gray)
        
        if star_match is None:
            return {"detected": False, "confidence": 0.0, "corner": corner_name, "location": None}
        
        total_score = star_match["base_score"]
        
        symmetry = self._check_symmetry(gray, star_match["location"])
        if symmetry > 0.65:
            total_score += 0.25
        elif symmetry > 0.50:
            total_score += 0.15
        
        if self._verify_star_pattern(gray, star_match["location"]):
            total_score += 0.20
        
        return {
            "detected": False,
            "confidence": min(1.0, total_score),
            "corner": corner_name,
            "location": star_match["location"]
        }
    
    def _find_gemini_star_shape(self, gray: np.ndarray) -> Optional[Dict]:
        for thresh in [140, 150, 160, 170, 180]:
            mask = (gray > thresh).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 200 or area > 10000:
                    continue
                
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                
                circularity = 4 * np.pi * area / (perimeter ** 2)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                x, y, bw, bh = cv2.boundingRect(contour)
                aspect = bw / bh if bh > 0 else 0
                
                is_gemini_star = (
                    0.38 < circularity < 0.55 and
                    0.60 < solidity < 0.78 and
                    0.85 < aspect < 1.18 and
                    bw >= 25 and bh >= 25
                )
                
                if is_gemini_star:
                    base_score = 0.50
                    if 0.42 < circularity < 0.50:
                        base_score += 0.10
                    if 0.65 < solidity < 0.73:
                        base_score += 0.10
                    if 0.92 < aspect < 1.08:
                        base_score += 0.05
                    
                    return {
                        "contour": contour,
                        "location": (x, y, bw, bh),
                        "circularity": circularity,
                        "solidity": solidity,
                        "area": area,
                        "base_score": base_score
                    }
        return None
    
    def _check_symmetry(self, gray: np.ndarray, location: Tuple) -> float:
        if location is None:
            return 0.0
        
        x, y, w, h = location
        cx, cy = x + w // 2, y + h // 2
        half = max(w, h) // 2 + 3
        
        h_img, w_img = gray.shape
        if cx - half < 0 or cx + half >= w_img or cy - half < 0 or cy + half >= h_img:
            return 0.0
        
        roi = gray[cy - half:cy + half, cx - half:cx + half].astype(np.float32)
        if roi.size == 0 or roi.shape[0] < 8:
            return 0.0
        
        rot90 = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
        rot180 = cv2.rotate(roi, cv2.ROTATE_180)
        
        min_size = min(roi.shape[0], roi.shape[1], rot90.shape[0], rot90.shape[1])
        roi = roi[:min_size, :min_size]
        rot90 = rot90[:min_size, :min_size]
        rot180 = rot180[:min_size, :min_size]
        
        diff90 = np.mean(np.abs(roi - rot90)) / 255
        diff180 = np.mean(np.abs(roi - rot180)) / 255
        
        return max(0, 1 - (diff90 + diff180) / 2 * 3)
    
    def _verify_star_pattern(self, gray: np.ndarray, location: Tuple) -> bool:
        if location is None:
            return False
        
        x, y, w, h = location
        cx, cy = x + w // 2, y + h // 2
        radius = max(w, h) // 2
        
        h_img, w_img = gray.shape
        if cx - radius < 0 or cx + radius >= w_img or cy - radius < 0 or cy + radius >= h_img:
            return False
        
        center_val = gray[cy, cx] if 0 <= cy < h_img and 0 <= cx < w_img else 0
        edge_vals = []
        
        for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
            rad = np.radians(angle)
            ex = int(cx + radius * 0.8 * np.cos(rad))
            ey = int(cy + radius * 0.8 * np.sin(rad))
            if 0 <= ey < h_img and 0 <= ex < w_img:
                edge_vals.append(gray[ey, ex])
        
        if len(edge_vals) < 4:
            return False
        
        return center_val > np.mean(edge_vals) + 15
    
    def analyze_batch(self, images: List[Union[np.ndarray, Image.Image]]) -> List[Dict[str, Any]]:
        return [self.detect(img) for img in images]


class CarDetector:
    
    def __init__(self):
        self.car_cascade = None
        self.cascade_available = False
    
    def detect(self, image: np.ndarray) -> Tuple[bool, float, List]:
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape
        indicators = 0
        total_checks = 4
        
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
        
        if lines is not None:
            horizontal_lines = sum(1 for l in lines if abs(np.arctan2(l[0][3]-l[0][1], l[0][2]-l[0][0]) * 180 / np.pi) < 15 or abs(np.arctan2(l[0][3]-l[0][1], l[0][2]-l[0][0]) * 180 / np.pi) > 165)
            vertical_lines = sum(1 for l in lines if 75 < abs(np.arctan2(l[0][3]-l[0][1], l[0][2]-l[0][0]) * 180 / np.pi) < 105)
            if horizontal_lines > 5 and vertical_lines > 3:
                indicators += 1
        
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            metallic_mask = cv2.inRange(hsv, np.array([0, 0, 100]), np.array([180, 50, 200]))
            if np.mean(metallic_mask > 0) > 0.15:
                indicators += 1
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_contours = [c for c in contours if cv2.contourArea(c) > (h * w * 0.01)]
        for contour in large_contours[:5]:
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect = cw / ch if ch > 0 else 0
            if 1.5 < aspect < 4.0:
                indicators += 1
                break
        
        lower_third = gray[int(h*0.6):, :]
        if lower_third.size > 0:
            circles = cv2.HoughCircles(lower_third, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=30, minRadius=10, maxRadius=100)
            if circles is not None and len(circles[0]) >= 2:
                indicators += 1
        
        confidence = indicators / total_checks
        return confidence >= 0.5, confidence, []


class GlassDetector:
    
    def detect(self, image: np.ndarray) -> Tuple[bool, float, str]:
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        else:
            gray = image.copy()
            hsv = None
        
        indicators = 0
        total_checks = 4
        h, w = gray.shape
        
        block_size = 32
        low_texture_blocks = sum(1 for i in range(0, h - block_size, block_size) 
                                 for j in range(0, w - block_size, block_size) 
                                 if np.std(gray[i:i+block_size, j:j+block_size]) < 20)
        total_blocks = max(1, ((h - block_size) // block_size) * ((w - block_size) // block_size))
        if low_texture_blocks / total_blocks > 0.3:
            indicators += 1
        
        if hsv is not None:
            low_sat_ratio = np.mean(hsv[:, :, 1] < 40)
            if low_sat_ratio > 0.4:
                indicators += 1
        
        _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        bright_ratio = np.mean(bright > 0)
        if 0.05 < bright_ratio < 0.4:
            indicators += 1
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        if np.mean(magnitude < 10) > 0.5:
            indicators += 1
        
        confidence = indicators / total_checks
        is_glass = confidence >= 0.5
        glass_type = "window" if bright_ratio > 0.15 else "dark_glass" if is_glass else "unknown"
        
        return is_glass, confidence, glass_type


class ReflectionDetector:
    
    def __init__(self, logger: AnalysisLogger = None):
        self.logger = logger or AnalysisLogger()
    
    def detect(self, image: np.ndarray) -> Tuple[float, bool, Dict]:
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1]
        else:
            gray = image
            saturation = np.zeros_like(gray)
        
        _, bright = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
        _, low_sat = cv2.threshold(saturation, 50, 255, cv2.THRESH_BINARY_INV)
        reflection_mask = cv2.bitwise_and(bright, low_sat)
        percent = np.mean(reflection_mask > 0)
        
        contours, _ = cv2.findContours(reflection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        reflection_info = {
            "percent": percent,
            "num_regions": len(contours),
            "is_specular": False,
            "is_diffuse": False
        }
        
        if len(contours) > 0:
            areas = [cv2.contourArea(c) for c in contours]
            max_area = max(areas)
            total_area = sum(areas)
            if max_area > total_area * 0.5:
                reflection_info["is_specular"] = True
            else:
                reflection_info["is_diffuse"] = True
        
        is_significant = percent > 0.10
        
        if self.logger:
            self.logger.log("REFLECTION", f"Reflexo: {percent*100:.1f}%", reflection_info)
        
        return percent, is_significant, reflection_info


class SmoothSurfaceDetector:
    
    def __init__(self, logger: AnalysisLogger = None):
        self.logger = logger or AnalysisLogger()
    
    def detect(self, image: np.ndarray) -> Tuple[float, bool, str]:
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        block_size = 32
        h, w = gray.shape
        block_stds = []
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                block_stds.append(np.std(block))
        
        mean_std = np.mean(block_stds) if block_stds else 50
        std_of_stds = np.std(block_stds) if block_stds else 50
        
        if mean_std < 15 and std_of_stds < 10:
            is_smooth, smooth_percent, surface_type = True, 0.9, "glass"
        elif mean_std < 25 and std_of_stds < 15:
            is_smooth, smooth_percent, surface_type = True, 0.7, "painted_wall"
        elif mean_std < 35:
            is_smooth, smooth_percent, surface_type = True, 0.5, "semi_smooth"
        else:
            is_smooth, smooth_percent, surface_type = False, 0.0, "textured"
        
        if self.logger:
            self.logger.log("SMOOTH", f"Superfície: {surface_type}", {
                "smooth_percent": round(smooth_percent, 2),
                "surface_type": surface_type,
                "mean_std": round(mean_std, 2)
            })
        
        return smooth_percent, is_smooth, surface_type


class DocumentDetector:
    
    def __init__(self, logger: AnalysisLogger = None):
        self.logger = logger or AnalysisLogger()
    
    def detect(self, image: np.ndarray) -> Tuple[bool, float, List[Dict]]:
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        h, w = image.shape[:2]
        total_area = h * w
        
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower_white = np.array([0, 0, 195])
        upper_white = np.array([180, 50, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        kernel = np.ones((9, 9), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        documents = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            area_ratio = area / total_area
            
            if area_ratio < 0.005 or area_ratio > 0.6:
                continue
            
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect_ratio = max(cw, ch) / min(cw, ch) if min(cw, ch) > 0 else 0
            
            if 1.0 <= aspect_ratio <= 4.0:
                documents.append({
                    "bbox": (x, y, cw, ch),
                    "area_percent": area_ratio * 100
                })
        
        has_document = len(documents) > 0
        confidence = min(1.0, len(documents) * 0.3 + (0.5 if has_document else 0))
        
        if self.logger:
            self.logger.log("DOCUMENT", f"Documentos: {len(documents)}", {
                "count": len(documents),
                "has_document": has_document
            })
        
        return has_document, confidence, documents


class SceneClassifier:
    
    def __init__(self, logger: AnalysisLogger = None):
        self.logger = logger or AnalysisLogger()
        self.car_detector = CarDetector()
        self.glass_detector = GlassDetector()
        self.reflection_detector = ReflectionDetector(logger)
        self.smooth_detector = SmoothSurfaceDetector(logger)
        self.document_detector = DocumentDetector(logger)
    
    def classify(self, image: np.ndarray) -> Tuple[SceneType, Dict[str, Any]]:
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        has_car, car_conf, _ = self.car_detector.detect(image)
        has_glass, glass_conf, glass_type = self.glass_detector.detect(image)
        reflection_pct, has_reflection, reflection_info = self.reflection_detector.detect(image)
        smooth_pct, is_smooth, surface_type = self.smooth_detector.detect(image)
        has_document, doc_conf, documents = self.document_detector.detect(image)
        
        detection_info = {
            "car": {"detected": has_car, "confidence": car_conf},
            "glass": {"detected": has_glass, "confidence": glass_conf, "type": glass_type},
            "reflection": {"detected": has_reflection, "percent": reflection_pct, "info": reflection_info},
            "smooth": {"detected": is_smooth, "percent": smooth_pct, "type": surface_type},
            "document": {"detected": has_document, "confidence": doc_conf, "count": len(documents)}
        }
        
        scene_type = self._determine_scene_type(
            has_car, has_glass, has_reflection, is_smooth, has_document, surface_type
        )
        
        self.logger.log("SCENE_CLASSIFICATION", f"Cena: {scene_type.value}", detection_info)
        
        return scene_type, detection_info
    
    def _determine_scene_type(self, has_car: bool, has_glass: bool, 
                              has_reflection: bool, is_smooth: bool,
                              has_document: bool, surface_type: str) -> SceneType:
        
        is_wall = surface_type in ["painted_wall", "semi_smooth"]
        
        if has_car and has_glass and has_reflection:
            return SceneType.CAR_GLASS_REFLECTION
        if has_car and has_glass:
            return SceneType.CAR_GLASS
        if has_car and has_reflection:
            return SceneType.CAR_REFLECTION
        if has_car and is_wall:
            return SceneType.CAR_SMOOTH_WALL
        if has_car and has_document:
            return SceneType.CAR_DOCUMENT
        if has_glass and has_reflection:
            return SceneType.GLASS_REFLECTION
        if has_glass and is_wall:
            return SceneType.GLASS_WALL
        if has_glass and has_document:
            return SceneType.GLASS_DOCUMENT
        if has_car:
            return SceneType.CAR
        if has_glass:
            return SceneType.GLASS
        if has_document:
            return SceneType.DOCUMENT
        if is_smooth:
            return SceneType.SMOOTH_SURFACE
        
        return SceneType.UNKNOWN


class TextureAnalyzer:
    
    def __init__(self, P: int = 8, R: int = 1, block_size: int = 16, 
                 threshold: float = 0.45, logger: AnalysisLogger = None,
                 use_clahe: bool = False, clahe_clip_limit: float = 2.0):
        self.P = P
        self.R = R
        self.block_size = block_size
        self.threshold = threshold
        self.logger = logger or AnalysisLogger()
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
    
    def analyze_image(self, image: np.ndarray, use_clahe: bool = None) -> Dict[str, Any]:
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        clahe_enabled = use_clahe if use_clahe is not None else self.use_clahe
        if clahe_enabled:
            gray = apply_clahe(gray, self.clahe_clip_limit)
        
        lbp = local_binary_pattern(gray, self.P, self.R, method="uniform")
        
        height, width = lbp.shape
        rows = max(1, height // self.block_size)
        cols = max(1, width // self.block_size)
        
        variance_map = np.zeros((rows, cols))
        entropy_map = np.zeros((rows, cols))
        uniformity_map = np.zeros((rows, cols))
        
        for i in range(0, height - self.block_size + 1, self.block_size):
            for j in range(0, width - self.block_size + 1, self.block_size):
                block = lbp[i:i+self.block_size, j:j+self.block_size]
                
                hist, _ = np.histogram(block, bins=10, range=(0, 10))
                hist = hist.astype("float") / (hist.sum() + 1e-7)
                
                block_entropy = entropy(hist)
                max_entropy = np.log(10)
                norm_entropy = block_entropy / max_entropy if max_entropy > 0 else 0
                
                block_variance = np.var(block) / 255.0
                
                max_hist_value = np.max(hist)
                uniformity_penalty = 1.0 - max_hist_value
                
                row_idx = i // self.block_size
                col_idx = j // self.block_size
                
                if row_idx < rows and col_idx < cols:
                    variance_map[row_idx, col_idx] = block_variance
                    entropy_map[row_idx, col_idx] = norm_entropy
                    uniformity_map[row_idx, col_idx] = uniformity_penalty
        
        naturalness_map = (entropy_map * 0.30 + 
                          variance_map * 0.40 + 
                          uniformity_map * 0.30)
        
        suspicious_mask = naturalness_map < self.threshold
        mean_naturalness = np.mean(naturalness_map)
        suspicious_ratio = np.mean(suspicious_mask)
        
        if suspicious_ratio > 0.05:
            penalty_factor = 1.0 - (suspicious_ratio * 1.1)
        else:
            penalty_factor = 1.0 - (suspicious_ratio * 0.5)
        
        penalty_factor = max(0.4, penalty_factor)
        
        score = int(mean_naturalness * penalty_factor * 100)
        score = max(0, min(100, score))
        
        norm_for_display = cv2.normalize(naturalness_map, None, 0, 1, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap((norm_for_display * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
        
        visual_report = cv2.addWeighted(image, 0.6, heatmap_resized, 0.4, 0)
        
        self.logger.log("TEXTURE", f"Score: {score}/100 (CLAHE: {clahe_enabled})", {
            "score": score,
            "suspicious_ratio": round(suspicious_ratio, 3),
            "mean_naturalness": round(mean_naturalness, 3),
            "clahe": clahe_enabled
        })
        
        return {
            "score": score,
            "suspicious_ratio": suspicious_ratio,
            "visual_report": visual_report,
            "heatmap": heatmap_resized,
            "naturalness_map": naturalness_map,
            "clahe_used": clahe_enabled
        }


class EdgeAnalyzer:
    
    def __init__(self, clahe_clip_limit: float = 2.0, logger: AnalysisLogger = None,
                 use_clahe: bool = True):
        self.clahe_clip_limit = clahe_clip_limit
        self.logger = logger or AnalysisLogger()
        self.use_clahe = use_clahe
    
    def analyze_image(self, image: np.ndarray, use_clahe: bool = None) -> Dict[str, Any]:
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        clahe_enabled = use_clahe if use_clahe is not None else self.use_clahe
        if clahe_enabled:
            gray = apply_clahe(gray, self.clahe_clip_limit)
        
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        direction = np.arctan2(gradient_y, gradient_x)
        
        block_size = 32
        height, width = gray.shape
        coherence_scores = []
        
        for i in range(0, height - block_size, block_size):
            for j in range(0, width - block_size, block_size):
                block_dir = direction[i:i+block_size, j:j+block_size]
                block_mag = magnitude[i:i+block_size, j:j+block_size]
                
                mask = block_mag > np.percentile(block_mag, 70)
                if np.sum(mask) > 10:
                    dirs = block_dir[mask]
                    coherence = 1.0 - (np.std(dirs) / np.pi)
                    coherence_scores.append(max(0, coherence))
        
        mean_coherence = np.mean(coherence_scores) if coherence_scores else 0.5
        edge_density = np.mean(magnitude > 30)
        
        score = int((mean_coherence * 0.6 + edge_density * 0.4) * 100)
        score = max(0, min(100, score))
        
        self.logger.log("EDGE", f"Score: {score}/100 (CLAHE: {clahe_enabled})", {
            "score": score,
            "mean_coherence": round(mean_coherence, 3),
            "edge_density": round(edge_density, 3),
            "clahe": clahe_enabled
        })
        
        return {
            "score": score, 
            "coherence": mean_coherence, 
            "density": edge_density,
            "clahe_used": clahe_enabled
        }


class NoiseAnalyzer:
    
    def __init__(self, block_size: int = 32, clahe_clip_limit: float = 2.0, 
                 logger: AnalysisLogger = None, use_clahe: bool = True):
        self.block_size = block_size
        self.clahe_clip_limit = clahe_clip_limit
        self.logger = logger or AnalysisLogger()
        self.use_clahe = use_clahe
    
    def analyze_image(self, image: np.ndarray, use_clahe: bool = None) -> Dict[str, Any]:
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        clahe_enabled = use_clahe if use_clahe is not None else self.use_clahe
        if clahe_enabled:
            gray = apply_clahe(gray, self.clahe_clip_limit)
        
        height, width = gray.shape
        noise_levels = []
        
        for i in range(0, height - self.block_size, self.block_size):
            for j in range(0, width - self.block_size, self.block_size):
                block = gray[i:i+self.block_size, j:j+self.block_size]
                
                try:
                    sigma = estimate_sigma(block, average_sigmas=True, channel_axis=None)
                    noise_levels.append(sigma)
                except Exception:
                    noise_levels.append(np.std(block))
        
        if not noise_levels:
            return {"score": 50, "noise_cv": 0.5, "clahe_used": clahe_enabled}
        
        noise_mean = np.mean(noise_levels)
        noise_std = np.std(noise_levels)
        
        noise_cv = noise_std / noise_mean if noise_mean > 0 else 0
        
        if noise_cv < 0.2:
            score = 60
        elif noise_cv <= 0.6:
            score = int(80 - (noise_cv - 0.2) * 50)
        else:
            score = max(20, int(50 - (noise_cv - 0.6) * 60))
        
        score = max(0, min(100, score))
        
        self.logger.log("NOISE", f"Score: {score}/100 (CLAHE: {clahe_enabled})", {
            "score": score,
            "noise_cv": round(noise_cv, 3),
            "noise_mean": round(noise_mean, 3),
            "clahe": clahe_enabled
        })
        
        return {
            "score": score, 
            "noise_cv": noise_cv, 
            "noise_mean": noise_mean,
            "clahe_used": clahe_enabled
        }


class LightingAnalyzer:
    
    def __init__(self, clahe_clip_limit: float = 2.0, logger: AnalysisLogger = None,
                 use_clahe: bool = True):
        self.clahe_clip_limit = clahe_clip_limit
        self.logger = logger or AnalysisLogger()
        self.use_clahe = use_clahe
    
    def analyze_image(self, image: np.ndarray, use_clahe: bool = None) -> Dict[str, Any]:
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        clahe_enabled = use_clahe if use_clahe is not None else self.use_clahe
        if clahe_enabled:
            gray = apply_clahe(gray, self.clahe_clip_limit)
        
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        smoothness = 1.0 - (np.std(grad_mag) / (np.mean(grad_mag) + 1e-7))
        smoothness = max(0, min(1, smoothness))
        
        edges = cv2.Canny(blurred, 30, 100)
        abrupt_transitions = np.mean(edges > 0)
        
        base_score = smoothness * 25
        penalty = abrupt_transitions * 10
        score = int(max(0, min(30, base_score - penalty + 5)))
        
        self.logger.log("LIGHTING", f"Score: {score}/30 (CLAHE: {clahe_enabled})", {
            "score": score,
            "smoothness": round(smoothness, 3),
            "abrupt_transitions": round(abrupt_transitions, 3),
            "clahe": clahe_enabled
        })
        
        return {
            "score": score, 
            "smoothness": smoothness,
            "clahe_used": clahe_enabled
        }


class CarGlassSpecificAnalyzer:
    
    def __init__(self, logger: AnalysisLogger = None):
        self.logger = logger or AnalysisLogger()
        self.th_manipulated = 50
    
    def analyze_artifacts(self, gray: np.ndarray) -> float:
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        diff = np.abs(gray.astype(float) - blurred.astype(float))
        block_size = 8
        h, w = gray.shape
        block_vars = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = diff[i:i+block_size, j:j+block_size]
                block_vars.append(np.var(block))
        return np.mean(block_vars) if block_vars else 0
    
    def analyze_edge_kurtosis(self, gray: np.ndarray) -> float:
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        return kurtosis(magnitude.flatten())
    
    def analyze_noise_std(self, gray: np.ndarray) -> float:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray.astype(float) - blurred.astype(float)
        return np.std(noise)
    
    def analyze_gradient_uniformity(self, gray: np.ndarray) -> float:
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        angles = np.arctan2(gy, gx) * 180 / np.pi
        hist, _ = np.histogram(angles.flatten(), bins=36, range=(-180, 180))
        hist = hist / (hist.sum() + 1e-7)
        return np.max(hist)
    
    def analyze_saturation_variation(self, image: np.ndarray) -> float:
        if len(image.shape) != 3:
            return 0
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        return np.std(hsv[:,:,1])
    
    def analyze_noise_kurtosis(self, gray: np.ndarray) -> float:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray.astype(float) - blurred.astype(float)
        return kurtosis(noise.flatten())
    
    def analyze(self, image: np.ndarray) -> Dict[str, Any]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        artifact = self.analyze_artifacts(gray)
        edge_kurt = self.analyze_edge_kurtosis(gray)
        noise_std = self.analyze_noise_std(gray)
        dir_uniform = self.analyze_gradient_uniformity(gray)
        sat_std = self.analyze_saturation_variation(image)
        noise_kurt = self.analyze_noise_kurtosis(gray)
        
        manipulation_score = 0
        artifact_norm = min(100, (artifact / 40) * 100)
        manipulation_score += artifact_norm * 0.30
        edge_norm = max(0, 100 - (edge_kurt / 35) * 100)
        manipulation_score += edge_norm * 0.20
        noise_std_norm = min(100, (noise_std / 15) * 100)
        manipulation_score += noise_std_norm * 0.15
        dir_norm = min(100, (dir_uniform / 0.25) * 100)
        manipulation_score += dir_norm * 0.15
        sat_norm = min(100, (sat_std / 60) * 100)
        manipulation_score += sat_norm * 0.10
        noise_kurt_norm = max(0, 100 - (noise_kurt / 25) * 100)
        manipulation_score += noise_kurt_norm * 0.10
        
        manipulation_score = min(100, max(0, manipulation_score))
        
        if manipulation_score >= self.th_manipulated:
            verdict = "MANIPULADA"
            confidence = int(60 + (manipulation_score - 50) * 0.7)
        else:
            verdict = "NATURAL"
            confidence = int(60 + (50 - manipulation_score) * 0.7)
        
        self.logger.log("CAR_GLASS_SPECIFIC", f"Score: {manipulation_score:.1f}, Veredicto: {verdict}", {
            "manipulation_score": round(manipulation_score, 1),
            "verdict": verdict,
            "artifact": round(artifact, 2),
            "edge_kurt": round(edge_kurt, 2),
            "noise_std": round(noise_std, 2),
            "dir_uniform": round(dir_uniform, 3),
            "sat_std": round(sat_std, 2),
            "noise_kurt": round(noise_kurt, 2)
        })
        
        return {
            "verdict": verdict,
            "confidence": min(95, confidence),
            "manipulation_score": round(manipulation_score, 1),
            "reason": f"Score de manipulação: {manipulation_score:.1f}/100",
            "metrics": {
                "artifact": round(artifact, 2),
                "edge_kurtosis": round(edge_kurt, 2),
                "noise_std": round(noise_std, 2),
                "gradient_uniformity": round(dir_uniform, 3),
                "saturation_std": round(sat_std, 2),
                "noise_kurtosis": round(noise_kurt, 2)
            }
        }


class MirrorGlass:
    
    def __init__(self, detection_mode: str = "Balanceado", manual_config: ManualConfig = None):
        self.logger = AnalysisLogger()
        self.detection_mode = detection_mode
        self.manual_config = manual_config or ManualConfig()
        
        clahe_cfg = self.manual_config.clahe_config or CLAHEConfig()
        
        self.scene_classifier = SceneClassifier(self.logger)
        self.texture_analyzer = TextureAnalyzer(
            logger=self.logger, 
            use_clahe=clahe_cfg.texture_clahe,
            clahe_clip_limit=clahe_cfg.clip_limit
        )
        self.edge_analyzer = EdgeAnalyzer(
            logger=self.logger, 
            use_clahe=clahe_cfg.edge_clahe,
            clahe_clip_limit=clahe_cfg.clip_limit
        )
        self.noise_analyzer = NoiseAnalyzer(
            logger=self.logger, 
            use_clahe=clahe_cfg.noise_clahe,
            clahe_clip_limit=clahe_cfg.clip_limit
        )
        self.lighting_analyzer = LightingAnalyzer(
            logger=self.logger, 
            use_clahe=clahe_cfg.lighting_clahe,
            clahe_clip_limit=clahe_cfg.clip_limit
        )
        self.reflection_detector = ReflectionDetector(logger=self.logger)
        self.smooth_detector = SmoothSurfaceDetector(logger=self.logger)
        self.car_glass_analyzer = CarGlassSpecificAnalyzer(logger=self.logger)
        
        watermark_sensitivity = {"Conservador": "conservative", "Agressivo": "aggressive"}.get(detection_mode, "balanced")
        self.watermark_detector = GeminiWatermarkDetector(sensitivity=watermark_sensitivity)
        
        self._base_thresholds = self._get_base_thresholds()
    
    def _get_base_thresholds(self) -> Dict:
        if self.detection_mode == "Conservador":
            return {
                "th_texture_manipulated": 40,
                "th_texture_natural": 65,
                "th_edge_manipulated": 30,
                "th_noise_manipulated": 30,
                "th_lighting_manipulated": 8,
                "th_final_suspicious": 45
            }
        elif self.detection_mode == "Agressivo":
            return {
                "th_texture_manipulated": 55,
                "th_texture_natural": 80,
                "th_edge_manipulated": 45,
                "th_noise_manipulated": 45,
                "th_lighting_manipulated": 12,
                "th_final_suspicious": 60
            }
        else:
            return {
                "th_texture_manipulated": 45,
                "th_texture_natural": 75,
                "th_edge_manipulated": 35,
                "th_noise_manipulated": 35,
                "th_lighting_manipulated": 8,
                "th_final_suspicious": 50
            }
    
    def analyze(self, image, show_logs: bool = False) -> Dict[str, Any]:
        self.logger = AnalysisLogger()
        self.car_glass_analyzer.logger = self.logger
        
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
        
        if self.manual_config.enable_watermark_detection:
            watermark_result = self.watermark_detector.detect(image_array)
            
            if watermark_result["gemini_detected"]:
                self.logger.log("WATERMARK", "Gemini watermark detected - MANIPULADA", watermark_result)
                
                result = {
                    "verdict": "MANIPULADA",
                    "confidence": int(watermark_result["confidence"] * 100),
                    "reason": watermark_result["reason"],
                    "main_score": 0,
                    "all_scores": {"watermark": watermark_result["confidence"] * 100},
                    "validation_chain": ["watermark"],
                    "phases_executed": 0,
                    "visual_report": None,
                    "heatmap": None,
                    "percent_suspicious": 100,
                    "detailed_reason": f"Watermark do Gemini detectado com {watermark_result['confidence']*100:.0f}% de confiança",
                    "logs": self.logger.get_logs(),
                    "scene_type": "watermark_detected",
                    "gemini_watermark": watermark_result,
                    "detection_info": {},
                    "weight_config": {},
                    "manual_config": {"auto_mode": self.manual_config.auto_mode, "watermark_detection": True}
                }
                
                if show_logs:
                    self._print_logs(result)
                
                return result
        
        if self.manual_config.auto_mode:
            scene_type, detection_info = self.scene_classifier.classify(image_array)
            
            if self.manual_config.force_scene_type:
                scene_type = self.manual_config.force_scene_type
            
            weight_config = WeightPresets.get_preset(scene_type)
        else:
            scene_type = SceneType.UNKNOWN
            detection_info = {}
            
            if self.manual_config.custom_weights:
                weight_config = self.manual_config.custom_weights
            else:
                weight_config = WeightConfig()
        
        if not self.manual_config.enable_boost_reflection:
            weight_config.boost_reflection = 0
        if not self.manual_config.enable_boost_smooth:
            weight_config.boost_smooth = 0
        if not self.manual_config.enable_boost_glass:
            weight_config.boost_glass = 0
        
        use_car_glass_specific = (
            self.manual_config.use_car_glass_specific and 
            scene_type == SceneType.CAR_GLASS
        )
        
        if use_car_glass_specific:
            self.logger.log("ANALYSIS_MODE", "Usando análise específica para CAR_GLASS", {"scene_type": scene_type.value})
            result = self._run_car_glass_analysis(image_array, scene_type, detection_info)
        else:
            result = self._run_analysis(image_array, weight_config, scene_type)
        
        clahe_cfg = self.manual_config.clahe_config or CLAHEConfig()
        
        result["scene_type"] = scene_type.value
        result["detection_info"] = detection_info
        result["weight_config"] = {
            "weight_texture": weight_config.weight_texture,
            "weight_edge": weight_config.weight_edge,
            "weight_noise": weight_config.weight_noise,
            "weight_lighting": weight_config.weight_lighting,
            "boost_reflection": weight_config.boost_reflection,
            "boost_smooth": weight_config.boost_smooth,
            "boost_glass": weight_config.boost_glass
        }
        result["manual_config"] = {
            "auto_mode": self.manual_config.auto_mode,
            "enable_texture": self.manual_config.enable_texture,
            "enable_edge": self.manual_config.enable_edge,
            "enable_noise": self.manual_config.enable_noise,
            "enable_lighting": self.manual_config.enable_lighting,
            "use_car_glass_specific": use_car_glass_specific,
            "watermark_detection": self.manual_config.enable_watermark_detection
        }
        result["clahe_status"] = {
            "texture": clahe_cfg.texture_clahe,
            "edge": clahe_cfg.edge_clahe,
            "noise": clahe_cfg.noise_clahe,
            "lighting": clahe_cfg.lighting_clahe
        }
        result["gemini_watermark"] = {"gemini_detected": False}
        
        if show_logs:
            self._print_logs(result)
        
        return result
    
    def _run_car_glass_analysis(self, image: np.ndarray, scene_type: SceneType, 
                                 detection_info: Dict) -> Dict[str, Any]:
        cg_result = self.car_glass_analyzer.analyze(image)
        texture_result = self.texture_analyzer.analyze_image(image)
        
        return {
            "verdict": cg_result["verdict"],
            "confidence": cg_result["confidence"],
            "reason": cg_result["reason"],
            "main_score": int(100 - cg_result["manipulation_score"]),
            "all_scores": {
                "car_glass_specific": cg_result["manipulation_score"],
                "texture": texture_result["score"]
            },
            "validation_chain": ["car_glass_specific"],
            "phases_executed": 1,
            "visual_report": texture_result.get("visual_report"),
            "heatmap": texture_result.get("heatmap"),
            "percent_suspicious": int(cg_result["manipulation_score"]),
            "detailed_reason": cg_result["reason"],
            "logs": self.logger.get_logs(),
            "car_glass_metrics": cg_result.get("metrics", {})
        }
    
    def _run_analysis(self, image: np.ndarray, config: WeightConfig, 
                      scene_type: SceneType) -> Dict[str, Any]:
        
        all_scores = {}
        validation_chain = []
        clahe_status = {}
        
        reflection_pct, has_reflection, reflection_info = self.reflection_detector.detect(image)
        smooth_pct, is_smooth, surface_type = self.smooth_detector.detect(image)
        
        all_scores['reflection'] = reflection_pct * 100
        all_scores['smooth'] = smooth_pct * 100
        
        reflection_percent = reflection_pct * 100
        
        texture_result = None
        texture_score = 50
        if self.manual_config.enable_texture:
            texture_result = self.texture_analyzer.analyze_image(image)
            texture_score = texture_result['score']
            all_scores['texture'] = texture_score
            validation_chain.append('texture')
            clahe_status['texture'] = texture_result.get('clahe_used', False)
        
        # NOVA REGRA: CAR_GLASS_REFLECTION com textura suspeita > 40%
        if texture_result is not None:
            texture_suspicious_percent = texture_result["suspicious_ratio"] * 100
            all_scores["texture_suspicious_percent"] = texture_suspicious_percent
            
            if scene_type == SceneType.CAR_GLASS_REFLECTION and texture_suspicious_percent <= 10 and reflection_percent > 20.99 and reflection_percent < 40 or texture_suspicious_percent < 1 and reflection_percent <= 18:
                self.logger.log("CAR_GLASS_TEXTURE_RULE", 
                    f"Regra de textura ativada: {texture_suspicious_percent:.1f} em CAR_GLASS_REFLECTION",
                    {
                        "texture_suspicious_percent": round(texture_suspicious_percent, 2),
                        "scene_type": scene_type.value,
                        "threshold": 40
                    })
                return self._build_response(
                    verdict="MANIPULADA",
                    confidence=90,
                    reason=f"Textura artificial suspeita na cena car_glass_reflection ({texture_suspicious_percent:.1f}%)",
                    main_score=int(texture_score),
                    all_scores=all_scores,
                    validation_chain=validation_chain,
                    phases_executed=1,
                    visual_report=texture_result.get('visual_report'),
                    heatmap=texture_result.get('heatmap'),
                    clahe_status=clahe_status
                )
        
        weighted_texture = texture_score * config.weight_texture
        
        if self.manual_config.enable_boost_reflection and has_reflection:
            weighted_texture += config.boost_reflection
        
        if self.manual_config.enable_boost_smooth and is_smooth:
            weighted_texture += config.boost_smooth
        
        force_continue_to_phase3 = (
            scene_type == SceneType.CAR_GLASS_REFLECTION and
            reflection_percent > 12 and
            reflection_percent < 25
        )
        
        if weighted_texture < config.th_texture_manipulated and not force_continue_to_phase3:
            return self._build_response(
                verdict="MANIPULADA",
                confidence=92,
                reason="Textura artificial detectada",
                main_score=int(weighted_texture),
                all_scores=all_scores,
                validation_chain=validation_chain,
                phases_executed=1,
                visual_report=texture_result['visual_report'] if texture_result else None,
                heatmap=texture_result['heatmap'] if texture_result else None,
                clahe_status=clahe_status
            )
        
        if weighted_texture > config.th_texture_natural and not force_continue_to_phase3:
            return self._build_response(
                verdict="NATURAL",
                confidence=85,
                reason="Textura natural confirmada",
                main_score=int(weighted_texture),
                all_scores=all_scores,
                validation_chain=validation_chain,
                phases_executed=1,
                visual_report=texture_result['visual_report'] if texture_result else None,
                heatmap=texture_result['heatmap'] if texture_result else None,
                clahe_status=clahe_status
            )
        
        edge_score = 50
        if self.manual_config.enable_edge:
            edge_result = self.edge_analyzer.analyze_image(image)
            edge_score = edge_result['score']
            all_scores['edge'] = edge_score
            validation_chain.append('edge')
            clahe_status['edge'] = edge_result.get('clahe_used', False)
        
        weighted_edge = edge_score * config.weight_edge
        if is_smooth:
            weighted_edge += 10
        
        if weighted_edge < config.th_edge_manipulated and not force_continue_to_phase3:
            return self._build_response(
                verdict="MANIPULADA",
                confidence=88,
                reason="Bordas artificiais detectadas",
                main_score=int((weighted_texture + weighted_edge) / 2),
                all_scores=all_scores,
                validation_chain=validation_chain,
                phases_executed=2,
                visual_report=texture_result['visual_report'] if texture_result else None,
                heatmap=texture_result['heatmap'] if texture_result else None,
                clahe_status=clahe_status
            )
        
        noise_score = 50
        if self.manual_config.enable_noise:
            noise_result = self.noise_analyzer.analyze_image(image)
            noise_score = noise_result['score']
            all_scores['noise'] = noise_score
            validation_chain.append('noise')
            clahe_status['noise'] = noise_result.get('clahe_used', False)
        
        weighted_noise = noise_score * config.weight_noise
        
        if self.manual_config.enable_boost_glass and is_smooth and reflection_pct > 0.10:
            weighted_noise += 15

        combined_manipulation_detected = False
        
        if scene_type == SceneType.CAR_GLASS_REFLECTION:
            combined_manipulation_detected = (
                noise_score >= 60 and
                reflection_percent < 19 and
                reflection_percent > 12 and
                texture_score > 20 and
                texture_score < 50
            )
            
        if combined_manipulation_detected:
            if scene_type == SceneType.CAR_GLASS_REFLECTION:
                self.logger.log("COMBINED_RULE", "Regra combinada ativada (car_glass_reflection): ruído>=60, reflexo 12-19%, textura>20", {
                    "noise_score": noise_score,
                    "reflection_percent": round(reflection_percent, 2),
                    "texture_score": texture_score,
                    "scene_type": scene_type.value
                })
                return self._build_response(
                    verdict="MANIPULADA",
                    confidence=90,
                    reason="Combinação suspeita: alto ruído, baixo reflexo e textura artificial",
                    main_score=int((weighted_texture + weighted_edge + weighted_noise) / 3),
                    all_scores=all_scores,
                    validation_chain=validation_chain,
                    phases_executed=3,
                    visual_report=texture_result['visual_report'] if texture_result else None,
                    heatmap=texture_result['heatmap'] if texture_result else None,
                    clahe_status=clahe_status
                )
            
        
        if weighted_noise < config.th_noise_manipulated:
            return self._build_response(
                verdict="MANIPULADA",
                confidence=85,
                reason="Ruído inconsistente detectado",
                main_score=int((weighted_texture + weighted_edge + weighted_noise) / 3),
                all_scores=all_scores,
                validation_chain=validation_chain,
                phases_executed=3,
                visual_report=texture_result['visual_report'] if texture_result else None,
                heatmap=texture_result['heatmap'] if texture_result else None,
                clahe_status=clahe_status
            )
        
        lighting_score = 15
        if self.manual_config.enable_lighting:
            lighting_result = self.lighting_analyzer.analyze_image(image)
            lighting_score = lighting_result['score']
            all_scores['lighting'] = lighting_score
            validation_chain.append('lighting')
            clahe_status['lighting'] = lighting_result.get('clahe_used', False)
        
        adjusted_lighting = lighting_score
        if self.manual_config.enable_boost_glass and is_smooth and reflection_pct > 0.10:
            adjusted_lighting += 5
        
        if adjusted_lighting < config.th_lighting_manipulated:
            return self._build_response(
                verdict="MANIPULADA",
                confidence=80,
                reason="Iluminação inconsistente detectada",
                main_score=int((weighted_texture + weighted_edge + weighted_noise) / 3),
                all_scores=all_scores,
                validation_chain=validation_chain,
                phases_executed=4,
                visual_report=texture_result['visual_report'] if texture_result else None,
                heatmap=texture_result['heatmap'] if texture_result else None,
                clahe_status=clahe_status
            )
        
        lighting_normalized = (lighting_score / 30) * 100
        
        weighted_final = (
            weighted_texture * 0.50 +
            weighted_edge * 0.25 +
            weighted_noise * 0.15 +
            lighting_normalized * 0.10
        )
        
        if self.manual_config.enable_boost_reflection and has_reflection:
            weighted_final += 5
        
        if weighted_final < config.th_final_suspicious:
            verdict = "SUSPEITA"
            confidence = 70
            reason = "Múltiplos indicadores ambíguos"
        else:
            verdict = "NATURAL"
            confidence = 75
            reason = "Passou por todas as validações"
        
        return self._build_response(
            verdict=verdict,
            confidence=confidence,
            reason=reason,
            main_score=int(weighted_final),
            all_scores=all_scores,
            validation_chain=validation_chain,
            phases_executed=4,
            visual_report=texture_result['visual_report'] if texture_result else None,
            heatmap=texture_result['heatmap'] if texture_result else None,
            clahe_status=clahe_status
        )
    
    def _build_response(self, verdict: str, confidence: int, reason: str,
                       main_score: int, all_scores: Dict, validation_chain: List,
                       phases_executed: int, visual_report: np.ndarray,
                       heatmap: np.ndarray, clahe_status: Dict = None) -> Dict[str, Any]:
        
        self.logger.log("VERDICT", f"{verdict} ({confidence}%)", {
            "verdict": verdict,
            "confidence": confidence,
            "reason": reason,
            "score": main_score,
            "phases": phases_executed
        })
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "reason": reason,
            "main_score": main_score,
            "all_scores": all_scores,
            "validation_chain": validation_chain,
            "phases_executed": phases_executed,
            "visual_report": visual_report,
            "heatmap": heatmap,
            "percent_suspicious": 100 - main_score,
            "detailed_reason": f"Score: {main_score}/100. {reason}. Fases: {phases_executed}",
            "logs": self.logger.get_logs(),
            "clahe_status": clahe_status or {}
        }
    
    def _print_logs(self, result: Dict):
        print("\n" + "="*60)
        print("LOGS DA ANÁLISE - MirrorGlass V7 (com Gemini Detection)")
        print("="*60)
        print(f"Cena detectada: {result.get('scene_type', 'unknown')}")
        print(f"Modo automático: {result['manual_config']['auto_mode']}")
        print(f"Watermark Gemini: {result.get('gemini_watermark', {}).get('gemini_detected', False)}")
        print(f"Análise CAR_GLASS específica: {result['manual_config'].get('use_car_glass_specific', False)}")
        print(f"CLAHE Status: {result.get('clahe_status', {})}")
        print("-"*60)
        for log in result['logs']:
            print(f"[{log['phase']}] {log['message']}")
        print("="*60 + "\n")


def create_default_manual_config() -> ManualConfig:
    return ManualConfig(
        auto_mode=True,
        enable_texture=True,
        enable_edge=True,
        enable_noise=True,
        enable_lighting=True,
        enable_boost_reflection=True,
        enable_boost_smooth=True,
        enable_boost_glass=True,
        enable_watermark_detection=True,
        use_car_glass_specific=True,
        clahe_config=CLAHEConfig()
    )


if __name__ == "__main__":
    print("\n" + "="*60)
    print("MirrorGlass V7 - Com Detecção de Watermark Gemini")
    print("="*60)
    
    config = create_default_manual_config()
    analyzer = MirrorGlass(detection_mode="Balanceado", manual_config=config)
    
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = analyzer.analyze(test_img)
    
    print(f"Verdict: {result['verdict']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Reason: {result['reason']}")
