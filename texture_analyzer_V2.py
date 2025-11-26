#!/usr/bin/env python3
"""
MirrorGlass V4.8 - Anti-Fake-Noise System (CORREÇÃO VISUAL)
==================================================

🚨 CORREÇÃO V4.8.1:
✅ SEMPRE gera visual_report e heatmap (mesmo em decisões rápidas)
✅ Usuário sempre vê a imagem analisada

Autor: MirrorGlass AI Detection System
Versão: 4.8.1 - Visual Fix
Data: 25/11/2025
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.restoration import estimate_sigma
from scipy.stats import entropy
from PIL import Image
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any


# ============================================================================
# SISTEMA DE LOGGING
# ============================================================================

class AnalysisLogger:
    """Sistema de logging detalhado para debug."""
    
    def __init__(self):
        self.logs = []
    
    def log(self, phase: str, message: str, data: Dict = None):
        """Adiciona log estruturado."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "message": message,
            "data": data or {}
        }
        self.logs.append(log_entry)
    
    def get_logs(self) -> List[Dict]:
        """Retorna todos os logs."""
        return self.logs
    
    def export_json(self) -> str:
        """Exporta logs como JSON."""
        return json.dumps(self.logs, indent=2, ensure_ascii=False)
    
    def print_summary(self):
        """Imprime resumo dos logs."""
        for log in self.logs:
            if log['phase'] in ['PERFECTION', 'REGIONAL', 'REFLECTION_DOMINANT', 'VERDICT', 'BOOST']:
                print(f"[{log['phase']}] {log['message']}")
                if log['data']:
                    for k, v in log['data'].items():
                        if isinstance(v, float):
                            print(f"  └─ {k}: {v:.3f}")
                        else:
                            print(f"  └─ {k}: {v}")


# ============================================================================
# DETECTOR DE REFLEXOS MELHORADO (V4.6 - CRÍTICO!)
# ============================================================================

class ReflectionMaskV46:
    """V4.6: ReflectionMask 3x mais inteligente."""
    
    def __init__(self, logger: AnalysisLogger = None):
        self.logger = logger or AnalysisLogger()
    
    def detect_reflection(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Detecta reflexos com 3 camadas de análise."""
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # CAMADA 1: BRILHO MODERADO
        _, bright_mask = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
        
        # CAMADA 2: BAIXA SATURAÇÃO
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1]
            _, low_sat_mask = cv2.threshold(saturation, 50, 255, cv2.THRESH_BINARY_INV)
        else:
            low_sat_mask = np.zeros_like(gray)
        
        # CAMADA 3: BAIXA VARIÂNCIA LOCAL
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        diff = cv2.absdiff(gray, blur)
        low_variance_mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY_INV)[1]
        
        # COMBINAR AS 3 CAMADAS
        reflection_mask = cv2.bitwise_and(bright_mask, low_sat_mask)
        reflection_mask = cv2.bitwise_and(reflection_mask, low_variance_mask)
        
        # Limpeza morfológica
        kernel = np.ones((9, 9), np.uint8)
        reflection_mask = cv2.morphologyEx(reflection_mask, cv2.MORPH_CLOSE, kernel)
        reflection_mask = cv2.dilate(reflection_mask, np.ones((5, 5), np.uint8), iterations=1)
        
        percent_reflection = np.mean(reflection_mask > 0)
        
        self.logger.log("REFLECTION", f"Reflexo detectado (3 camadas): {percent_reflection*100:.1f}%", {
            "percent": round(percent_reflection, 3)
        })
        
        return reflection_mask, percent_reflection
    
    def get_non_reflective_mask(self, image: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Retorna máscara de áreas válidas."""
        reflection_mask, percent_reflection = self.detect_reflection(image)
        non_reflective = cv2.bitwise_not(reflection_mask)
        percent_valid = 1.0 - percent_reflection
        return non_reflective, percent_valid, percent_reflection


# ============================================================================
# DETECTOR DE SUPERFÍCIES LISAS
# ============================================================================

class SmoothSurfaceDetector:
    """Detecta superfícies naturalmente lisas."""
    
    def __init__(self, logger: AnalysisLogger = None):
        self.logger = logger or AnalysisLogger()
        self.std_threshold = 18
        self.mean_std_threshold = 35
        self.saturation_threshold = 35
    
    def detect_smooth_surface(self, image: np.ndarray) -> Tuple[bool, float, str]:
        """Detecta se a imagem contém superfícies naturalmente lisas."""
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        
        block_size = 32
        block_stds = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                block_stds.append(np.std(block))
        
        std_of_stds = np.std(block_stds) if len(block_stds) > 0 else 0
        mean_std = np.mean(block_stds) if len(block_stds) > 0 else 0
        
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        strong_edges_percent = np.mean(magnitude > 50)
        
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            saturation_std = np.std(hsv[:, :, 1])
        else:
            saturation_std = 0
        
        is_smooth = False
        smooth_percent = 0
        surface_type = "textured"
        
        # Detectar VIDRO primeiro
        if mean_std < 10 and saturation_std < 15:
            is_smooth = True
            smooth_percent = 0.95
            surface_type = "smooth_glass"
        # Superfície pintada
        elif std_of_stds < self.std_threshold and mean_std < self.mean_std_threshold:
            is_smooth = True
            smooth_percent = 1.0 - (std_of_stds / self.std_threshold)
            surface_type = "smooth_painted"
        # Vidro/metal
        elif strong_edges_percent < 0.05 and mean_std < 28:
            is_smooth = True
            smooth_percent = 0.8
            surface_type = "smooth_glass"
        # Cor uniforme
        elif saturation_std < self.saturation_threshold and mean_std < 40:
            is_smooth = True
            smooth_percent = 0.7
            surface_type = "smooth_painted"
        
        self.logger.log("SmoothSurface", f"Detectado: {surface_type}", {
            "is_smooth": is_smooth,
            "smooth_percent": round(smooth_percent, 2),
            "surface_type": surface_type
        })
        
        return is_smooth, smooth_percent, surface_type


# ============================================================================
# DETECTOR REGIONAL
# ============================================================================

class RegionalUniformityDetectorV46:
    """V4.6: Detector regional que considera reflexo."""
    
    def __init__(self, logger: AnalysisLogger = None):
        self.logger = logger or AnalysisLogger()
        self.grid_size = 4
    
    def analyze_regional_uniformity(self, naturalness_map: np.ndarray, 
                                   percent_reflection: float = 0.0) -> Dict[str, Any]:
        """Analisa uniformidade por região COM CONSCIÊNCIA DE REFLEXO."""
        h, w = naturalness_map.shape
        region_h = h // self.grid_size
        region_w = w // self.grid_size
        
        region_uniformities = []
        uniform_regions = 0
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                start_h = i * region_h
                end_h = start_h + region_h if i < self.grid_size - 1 else h
                start_w = j * region_w
                end_w = start_w + region_w if j < self.grid_size - 1 else w
                
                region = naturalness_map[start_h:end_h, start_w:end_w]
                
                region_mean = np.mean(region)
                region_std = np.std(region)
                
                region_uniformities.append({
                    'mean': region_mean,
                    'std': region_std,
                    'position': (i, j)
                })
                
                if region_std < 0.08:
                    uniform_regions += 1
        
        percent_uniform_regions = uniform_regions / len(region_uniformities)
        
        region_means = [r['mean'] for r in region_uniformities]
        region_stds = [r['std'] for r in region_uniformities]
        
        inter_region_std = np.std(region_means)
        avg_intra_region_std = np.mean(region_stds)
        
        is_localized = False
        is_global = False
        uniformity_type = "MIXED"
        
        # REGRA NOVA: Uniformidade + reflexo = VIDRO (LOCALIZED)
        if percent_uniform_regions > 0.65 and percent_reflection > 0.20:
            is_localized = True
            uniformity_type = "LOCALIZED"
        # UNIFORMIDADE LOCALIZADA (original)
        elif percent_uniform_regions < 0.70 and inter_region_std > 0.06:
            is_localized = True
            uniformity_type = "LOCALIZED"
        # UNIFORMIDADE GLOBAL (IA sintética)
        elif percent_uniform_regions > 0.78 and inter_region_std < 0.09:
            is_global = True
            uniformity_type = "GLOBAL"
        # INTERMEDIÁRIO
        elif percent_uniform_regions >= 0.70 and percent_uniform_regions <= 0.78:
            if inter_region_std < 0.06:
                is_global = True
                uniformity_type = "GLOBAL"
            else:
                is_localized = True
                uniformity_type = "LOCALIZED"
        
        return {
            "uniformity_type": uniformity_type,
            "is_localized": is_localized,
            "is_global": is_global,
            "percent_uniform_regions": percent_uniform_regions,
            "inter_region_std": inter_region_std,
            "avg_intra_region_std": avg_intra_region_std,
            "uniform_regions_count": uniform_regions,
            "total_regions": len(region_uniformities)
        }


# ============================================================================
# ANALISADOR DE TEXTURA
# ============================================================================

class TextureAnalyzer:
    """Análise de texturas usando LBP."""
    
    def __init__(self, P: int = 8, R: int = 1, block_size: int = 16, 
                 threshold: float = 0.38, logger: AnalysisLogger = None):
        self.P = P
        self.R = R
        self.block_size = block_size
        self.threshold = threshold
        self.logger = logger or AnalysisLogger()
    
    def calculate_lbp(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calcula LBP."""
        if isinstance(image, Image.Image):
            img_gray = np.array(image.convert('L'))
        elif len(image.shape) > 2:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = image.copy()
        
        lbp = local_binary_pattern(img_gray, self.P, self.R, method="uniform")
        
        n_bins = self.P + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype("float") / (hist.sum() + 1e-7)
        
        return lbp, hist
    
    def analyze_texture_variance(self, image: np.ndarray, is_smooth_surface: bool = False, 
                                smooth_percent: float = 0.0) -> Dict[str, Any]:
        """Análise de textura."""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        lbp_image, _ = self.calculate_lbp(image)
        height, width = lbp_image.shape
        rows = max(1, height // self.block_size)
        cols = max(1, width // self.block_size)
        
        variance_map = np.zeros((rows, cols))
        entropy_map = np.zeros((rows, cols))
        uniformity_map = np.zeros((rows, cols))
        
        for i in range(0, height - self.block_size + 1, self.block_size):
            for j in range(0, width - self.block_size + 1, self.block_size):
                block = lbp_image[i:i+self.block_size, j:j+self.block_size]
                
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
        
        if is_smooth_surface:
            naturalness_map = (entropy_map * 0.30 +
                              variance_map * 0.50 +
                              uniformity_map * 0.20)
        else:
            naturalness_map = (entropy_map * 0.50 +
                              variance_map * 0.25 +
                              uniformity_map * 0.25)
        
        suspicious_mask = naturalness_map < self.threshold
        mean_naturalness = np.mean(naturalness_map)
        suspicious_ratio = np.mean(suspicious_mask)
        std_of_naturalness = np.std(naturalness_map)
        
        if is_smooth_surface:
            penalty_multiplier = 0.5 * smooth_percent
        else:
            penalty_multiplier = 1.1 if suspicious_ratio > 0.05 else 0.7
        
        if suspicious_ratio > 0.05:
            penalty_factor = 1.0 - (suspicious_ratio * penalty_multiplier)
        else:
            penalty_factor = 1.0 - (suspicious_ratio * (penalty_multiplier * 0.7))
        
        penalty_factor = max(0.5 if is_smooth_surface else 0.4, penalty_factor)
        
        naturalness_score = int(mean_naturalness * penalty_factor * 100)
        naturalness_score = max(0, min(100, naturalness_score))
        
        norm_for_display = cv2.normalize(naturalness_map, None, 0, 1, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap((norm_for_display * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        return {
            "variance_map": variance_map,
            "naturalness_map": naturalness_map,
            "suspicious_mask": suspicious_mask,
            "naturalness_score": naturalness_score,
            "heatmap": heatmap,
            "suspicious_ratio": suspicious_ratio,
            "mean_naturalness_raw": mean_naturalness,
            "std_naturalness": std_of_naturalness
        }
    
    def generate_visual_report(self, image: np.ndarray, analysis_results: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Gera relatório visual."""
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        naturalness_map = analysis_results["naturalness_map"]
        suspicious_mask = analysis_results["suspicious_mask"]
        score = analysis_results["naturalness_score"]
        
        height, width = image.shape[:2]
        naturalness_map_resized = cv2.resize(naturalness_map, (width, height), 
                                           interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(suspicious_mask.astype(np.uint8), (width, height), 
                                 interpolation=cv2.INTER_NEAREST)
        
        norm_for_display = cv2.normalize(naturalness_map_resized, None, 0, 1, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap((norm_for_display * 255).astype(np.uint8), 
                                    cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        highlighted = overlay.copy()
        
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(highlighted, contours, -1, (0, 0, 255), 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(highlighted, f"Score: {score}/100", (10, 30), font, 0.7, (255, 255, 255), 2)
        
        return highlighted, heatmap
    
    def analyze_image(self, image: np.ndarray, is_smooth_surface: bool = False, 
                     smooth_percent: float = 0.0) -> Dict[str, Any]:
        """Análise completa."""
        analysis_results = self.analyze_texture_variance(image, is_smooth_surface, smooth_percent)
        visual_report, heatmap = self.generate_visual_report(image, analysis_results)
        
        score = analysis_results["naturalness_score"]
        percent_suspicious = float(np.mean(analysis_results["suspicious_mask"]) * 100)
        
        return {
            "score": score,
            "percent_suspicious": percent_suspicious,
            "visual_report": visual_report,
            "heatmap": heatmap,
            "analysis_results": analysis_results
        }


# ============================================================================
# ANALISADORES AUXILIARES
# ============================================================================

class EdgeAnalyzer:
    """Análise de bordas."""
    
    def __init__(self, use_clahe: bool = True, logger: AnalysisLogger = None):
        self.use_clahe = use_clahe
        self.logger = logger or AnalysisLogger()
    
    def apply_clahe(self, img_gray: np.ndarray) -> np.ndarray:
        if not self.use_clahe:
            return img_gray
        if img_gray.dtype != np.uint8:
            img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img_gray)
    
    def analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Análise de bordas."""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        gray = self.apply_clahe(gray)
        
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        edge_score = int(np.mean(magnitude) / 255.0 * 100)
        edge_score = max(0, min(100, edge_score))
        
        return {"edge_score": edge_score}


class NoiseAnalyzer:
    """Análise de ruído."""
    
    def __init__(self, block_size: int = 32, use_clahe: bool = True, logger: AnalysisLogger = None):
        self.block_size = block_size
        self.use_clahe = use_clahe
        self.logger = logger or AnalysisLogger()
    
    def apply_clahe(self, img_gray: np.ndarray) -> np.ndarray:
        if not self.use_clahe:
            return img_gray
        if img_gray.dtype != np.uint8:
            img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img_gray)
    
    def analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Análise de ruído."""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        gray = self.apply_clahe(gray)
        height, width = gray.shape
        
        rows = max(1, height // self.block_size)
        cols = max(1, width // self.block_size)
        noise_map = np.zeros((rows, cols))
        
        for i in range(0, height - self.block_size + 1, self.block_size):
            for j in range(0, width - self.block_size + 1, self.block_size):
                block = gray[i:i+self.block_size, j:j+self.block_size]
                
                row_idx = i // self.block_size
                col_idx = j // self.block_size
                
                if row_idx >= rows or col_idx >= cols:
                    continue
                
                try:
                    block_sigma = estimate_sigma(block, average_sigmas=True, channel_axis=None)
                    noise_map[row_idx, col_idx] = block_sigma
                except:
                    noise_map[row_idx, col_idx] = np.std(block)
        
        noise_mean = np.mean(noise_map)
        noise_std = np.std(noise_map)
        noise_cv = noise_std / noise_mean if noise_mean > 0 else 0
        
        if noise_cv < 0.2:
            noise_consistency_score = 30
        elif noise_cv <= 0.8:
            normalized = (noise_cv - 0.2) / 0.6
            noise_consistency_score = int(60 + (1 - normalized) * 25)
        else:
            noise_consistency_score = max(20, int(60 - (noise_cv - 0.8) * 50))
        
        return {"noise_score": noise_consistency_score}


# ============================================================================
# ANALISADOR SEQUENCIAL V4.8.1 - COM VISUAL FIX
# ============================================================================

class SequentialAnalyzer:
    """
    Sistema V4.8.1 - SEMPRE gera visual_report
    """
    
    def __init__(self):
        self.logger = AnalysisLogger()
        self.smooth_detector = SmoothSurfaceDetector(self.logger)
        self.reflection_detector = ReflectionMaskV46(self.logger)
        self.regional_detector = RegionalUniformityDetectorV46(self.logger)
        self.texture_analyzer = TextureAnalyzer(logger=self.logger)
        self.edge_analyzer = EdgeAnalyzer(use_clahe=True, logger=self.logger)
        self.noise_analyzer = NoiseAnalyzer(use_clahe=True, logger=self.logger)
    
    def analyze_sequential(self, image) -> Dict[str, Any]:
        """Análise sequencial com visual sempre gerado."""
        
        self.logger.log("START", "Iniciando análise V4.8.1 - Visual Fix", {})
        
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
        
        validation_chain = []
        all_scores = {}
        
        # FASE 0A: Detecção de superfície lisa
        is_smooth, smooth_percent, surface_type = self.smooth_detector.detect_smooth_surface(image_array)
        all_scores['smooth_surface'] = round(smooth_percent * 100, 1)
        all_scores['surface_type'] = surface_type
        validation_chain.append('smooth_detection')
        
        # FASE 0B: Detecção de reflexo
        non_reflective_mask, percent_valid, percent_reflection = \
            self.reflection_detector.get_non_reflective_mask(image_array)
        
        all_scores['reflection'] = round(percent_reflection * 100, 1)
        validation_chain.append('reflection')
        
        # ===================================================================
        # PROTEÇÃO V4.7: REFLEXO DOMINANTE > 35% - GERA VISUAL!
        # ===================================================================
        if percent_reflection > 0.35:
            self.logger.log("REFLECTION_DOMINANT", "🛡️ Reflexo dominante (>35%) - foto real!", {
                "percent_reflection": round(percent_reflection * 100, 1),
                "conclusion": "NEVER_AI"
            })
            
            # 🔥 CORREÇÃO V4.8.1: GERAR VISUAL REPORT MESMO COM DECISÃO RÁPIDA
            texture_result = self.texture_analyzer.analyze_image(image_array, is_smooth, smooth_percent)
            
            return {
                "verdict": "NATURAL",
                "confidence": 92,
                "reason": "Reflexo dominante em superfície de vidro - foto autêntica",
                "main_score": 85,
                "all_scores": all_scores,
                "validation_chain": validation_chain,
                "phases_executed": 1,
                "visual_report": texture_result['visual_report'],  # ✅ AGORA TEM IMAGEM!
                "heatmap": texture_result['heatmap'],             # ✅ AGORA TEM IMAGEM!
                "percent_suspicious": texture_result['percent_suspicious'],
                "detailed_reason": "Reflexo > 35% indica foto real de vidro/espelho",
                "logs": self.logger.get_logs()
            }
        
        # Determinar boost
        if percent_reflection > 0.25:
            analysis_mode = "GLASS_DOMINANT"
            texture_boost = 1.0
        elif is_smooth:
            analysis_mode = "SMOOTH_SURFACE"
            texture_boost = 1.3 + (smooth_percent * 0.3)
        elif percent_reflection >= 0.10:
            analysis_mode = "MODERATE_REFLECTION"
            texture_boost = 1.2
        else:
            analysis_mode = "NORMAL"
            texture_boost = 1.0
        
        # FASE 1: TEXTURA
        texture_result = self.texture_analyzer.analyze_image(image_array, is_smooth, smooth_percent)
        texture_score = texture_result['score']
        texture_score_original = texture_score
        
        if texture_boost > 1.0:
            texture_score = min(100, int(texture_score * texture_boost))
        
        all_scores['texture'] = texture_score
        validation_chain.append('texture')
        
        # ANÁLISE REGIONAL
        naturalness_map = texture_result['analysis_results']['naturalness_map']
        regional_analysis = self.regional_detector.analyze_regional_uniformity(
            naturalness_map, 
            percent_reflection
        )
        
        all_scores['uniformity_type'] = regional_analysis['uniformity_type']
        all_scores['is_localized'] = regional_analysis['is_localized']
        all_scores['percent_uniform_regions'] = round(regional_analysis['percent_uniform_regions'] * 100, 1)
        
        # DETECTOR DE PERFEIÇÃO
        std_naturalness = texture_result['analysis_results']['std_naturalness']
        suspicious_ratio = texture_result['percent_suspicious'] / 100.0
        
        perfection_detected = False
        perfection_level = "NOT_DETECTED"
        
        if (regional_analysis['uniformity_type'] == "GLOBAL" and 
            regional_analysis['percent_uniform_regions'] >= 0.99):
            perfection_detected = True
            perfection_level = "DETECTED_EXTREME"
        elif regional_analysis['is_localized']:
            perfection_detected = False
        elif (regional_analysis['is_global'] and 
              texture_score >= 55 and 
              suspicious_ratio < 0.05 and 
              std_naturalness < 0.10):
            perfection_detected = True
            perfection_level = "DETECTED_EXTREME"
        elif (regional_analysis['percent_uniform_regions'] > 0.65 and
              texture_score >= 55 and 
              suspicious_ratio < 0.10 and 
              std_naturalness < 0.12):
            perfection_detected = True
            perfection_level = "DETECTED_HIGH"
        elif (texture_score >= 72 and 
              suspicious_ratio < 0.15 and
              not regional_analysis['is_localized']):
            perfection_detected = True
            perfection_level = "DETECTED_SCORE"
        
        all_scores['perfection_flag'] = perfection_level
        
        # Thresholds
        threshold_low = 30 if is_smooth else 35
        threshold_high = 70
        
        # Decisão Fase 1
        if texture_score < threshold_low:
            return self._build_response(
                "MANIPULADA", 95, "Textura artificial detectada",
                texture_score, all_scores, validation_chain, 1, texture_result
            )
        
        if texture_score > threshold_high and not perfection_detected:
            return self._build_response(
                "NATURAL", 85, "Textura natural com alta variabilidade",
                texture_score, all_scores, validation_chain, 1, texture_result
            )
        
        # FASE 2: BORDAS
        edge_result = self.edge_analyzer.analyze_image(image_array)
        edge_score = edge_result['edge_score']
        
        if is_smooth and edge_score < 40:
            edge_score = min(100, int(edge_score * 1.5))
        
        all_scores['edge'] = edge_score
        validation_chain.append('edge')
        
        # FASE 3: RUÍDO
        noise_result = self.noise_analyzer.analyze_image(image_array)
        noise_score = noise_result['noise_score']
        
        all_scores['noise'] = noise_score
        validation_chain.append('noise')
        
        # GLOBAL + NOISE ALTO = IA MODERNA
        uniformity_type = regional_analysis['uniformity_type']
        
        if (uniformity_type == "GLOBAL" and 
            noise_score >= 65 and
            regional_analysis['percent_uniform_regions'] >= 0.90):
            
            return self._build_response(
                "MANIPULADA",
                90,
                "Ruído sintético uniforme detectado - padrão de IA moderna",
                int(noise_score),
                all_scores,
                validation_chain,
                3,
                texture_result,
            )
        
        # REGRA DE SEGURANÇA - FOTO REAL COM BORDAS SUAVES
        if (
            noise_score >= 55 and
            texture_score >= 40 and
            edge_score < 32 and
            uniformity_type != "GLOBAL" and
            suspicious_ratio <= 0.15
        ):
            glass_like = percent_reflection > 0.10
            
            reason = "Textura e ruído naturais com bordas suaves (foto real)"
            if glass_like:
                reason = "Vidro com reflexo e ruído natural; bordas suaves"
            
            return self._build_response(
                "NATURAL",
                85,
                reason,
                int((texture_score + noise_score) / 2),
                all_scores,
                validation_chain,
                3,
                texture_result,
            )
        
        # DECISÃO FINAL COM PESO DINÂMICO
        if percent_reflection > 0.25:
            weighted_score = (
                texture_score * 0.35 +
                edge_score * 0.30 +
                noise_score * 0.35
            )
        else:
            weighted_score = (
                texture_score * 0.50 +
                edge_score * 0.30 +
                noise_score * 0.20
            )
        
        # LÓGICA COM PERFEIÇÃO
        if perfection_detected:
            if perfection_level == "DETECTED_EXTREME":
                if noise_score >= 70 and edge_score >= 45:
                    verdict, confidence = "SUSPEITA", 75
                    reason = "Uniformidade extrema mas ruído excelente"
                else:
                    verdict, confidence = "MANIPULADA", 92
                    reason = "Uniformidade sintética extrema - IA detectada"
            elif perfection_level == "DETECTED_HIGH":
                if noise_score >= 75 and edge_score >= 50:
                    verdict, confidence = "NATURAL", 78
                    reason = "Uniformidade alta mas validada"
                elif noise_score >= 65:
                    verdict, confidence = "SUSPEITA", 80
                    reason = "Uniformidade sintética detectada"
                else:
                    verdict, confidence = "MANIPULADA", 88
                    reason = "Uniformidade sintética confirmada"
            else:
                if noise_score >= 65:
                    verdict, confidence = "SUSPEITA", 75
                    reason = "Score alto - possível IA"
                else:
                    verdict, confidence = "MANIPULADA", 85
                    reason = "Score alto com ruído artificial"
        elif is_smooth and regional_analysis['uniformity_type'] != "GLOBAL":
            if weighted_score > 48:
                verdict, confidence = "NATURAL", 80
                reason = f"Superfície lisa natural ({surface_type})"
            elif weighted_score > 38:
                verdict, confidence = "SUSPEITA", 70
                reason = "Superfície lisa - revisão recomendada"
            else:
                verdict, confidence = "MANIPULADA", 75
                reason = "IA detectada"
        else:
            if texture_score < 38 and edge_score < 32:
                verdict, confidence = "MANIPULADA", 90
                reason = "Textura e bordas artificiais"
            elif noise_score >= 60 and 38 <= texture_score < 48 and edge_score >= 30:
                verdict, confidence = "SUSPEITA", 70
                reason = "Ruído natural, textura afetada por compressão"
            elif texture_score >= 45 and edge_score < 32:
                if noise_score < 55 or regional_analysis["uniformity_type"] == "GLOBAL":
                    verdict, confidence = "MANIPULADA", 88
                    reason = "Bordas fracas e ruído artificial"
                else:
                    verdict, confidence = "SUSPEITA", 72
                    reason = "Bordas suaves mas ruído natural"
            elif weighted_score > 62 and noise_score >= 55 and texture_score > 55:
                verdict, confidence = "NATURAL", 82
                reason = "Texturas e ruído naturais"
            elif weighted_score > 55 and noise_score >= 55:
                verdict, confidence = "SUSPEITA", 70
                reason = "Características mistas"
            elif weighted_score < 45:
                verdict, confidence = "MANIPULADA", 80
                reason = "Score ponderado indica artifícios"
            elif weighted_score > 52:
                verdict, confidence = "INCONCLUSIVA", 60
                reason = "Características ambíguas"
            else:
                verdict, confidence = "SUSPEITA", 70
                reason = "Indicadores ambíguos"
        
        return self._build_response(
            verdict, confidence, reason,
            int(weighted_score), all_scores, validation_chain, 3, texture_result
        )
    
    def _build_response(self, verdict: str, confidence: int, reason: str, 
                       main_score: int, all_scores: Dict, validation_chain: List,
                       phases: int, texture_result: Dict) -> Dict[str, Any]:
        """Constrói resposta padronizada."""
        return {
            "verdict": verdict,
            "confidence": confidence,
            "reason": reason,
            "main_score": main_score,
            "all_scores": all_scores,
            "validation_chain": validation_chain,
            "phases_executed": phases,
            "visual_report": texture_result['visual_report'] if texture_result else None,
            "heatmap": texture_result['heatmap'] if texture_result else None,
            "percent_suspicious": texture_result['percent_suspicious'] if texture_result else 0,
            "detailed_reason": f"Score: {main_score}/100. {reason}",
            "logs": self.logger.get_logs()
        }


# ============================================================================
# CLASSE PRINCIPAL
# ============================================================================

class MirrorGlass:
    """API simplificada do MirrorGlass V4.8.1"""
    
    def __init__(self):
        self.analyzer = SequentialAnalyzer()
    
    def analisar(self, image, mostrar_logs: bool = False) -> Dict[str, Any]:
        """Analisa uma imagem para detectar manipulação por IA."""
        resultado = self.analyzer.analyze_sequential(image)
        
        if mostrar_logs:
            print("\n" + "="*80)
            print("LOGS DA ANÁLISE V4.8.1 - VISUAL FIX")
            print("="*80 + "\n")
            self.analyzer.logger.print_summary()
            print("\n" + "="*80)
        
        return resultado


if __name__ == "__main__":
    print("\n" + "="*80)
    print("MirrorGlass V4.8.1 - Visual Fix")
    print("="*80)
    print("\n✅ Correção: Imagens SEMPRE aparecem, mesmo em decisões rápidas")
    print("\n" + "="*80 + "\n")