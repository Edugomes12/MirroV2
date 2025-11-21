# texture_analyzer.py
# Sistema de Análise Sequencial com Validação em Cadeia
# Versão: 4.3.0 - Com Detecção de Reflexo

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.restoration import estimate_sigma
from scipy.stats import entropy
from PIL import Image
import io
import base64


class ReflectionMask:
    """
    Detecta áreas com reflexo (vidro, parabrisas, superfícies brilhantes).
    
    Reflexo possui três características:
    1. Brilho muito alto e saturado
    2. Gradientes fortes em apenas 1 direção
    3. Baixa entropia local
    """
    
    def __init__(self, brightness_thresh=220, entropy_thresh=0.15, gradient_thresh=80):
        self.brightness_thresh = brightness_thresh
        self.entropy_thresh = entropy_thresh
        self.gradient_thresh = gradient_thresh

    def compute_local_entropy(self, gray, block_size=16):
        """Calcula entropia local por blocos."""
        h, w = gray.shape
        entropy_map = np.zeros((h // block_size, w // block_size))
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                hist, _ = np.histogram(block.ravel(), bins=256, range=(0, 256))
                hist = hist.astype("float") / (hist.sum() + 1e-7)
                ent = entropy(hist)
                row_idx = i // block_size
                col_idx = j // block_size
                if row_idx < entropy_map.shape[0] and col_idx < entropy_map.shape[1]:
                    entropy_map[row_idx, col_idx] = ent
        
        return entropy_map

    def detect_reflection(self, image):
        """
        Detecta áreas com reflexo na imagem.
        
        Returns:
            mask: Máscara onde 255 = área com reflexo
            percent: Porcentagem da imagem com reflexo
        """
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape

        # 1) Brightness map (reflexo é muito claro)
        _, bright_mask = cv2.threshold(gray, self.brightness_thresh, 255, cv2.THRESH_BINARY)

        # 2) Gradiente especular (bordas fortes)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        magnitude = cv2.magnitude(grad_x, grad_y)
        gradient_mask = (magnitude > self.gradient_thresh).astype(np.uint8) * 255

        # 3) Detecção de saturação (reflexo costuma ter baixa saturação em HSV)
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1]
            low_sat_mask = cv2.threshold(saturation, 30, 255, cv2.THRESH_BINARY_INV)[1]
            bright_and_low_sat = cv2.bitwise_and(bright_mask, low_sat_mask)
        else:
            bright_and_low_sat = bright_mask

        # 4) Combinação: áreas claras com baixa saturação OU brilho extremo com gradiente
        combined = cv2.bitwise_or(bright_and_low_sat, cv2.bitwise_and(bright_mask, gradient_mask))

        # 5) Limpeza morfológica
        kernel = np.ones((7, 7), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        
        # Dilatar para pegar bordas do reflexo
        combined = cv2.dilate(combined, np.ones((5, 5), np.uint8), iterations=2)

        # Calcular porcentagem
        percent_reflection = np.mean(combined > 0)

        return combined, percent_reflection

    def get_non_reflective_mask(self, image):
        """
        Retorna máscara das áreas SEM reflexo (para análise).
        
        Returns:
            mask: Máscara onde 255 = área válida para análise (sem reflexo)
            percent_valid: Porcentagem da imagem válida para análise
        """
        reflection_mask, percent_reflection = self.detect_reflection(image)
        non_reflective = cv2.bitwise_not(reflection_mask)
        percent_valid = 1.0 - percent_reflection
        return non_reflective, percent_valid, percent_reflection


class DocumentDetector:
    """
    Detecta áreas com documentos/papéis na imagem.
    Papel branco com texto não deve ser confundido com IA.
    """
    
    def __init__(self, white_thresh=160, min_area_percent=1.0):
        self.white_thresh = white_thresh  # 160 para pegar papel/vidro claro
        self.min_area_percent = min_area_percent
    
    def detect_document(self, image):
        """Detecta documento/papel na imagem."""
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape
        
        # Detectar áreas brancas (papel)
        _, white_mask = cv2.threshold(gray, self.white_thresh, 255, cv2.THRESH_BINARY)
        
        # Detectar texto (áreas escuras)
        _, dark_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        
        # Encontrar contornos de áreas brancas
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        total_doc_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            area_percent = area / (h * w) * 100
            
            if area_percent >= self.min_area_percent:
                mask_temp = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask_temp, [contour], -1, 255, -1)
                
                text_area = cv2.bitwise_and(dark_mask, mask_temp)
                text_percent = np.sum(text_area > 0) / (np.sum(mask_temp > 0) + 1) * 100
                
                if text_percent > 3:
                    total_doc_area += area
        
        percent_document = total_doc_area / (h * w)
        has_document = percent_document >= (self.min_area_percent / 100)
        
        return percent_document, has_document


class TextureAnalyzer:
    """Análise de texturas usando LBP - DETECTOR PRIMÁRIO (SEM CLAHE)."""
    
    def __init__(self, P=8, R=1, block_size=16, threshold=0.38):  # Balanceado: 0.38
        self.P = P
        self.R = R
        self.block_size = block_size
        self.threshold = threshold
    
    def calculate_lbp(self, image):
        """Calcula LBP SEM CLAHE - textura pura"""
        if isinstance(image, Image.Image):
            img_gray = np.array(image.convert('L'))
        elif len(image.shape) > 2:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = image.copy()
        
        # CRITICAL: SEM CLAHE para detectar uniformidade de IA!
        lbp = local_binary_pattern(img_gray, self.P, self.R, method="uniform")
        
        n_bins = self.P + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype("float") / (hist.sum() + 1e-7)
        
        return lbp, hist
    
    def analyze_texture_variance(self, image):
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
        
        # BALANCED: Pesos equilibrados
        # Entropia detecta complexidade, variância detecta uniformidade de IA
        naturalness_map = (entropy_map * 0.50 +  # Reduzido de 0.50
                          variance_map * 0.25 +   # Aumentado de 0.25 
                          uniformity_map * 0.25)  # Reduzido de 0.25
        
        # NÃO NORMALIZAR! Manter valores absolutos 0-1
        
        suspicious_mask = naturalness_map < self.threshold
        
        mean_naturalness = np.mean(naturalness_map)
        suspicious_ratio = np.mean(suspicious_mask)
        
        # BALANCED: Penalização média entre original (3.0) e suave (0.9)
        # Detecta IA sem matar fotos reais com JPEG
        if suspicious_ratio > 0.05:  # Se > 5% da imagem é suspeita
            penalty_factor = 1.0 - (suspicious_ratio * 1.1)  # Balanceado: 1.1
        else:
            penalty_factor = 1.0 - (suspicious_ratio * 0.7)  # Balanceado: 0.7
        
        penalty_factor = max(0.4, penalty_factor)  # Aumentado de 0.2 para 0.4 - menos agressivo
        
        naturalness_score = int(mean_naturalness * penalty_factor * 100)
        naturalness_score = max(0, min(100, naturalness_score))
        
        # Heatmap para visualização (aqui SIM normalizar só para cores)
        norm_for_display = cv2.normalize(naturalness_map, None, 0, 1, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap((norm_for_display * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        return {
            "variance_map": variance_map,
            "naturalness_map": naturalness_map,  # Valores absolutos!
            "suspicious_mask": suspicious_mask,
            "naturalness_score": naturalness_score,
            "heatmap": heatmap,
            "suspicious_ratio": suspicious_ratio,
            "mean_naturalness_raw": mean_naturalness  # Para debug
        }
    
    def classify_naturalness(self, score):
        # BALANCED: Thresholds intermediários
        if score <= 45:  # Meio-termo entre 40 e 50
            return "Alta chance de manipulação", "Textura artificial detectada"
        elif score <= 65:  # Meio-termo entre 60 e 68
            return "Textura suspeita", "Revisão manual sugerida"
        else:
            return "Textura natural", "Baixa chance de manipulação"
    
    def generate_visual_report(self, image, analysis_results):
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
        
        # Para visualização, normalizar
        norm_for_display = cv2.normalize(naturalness_map_resized, None, 0, 1, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap((norm_for_display * 255).astype(np.uint8), 
                                    cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        highlighted = overlay.copy()
        
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(highlighted, contours, -1, (0, 0, 255), 2)
        
        category, description = self.classify_naturalness(score)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(highlighted, f"Score: {score}/100", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(highlighted, category, (10, 60), font, 0.7, (255, 255, 255), 2)
        
        return highlighted, heatmap
    
    def analyze_image(self, image):
        analysis_results = self.analyze_texture_variance(image)
        visual_report, heatmap = self.generate_visual_report(image, analysis_results)
        
        score = analysis_results["naturalness_score"]
        category, description = self.classify_naturalness(score)
        percent_suspicious = float(np.mean(analysis_results["suspicious_mask"]) * 100)
        
        return {
            "score": score,
            "category": category,
            "description": description,
            "percent_suspicious": percent_suspicious,
            "visual_report": visual_report,
            "heatmap": heatmap,
            "analysis_results": analysis_results,
            "clahe_enabled": False
        }


class EdgeAnalyzer:
    """Análise de bordas COM CLAHE - útil para revelar transições."""
    
    def __init__(self, block_size=16, edge_threshold_low=50, edge_threshold_high=150,
                 use_clahe=True, clahe_clip_limit=2.0, clahe_tile_size=8):
        self.block_size = block_size
        self.edge_threshold_low = edge_threshold_low
        self.edge_threshold_high = edge_threshold_high
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
    
    def apply_clahe(self, img_gray):
        if not self.use_clahe:
            return img_gray
        if img_gray.dtype != np.uint8:
            img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit,
                                tileGridSize=(self.clahe_tile_size, self.clahe_tile_size))
        return clahe.apply(img_gray)
    
    def _convert_to_gray(self, image):
        if isinstance(image, Image.Image):
            gray = np.array(image.convert('L'))
        elif len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        return self.apply_clahe(gray)
    
    def compute_gradients(self, image):
        gray = self._convert_to_gray(image)
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        direction = np.arctan2(gradient_y, gradient_x)
        
        return {
            "magnitude": magnitude,
            "direction": direction
        }
    
    def analyze_edge_coherence(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        gray = self._convert_to_gray(image)
        height, width = gray.shape
        
        gradients = self.compute_gradients(image)
        magnitude = gradients["magnitude"]
        direction = gradients["direction"]
        
        rows = max(1, height // self.block_size)
        cols = max(1, width // self.block_size)
        
        coherence_map = np.zeros((rows, cols))
        edge_density_map = np.zeros((rows, cols))
        
        for i in range(0, height - self.block_size + 1, self.block_size):
            for j in range(0, width - self.block_size + 1, self.block_size):
                block_mag = magnitude[i:i+self.block_size, j:j+self.block_size]
                block_dir = direction[i:i+self.block_size, j:j+self.block_size]
                
                row_idx = i // self.block_size
                col_idx = j // self.block_size
                
                if row_idx >= rows or col_idx >= cols:
                    continue
                
                edge_density = np.mean(block_mag) / 255.0
                edge_density_map[row_idx, col_idx] = edge_density
                
                if np.sum(block_mag > 10) > 10:
                    significant_pixels = block_mag > np.percentile(block_mag, 70)
                    if np.any(significant_pixels):
                        directions_sig = block_dir[significant_pixels]
                        mean_cos = np.mean(np.cos(directions_sig))
                        mean_sin = np.mean(np.sin(directions_sig))
                        circular_variance = 1 - np.sqrt(mean_cos**2 + mean_sin**2)
                        coherence_map[row_idx, col_idx] = 1 - circular_variance
                    else:
                        coherence_map[row_idx, col_idx] = 0.5
                else:
                    coherence_map[row_idx, col_idx] = 0.5
        
        coherence_normalized = cv2.normalize(coherence_map, None, 0, 1, cv2.NORM_MINMAX)
        edge_density_normalized = cv2.normalize(edge_density_map, None, 0, 1, cv2.NORM_MINMAX)
        edge_naturalness = coherence_normalized * 0.6 + edge_density_normalized * 0.4
        edge_score = int(np.mean(edge_naturalness) * 100)
        
        return {
            "edge_score": edge_score
        }
    
    def analyze_image(self, image):
        coherence_results = self.analyze_edge_coherence(image)
        edge_score = coherence_results["edge_score"]
        
        if edge_score <= 40:
            category = "Bordas artificiais"
            description = "Alta probabilidade de manipulação"
        elif edge_score <= 65:
            category = "Bordas suspeitas"
            description = "Requer verificação"
        else:
            category = "Bordas naturais"
            description = "Baixa probabilidade de manipulação"
        
        return {
            "edge_score": edge_score,
            "category": category,
            "description": description,
            "clahe_enabled": self.use_clahe
        }


class NoiseAnalyzer:
    """Análise de ruído COM CLAHE."""
    
    def __init__(self, block_size=32, use_clahe=True, clahe_clip_limit=2.0, clahe_tile_size=8):
        self.block_size = block_size
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
    
    def apply_clahe(self, img_gray):
        if not self.use_clahe:
            return img_gray
        if img_gray.dtype != np.uint8:
            img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit,
                                tileGridSize=(self.clahe_tile_size, self.clahe_tile_size))
        return clahe.apply(img_gray)
    
    def _convert_to_gray(self, image):
        if isinstance(image, Image.Image):
            gray = np.array(image.convert('L'))
        elif len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        return self.apply_clahe(gray)
    
    def analyze_local_noise(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        gray = self._convert_to_gray(image)
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
        
        # FIX CRÍTICO: Fórmula original estava quebrada!
        # CV de 0.3-0.8 é NORMAL em imagens reais (variação natural entre áreas)
        # IA tende a ter CV mais uniforme (0.2-0.4) ou muito inconsistente (>1.0)
        
        if noise_cv < 0.2:
            # Muito uniforme = suspeito (IA)
            noise_consistency_score = 30
        elif noise_cv <= 0.8:
            # Variação normal = natural
            # Mapear 0.2-0.8 para scores 60-85
            normalized = (noise_cv - 0.2) / 0.6  # 0 a 1
            noise_consistency_score = int(60 + (1 - normalized) * 25)  # 85 quando CV=0.2, 60 quando CV=0.8
        else:
            # Muito inconsistente = suspeito
            noise_consistency_score = max(20, int(60 - (noise_cv - 0.8) * 50))
        
        return noise_consistency_score
    
    def analyze_image(self, image):
        noise_score = self.analyze_local_noise(image)
        
        if noise_score <= 40:
            category = "Ruído artificial"
            description = "Alta probabilidade de manipulação"
        elif noise_score <= 65:
            category = "Ruído inconsistente"
            description = "Requer verificação"
        else:
            category = "Ruído natural"
            description = "Baixa probabilidade de manipulação"
        
        return {
            "noise_score": noise_score,
            "category": category,
            "description": description,
            "clahe_enabled": self.use_clahe
        }


class LightingAnalyzer:
    """Analisador de iluminação COM CLAHE."""
    
    def __init__(self, use_clahe=True, clahe_clip_limit=2.0, clahe_tile_size=8):
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
    
    def apply_clahe(self, img_gray):
        if not self.use_clahe:
            return img_gray
        if img_gray.dtype != np.uint8:
            img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit,
                                tileGridSize=(self.clahe_tile_size, self.clahe_tile_size))
        return clahe.apply(img_gray)
    
    def _convert_to_gray(self, image):
        if isinstance(image, Image.Image):
            gray = np.array(image.convert('L'))
        elif len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        return self.apply_clahe(gray)
    
    def analyze_image(self, image):
        gray = self._convert_to_gray(image)
        
        # Análise simplificada de iluminação
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        smoothness = 1.0 / (np.std(magnitude) + 1)
        lighting_score = int(min(smoothness * 50, 30))  # Score simplificado
        
        if lighting_score >= 20:
            category = "Iluminação natural"
            description = "Física consistente"
        elif lighting_score >= 10:
            category = "Iluminação aceitável"
            description = "Pequenas inconsistências"
        else:
            category = "Iluminação suspeita"
            description = "Inconsistências detectadas"
        
        return {
            "lighting_score": lighting_score,
            "category": category,
            "description": description,
            "clahe_enabled": self.use_clahe
        }


class SequentialAnalyzer:
    """Sistema de Análise Sequencial - Validação em Cadeia com Detecção de Reflexo e Documento"""
    
    def __init__(self):
        self.texture_analyzer = TextureAnalyzer()
        self.edge_analyzer = EdgeAnalyzer(use_clahe=True)
        self.noise_analyzer = NoiseAnalyzer(use_clahe=True)
        self.lighting_analyzer = LightingAnalyzer(use_clahe=True)
        self.reflection_detector = ReflectionMask()
        self.document_detector = DocumentDetector()  # NOVO
    
    def analyze_sequential(self, image):
        """Análise sequencial com validação em cadeia e compensação de reflexo/documento."""
        validation_chain = []
        all_scores = {}
        
        # ========================================
        # FASE 0A: DETECÇÃO DE REFLEXO
        # ========================================
        non_reflective_mask, percent_valid, percent_reflection = \
            self.reflection_detector.get_non_reflective_mask(image)
        
        all_scores['reflection'] = round(percent_reflection * 100, 1)
        validation_chain.append('reflection')
        
        # ========================================
        # FASE 0B: DETECÇÃO DE DOCUMENTO (NOVO)
        # ========================================
        percent_document, has_document = self.document_detector.detect_document(image)
        all_scores['document'] = round(percent_document * 100, 1)
        
        # Determinar modo de análise baseado em reflexo
        if percent_reflection >= 0.30:
            reflection_mode = "IGNORE"
            reflection_boost = 1.4
        elif percent_reflection >= 0.10:
            reflection_mode = "HEAVY_COMPENSATE"
            reflection_boost = 1.25
        elif percent_reflection >= 0.05:
            reflection_mode = "LIGHT_COMPENSATE"
            reflection_boost = 1.1
        else:
            reflection_mode = "NORMAL"
            reflection_boost = 1.0
        
        # NOVO: Se tem documento, aplicar boost adicional
        document_boost = 1.0
        if has_document:
            document_boost = 1.15
            if percent_document > 0.10:
                document_boost = 1.25
        
        # ========================================
        # FASE 1: DETECTOR PRIMÁRIO (Textura)
        # ========================================
        texture_result = self.texture_analyzer.analyze_image(image)
        texture_score = texture_result['score']
        
        # Aplicar compensação de reflexo na textura
        if reflection_mode != "NORMAL":
            texture_score = min(100, int(texture_score * reflection_boost))
        
        # Aplicar compensação de documento (papel branco não é IA)
        if has_document:
            texture_score = min(100, int(texture_score * document_boost))
        
        all_scores['texture'] = texture_score
        validation_chain.append('texture')
        
        # BALANCED: Threshold ajustado para considerar compressão JPEG e documentos
        # Com documento, ser mais tolerante
        texture_threshold_low = 30 if has_document else 35
        
        if texture_score < texture_threshold_low:  # Só rejeita se MUITO baixo
            return {
                "verdict": "MANIPULADA",
                "confidence": 95,
                "reason": "Textura artificial detectada",
                "main_score": texture_score,
                "all_scores": all_scores,
                "validation_chain": validation_chain,
                "phases_executed": 1,
                "visual_report": texture_result['visual_report'],
                "heatmap": texture_result['heatmap'],
                "percent_suspicious": texture_result['percent_suspicious'],
                "detailed_reason": f"Score {texture_score}/100 indica textura artificial típica de IA."
            }
        
        if texture_score > 70:  # Ajustado para 70 (mais rigoroso)
            return {
                "verdict": "NATURAL",
                "confidence": 85,
                "reason": "Textura natural com alta variabilidade",
                "main_score": texture_score,
                "all_scores": all_scores,
                "validation_chain": validation_chain,
                "phases_executed": 1,
                "visual_report": texture_result['visual_report'],
                "heatmap": texture_result['heatmap'],
                "percent_suspicious": texture_result['percent_suspicious'],
                "detailed_reason": f"Score {texture_score}/100 indica textura natural."
            }
        
        # ========================================
        # FASE 2: VALIDADOR DE BORDAS
        # ========================================
        edge_result = self.edge_analyzer.analyze_image(image)
        edge_score = edge_result['edge_score']
        
        # Aplicar compensação de reflexo nas bordas
        # Reflexo causa bordas caóticas que parecem artificiais
        if reflection_mode in ["IGNORE", "HEAVY_COMPENSATE"]:
            edge_score_original = edge_score
            if edge_score < 35:
                # Reflexo pesado: suavizar penalidade de bordas
                edge_score = int((edge_score + 50) / 2)  # Média com 50
            else:
                edge_score = min(100, int(edge_score * reflection_boost))
        elif reflection_mode == "LIGHT_COMPENSATE":
            edge_score = min(100, int(edge_score * reflection_boost))
        
        all_scores['edge'] = edge_score
        validation_chain.append('edge')
        
        # BALANCED: Fase 2 considera contexto da textura E reflexo
        # Se há muito reflexo, ser mais tolerante com bordas ruins
        
        edge_threshold_low = 30 if reflection_mode == "NORMAL" else 25
        edge_threshold_medium = 40 if reflection_mode == "NORMAL" else 35
        
        if edge_score < edge_threshold_low:
            # Bordas MUITO ruins = forte indicador de IA (mesmo com reflexo)
            if texture_score < 55:
                # Bordas ruins + textura não-excelente = IA confirmada
                return {
                    "verdict": "MANIPULADA",
                    "confidence": 90,
                    "reason": "Bordas artificiais detectadas",
                    "main_score": texture_score,
                    "all_scores": all_scores,
                    "validation_chain": validation_chain,
                    "phases_executed": 2,
                    "visual_report": texture_result['visual_report'],
                    "heatmap": texture_result['heatmap'],
                    "percent_suspicious": texture_result['percent_suspicious'],
                    "detailed_reason": f"Bordas artificiais ({edge_score}/100) confirmam suspeita de textura ({texture_score}/100)."
                }
        
        elif edge_score < edge_threshold_medium:
            # Se textura estava OK (38-50), bordas ruins podem ser compressão
            if texture_score >= 38:
                # Continuar para Fase 3 (não decidir ainda)
                pass
            else:
                # Textura MUITO ruim + bordas ruins = IA confirmada
                return {
                    "verdict": "MANIPULADA",
                    "confidence": 90,
                    "reason": "Textura duvidosa + bordas artificiais",
                    "main_score": texture_score,
                    "all_scores": all_scores,
                    "validation_chain": validation_chain,
                    "phases_executed": 2,
                    "visual_report": texture_result['visual_report'],
                    "heatmap": texture_result['heatmap'],
                    "percent_suspicious": texture_result['percent_suspicious'],
                    "detailed_reason": f"Textura artificial ({texture_score}/100) confirmada por bordas artificiais ({edge_score}/100)."
                }
        
        # ========================================
        # FASE 3: VALIDADOR DE RUÍDO
        # ========================================
        noise_result = self.noise_analyzer.analyze_image(image)
        noise_score = noise_result['noise_score']
        
        # Aplicar compensação de reflexo no ruído
        # Reflexo reduz ruído artificialmente (áreas "lavadas")
        if reflection_mode in ["IGNORE", "HEAVY_COMPENSATE"]:
            noise_score_original = noise_score
            if noise_score < 45:
                # Reflexo pesado: aumentar score de ruído
                noise_score = min(100, int(noise_score * 1.5))
            else:
                noise_score = min(100, int(noise_score * reflection_boost))
        elif reflection_mode == "LIGHT_COMPENSATE":
            noise_score = min(100, int(noise_score * reflection_boost))
        
        all_scores['noise'] = noise_score
        validation_chain.append('noise')
        
        # Threshold de ruído ajustado pelo reflexo
        noise_threshold = 40 if reflection_mode == "NORMAL" else 30
        
        if noise_score < noise_threshold:
            return {
                "verdict": "MANIPULADA",
                "confidence": 85,
                "reason": "Múltiplos indicadores artificiais",
                "main_score": texture_score,
                "all_scores": all_scores,
                "validation_chain": validation_chain,
                "phases_executed": 3,
                "visual_report": texture_result['visual_report'],
                "heatmap": texture_result['heatmap'],
                "percent_suspicious": texture_result['percent_suspicious'],
                "detailed_reason": f"Textura suspeita ({texture_score}/100) + ruído artificial ({noise_score}/100)."
            }
        
        # ========================================
        # FASE 4: VALIDADOR DE FÍSICA (DESABILITADA)
        # ========================================
        # NOTA: LightingAnalyzer está dando score 0 para TODAS as imagens
        # Precisa recalibração. Por agora, desabilitado.
        
        #lighting_result = self.lighting_analyzer.analyze_image(image)
        #lighting_score = lighting_result['lighting_score']
        #all_scores['lighting'] = lighting_score
        #validation_chain.append('lighting')
        
        #if lighting_score < 10:
        #    return {
        #        "verdict": "MANIPULADA",
        #        "confidence": 80,
        #        "reason": "Física da iluminação impossível",
        #        "main_score": texture_score,
        #        "all_scores": all_scores,
        #        "validation_chain": validation_chain,
        #        "phases_executed": 4,
        #        "visual_report": texture_result['visual_report'],
        #        "heatmap": texture_result['heatmap'],
        #        "percent_suspicious": texture_result['percent_suspicious'],
        #        "detailed_reason": f"Iluminação inconsistente ({lighting_score}/100)."
        #    }
        
        # ========================================
        # DECISÃO FINAL: LÓGICA COM REFLEXO
        # ========================================
        
        weighted_score = (
            texture_score * 0.50 +
            edge_score * 0.30 +
            noise_score * 0.20
        )
        
        phases_count = 3
        
        # REGRA 0 (NOVA): Se muito reflexo, ser mais tolerante
        if reflection_mode in ["IGNORE", "HEAVY_COMPENSATE"]:
            # Com muito reflexo: priorizar ruído (menos afetado)
            if noise_score >= 55:
                verdict = "SUSPEITA"  # Não rejeitar, mandar para revisão
                confidence = 70
                reason = f"Imagem com {all_scores['reflection']:.0f}% reflexo - ruído sugere foto real"
            elif noise_score < 40 and texture_score < 40:
                verdict = "MANIPULADA"
                confidence = 80
                reason = "IA detectada mesmo com compensação de reflexo"
            else:
                verdict = "SUSPEITA"
                confidence = 65
                reason = f"Alto reflexo ({all_scores['reflection']:.0f}%) - revisão manual recomendada"
        
        # REGRA 1: IA óbvia (texture BEM ruim + edge ruim)
        elif texture_score < 38 and edge_score < 32:
            verdict = "MANIPULADA"
            confidence = 90
            reason = "Textura e bordas artificiais"
        
        # REGRA 2: Foto real provável (noise BOM + texture no range JPEG)
        elif noise_score >= 60 and 38 <= texture_score < 48 and edge_score >= 30:
            verdict = "SUSPEITA"
            confidence = 70
            reason = "Ruído natural detectado, textura afetada por compressão"
        
        # REGRA 3: IA moderna (texture OK mas edge MUITO ruim)
        elif texture_score >= 45 and edge_score < 32:
            verdict = "MANIPULADA"
            confidence = 88
            reason = "Bordas artificiais típicas de IA"
        
        # REGRA 3B (NOVA): IA avançada (Gemini/GPT-4o)
        # Detecta IAs com ruído sintético perfeito + scores medianos
        # Padrão: noise MUITO alto (>75), texture médio (50-70), edge médio (35-50)
        elif noise_score >= 75 and 50 <= texture_score <= 70 and 35 <= edge_score <= 50:
            verdict = "SUSPEITA"
            confidence = 75
            reason = "Possível IA avançada - ruído sintético detectado"
        
        # REGRA 4: Foto de boa qualidade (MAIS RIGOROSA)
        # Agora exige weighted_score > 62 (era 55) E texture > 55
        elif weighted_score > 62 and noise_score >= 55 and texture_score > 55:
            verdict = "NATURAL"
            confidence = 82
            reason = "Texturas e ruído naturais"
        
        # REGRA 4B: Score médio-alto vai para SUSPEITA (não NATURAL)
        elif weighted_score > 55 and noise_score >= 55:
            verdict = "SUSPEITA"
            confidence = 70
            reason = "Características mistas - revisão recomendada"
        
        # REGRA 5: Score ponderado (fallback)
        elif weighted_score < 45:
            verdict = "MANIPULADA"
            confidence = 80
            reason = "Score ponderado indica artifícios"
        elif weighted_score > 52:
            verdict = "INCONCLUSIVA"
            confidence = 60
            reason = "Características ambíguas"
        else:
            verdict = "SUSPEITA"
            confidence = 70
            reason = "Indicadores ambíguos - revisão recomendada"
        
        # Adicionar info de reflexo na razão detalhada
        reflection_info = f" [Reflexo: {all_scores['reflection']:.1f}%]" if all_scores['reflection'] > 5 else ""
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "reason": reason,
            "main_score": int(weighted_score),
            "all_scores": all_scores,
            "validation_chain": validation_chain,
            "phases_executed": phases_count,
            "visual_report": texture_result['visual_report'],
            "heatmap": texture_result['heatmap'],
            "percent_suspicious": texture_result['percent_suspicious'],
            "detailed_reason": f"Score ponderado: {int(weighted_score)}/100.{reflection_info}"
        }


def get_image_download_link(img, filename, text):
    if isinstance(img, np.ndarray):
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            img_pil = Image.fromarray(img)
    else:
        img_pil = img
    
    buf = io.BytesIO()
    img_pil.save(buf, format='JPEG', quality=95)
    buf.seek(0)
    
    img_str = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/jpeg;base64,{img_str}" download="{filename}">{text}</a>'
    return href