"""
Sistema de Detecção e Filtragem de Reflexos
Integração com MirrorGlass V4
"""

import cv2
import numpy as np
from PIL import Image


class ReflectionDetector:
    """
    Detector especializado em reflexos especulares.
    Identifica áreas com características de reflexo antes da análise de avarias.
    """
    
    def __init__(
        self,
        intensity_threshold=200,      # Reflexos são muito brilhantes
        saturation_threshold=0.3,     # Reflexos têm baixa saturação
        variance_threshold=1500,      # Reflexos têm alta variância local
        min_reflection_area=100,      # Área mínima em pixels
        edge_coherence_threshold=0.65 # Reflexos têm bordas coerentes
    ):
        self.intensity_threshold = intensity_threshold
        self.saturation_threshold = saturation_threshold
        self.variance_threshold = variance_threshold
        self.min_reflection_area = min_reflection_area
        self.edge_coherence_threshold = edge_coherence_threshold
    
    def _convert_to_arrays(self, image):
        """Converte PIL Image ou array numpy para formato padrão"""
        if isinstance(image, Image.Image):
            return np.array(image.convert('RGB'))
        return image.copy()
    
    def detect_by_intensity(self, image):
        """
        MÉTODO 1: Detecta reflexos por intensidade extrema
        Reflexos têm valores RGB muito altos (próximos de 255)
        """
        img = self._convert_to_arrays(image)
        
        # Converter para HSV para separar intensidade
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # Reflexos: VALUE alto + SATURATION baixa
        high_intensity = v > self.intensity_threshold
        low_saturation = s < (self.saturation_threshold * 255)
        
        # Combinar condições
        reflection_mask = np.logical_and(high_intensity, low_saturation)
        
        return reflection_mask.astype(np.uint8)
    
    def detect_by_variance(self, image, block_size=32):
        """
        MÉTODO 2: Detecta reflexos por alta variância local
        Reflexos de céu/nuvens têm transições abruptas
        """
        img = self._convert_to_arrays(image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        height, width = gray.shape
        variance_map = np.zeros_like(gray, dtype=np.float32)
        
        # Calcular variância em janelas deslizantes
        for i in range(0, height - block_size, block_size // 2):
            for j in range(0, width - block_size, block_size // 2):
                block = gray[i:i+block_size, j:j+block_size]
                var = np.var(block)
                variance_map[i:i+block_size, j:j+block_size] = var
        
        # Reflexos têm variância muito alta
        reflection_mask = variance_map > self.variance_threshold
        
        return reflection_mask.astype(np.uint8)
    
    def detect_by_edge_coherence(self, image):
        """
        MÉTODO 3: Detecta reflexos por coerência de bordas
        Reflexos têm bordas com direções muito uniformes
        """
        img = self._convert_to_arrays(image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Calcular gradientes
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Analisar coerência em blocos
        height, width = gray.shape
        block_size = 32
        coherence_map = np.zeros((height, width))
        
        for i in range(0, height - block_size, block_size):
            for j in range(0, width - block_size, block_size):
                block_mag = magnitude[i:i+block_size, j:j+block_size]
                block_dir = direction[i:i+block_size, j:j+block_size]
                
                # Só analisar onde há bordas significativas
                if np.sum(block_mag > 10) > 20:
                    # Calcular coerência circular
                    mean_cos = np.mean(np.cos(block_dir))
                    mean_sin = np.mean(np.sin(block_dir))
                    coherence = np.sqrt(mean_cos**2 + mean_sin**2)
                    
                    coherence_map[i:i+block_size, j:j+block_size] = coherence
        
        # Reflexos têm alta coerência
        reflection_mask = coherence_map > self.edge_coherence_threshold
        
        return reflection_mask.astype(np.uint8)
    
    def detect_by_spectral_analysis(self, image):
        """
        MÉTODO 4: Detecta reflexos por análise espectral
        Reflexos de céu têm padrão de cor específico (azul/branco)
        """
        img = self._convert_to_arrays(image)
        
        # Converter para LAB para análise de cor
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Reflexos de céu: alto L (brilho), baixo A e B (pouca cor)
        high_luminance = l > 180
        low_chroma = (np.abs(a - 128) < 20) & (np.abs(b - 128) < 20)
        
        sky_reflection = np.logical_and(high_luminance, low_chroma)
        
        return sky_reflection.astype(np.uint8)
    
    def combine_methods(self, image):
        """
        Combina todos os métodos para detecção robusta
        Usa votação majoritária
        """
        # Executar todos os detectores
        mask1 = self.detect_by_intensity(image)
        mask2 = self.detect_by_variance(image)
        mask3 = self.detect_by_edge_coherence(image)
        mask4 = self.detect_by_spectral_analysis(image)
        
        # Votação: se 2+ métodos concordam, é reflexo
        combined = mask1 + mask2 + mask3 + mask4
        reflection_mask = (combined >= 2).astype(np.uint8)
        
        # Pós-processamento: remover ruído
        kernel = np.ones((5, 5), np.uint8)
        reflection_mask = cv2.morphologyEx(reflection_mask, cv2.MORPH_CLOSE, kernel)
        reflection_mask = cv2.morphologyEx(reflection_mask, cv2.MORPH_OPEN, kernel)
        
        # Remover regiões muito pequenas
        contours, _ = cv2.findContours(
            reflection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_reflection_area:
                cv2.drawContours(reflection_mask, [contour], -1, 0, -1)
        
        return reflection_mask
    
    def calculate_reflection_score(self, image):
        """
        Calcula um score indicando probabilidade de reflexo
        0-100: 0=sem reflexo, 100=reflexo extremo
        """
        reflection_mask = self.combine_methods(image)
        
        # Calcular percentual da imagem com reflexo
        img = self._convert_to_arrays(image)
        total_pixels = img.shape[0] * img.shape[1]
        reflection_pixels = np.sum(reflection_mask > 0)
        
        reflection_percentage = (reflection_pixels / total_pixels) * 100
        
        # Calcular intensidade média nas áreas de reflexo
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if reflection_pixels > 0:
            mean_intensity = np.mean(gray[reflection_mask > 0])
            intensity_score = (mean_intensity / 255) * 100
        else:
            intensity_score = 0
        
        # Score final: combinação de área e intensidade
        final_score = (reflection_percentage * 0.6 + intensity_score * 0.4)
        final_score = min(100, final_score)
        
        return {
            'reflection_score': int(final_score),
            'reflection_percentage': float(reflection_percentage),
            'mean_intensity': float(intensity_score),
            'has_reflection': final_score > 30
        }
    
    def create_clean_image(self, image, method='inpaint'):
        """
        Remove reflexos da imagem
        method: 'inpaint', 'blur', 'darken'
        """
        img = self._convert_to_arrays(image)
        reflection_mask = self.combine_methods(image)
        
        if method == 'inpaint':
            # Inpainting: preenche áreas de reflexo
            clean = cv2.inpaint(img, reflection_mask, 3, cv2.INPAINT_TELEA)
        
        elif method == 'blur':
            # Blur: suaviza reflexos
            blurred = cv2.GaussianBlur(img, (15, 15), 0)
            clean = np.where(reflection_mask[:, :, None] > 0, blurred, img)
        
        elif method == 'darken':
            # Darken: reduz intensidade dos reflexos
            darkened = (img * 0.5).astype(np.uint8)
            clean = np.where(reflection_mask[:, :, None] > 0, darkened, img)
        
        else:
            clean = img
        
        return clean, reflection_mask
    
    def generate_visual_report(self, image):
        """Gera visualização da detecção de reflexos"""
        img = self._convert_to_arrays(image)
        reflection_mask = self.combine_methods(image)
        
        # Criar heatmap
        heatmap = cv2.applyColorMap(reflection_mask * 255, cv2.COLORMAP_HOT)
        
        # Overlay
        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        
        # Desenhar contornos
        contours, _ = cv2.findContours(
            reflection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        
        # Adicionar texto
        score_info = self.calculate_reflection_score(image)
        text = f"Reflection Score: {score_info['reflection_score']}/100"
        cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (255, 255, 255), 2)
        
        return overlay, heatmap
    
    def analyze_image(self, image):
        """
        Análise completa de reflexos
        Retorna: dicionário com todas as informações
        """
        reflection_mask = self.combine_methods(image)
        score_info = self.calculate_reflection_score(image)
        visual_report, heatmap = self.generate_visual_report(image)
        
        # Classificação
        score = score_info['reflection_score']
        if score > 60:
            category = "Reflexo Extremo"
            description = "Imagem dominada por reflexos - não analisar"
            recommendation = "SKIP_ANALYSIS"
        elif score > 30:
            category = "Reflexo Moderado"
            description = "Reflexos significativos detectados"
            recommendation = "CLEAN_BEFORE_ANALYSIS"
        else:
            category = "Reflexo Mínimo"
            description = "Poucos reflexos - pode analisar normalmente"
            recommendation = "PROCEED"
        
        return {
            'reflection_score': score,
            'category': category,
            'description': description,
            'recommendation': recommendation,
            'reflection_percentage': score_info['reflection_percentage'],
            'reflection_mask': reflection_mask,
            'visual_report': visual_report,
            'heatmap': heatmap
        }
