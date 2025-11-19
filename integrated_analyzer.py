"""
IMPLEMENTACAO FINAL: Sistema Integrado V2
Filtro de Reflexos + Detecção de IA (MirrorGlass V4)

Este arquivo substitui/complementa o texture_analyzer.py atual
"""

import cv2
import numpy as np
from PIL import Image
from reflection_filter import ReflectionDetector


class IntegratedAnalyzer:
    """
    Analisador integrado que:
    1. Detecta e remove reflexos (FASE 0 - NOVA)
    2. Analisa se imagem foi modificada por IA (FASES 1-4)
    """
    
    def __init__(self, enable_reflection_filter=True):
        """
        Args:
            enable_reflection_filter: Se True, ativa pré-filtro de reflexos
        """
        self.enable_reflection_filter = enable_reflection_filter
        
        # Importar analisadores do MirrorGlass V4
        try:
            from texture_analyzer import (
                TextureAnalyzer, EdgeAnalyzer,
                NoiseAnalyzer, LightingAnalyzer
            )
            self.texture_analyzer = TextureAnalyzer()
            self.edge_analyzer = EdgeAnalyzer(use_clahe=True)
            self.noise_analyzer = NoiseAnalyzer(use_clahe=True)
            self.lighting_analyzer = LightingAnalyzer(use_clahe=True)
            self.has_mirrorglassv4 = True
        except ImportError:
            print("⚠️  MirrorGlass V4 não encontrado. Usando análise simplificada.")
            self.has_mirrorglassv4 = False
        
        # Detector de reflexos
        if self.enable_reflection_filter:
            self.reflection_detector = ReflectionDetector(
                intensity_threshold=200,
                saturation_threshold=0.3,
                variance_threshold=1500
            )
    
    def analyze_sequential(self, image):
        """
        Análise completa com pré-filtro de reflexos
        
        Returns:
            dict com resultado completo
        """
        
        # Converter imagem se necessário
        if isinstance(image, Image.Image):
            img_array = np.array(image.convert('RGB'))
        else:
            img_array = image.copy()
        
        validation_chain = []
        all_scores = {}
        
        # ==========================================
        # FASE 0: PRÉ-FILTRO DE REFLEXOS (NOVA!)
        # ==========================================
        
        if self.enable_reflection_filter:
            reflection_result = self.reflection_detector.analyze_image(img_array)
            reflection_score = reflection_result['reflection_score']
            all_scores['reflection'] = reflection_score
            validation_chain.append('reflection_filter')
            
            # Caso 1: Reflexo EXTREMO - Não analisar
            if reflection_result['recommendation'] == 'SKIP_ANALYSIS':
                return {
                    "verdict": "NÃO ANALISÁVEL",
                    "confidence": 95,
                    "reason": "Reflexo extremo detectado",
                    "main_score": 0,
                    "all_scores": all_scores,
                    "validation_chain": validation_chain,
                    "phases_executed": 1,
                    "visual_report": reflection_result['visual_report'],
                    "heatmap": reflection_result['heatmap'],
                    "percent_suspicious": reflection_result['reflection_percentage'],
                    "detailed_reason": f"Imagem com {reflection_score}% de reflexo - impossível analisar."
                }
            
            # Caso 2: Reflexo MODERADO - Limpar antes de analisar
            elif reflection_result['recommendation'] == 'CLEAN_BEFORE_ANALYSIS':
                img_clean, mask = self.reflection_detector.create_clean_image(
                    img_array, method='inpaint'
                )
                img_to_analyze = img_clean
                used_cleaned = True
                
            # Caso 3: Reflexo MÍNIMO - Usar original
            else:
                img_to_analyze = img_array
                used_cleaned = False
        
        else:
            # Filtro desabilitado - usar original
            img_to_analyze = img_array
            used_cleaned = False
        
        # ==========================================
        # FASES 1-4: DETECÇÃO DE IA (MirrorGlass V4)
        # ==========================================
        
        if self.has_mirrorglassv4:
            # Usar sistema completo do MirrorGlass V4
            ai_result = self._analyze_with_mirrorglassv4(img_to_analyze, all_scores, validation_chain)
        else:
            # Análise simplificada (fallback)
            ai_result = self._analyze_simplified(img_to_analyze, all_scores, validation_chain)
        
        # Adicionar informação sobre uso de filtro
        ai_result['reflection_filter_used'] = self.enable_reflection_filter
        ai_result['image_cleaned'] = used_cleaned
        
        return ai_result
    
    def _analyze_with_mirrorglassv4(self, image, all_scores, validation_chain):
        """Análise completa usando MirrorGlass V4"""
        
        # FASE 1: Textura
        texture_result = self.texture_analyzer.analyze_image(image)
        texture_score = texture_result['score']
        all_scores['texture'] = texture_score
        validation_chain.append('texture')
        
        if texture_score < 50:
            return {
                "verdict": "MANIPULADA",
                "confidence": 95,
                "reason": "Textura artificial detectada",
                "main_score": texture_score,
                "all_scores": all_scores,
                "validation_chain": validation_chain,
                "phases_executed": len(validation_chain),
                "visual_report": texture_result['visual_report'],
                "heatmap": texture_result['heatmap'],
                "percent_suspicious": texture_result['percent_suspicious'],
                "detailed_reason": f"Score {texture_score}/100 indica textura artificial."
            }
        
        if texture_score > 75:
            return {
                "verdict": "NATURAL",
                "confidence": 85,
                "reason": "Textura natural com alta variabilidade",
                "main_score": texture_score,
                "all_scores": all_scores,
                "validation_chain": validation_chain,
                "phases_executed": len(validation_chain),
                "visual_report": texture_result['visual_report'],
                "heatmap": texture_result['heatmap'],
                "percent_suspicious": texture_result['percent_suspicious'],
                "detailed_reason": f"Score {texture_score}/100 indica textura natural."
            }
        
        # FASE 2: Bordas
        edge_result = self.edge_analyzer.analyze_image(image)
        edge_score = edge_result['edge_score']
        all_scores['edge'] = edge_score
        validation_chain.append('edge')
        
        if edge_score < 40:
            return {
                "verdict": "MANIPULADA",
                "confidence": 90,
                "reason": "Bordas artificiais detectadas",
                "main_score": texture_score,
                "all_scores": all_scores,
                "validation_chain": validation_chain,
                "phases_executed": len(validation_chain),
                "visual_report": texture_result['visual_report'],
                "heatmap": texture_result['heatmap'],
                "percent_suspicious": texture_result['percent_suspicious'],
                "detailed_reason": f"Bordas artificiais confirmam suspeita."
            }
        
        # FASE 3: Ruído
        noise_result = self.noise_analyzer.analyze_image(image)
        noise_score = noise_result['noise_score']
        all_scores['noise'] = noise_score
        validation_chain.append('noise')
        
        if noise_score < 40:
            return {
                "verdict": "MANIPULADA",
                "confidence": 85,
                "reason": "Ruído artificial detectado",
                "main_score": texture_score,
                "all_scores": all_scores,
                "validation_chain": validation_chain,
                "phases_executed": len(validation_chain),
                "visual_report": texture_result['visual_report'],
                "heatmap": texture_result['heatmap'],
                "percent_suspicious": texture_result['percent_suspicious'],
                "detailed_reason": f"Ruído inconsistente."
            }
        
        # FASE 4: Iluminação
        lighting_result = self.lighting_analyzer.analyze_image(image)
        lighting_score = lighting_result['lighting_score']
        all_scores['lighting'] = lighting_score
        validation_chain.append('lighting')
        
        if lighting_score < 10:
            return {
                "verdict": "MANIPULADA",
                "confidence": 80,
                "reason": "Iluminação física impossível",
                "main_score": texture_score,
                "all_scores": all_scores,
                "validation_chain": validation_chain,
                "phases_executed": len(validation_chain),
                "visual_report": texture_result['visual_report'],
                "heatmap": texture_result['heatmap'],
                "percent_suspicious": texture_result['percent_suspicious'],
                "detailed_reason": f"Iluminação inconsistente."
            }
        
        # Score ponderado final
        weighted_score = (
            texture_score * 0.50 +
            edge_score * 0.25 +
            noise_score * 0.15 +
            lighting_score * 0.10
        )
        
        if weighted_score < 55:
            verdict = "SUSPEITA"
            confidence = 70
            reason = "Múltiplos indicadores ambíguos"
        else:
            verdict = "NATURAL"
            confidence = 75
            reason = "Análise inconclusiva - provável natural"
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "reason": reason,
            "main_score": int(weighted_score),
            "all_scores": all_scores,
            "validation_chain": validation_chain,
            "phases_executed": len(validation_chain),
            "visual_report": texture_result['visual_report'],
            "heatmap": texture_result['heatmap'],
            "percent_suspicious": texture_result['percent_suspicious'],
            "detailed_reason": f"Score ponderado: {int(weighted_score)}/100."
        }
    
    def _analyze_simplified(self, image, all_scores, validation_chain):
        """Análise simplificada (fallback quando MirrorGlass V4 não disponível)"""
        
        # Análise básica de textura
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        variance = np.var(gray)
        mean_intensity = np.mean(gray)
        
        # Score simplificado baseado em variância
        # Imagens naturais têm variância > 1000
        # IA generativa tende a ter variância < 500
        
        if variance > 1500:
            score = 75
            verdict = "NATURAL"
            confidence = 70
        elif variance < 500:
            score = 30
            verdict = "SUSPEITA"
            confidence = 65
        else:
            score = 55
            verdict = "INCONCLUSIVA"
            confidence = 50
        
        all_scores['texture_simple'] = int(score)
        validation_chain.append('simplified_analysis')
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "reason": "Análise simplificada (MirrorGlass V4 não disponível)",
            "main_score": int(score),
            "all_scores": all_scores,
            "validation_chain": validation_chain,
            "phases_executed": len(validation_chain),
            "visual_report": image,
            "heatmap": image,
            "percent_suspicious": 0,
            "detailed_reason": f"Variância: {variance:.0f}"
        }


# ==============================================
# EXEMPLO DE USO
# ==============================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTE: Sistema Integrado V2")
    print("="*60)
    
    # Carregar imagem do caminhão
    image_path = "/mnt/user-data/uploads/Image__2_.png"
    image = Image.open(image_path)
    
    print(f"\n✅ Imagem carregada: {image.size}")
    
    # Criar analisador integrado
    analyzer = IntegratedAnalyzer(enable_reflection_filter=True)
    
    # Análise completa
    print("\n🔬 Iniciando análise sequencial com filtro de reflexos...")
    result = analyzer.analyze_sequential(image)
    
    print("\n" + "="*60)
    print("📊 RESULTADO")
    print("="*60)
    print(f"Veredito: {result['verdict']}")
    print(f"Confiança: {result['confidence']}%")
    print(f"Score: {result['main_score']}/100")
    print(f"Razão: {result['reason']}")
    print(f"Fases executadas: {result['phases_executed']}")
    print(f"Cadeia de validação: {' → '.join(result['validation_chain'])}")
    print(f"Filtro de reflexos usado: {result['reflection_filter_used']}")
    print(f"Imagem foi limpa: {result['image_cleaned']}")
    print("="*60)
    
    print("\n✨ Teste concluído!")
