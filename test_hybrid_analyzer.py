"""
ANALISADOR HÍBRIDO - Melhor dos Dois Mundos!

Estratégia:
- Usa IMAGEM ORIGINAL para análise de bordas (sem artefatos de inpainting)
- Usa IMAGEM LIMPA para análise de textura (sem reflexos)
- Combina os melhores resultados de cada análise
"""

import sys
import numpy as np
from PIL import Image
from datetime import datetime

sys.path.append('/home/claude')
sys.path.append('/mnt/user-data/outputs')

from reflection_filter import ReflectionDetector


class HybridAnalyzer:
    """
    Analisador híbrido inteligente:
    - Original: para bordas, ruído, iluminação
    - Limpa: para textura
    """
    
    def __init__(self, enable_reflection_filter=True):
        self.enable_reflection_filter = enable_reflection_filter
        self.reflection_detector = ReflectionDetector()
        
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
    
    def analyze_sequential(self, image):
        """
        Análise híbrida inteligente
        """
        
        # Converter imagem
        if isinstance(image, Image.Image):
            img_original = np.array(image.convert('RGB'))
        else:
            img_original = image.copy()
        
        validation_chain = []
        all_scores = {}
        
        print("\n" + "="*70)
        print("🔬 ANÁLISE HÍBRIDA INICIADA")
        print("="*70)
        
        # ==========================================
        # FASE 0: PRÉ-FILTRO DE REFLEXOS
        # ==========================================
        
        img_clean = None
        image_was_cleaned = False
        
        if self.enable_reflection_filter:
            print("\n📍 FASE 0: Análise de Reflexos")
            
            reflection_result = self.reflection_detector.analyze_image(img_original)
            reflection_score = reflection_result['reflection_score']
            all_scores['reflection'] = reflection_score
            validation_chain.append('reflection_filter')
            
            print(f"   Score de reflexo: {reflection_score}/100")
            print(f"   Recomendação: {reflection_result['recommendation']}")
            
            # Reflexo extremo
            if reflection_result['recommendation'] == 'SKIP_ANALYSIS':
                print("\n⛔ Reflexo EXTREMO - Não é possível analisar")
                return self._build_result(
                    "NÃO ANALISÁVEL", 95,
                    "Reflexo extremo detectado", 0,
                    all_scores, validation_chain,
                    None, False, img_original,
                    f"Imagem com {reflection_score}% de reflexo."
                )
            
            # Reflexo moderado - criar versão limpa
            elif reflection_result['recommendation'] == 'CLEAN_BEFORE_ANALYSIS':
                print(f"\n🧹 Criando versão limpa para análise de TEXTURA...")
                img_clean, mask = self.reflection_detector.create_clean_image(
                    img_original, method='inpaint'
                )
                image_was_cleaned = True
                print(f"   ✅ Versão limpa criada ({reflection_result['reflection_percentage']:.2f}% removido)")
            
            else:
                print("\n✅ Reflexos mínimos - usar original para tudo")
                image_was_cleaned = False
        
        # ==========================================
        # ESTRATÉGIA HÍBRIDA
        # ==========================================
        
        if image_was_cleaned:
            print("\n" + "="*70)
            print("🔀 ESTRATÉGIA HÍBRIDA ATIVADA")
            print("="*70)
            print("   📸 ORIGINAL → Bordas, Ruído, Iluminação")
            print("   🧹 LIMPA    → Textura")
            print("="*70)
            
            img_for_texture = img_clean      # ← Limpa (sem reflexos)
            img_for_edges = img_original     # ← Original (sem artefatos)
            img_for_noise = img_original     # ← Original (sem artefatos)
            img_for_lighting = img_original  # ← Original (sem artefatos)
        else:
            # Sem reflexos significativos - usar original para tudo
            img_for_texture = img_original
            img_for_edges = img_original
            img_for_noise = img_original
            img_for_lighting = img_original
        
        # ==========================================
        # ANÁLISE COM MIRRORGLASSV4
        # ==========================================
        
        if not self.has_mirrorglassv4:
            return self._analyze_simplified(
                img_original, img_clean, all_scores, 
                validation_chain, image_was_cleaned
            )
        
        # FASE 1: Textura (usa imagem LIMPA se disponível)
        print("\n📍 FASE 1: Análise de Textura")
        print(f"   Usando imagem: {'LIMPA' if image_was_cleaned else 'ORIGINAL'}")
        
        texture_result = self.texture_analyzer.analyze_image(img_for_texture)
        texture_score = texture_result['score']
        all_scores['texture'] = texture_score
        validation_chain.append('texture')
        
        print(f"   Score: {texture_score}/100")
        
        if texture_score < 50:
            print(f"   ❌ Textura artificial detectada!")
            return self._build_result(
                "MANIPULADA", 95, "Textura artificial detectada",
                texture_score, all_scores, validation_chain,
                texture_result, image_was_cleaned, img_original,
                f"Score {texture_score}/100 indica textura artificial."
            )
        
        if texture_score > 75:
            print(f"   ✅ Textura natural detectada!")
            return self._build_result(
                "NATURAL", 85, "Textura natural com alta variabilidade",
                texture_score, all_scores, validation_chain,
                texture_result, image_was_cleaned, img_original,
                f"Score {texture_score}/100 indica textura natural."
            )
        
        print(f"   ⚠️  Textura inconclusiva - continuando análise...")
        
        # FASE 2: Bordas (usa imagem ORIGINAL!)
        print("\n📍 FASE 2: Análise de Bordas")
        print(f"   Usando imagem: ORIGINAL (sem artefatos de inpainting)")
        
        edge_result = self.edge_analyzer.analyze_image(img_for_edges)
        edge_score = edge_result['edge_score']
        all_scores['edge'] = edge_score
        validation_chain.append('edge')
        
        print(f"   Score: {edge_score}/100")
        
        if edge_score < 40:
            print(f"   ❌ Bordas artificiais detectadas!")
            return self._build_result(
                "MANIPULADA", 90, "Bordas artificiais detectadas",
                texture_score, all_scores, validation_chain,
                texture_result, image_was_cleaned, img_original,
                f"Bordas artificiais confirmam suspeita."
            )
        
        print(f"   ✅ Bordas naturais")
        
        # FASE 3: Ruído (usa imagem ORIGINAL!)
        print("\n📍 FASE 3: Análise de Ruído")
        print(f"   Usando imagem: ORIGINAL")
        
        noise_result = self.noise_analyzer.analyze_image(img_for_noise)
        noise_score = noise_result['noise_score']
        all_scores['noise'] = noise_score
        validation_chain.append('noise')
        
        print(f"   Score: {noise_score}/100")
        
        if noise_score < 40:
            print(f"   ❌ Ruído artificial detectado!")
            return self._build_result(
                "MANIPULADA", 85, "Ruído artificial detectado",
                texture_score, all_scores, validation_chain,
                texture_result, image_was_cleaned, img_original,
                f"Ruído inconsistente."
            )
        
        print(f"   ✅ Ruído natural")
        
        # FASE 4: Iluminação (usa imagem ORIGINAL!)
        print("\n📍 FASE 4: Análise de Iluminação")
        print(f"   Usando imagem: ORIGINAL")
        
        lighting_result = self.lighting_analyzer.analyze_image(img_for_lighting)
        lighting_score = lighting_result['lighting_score']
        all_scores['lighting'] = lighting_score
        validation_chain.append('lighting')
        
        print(f"   Score: {lighting_score}/100")
        
        if lighting_score < 10:
            print(f"   ❌ Iluminação impossível!")
            return self._build_result(
                "MANIPULADA", 80, "Iluminação física impossível",
                texture_score, all_scores, validation_chain,
                texture_result, image_was_cleaned, img_original,
                f"Iluminação inconsistente."
            )
        
        print(f"   ✅ Iluminação natural")
        
        # ==========================================
        # DECISÃO FINAL PONDERADA
        # ==========================================
        
        print("\n📍 DECISÃO FINAL")
        
        weighted_score = (
            texture_score * 0.50 +
            edge_score * 0.25 +
            noise_score * 0.15 +
            lighting_score * 0.10
        )
        
        print(f"   Score ponderado: {int(weighted_score)}/100")
        print(f"   Componentes:")
        print(f"      Textura: {texture_score} × 50% = {texture_score * 0.50:.1f}")
        print(f"      Bordas:  {edge_score} × 25% = {edge_score * 0.25:.1f}")
        print(f"      Ruído:   {noise_score} × 15% = {noise_score * 0.15:.1f}")
        print(f"      Luz:     {lighting_score} × 10% = {lighting_score * 0.10:.1f}")
        
        if weighted_score < 55:
            verdict = "SUSPEITA"
            confidence = 70
            reason = "Múltiplos indicadores ambíguos"
        elif weighted_score < 65:
            verdict = "INCONCLUSIVA"
            confidence = 65
            reason = "Análise ambígua - revisar manualmente"
        else:
            verdict = "NATURAL"
            confidence = 80
            reason = "Todos os indicadores apontam para imagem natural"
        
        print(f"\n   🎯 Veredito: {verdict} ({confidence}% confiança)")
        
        return self._build_result(
            verdict, confidence, reason,
            int(weighted_score), all_scores, validation_chain,
            texture_result, image_was_cleaned, img_original,
            f"Score ponderado: {int(weighted_score)}/100."
        )
    
    def _analyze_simplified(self, img_original, img_clean, all_scores, 
                           validation_chain, image_was_cleaned):
        """Análise simplificada (fallback)"""
        
        # Usar imagem limpa para textura se disponível
        img_to_analyze = img_clean if image_was_cleaned else img_original
        
        gray = np.array(Image.fromarray(img_to_analyze).convert('L'))
        variance = np.var(gray)
        
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
        
        return self._build_result(
            verdict, confidence,
            "Análise simplificada (MirrorGlass V4 não disponível)",
            int(score), all_scores, validation_chain,
            None, image_was_cleaned, img_original,
            f"Variância: {variance:.0f}"
        )
    
    def _build_result(self, verdict, confidence, reason, main_score,
                     all_scores, validation_chain, texture_result,
                     image_cleaned, img_original, detailed_reason):
        """Helper para construir resultado"""
        
        if texture_result:
            visual_report = texture_result['visual_report']
            heatmap = texture_result['heatmap']
            percent_suspicious = texture_result['percent_suspicious']
        else:
            visual_report = img_original
            heatmap = img_original
            percent_suspicious = 0
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "reason": reason,
            "main_score": main_score,
            "all_scores": all_scores,
            "validation_chain": validation_chain,
            "phases_executed": len(validation_chain),
            "visual_report": visual_report,
            "heatmap": heatmap,
            "percent_suspicious": percent_suspicious,
            "detailed_reason": detailed_reason,
            "reflection_filter_used": self.enable_reflection_filter,
            "image_cleaned": image_cleaned,
            "strategy": "HYBRID" if image_cleaned else "STANDARD"
        }


# ==============================================
# TESTE COMPLETO
# ==============================================

def test_hybrid_analyzer():
    """Teste do analisador híbrido"""
    
    print("\n" + "="*70)
    print("🧪 TESTE: ANALISADOR HÍBRIDO")
    print("="*70)
    print("\nESTRATÉGIA:")
    print("   • IMAGEM ORIGINAL → Bordas, Ruído, Iluminação")
    print("   • IMAGEM LIMPA    → Textura")
    print("="*70)
    
    # Carregar imagem
    image_path = "C:\\Users\\efelipe\\OneDrive - CARGLASS AUTOMOTIVA LTDA\\Documentos\\Projetos_python\\MirrorV2-main\\img\\Caminhão\\Image (1).png"
    image = Image.open(image_path)
    
    print(f"\n✅ Imagem carregada: {image.size}")
    
    # Teste 1: Analisador híbrido
    print("\n" + "="*70)
    print("TESTE 1: ANALISADOR HÍBRIDO")
    print("="*70)
    
    analyzer_hybrid = HybridAnalyzer(enable_reflection_filter=True)
    result_hybrid = analyzer_hybrid.analyze_sequential(image)
    
    print("\n" + "="*70)
    print("📊 RESULTADO FINAL - ANALISADOR HÍBRIDO")
    print("="*70)
    print(f"Veredito: {result_hybrid['verdict']}")
    print(f"Confiança: {result_hybrid['confidence']}%")
    print(f"Score: {result_hybrid['main_score']}/100")
    print(f"Razão: {result_hybrid['reason']}")
    print(f"Estratégia usada: {result_hybrid['strategy']}")
    print(f"Fases executadas: {result_hybrid['phases_executed']}")
    print(f"Cadeia: {' → '.join(result_hybrid['validation_chain'])}")
    print(f"Imagem foi limpa: {result_hybrid['image_cleaned']}")
    
    if 'all_scores' in result_hybrid:
        print(f"\n📊 Scores detalhados:")
        for key, value in result_hybrid['all_scores'].items():
            print(f"   {key}: {value}/100")
    
    print("="*70)
    
    # Comparar com análise original (se tiver MirrorGlass V4)
    print("\n💡 VANTAGENS DO ANALISADOR HÍBRIDO:")
    print("   ✅ Usa original para bordas → Sem artefatos de inpainting")
    print("   ✅ Usa limpa para textura → Sem interferência de reflexos")
    print("   ✅ Melhor dos dois mundos!")
    
    return result_hybrid


if __name__ == "__main__":
    result = test_hybrid_analyzer()
    
    print("\n✨ Teste concluído!")
    
    # Salvar timestamp para relatório
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n📝 Timestamp: {timestamp}")
