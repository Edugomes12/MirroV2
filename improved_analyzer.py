"""
ANÁLISE E AJUSTE: Tratamento de Artefatos de Inpainting

O problema: Inpainting pode criar bordas artificiais que são detectadas
como manipulação por IA, criando um "falso positivo invertido".

Solução: Ajustar thresholds e adicionar validação extra após inpainting.
"""

from integrated_analyzer import IntegratedAnalyzer
from reflection_filter import ReflectionDetector
from PIL import Image
import numpy as np


class ImprovedIntegratedAnalyzer(IntegratedAnalyzer):
    """
    Versão melhorada que trata artefatos de inpainting
    """
    
    def __init__(self, enable_reflection_filter=True, 
                 edge_threshold_adjustment=True):
        super().__init__(enable_reflection_filter)
        self.edge_threshold_adjustment = edge_threshold_adjustment
    
    def analyze_sequential(self, image):
        """
        Análise com ajuste para artefatos de inpainting
        """
        
        # Converter imagem
        if isinstance(image, Image.Image):
            img_array = np.array(image.convert('RGB'))
        else:
            img_array = image.copy()
        
        validation_chain = []
        all_scores = {}
        image_was_cleaned = False
        
        # ==========================================
        # FASE 0: PRÉ-FILTRO DE REFLEXOS
        # ==========================================
        
        if self.enable_reflection_filter:
            reflection_result = self.reflection_detector.analyze_image(img_array)
            reflection_score = reflection_result['reflection_score']
            all_scores['reflection'] = reflection_score
            validation_chain.append('reflection_filter')
            
            # Reflexo extremo
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
                    "detailed_reason": f"Imagem com {reflection_score}% de reflexo - impossível analisar.",
                    "reflection_filter_used": True,
                    "image_cleaned": False
                }
            
            # Reflexo moderado - limpar
            elif reflection_result['recommendation'] == 'CLEAN_BEFORE_ANALYSIS':
                img_clean, mask = self.reflection_detector.create_clean_image(
                    img_array, method='inpaint'
                )
                img_to_analyze = img_clean
                image_was_cleaned = True
                
                print(f"\n🧹 Imagem limpa automaticamente")
                print(f"   Área com reflexo removida: {reflection_result['reflection_percentage']:.2f}%")
                
            else:
                img_to_analyze = img_array
                image_was_cleaned = False
        
        else:
            img_to_analyze = img_array
            image_was_cleaned = False
        
        # ==========================================
        # ANÁLISE COM MIRRORGLASSV4 (AJUSTADA)
        # ==========================================
        
        if not self.has_mirrorglassv4:
            return self._analyze_simplified(img_to_analyze, all_scores, validation_chain)
        
        # FASE 1: Textura
        texture_result = self.texture_analyzer.analyze_image(img_to_analyze)
        texture_score = texture_result['score']
        all_scores['texture'] = texture_score
        validation_chain.append('texture')
        
        if texture_score < 50:
            return self._build_result(
                "MANIPULADA", 95, "Textura artificial detectada",
                texture_score, all_scores, validation_chain,
                texture_result, image_was_cleaned,
                f"Score {texture_score}/100 indica textura artificial."
            )
        
        if texture_score > 75:
            return self._build_result(
                "NATURAL", 85, "Textura natural com alta variabilidade",
                texture_score, all_scores, validation_chain,
                texture_result, image_was_cleaned,
                f"Score {texture_score}/100 indica textura natural."
            )
        
        # FASE 2: Bordas (COM AJUSTE!)
        edge_result = self.edge_analyzer.analyze_image(img_to_analyze)
        edge_score = edge_result['edge_score']
        all_scores['edge'] = edge_score
        validation_chain.append('edge')
        
        # AJUSTE CRÍTICO: Se imagem foi limpa, ser mais permissivo com bordas
        if self.edge_threshold_adjustment and image_was_cleaned:
            edge_threshold = 35  # Mais permissivo (era 40)
            print(f"\n⚙️  Ajuste ativado: Threshold de bordas reduzido para {edge_threshold}")
            print(f"   Motivo: Inpainting pode criar artefatos de borda")
        else:
            edge_threshold = 40  # Padrão
        
        if edge_score < edge_threshold:
            # VALIDAÇÃO EXTRA: Se imagem foi limpa E score de textura é OK
            if image_was_cleaned and texture_score >= 65:
                print(f"\n⚠️  Bordas artificiais detectadas (score: {edge_score})")
                print(f"   MAS textura é boa ({texture_score}/100)")
                print(f"   Provável artefato de inpainting - continuando análise...")
                
                # Continuar para próxima fase ao invés de parar
                pass
            else:
                return self._build_result(
                    "MANIPULADA", 90, "Bordas artificiais detectadas",
                    texture_score, all_scores, validation_chain,
                    texture_result, image_was_cleaned,
                    f"Bordas artificiais confirmam suspeita."
                )
        
        # FASE 3: Ruído
        noise_result = self.noise_analyzer.analyze_image(img_to_analyze)
        noise_score = noise_result['noise_score']
        all_scores['noise'] = noise_score
        validation_chain.append('noise')
        
        if noise_score < 40:
            # Mesma lógica: se imagem foi limpa, ser mais cuidadoso
            if image_was_cleaned and texture_score >= 65:
                print(f"\n⚠️  Ruído artificial detectado (score: {noise_score})")
                print(f"   MAS textura é boa ({texture_score}/100)")
                print(f"   Continuando análise...")
                pass
            else:
                return self._build_result(
                    "MANIPULADA", 85, "Ruído artificial detectado",
                    texture_score, all_scores, validation_chain,
                    texture_result, image_was_cleaned,
                    f"Ruído inconsistente."
                )
        
        # FASE 4: Iluminação
        lighting_result = self.lighting_analyzer.analyze_image(img_to_analyze)
        lighting_score = lighting_result['lighting_score']
        all_scores['lighting'] = lighting_score
        validation_chain.append('lighting')
        
        if lighting_score < 10:
            return self._build_result(
                "MANIPULADA", 80, "Iluminação física impossível",
                texture_score, all_scores, validation_chain,
                texture_result, image_was_cleaned,
                f"Iluminação inconsistente."
            )
        
        # DECISÃO FINAL PONDERADA (COM AJUSTE!)
        # Se imagem foi limpa, dar mais peso à textura
        if image_was_cleaned:
            weighted_score = (
                texture_score * 0.60 +   # Aumentado de 0.50
                edge_score * 0.20 +      # Reduzido de 0.25
                noise_score * 0.12 +     # Reduzido de 0.15
                lighting_score * 0.08    # Reduzido de 0.10
            )
            print(f"\n⚙️  Pesos ajustados (imagem foi limpa):")
            print(f"   Textura: 60% | Bordas: 20% | Ruído: 12% | Luz: 8%")
        else:
            weighted_score = (
                texture_score * 0.50 +
                edge_score * 0.25 +
                noise_score * 0.15 +
                lighting_score * 0.10
            )
        
        # Decidir baseado no score ponderado
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
            confidence = 75
            reason = "Score ponderado indica imagem natural"
        
        # AJUSTE FINAL: Se imagem foi limpa e textura é boa, favorecer NATURAL
        if image_was_cleaned and texture_score >= 68 and verdict in ["SUSPEITA", "INCONCLUSIVA"]:
            print(f"\n✅ Ajuste final aplicado:")
            print(f"   Textura boa ({texture_score}) + Imagem limpa")
            print(f"   Veredito ajustado: NATURAL")
            verdict = "NATURAL"
            confidence = 75
            reason = "Textura natural - artefatos de inpainting desconsiderados"
        
        return self._build_result(
            verdict, confidence, reason,
            int(weighted_score), all_scores, validation_chain,
            texture_result, image_was_cleaned,
            f"Score ponderado: {int(weighted_score)}/100."
        )
    
    def _build_result(self, verdict, confidence, reason, main_score,
                     all_scores, validation_chain, texture_result,
                     image_cleaned, detailed_reason):
        """Helper para construir resultado"""
        return {
            "verdict": verdict,
            "confidence": confidence,
            "reason": reason,
            "main_score": main_score,
            "all_scores": all_scores,
            "validation_chain": validation_chain,
            "phases_executed": len(validation_chain),
            "visual_report": texture_result['visual_report'],
            "heatmap": texture_result['heatmap'],
            "percent_suspicious": texture_result['percent_suspicious'],
            "detailed_reason": detailed_reason,
            "reflection_filter_used": self.enable_reflection_filter,
            "image_cleaned": image_cleaned
        }


# ==============================================
# TESTE COMPARATIVO
# ==============================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTE COMPARATIVO: Versão Original vs Ajustada")
    print("="*70)
    
    image_path = "/mnt/user-data/uploads/Image__2_.png"
    image = Image.open(image_path)
    
    print(f"\n✅ Imagem carregada: {image.size}")
    
    # Teste 1: Versão Original
    print("\n" + "="*70)
    print("TESTE 1: ANALISADOR ORIGINAL")
    print("="*70)
    
    analyzer_original = IntegratedAnalyzer(enable_reflection_filter=True)
    result_original = analyzer_original.analyze_sequential(image)
    
    print(f"\n📊 Resultado Original:")
    print(f"   Veredito: {result_original['verdict']}")
    print(f"   Confiança: {result_original['confidence']}%")
    print(f"   Score: {result_original['main_score']}/100")
    print(f"   Razão: {result_original['reason']}")
    
    # Teste 2: Versão Ajustada
    print("\n" + "="*70)
    print("TESTE 2: ANALISADOR AJUSTADO")
    print("="*70)
    
    analyzer_improved = ImprovedIntegratedAnalyzer(
        enable_reflection_filter=True,
        edge_threshold_adjustment=True
    )
    result_improved = analyzer_improved.analyze_sequential(image)
    
    print(f"\n📊 Resultado Ajustado:")
    print(f"   Veredito: {result_improved['verdict']}")
    print(f"   Confiança: {result_improved['confidence']}%")
    print(f"   Score: {result_improved['main_score']}/100")
    print(f"   Razão: {result_improved['reason']}")
    
    # Comparação
    print("\n" + "="*70)
    print("COMPARAÇÃO")
    print("="*70)
    
    print(f"\nOriginal: {result_original['verdict']} ({result_original['confidence']}%)")
    print(f"Ajustado: {result_improved['verdict']} ({result_improved['confidence']}%)")
    
    if result_original['verdict'] != result_improved['verdict']:
        print(f"\n⚠️  DIFERENÇA DETECTADA!")
        print(f"   Versão ajustada corrigiu possível falso positivo causado por inpainting")
    else:
        print(f"\n✅ Ambas versões concordam no veredito")
    
    print("\n✨ Teste concluído!")
