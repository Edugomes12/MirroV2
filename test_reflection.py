"""
TEST_REFLECTION.PY - Teste Completo do Sistema Integrado
Valida todas as funcionalidades antes da implementação em produção
"""

import sys
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Adicionar diretórios ao path se necessário
sys.path.append('/home/claude')
sys.path.append('/mnt/user-data/outputs')

# Imports dos módulos
try:
    from reflection_filter import ReflectionDetector
    print("✅ ReflectionDetector importado com sucesso")
except ImportError as e:
    print(f"❌ Erro ao importar ReflectionDetector: {e}")
    sys.exit(1)

try:
    from integrated_analyzer import IntegratedAnalyzer
    print("✅ IntegratedAnalyzer importado com sucesso")
except ImportError as e:
    print(f"❌ Erro ao importar IntegratedAnalyzer: {e}")
    sys.exit(1)


class TestRunner:
    """Executor de testes para o sistema integrado"""
    
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None
        self.test_results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def load_image(self):
        """Teste 1: Carregar imagem"""
        print("\n" + "="*70)
        print("TESTE 1: CARREGAMENTO DE IMAGEM")
        print("="*70)
        
        try:
            self.image = Image.open(self.image_path)
            print(f"✅ PASSOU - Imagem carregada")
            print(f"   Tamanho: {self.image.size[0]}x{self.image.size[1]} pixels")
            print(f"   Formato: {self.image.format}")
            print(f"   Modo: {self.image.mode}")
            self.test_results.append(("Carregamento", True, "OK"))
            return True
        except Exception as e:
            print(f"❌ FALHOU - Erro: {e}")
            self.test_results.append(("Carregamento", False, str(e)))
            return False
    
    def test_reflection_detector(self):
        """Teste 2: Detector de reflexos"""
        print("\n" + "="*70)
        print("TESTE 2: DETECTOR DE REFLEXOS")
        print("="*70)
        
        try:
            detector = ReflectionDetector()
            print("✅ Detector criado com sucesso")
            
            # Testar análise
            result = detector.analyze_image(self.image)
            
            print(f"\n📊 Resultados da análise:")
            print(f"   Score de reflexo: {result['reflection_score']}/100")
            print(f"   Categoria: {result['category']}")
            print(f"   Área com reflexo: {result['reflection_percentage']:.2f}%")
            print(f"   Recomendação: {result['recommendation']}")
            
            # Validações
            assert 0 <= result['reflection_score'] <= 100, "Score fora do range 0-100"
            assert result['category'] in ['Reflexo Extremo', 'Reflexo Moderado', 'Reflexo Mínimo'], "Categoria inválida"
            assert result['recommendation'] in ['SKIP_ANALYSIS', 'CLEAN_BEFORE_ANALYSIS', 'PROCEED'], "Recomendação inválida"
            
            print("\n✅ PASSOU - Todas as validações OK")
            self.test_results.append(("Detector de Reflexos", True, 
                                     f"Score: {result['reflection_score']}, {result['category']}"))
            
            return result
            
        except Exception as e:
            print(f"\n❌ FALHOU - Erro: {e}")
            self.test_results.append(("Detector de Reflexos", False, str(e)))
            return None
    
    def test_reflection_methods(self):
        """Teste 3: Métodos individuais de detecção"""
        print("\n" + "="*70)
        print("TESTE 3: MÉTODOS INDIVIDUAIS DE DETECÇÃO")
        print("="*70)
        
        try:
            detector = ReflectionDetector()
            img_array = np.array(self.image.convert('RGB'))
            
            methods = [
                ("Intensidade", detector.detect_by_intensity),
                ("Variância", detector.detect_by_variance),
                ("Coerência de Bordas", detector.detect_by_edge_coherence),
                ("Análise Espectral", detector.detect_by_spectral_analysis)
            ]
            
            all_passed = True
            
            for method_name, method_func in methods:
                try:
                    mask = method_func(img_array)
                    
                    # Validações
                    assert mask.dtype == np.uint8, f"{method_name}: Tipo incorreto"
                    assert mask.shape == img_array.shape[:2], f"{method_name}: Shape incorreto"
                    assert np.all((mask == 0) | (mask == 1)), f"{method_name}: Valores não binários"
                    
                    percentage = (np.sum(mask > 0) / mask.size) * 100
                    print(f"✅ {method_name}: {percentage:.2f}% detectado")
                    
                except Exception as e:
                    print(f"❌ {method_name}: Falhou - {e}")
                    all_passed = False
            
            if all_passed:
                print("\n✅ PASSOU - Todos os métodos funcionando")
                self.test_results.append(("Métodos Individuais", True, "4/4 métodos OK"))
            else:
                print("\n⚠️  PASSOU PARCIALMENTE - Alguns métodos falharam")
                self.test_results.append(("Métodos Individuais", True, "Alguns falharam"))
            
            return all_passed
            
        except Exception as e:
            print(f"\n❌ FALHOU - Erro: {e}")
            self.test_results.append(("Métodos Individuais", False, str(e)))
            return False
    
    def test_image_cleaning(self):
        """Teste 4: Limpeza de imagem (inpainting)"""
        print("\n" + "="*70)
        print("TESTE 4: LIMPEZA DE IMAGEM (INPAINTING)")
        print("="*70)
        
        try:
            detector = ReflectionDetector()
            img_array = np.array(self.image.convert('RGB'))
            
            methods = ['inpaint', 'blur', 'darken']
            all_passed = True
            
            for method in methods:
                try:
                    clean_img, mask = detector.create_clean_image(img_array, method=method)
                    
                    # Validações
                    assert clean_img.shape == img_array.shape, f"{method}: Shape diferente"
                    assert clean_img.dtype == img_array.dtype, f"{method}: Tipo diferente"
                    assert not np.array_equal(clean_img, img_array), f"{method}: Imagem não foi modificada"
                    
                    print(f"✅ Método '{method}': OK")
                    
                except Exception as e:
                    print(f"❌ Método '{method}': Falhou - {e}")
                    all_passed = False
            
            if all_passed:
                print("\n✅ PASSOU - Todos os métodos de limpeza funcionando")
                self.test_results.append(("Limpeza de Imagem", True, "3/3 métodos OK"))
            else:
                print("\n⚠️  PASSOU PARCIALMENTE")
                self.test_results.append(("Limpeza de Imagem", True, "Alguns falharam"))
            
            return all_passed
            
        except Exception as e:
            print(f"\n❌ FALHOU - Erro: {e}")
            self.test_results.append(("Limpeza de Imagem", False, str(e)))
            return False
    
    def test_integrated_analyzer_without_filter(self):
        """Teste 5: Analisador integrado SEM filtro"""
        print("\n" + "="*70)
        print("TESTE 5: ANÁLISE SEM FILTRO DE REFLEXOS (Baseline)")
        print("="*70)
        
        try:
            analyzer = IntegratedAnalyzer(enable_reflection_filter=False)
            print("✅ Analisador criado (filtro DESATIVADO)")
            
            result = analyzer.analyze_sequential(self.image)
            
            print(f"\n📊 Resultados SEM filtro:")
            print(f"   Veredito: {result['verdict']}")
            print(f"   Confiança: {result['confidence']}%")
            print(f"   Score: {result['main_score']}/100")
            print(f"   Fases executadas: {result['phases_executed']}")
            print(f"   Filtro usado: {result.get('reflection_filter_used', False)}")
            
            # Validações
            assert 'verdict' in result, "Falta campo 'verdict'"
            assert 'confidence' in result, "Falta campo 'confidence'"
            assert 'main_score' in result, "Falta campo 'main_score'"
            assert result['reflection_filter_used'] == False, "Filtro deveria estar desativado"
            
            print("\n✅ PASSOU - Análise sem filtro funcionando")
            self.test_results.append(("Análise sem Filtro", True, 
                                     f"{result['verdict']}, Score: {result['main_score']}"))
            
            return result
            
        except Exception as e:
            print(f"\n❌ FALHOU - Erro: {e}")
            self.test_results.append(("Análise sem Filtro", False, str(e)))
            return None
    
    def test_integrated_analyzer_with_filter(self):
        """Teste 6: Analisador integrado COM filtro"""
        print("\n" + "="*70)
        print("TESTE 6: ANÁLISE COM FILTRO DE REFLEXOS (Sistema V2)")
        print("="*70)
        
        try:
            analyzer = IntegratedAnalyzer(enable_reflection_filter=True)
            print("✅ Analisador criado (filtro ATIVADO)")
            
            result = analyzer.analyze_sequential(self.image)
            
            print(f"\n📊 Resultados COM filtro:")
            print(f"   Veredito: {result['verdict']}")
            print(f"   Confiança: {result['confidence']}%")
            print(f"   Score: {result['main_score']}/100")
            print(f"   Fases executadas: {result['phases_executed']}")
            print(f"   Filtro usado: {result.get('reflection_filter_used', True)}")
            print(f"   Imagem limpa: {result.get('image_cleaned', False)}")
            
            # Validações
            assert 'verdict' in result, "Falta campo 'verdict'"
            assert 'confidence' in result, "Falta campo 'confidence'"
            assert 'main_score' in result, "Falta campo 'main_score'"
            assert result['reflection_filter_used'] == True, "Filtro deveria estar ativado"
            
            print("\n✅ PASSOU - Análise com filtro funcionando")
            self.test_results.append(("Análise com Filtro", True, 
                                     f"{result['verdict']}, Score: {result['main_score']}"))
            
            return result
            
        except Exception as e:
            print(f"\n❌ FALHOU - Erro: {e}")
            self.test_results.append(("Análise com Filtro", False, str(e)))
            return None
    
    def test_comparison(self, result_without, result_with):
        """Teste 7: Comparação de resultados"""
        print("\n" + "="*70)
        print("TESTE 7: COMPARAÇÃO - SEM FILTRO vs COM FILTRO")
        print("="*70)
        
        if not result_without or not result_with:
            print("❌ Não foi possível comparar (faltam resultados)")
            self.test_results.append(("Comparação", False, "Resultados faltando"))
            return False
        
        try:
            score_without = result_without['main_score']
            score_with = result_with['main_score']
            diff = score_with - score_without
            
            print(f"\n📊 Comparação de Scores:")
            print(f"   SEM filtro: {score_without}/100")
            print(f"   COM filtro: {score_with}/100")
            print(f"   Diferença: {diff:+d} pontos")
            
            print(f"\n📊 Comparação de Vereditos:")
            print(f"   SEM filtro: {result_without['verdict']}")
            print(f"   COM filtro: {result_with['verdict']}")
            
            # Análise do resultado
            if diff > 15:
                print(f"\n✅ MELHORIA SIGNIFICATIVA!")
                print(f"   O filtro melhorou o score em {diff} pontos")
                print(f"   Reflexos estavam causando falso positivo")
                self.test_results.append(("Comparação", True, f"Melhoria de {diff} pontos"))
            elif diff > 0:
                print(f"\n✅ MELHORIA MODERADA")
                print(f"   O filtro melhorou o score em {diff} pontos")
                self.test_results.append(("Comparação", True, f"Melhoria de {diff} pontos"))
            elif diff == 0:
                print(f"\n✅ SEM DIFERENÇA")
                print(f"   Reflexos não afetaram análise (esperado se poucos reflexos)")
                self.test_results.append(("Comparação", True, "Sem diferença"))
            else:
                print(f"\n⚠️  PIORA DE {abs(diff)} PONTOS")
                print(f"   Investigar se filtro está muito agressivo")
                self.test_results.append(("Comparação", True, f"Piora de {abs(diff)} pontos"))
            
            return True
            
        except Exception as e:
            print(f"\n❌ FALHOU - Erro: {e}")
            self.test_results.append(("Comparação", False, str(e)))
            return False
    
    def generate_visual_report(self, reflection_result, result_without, result_with):
        """Teste 8: Gerar relatório visual"""
        print("\n" + "="*70)
        print("TESTE 8: GERAÇÃO DE RELATÓRIO VISUAL")
        print("="*70)
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Linha 1: Análise de reflexos
            img_array = np.array(self.image.convert('RGB'))
            
            axes[0, 0].imshow(img_array)
            axes[0, 0].set_title("Imagem Original", fontsize=12, fontweight='bold')
            axes[0, 0].axis('off')
            
            if reflection_result:
                axes[0, 1].imshow(reflection_result['visual_report'])
                axes[0, 1].set_title(f"Detecção de Reflexos\nScore: {reflection_result['reflection_score']}/100", 
                                    fontsize=12, fontweight='bold')
                axes[0, 1].axis('off')
                
                axes[0, 2].imshow(reflection_result['heatmap'])
                axes[0, 2].set_title(f"Heatmap\n{reflection_result['reflection_percentage']:.1f}% detectado", 
                                    fontsize=12, fontweight='bold')
                axes[0, 2].axis('off')
            
            # Linha 2: Comparação de análises
            if result_without:
                score_without = result_without['main_score']
                axes[1, 0].text(0.5, 0.5, 
                               f"SEM FILTRO\n\n{result_without['verdict']}\n\nScore: {score_without}/100\nConfiança: {result_without['confidence']}%",
                               ha='center', va='center', fontsize=11,
                               bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))
                axes[1, 0].set_title("Análise SEM Filtro", fontsize=12, fontweight='bold')
                axes[1, 0].axis('off')
            
            if result_with:
                score_with = result_with['main_score']
                axes[1, 1].text(0.5, 0.5,
                               f"COM FILTRO\n\n{result_with['verdict']}\n\nScore: {score_with}/100\nConfiança: {result_with['confidence']}%",
                               ha='center', va='center', fontsize=11,
                               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
                axes[1, 1].set_title("Análise COM Filtro", fontsize=12, fontweight='bold')
                axes[1, 1].axis('off')
            
            # Comparação
            if result_without and result_with:
                diff = result_with['main_score'] - result_without['main_score']
                comparison_text = f"""
COMPARAÇÃO FINAL:

Score SEM filtro: {result_without['main_score']}/100
Score COM filtro: {result_with['main_score']}/100

DIFERENÇA: {diff:+d} pontos

Veredito SEM: {result_without['verdict']}
Veredito COM: {result_with['verdict']}

CONCLUSÃO:
{'Filtro MELHOROU resultado!' if diff > 0 else 'Sem diferença significativa' if diff == 0 else 'Investigar piora'}
                """
                
                color = 'lightgreen' if diff > 0 else 'lightyellow' if diff == 0 else 'mistyrose'
                
                axes[1, 2].text(0.5, 0.5, comparison_text,
                               ha='center', va='center', fontsize=9,
                               family='monospace',
                               bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
                axes[1, 2].set_title("Comparação", fontsize=12, fontweight='bold')
                axes[1, 2].axis('off')
            
            plt.suptitle(f'RELATÓRIO DE TESTE - {self.timestamp}', 
                        fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            
            output_path = f"/mnt/user-data/outputs/TEST_REPORT_{self.timestamp}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            
            print(f"✅ Relatório visual salvo: {output_path}")
            self.test_results.append(("Relatório Visual", True, "Gerado com sucesso"))
            
            return output_path
            
        except Exception as e:
            print(f"❌ FALHOU - Erro: {e}")
            self.test_results.append(("Relatório Visual", False, str(e)))
            return None
    
    def print_summary(self):
        """Imprime resumo final dos testes"""
        print("\n" + "="*70)
        print("RESUMO FINAL DOS TESTES")
        print("="*70)
        
        total = len(self.test_results)
        passed = sum(1 for _, success, _ in self.test_results if success)
        failed = total - passed
        
        print(f"\n📊 Estatísticas:")
        print(f"   Total de testes: {total}")
        print(f"   ✅ Passou: {passed}")
        print(f"   ❌ Falhou: {failed}")
        print(f"   Taxa de sucesso: {(passed/total)*100:.1f}%")
        
        print(f"\n📋 Detalhamento:")
        for test_name, success, details in self.test_results:
            status = "✅" if success else "❌"
            print(f"   {status} {test_name}: {details}")
        
        print("\n" + "="*70)
        
        if failed == 0:
            print("🎉 TODOS OS TESTES PASSARAM!")
            print("✅ Sistema pronto para implementação em produção")
        elif passed > failed:
            print("⚠️  MAIORIA DOS TESTES PASSOU")
            print("⚠️  Revisar falhas antes de implementar")
        else:
            print("❌ MUITAS FALHAS DETECTADAS")
            print("❌ NÃO implementar em produção ainda")
        
        print("="*70)
        
        return failed == 0


def main():
    """Função principal de teste"""
    
    print("\n" + "="*70)
    print("🧪 TEST_REFLECTION.PY - SUITE DE TESTES COMPLETA")
    print("="*70)
    print("\nTestando sistema integrado: Filtro de Reflexos + Detecção de IA")
    print(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Caminho da imagem de teste
    image_path = "/mnt/user-data/uploads/Image__2_.png"
    
    if not os.path.exists(image_path):
        print(f"\n❌ ERRO: Imagem não encontrada em {image_path}")
        print("   Por favor, forneça uma imagem de teste válida")
        return False
    
    # Criar executor de testes
    tester = TestRunner(image_path)
    
    # Executar testes sequencialmente
    reflection_result = None
    result_without = None
    result_with = None
    
    # Teste 1: Carregar imagem
    if not tester.load_image():
        print("\n❌ ABORTADO: Não foi possível carregar imagem")
        return False
    
    # Teste 2: Detector de reflexos
    reflection_result = tester.test_reflection_detector()
    
    # Teste 3: Métodos individuais
    tester.test_reflection_methods()
    
    # Teste 4: Limpeza de imagem
    tester.test_image_cleaning()
    
    # Teste 5: Análise sem filtro
    result_without = tester.test_integrated_analyzer_without_filter()
    
    # Teste 6: Análise com filtro
    result_with = tester.test_integrated_analyzer_with_filter()
    
    # Teste 7: Comparação
    tester.test_comparison(result_without, result_with)
    
    # Teste 8: Relatório visual
    tester.generate_visual_report(reflection_result, result_without, result_with)
    
    # Resumo final
    success = tester.print_summary()
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
