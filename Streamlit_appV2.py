import streamlit as st
import numpy as np
from PIL import Image
import io
import base64
import json
import pandas as pd
import time
import cv2

from texture_analyzer_V2 import SequentialAnalyzer


# ============================================================================
# FUNÇÃO AUXILIAR PARA SERIALIZAÇÃO JSON
# ============================================================================
def numpy_safe_json(obj):
    """Converte tipos NumPy para tipos Python nativos para serialização JSON."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


st.set_page_config(
    page_title="MirrorGlass V2",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título
st.title("🔍 MirrorGlass V2")
st.markdown("**Sistema com Logs Detalhados e Compensação para Superfícies Lisas**")

# Sidebar com configurações
with st.sidebar:
    st.header("⚙️ Configurações")
    
    show_logs = st.checkbox("Mostrar Logs Detalhados", value=True)
    show_all_scores = st.checkbox("Mostrar Todos os Scores", value=True)
    
    st.markdown("---")
    st.markdown("### 📖 Como Funciona")
    st.markdown("""
    **Fase 0A:** Detecta superfícies lisas (carros, vidros)  
    **Fase 0B:** Detecta reflexos  
    **Fase 1:** Análise de textura (com boost para superfícies lisas)  
    **Fase 2:** Validação de bordas  
    **Fase 3:** Análise de ruído  
    
    **NOVO:** Superfícies naturalmente uniformes (pintura de carro, vidro) 
    recebem boost automático nos scores para evitar falsos positivos.
    """)

def get_verdict_emoji(verdict):
    if verdict == "MANIPULADA":
        return "🔴"
    elif verdict == "NATURAL":
        return "🟢"
    elif verdict == "SUSPEITA":
        return "🟡"
    else:
        return "⚪"

def analisar_sequencial(imagens, nomes):
    analyzer = SequentialAnalyzer()
    progress_bar = st.progress(0)
    status_text = st.empty()
    resultados = []
    
    for i, img in enumerate(imagens):
        progress = (i + 1) / len(imagens)
        progress_bar.progress(progress)
        status_text.text(f"Analisando {i+1}/{len(imagens)}: {nomes[i]}")
        
        try:
            report = analyzer.analyze_sequential(img)
            resultados.append({
                "nome": nomes[i],
                "verdict": report["verdict"],
                "confidence": report["confidence"],
                "reason": report["reason"],
                "main_score": report["main_score"],
                "all_scores": report["all_scores"],
                "validation_chain": report["validation_chain"],
                "phases_executed": report["phases_executed"],
                "visual_report": report["visual_report"],
                "heatmap": report["heatmap"],
                "percent_suspicious": report["percent_suspicious"],
                "detailed_reason": report["detailed_reason"],
                "logs": report.get("logs", [])  # NOVO!
            })
        except Exception as e:
            st.error(f"Erro: {nomes[i]} - {str(e)}")
            resultados.append({
                "nome": nomes[i],
                "verdict": "ERRO",
                "confidence": 0,
                "reason": f"Erro: {str(e)}",
                "main_score": 0,
                "all_scores": {},
                "validation_chain": [],
                "phases_executed": 0,
                "visual_report": None,
                "heatmap": None,
                "percent_suspicious": 0,
                "detailed_reason": "Falha na análise",
                "logs": []
            })
    
    progress_bar.empty()
    status_text.empty()
    
    return resultados

def exibir_logs(logs):
    """Exibe logs em formato estruturado."""
    if not logs:
        st.info("Sem logs disponíveis")
        return
    
    st.markdown("### 📋 Logs Detalhados")
    
    for log in logs:
        phase = log.get("phase", "")
        message = log.get("message", "")
        data = log.get("data", {})
        
        # Colorir por fase
        if phase == "START":
            st.success(f"**[{phase}]** {message}")
        elif phase in ["VERDICT", "DECISION"]:
            st.info(f"**[{phase}]** {message}")
        elif phase == "BOOST":
            st.warning(f"**[{phase}]** {message}")
        else:
            st.text(f"[{phase}] {message}")
        
        # Mostrar dados se existirem
        if data and isinstance(data, dict):
            with st.expander(f"Dados de {phase}", expanded=False):
                # Converter tipos NumPy para Python nativo
                safe_data = json.loads(json.dumps(data, default=numpy_safe_json))
                st.json(safe_data)

def exibir_resultados(resultados, show_logs_flag, show_all_scores_flag):
    if not resultados:
        st.info("Nenhum resultado disponível.")
        return None
    
    # Estatísticas no topo
    col1, col2, col3, col4 = st.columns(4)
    
    manipuladas = sum(1 for r in resultados if r["verdict"] == "MANIPULADA")
    naturais = sum(1 for r in resultados if r["verdict"] == "NATURAL")
    suspeitas = sum(1 for r in resultados if r["verdict"] in ["SUSPEITA", "INCONCLUSIVA"])
    avg_phases = np.mean([r["phases_executed"] for r in resultados if r["phases_executed"] > 0])
    
    with col1:
        st.metric("🔴 Manipuladas", manipuladas)
    with col2:
        st.metric("🟢 Naturais", naturais)
    with col3:
        st.metric("🟡 Suspeitas", suspeitas)
    with col4:
        st.metric("⚡ Fases Médias", f"{avg_phases:.1f}")
    
    st.markdown("---")
    
    # Resultados por imagem
    relatorio_dados = []
    
    for res in resultados:
        emoji = get_verdict_emoji(res["verdict"])
        
        st.markdown(f"## {emoji} {res['nome']}")
        
        if res["visual_report"] is None:
            st.error(f"❌ {res['reason']}")
            st.markdown("---")
            continue
        
        # Layout: 2 colunas
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(res["visual_report"], use_column_width=True, caption="Análise Visual")
            
            # Veredito
            if res["verdict"] == "MANIPULADA":
                st.error(f"**{res['verdict']}** ({res['confidence']}% confiança)")
            elif res["verdict"] == "NATURAL":
                st.success(f"**{res['verdict']}** ({res['confidence']}% confiança)")
            elif res["verdict"] == "SUSPEITA":
                st.warning(f"**{res['verdict']}** ({res['confidence']}% confiança)")
            else:
                st.info(f"**{res['verdict']}** ({res['confidence']}% confiança)")
            
            st.caption(res["detailed_reason"])
        
        with col2:
            st.image(res["heatmap"], use_column_width=True, caption="Mapa de Calor")
            
            # Info de fases
            if res['phases_executed'] == 1:
                st.success("⚡ Decidido na Fase 1")
            else:
                st.info(f"Fases executadas: {' → '.join(res['validation_chain'])}")
        
        # Scores detalhados
        if show_all_scores_flag and res["all_scores"]:
            st.markdown("### 📊 Scores Detalhados")
            
            # Criar dataframe dos scores
            scores_data = []
            for key, value in res["all_scores"].items():
                if key not in ['surface_type']:  # Pular strings
                    scores_data.append({
                        "Métrica": key.replace('_', ' ').title(),
                        "Valor": f"{value:.1f}" if isinstance(value, float) else value
                    })
            
            if scores_data:
                df_scores = pd.DataFrame(scores_data)
                st.dataframe(df_scores, use_container_width=True)
            
            # Info especial: superfície lisa
            if 'surface_type' in res["all_scores"]:
                surface_type = res["all_scores"]["surface_type"]
                smooth_percent = res["all_scores"].get("smooth_surface", 0)
                
                if smooth_percent > 30:
                    st.info(f"🔍 **Superfície detectada:** {surface_type} ({smooth_percent:.0f}% confiança)")
                    st.caption("ℹ️ Thresholds ajustados automaticamente para superfícies lisas")
        
        # Logs detalhados
        if show_logs_flag and res.get("logs"):
            with st.expander("📋 Ver Logs Detalhados", expanded=False):
                exibir_logs(res["logs"])
        
        st.markdown("---")
        
        # Adicionar ao relatório
        relatorio_dados.append({
            "Arquivo": res["nome"],
            "Veredito": res["verdict"],
            "Confiança (%)": res["confidence"],
            "Score": res["main_score"],
            "Fases": res["phases_executed"],
            "Textura": res["all_scores"].get("texture", 0),
            "Bordas": res["all_scores"].get("edge", 0),
            "Ruído": res["all_scores"].get("noise", 0),
            "Reflexo (%)": res["all_scores"].get("reflection", 0),
            "Superfície Lisa (%)": res["all_scores"].get("smooth_surface", 0)
        })
    
    # Tabela resumo
    if relatorio_dados:
        st.subheader("📊 Relatório Consolidado")
        df = pd.DataFrame(relatorio_dados)
        st.dataframe(df, use_container_width=True)
        
        # Downloads
        col_d1, col_d2, col_d3 = st.columns(3)
        
        with col_d1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Baixar CSV",
                data=csv,
                file_name=f"relatorio_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col_d2:
            # JSON resumido
            json_data = {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "versao": "4.2.0-fixed",
                "total": len(resultados),
                "manipuladas": manipuladas,
                "naturais": naturais,
                "suspeitas": suspeitas,
                "resultados": [
                    {
                        "nome": r["nome"],
                        "verdict": r["verdict"],
                        "confidence": r["confidence"],
                        "score": r["main_score"],
                        "phases": r["phases_executed"],
                        "scores": r["all_scores"]
                    }
                    for r in resultados
                ]
            }
            
            st.download_button(
                label="📥 Baixar JSON",
                data=json.dumps(json_data, indent=2, ensure_ascii=False, default=numpy_safe_json),
                file_name=f"relatorio_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col_d3:
            # JSON completo com logs
            json_full = {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "versao": "4.2.0-fixed",
                "total": len(resultados),
                "resultados_completos": [
                    {
                        "nome": r["nome"],
                        "verdict": r["verdict"],
                        "confidence": r["confidence"],
                        "scores": r["all_scores"],
                        "logs": r.get("logs", [])
                    }
                    for r in resultados
                ]
            }
            
            st.download_button(
                label="📥 Baixar JSON Completo (com logs)",
                data=json.dumps(json_full, indent=2, ensure_ascii=False, default=numpy_safe_json),
                file_name=f"relatorio_completo_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        return df
    return None

# Interface principal
st.markdown("### 📤 Upload de Imagens")
uploaded_files = st.file_uploader(
    "Selecione as imagens para análise",
    accept_multiple_files=True,
    type=['jpg', 'jpeg', 'png']
)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} imagens carregadas")
    
    if st.button("🚀 Iniciar Análise", type="primary", use_container_width=True):
        imagens = []
        nomes = []
        
        for arquivo in uploaded_files:
            try:
                img = Image.open(arquivo).convert('RGB')
                imagens.append(img)
                nomes.append(arquivo.name)
            except Exception as e:
                st.error(f"Erro ao abrir {arquivo.name}: {e}")
        
        if imagens:
            st.markdown("## 🔍 Resultados da Análise")
            resultados = analisar_sequencial(imagens, nomes)
            exibir_resultados(resultados, show_logs, show_all_scores)
        else:
            st.error("Nenhuma imagem válida para analisar.")
else:
    st.info("👆 Faça upload de imagens para começar")
    
    # Exemplo de como interpretar
    with st.expander("📖 Interpretação dos Resultados"):
        st.markdown("""
        ### Vereditos
        - 🔴 **MANIPULADA:** IA detectada com alta confiança
        - 🟢 **NATURAL:** Imagem autêntica confirmada
        - 🟡 **SUSPEITA:** Indicadores ambíguos - revisão manual recomendada
        
        ### Scores (0-100)
        - **Textura:** Uniformidade e variabilidade natural
        - **Bordas:** Coerência de transições
        - **Ruído:** Consistência entre regiões
        - **Reflexo:** Porcentagem com reflexo especular
        - **Superfície Lisa:** Confiança de superfície uniforme (carro, vidro)
        
        ### Thresholds Dinâmicos
        O sistema ajusta automaticamente os thresholds quando detecta:
        - 🚗 **Superfícies lisas** (pintura de carro, vidro): +30-60% boost
        - ✨ **Reflexos** (vidro quebrado, metal): +10-40% boost
        - 📄 **Documentos** (papel branco): +15-25% boost
        
        ### Exemplo Prático
        **Foto de carro branco:**
        - Score textura original: 35 (baixo)
        - Superfície lisa detectada: 80% (boost 1.5x)
        - Score final: 52 (aceitável para superfície lisa)
        - Veredito: NATURAL ✅
        """)

# Rodapé
st.markdown("---")
st.caption("MirrorGlass V2.0")
st.caption("⚠️ IMPORTANTE: Agora detecta e compensa automaticamente fotos de carros, vidros e superfícies uniformes!")