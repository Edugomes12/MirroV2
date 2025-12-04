import streamlit as st
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import time

from texture_analyzer import (
    TextureAnalyzer, EdgeAnalyzer, NoiseAnalyzer, 
    LightingAnalyzer, MirrorGlass,
    ManualConfig, CLAHEConfig, WeightConfig
)

st.set_page_config(
    page_title="MirrorGlass V2 - Análise em Lote",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.title("⚙️ Configurações")
    
    st.markdown("---")
    
    auto_mode = st.toggle("🤖 Análise Automática", value=True, 
                          help="Quando ativo, o sistema detecta automaticamente o tipo de cena e ajusta os parâmetros")
    
    st.markdown("---")
    
    if auto_mode:
        st.success("✅ Modo Automático Ativo")
        st.info("O sistema irá detectar automaticamente carros, vidros, reflexos e ajustar os pesos de análise.")
        
        detection_mode = st.selectbox(
            "Sensibilidade",
            ["Balanceado", "Conservador", "Agressivo"],
            index=0,
            help="Conservador = menos falsos positivos | Agressivo = detecta mais manipulações"
        )
        
        weight_texture = 1.0
        weight_edge = 1.0
        weight_noise = 1.0
        weight_lighting = 1.0
        
        clahe_texture = False
        clahe_edge = True
        clahe_noise = True
        clahe_lighting = True
        clahe_clip = 2.0
        
        enable_boost_reflection = True
        enable_boost_smooth = True
        enable_boost_glass = True
        
    else:
        st.warning("⚠️ Modo Manual Ativo")
        st.caption("Configure os parâmetros abaixo")
        
        detection_mode = st.selectbox(
            "Sensibilidade Base",
            ["Balanceado", "Conservador", "Agressivo"],
            index=0
        )
        
        st.markdown("---")
        st.subheader("🎚️ Pesos dos Analisadores")
        
        weight_texture = st.slider("Textura", 0.5, 1.5, 1.0, 0.05,
                                   help="Multiplica o score de textura")
        weight_edge = st.slider("Bordas", 0.5, 1.5, 1.0, 0.05,
                               help="Multiplica o score de bordas")
        weight_noise = st.slider("Ruído", 0.5, 1.5, 1.0, 0.05,
                                help="Multiplica o score de ruído")
        weight_lighting = st.slider("Iluminação", 0.5, 1.5, 1.0, 0.05,
                                   help="Multiplica o score de iluminação")
        
        st.markdown("---")
        st.subheader("🔬 CLAHE (Equalização)")
        
        clahe_texture = st.checkbox("Textura com CLAHE", value=False,
                                    help="DESLIGADO detecta melhor IA")
        clahe_edge = st.checkbox("Bordas com CLAHE", value=True)
        clahe_noise = st.checkbox("Ruído com CLAHE", value=True)
        clahe_lighting = st.checkbox("Iluminação com CLAHE", value=True)
        
        clahe_clip = st.slider("CLAHE Clip Limit", 1.0, 5.0, 2.0, 0.5)
        
        st.markdown("---")
        st.subheader("🚀 Boosts de Cena")
        
        enable_boost_reflection = st.checkbox("Boost Reflexo", value=True,
                                              help="Ajusta score para cenas com reflexo")
        enable_boost_smooth = st.checkbox("Boost Superfície Lisa", value=True,
                                          help="Ajusta score para superfícies lisas")
        enable_boost_glass = st.checkbox("Boost Vidro", value=True,
                                         help="Ajusta score para vidros")
    
    st.markdown("---")
    st.subheader("📊 Exibição")
    
    show_heatmaps = st.checkbox("Mostrar Heatmaps", value=True)
    show_details = st.checkbox("Mostrar Detalhes", value=False)
    cols_per_row = st.slider("Imagens por linha", 1, 4, 2)

st.title("🔍 MirrorGlass V2")
st.markdown("### Detector de Manipulação em Lote")

uploaded_files = st.file_uploader(
    "📤 Arraste suas imagens aqui",
    type=['jpg', 'jpeg', 'png'],
    accept_multiple_files=True,
    help="Selecione uma ou mais imagens para análise"
)

if uploaded_files:
    st.markdown("---")
    st.markdown(f"### 📁 {len(uploaded_files)} imagem(ns) carregada(s)")
    
    if st.button("🚀 Analisar Todas", type="primary", use_container_width=True):
        
        clahe_cfg = CLAHEConfig(
            texture_clahe=clahe_texture,
            edge_clahe=clahe_edge,
            noise_clahe=clahe_noise,
            lighting_clahe=clahe_lighting,
            clip_limit=clahe_clip
        )
        
        if auto_mode:
            manual_cfg = ManualConfig(
                auto_mode=True,
                clahe_config=clahe_cfg
            )
        else:
            custom_weights = WeightConfig(
                weight_texture=weight_texture,
                weight_edge=weight_edge,
                weight_noise=weight_noise,
                weight_lighting=weight_lighting
            )
            
            manual_cfg = ManualConfig(
                auto_mode=False,
                enable_boost_reflection=enable_boost_reflection,
                enable_boost_smooth=enable_boost_smooth,
                enable_boost_glass=enable_boost_glass,
                custom_weights=custom_weights,
                clahe_config=clahe_cfg
            )
        
        analyzer = MirrorGlass(
            detection_mode=detection_mode,
            manual_config=manual_cfg
        )
        
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Analisando {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
            
            image = Image.open(uploaded_file).convert('RGB')
            image_np = np.array(image)
            
            result = analyzer.analyze(image, show_logs=False)
            
            result['image'] = image
            result['image_np'] = image_np
            result['filename'] = uploaded_file.name
            results.append(result)
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        status_text.empty()
        progress_bar.empty()
        
        st.session_state.results = results
        st.success(f"✅ Análise concluída! {len(results)} imagens processadas.")

if 'results' in st.session_state and st.session_state.results:
    results = st.session_state.results
    
    st.markdown("---")
    st.markdown("## 📊 Resumo")
    
    manipuladas = sum(1 for r in results if r['verdict'] == 'MANIPULADA')
    suspeitas = sum(1 for r in results if r['verdict'] == 'SUSPEITA')
    naturais = sum(1 for r in results if r['verdict'] == 'NATURAL')
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total", len(results))
    with col2:
        st.metric("🔴 Manipuladas", manipuladas)
    with col3:
        st.metric("🟡 Suspeitas", suspeitas)
    with col4:
        st.metric("🟢 Naturais", naturais)
    
    st.markdown("---")
    
    filter_option = st.radio(
        "Filtrar por:",
        ["Todas", "🔴 Manipuladas", "🟡 Suspeitas", "🟢 Naturais"],
        horizontal=True
    )
    
    if filter_option == "🔴 Manipuladas":
        filtered_results = [r for r in results if r['verdict'] == 'MANIPULADA']
    elif filter_option == "🟡 Suspeitas":
        filtered_results = [r for r in results if r['verdict'] == 'SUSPEITA']
    elif filter_option == "🟢 Naturais":
        filtered_results = [r for r in results if r['verdict'] == 'NATURAL']
    else:
        filtered_results = results
    
    st.markdown(f"### Exibindo {len(filtered_results)} imagem(ns)")
    
    for i in range(0, len(filtered_results), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            if i + j < len(filtered_results):
                result = filtered_results[i + j]
                
                with col:
                    st.image(result['image'], caption=result['filename'], use_column_width=True)
                    
                    verdict = result['verdict']
                    confidence = result['confidence']
                    
                    if verdict == "MANIPULADA":
                        st.error(f"🔴 **{verdict}** ({confidence}%)")
                    elif verdict == "SUSPEITA":
                        st.warning(f"🟡 **{verdict}** ({confidence}%)")
                    else:
                        st.success(f"🟢 **{verdict}** ({confidence}%)")
                    
                    st.caption(f"📍 Cena: {result.get('scene_type', 'N/A')}")
                    st.caption(f"💡 {result['reason']}")
                    
                    if show_heatmaps and result.get('visual_report') is not None:
                        with st.expander("🗺️ Heatmap"):
                            st.image(result['visual_report'], use_column_width=True)
                    
                    if show_details:
                        with st.expander("📋 Detalhes"):
                            scores = result.get('all_scores', {})
                            score_display = {k: v for k, v in scores.items() 
                                           if isinstance(v, (int, float))}
                            
                            for name, score in score_display.items():
                                if isinstance(score, float):
                                    st.text(f"{name}: {score:.1f}")
                                else:
                                    st.text(f"{name}: {score}")
                            
                            st.text(f"Fases: {result.get('phases_executed', 'N/A')}")
                            st.text(f"Cadeia: {' → '.join(result.get('validation_chain', []))}")
    
    st.markdown("---")
    
    with st.expander("📥 Exportar Resultados"):
        export_data = []
        for r in results:
            scores = r.get('all_scores', {})
            export_data.append({
                'Arquivo': r['filename'],
                'Veredito': r['verdict'],
                'Confiança': r['confidence'],
                'Razão': r['reason'],
                'Cena': r.get('scene_type', 'N/A'),
                'Fases': r.get('phases_executed', 'N/A'),
                'Score Textura': scores.get('texture', 'N/A'),
                'Score Bordas': scores.get('edge', 'N/A'),
                'Score Ruído': scores.get('noise', 'N/A'),
                'Score Iluminação': scores.get('lighting', 'N/A'),
                'Reflexo %': scores.get('reflection', 'N/A')
            })
        
        df = pd.DataFrame(export_data)
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Baixar CSV",
            csv,
            "mirrorglass_resultados.csv",
            "text/csv",
            use_container_width=True
        )

else:
    st.markdown("---")
    st.info("👆 Faça upload de imagens e clique em **Analisar Todas** para começar")
    
    with st.expander("📖 Como usar"):
        st.markdown("""
        ### Passo a passo:
        
        1. **Configure** na sidebar esquerda:
           - 🤖 **Modo Automático**: detecta cenas automaticamente
           - ⚙️ **Modo Manual**: configure pesos e CLAHE manualmente
        
        2. **Arraste** suas imagens para a área de upload
        
        3. **Clique** em "Analisar Todas"
        
        4. **Filtre** os resultados por veredito
        
        5. **Exporte** para CSV se necessário
        
        ### Sobre os vereditos:
        
        - 🔴 **MANIPULADA**: Alta probabilidade de ser IA ou editada
        - 🟡 **SUSPEITA**: Requer análise manual adicional
        - 🟢 **NATURAL**: Provavelmente foto real
        
        ### Dicas:
        
        - **Modo Conservador**: Menos falsos positivos
        - **Modo Agressivo**: Detecta mais manipulações
        - **Textura sem CLAHE**: Melhor para detectar IA
        """)

st.markdown("---")
st.caption("MirrorGlass V2 | Análise em Lote | Dezembro 2025")