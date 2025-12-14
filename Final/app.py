import io
from pathlib import Path

import altair as alt
import folium
import geopandas as gpd
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from streamlit_folium import st_folium

@st.cache_data(show_spinner=False)
def load_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    """Cache CSV parse to avoid re-reading on every rerun/interaction."""
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def load_geojson(path_str: str) -> gpd.GeoDataFrame:
    """Cache GeoJSON read to prevent reloads and visual flicker."""
    return gpd.read_file(path_str)

def resolve_geojson_path() -> Path:
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / 'cr.json',
        Path.cwd() / 'cr.json',
        Path.cwd() / 'Final' / 'cr.json',
    ]
    for path in candidates:
        if path.exists():
            return path
    st.error("No se encontro el archivo de provincias 'cr.json'. Colocalo junto a app.py o en la raiz del repo.")
    st.stop()
    return candidates[0]

# === Configuración general ===
st.set_page_config(page_title="Mapa de Nacimientos CR", layout="wide")
st.title("🇨🇷 Visualización de Nacimientos y Educación en Costa Rica")

st.markdown("""
Explora cómo varían los **nacimientos por sexo** y el **nivel educativo de los padres** por provincia 🇨🇷  
🟦 Azul = Hombres 🟥 Rojo = Mujeres ⬜ Blanco = Igual  
También puedes ver la **educación del padre y la madre** con intensidad de color según los valores predominantes.  
Usa el *slider* inferior para seleccionar los años.
""")

# === Subida del dataset ===
uploaded_file = st.file_uploader("📂 Sube tu archivo CSV con los datos de nacimientos", type=["csv"])

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    df = load_uploaded_csv(file_bytes)
    df.columns = df.columns.str.strip()

    required_cols = {"Provocu", "Sexo", "Anotrab"}
    if not required_cols.issubset(df.columns):
        st.error("❌ Tu archivo debe contener las columnas 'Provocu', 'Sexo' y 'Anotrab'.")
        st.stop()

    # ---- Mapeos base
    prov_map = {1: "San José", 2: "Alajuela", 3: "Cartago", 4: "Heredia",
                5: "Guanacaste", 6: "Puntarenas", 7: "Limón"}
    sexo_map = {1: "Hombre", 2: "Mujer"}
    df["Provincia"] = df["Provocu"].map(prov_map)
    df["Sexo_nombre"] = df["Sexo"].map(sexo_map)

    # ---- Slider de años
    min_year, max_year = int(df["Anotrab"].min()), int(df["Anotrab"].max())
    selected_years = st.slider("Selecciona rango de años", min_year, max_year, (min_year, max_year), step=1)
    df = df[(df["Anotrab"] >= selected_years[0]) & (df["Anotrab"] <= selected_years[1])]

    # ---- Tabs (dejamos 1,2,3 como el código original; 4 es estacionalidad mejorada)
    tabs = st.tabs(["1. 👶 Nacimientos por sexo", "2. 🧑‍🎓 Educación del padre", "3. 👩‍🎓 Educación de la madre", "4. 📆 Estacionalidad","5. Correlación", "6. 🔮 Proyecciones"])

    # ---- GeoJSON
    geojson_path = resolve_geojson_path()
    gdf = load_geojson(str(geojson_path))
    gdf["name"] = gdf["name"].astype(str)

    # ==========================================================
    # TAB 1: NACIMIENTOS POR SEXO (igual a tu código)
    # ==========================================================
    with tabs[0]:
        st.subheader("👶 Nacimientos por sexo y provincia")

        summary = df.groupby(["Provincia", "Sexo_nombre"]).size().unstack(fill_value=0)
        summary["Total"] = summary.sum(axis=1)
        summary["% Hombres"] = (summary.get("Hombre", 0) / summary["Total"] * 100).round(2)
        summary["% Mujeres"] = (summary.get("Mujer", 0) / summary["Total"] * 100).round(2)
        summary["Diferencia"] = summary["% Hombres"] - summary["% Mujeres"]
        summary["Predominante"] = summary.apply(
            lambda r: "Hombres" if r["% Hombres"] > r["% Mujeres"]
            else "Mujeres" if r["% Mujeres"] > r["% Hombres"]
            else "Igual",
            axis=1
        )

        max_diff = abs(summary["Diferencia"]).max() or 1  # evita división 0
        def color_diff(row):
            diff = row["Diferencia"]; intensity = abs(diff) / max_diff
            if diff > 0:   # más hombres → azul
                return mcolors.to_hex((0*(1-intensity)+0, 0*(1-intensity)+0, 1))
            elif diff < 0: # más mujeres → rojo
                return mcolors.to_hex((1, 0*(1-intensity)+0, 0*(1-intensity)+0))
            else:
                return "#ffffff"

        summary["Color"] = summary.apply(color_diff, axis=1)
        summary.reset_index(inplace=True)

        gdf_merged = gdf.merge(summary, left_on="name", right_on="Provincia", how="left")

        m = folium.Map(location=[9.7489, -83.7534], zoom_start=7, tiles="cartodb positron")
        folium.GeoJson(
            gdf_merged,
            style_function=lambda f: {
                "fillColor": f["properties"].get("Color", "#cccccc"),
                "color": "black",
                "weight": 1,
                "fillOpacity": 0.7,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["name", "Predominante", "% Hombres", "% Mujeres", "Diferencia"],
                aliases=["Provincia", "Sexo predominante", "% Hombres", "% Mujeres", "Dif. %"],
                localize=True,
            ),
        ).add_to(m)
        st_folium(m, width=1000, height=600)

        st.subheader("📈 Evolución de nacimientos por año y sexo")
        timeline = df.groupby(["Anotrab", "Sexo_nombre"]).size().reset_index(name="Cantidad")
        fig_timeline = px.line(
            timeline, x="Anotrab", y="Cantidad", color="Sexo_nombre",
            color_discrete_map={"Hombre": "#0050B8", "Mujer": "#E13C3C"},
            markers=True, title="Tendencia de nacimientos por año"
        )
        fig_timeline.update_traces(line=dict(width=3))
        st.plotly_chart(fig_timeline, use_container_width=True)

        st.subheader(f"📊 Resumen por provincia — Años: {selected_years[0]}–{selected_years[1]}")
        st.dataframe(summary[["Provincia", "% Hombres", "% Mujeres", "Diferencia"]])

    # ==========================================================
    # TAB 2: EDUCACIÓN DEL PADRE (igual a tu código)
    # ==========================================================
    with tabs[1]:
        st.subheader("🧑‍🎓 Nivel educativo del padre por provincia")

        with st.expander("📘 Ver significado de los niveles educativos"):
            st.markdown("""
            **Escala de niveles educativos del padre:**
            - 0️⃣ Ninguno  
            - 1️⃣ Primaria incompleta  
            - 2️⃣ Primaria completa  
            - 3️⃣ Secundaria incompleta  
            - 4️⃣ Secundaria completa  
            - 5️⃣ Universitaria incompleta  
            - 6️⃣ Universitaria completa  
            - 8️⃣ Padre no declarado  
            - 9️⃣ Ignorado  
            """)

        nivel_padre = {
            0: "Ninguno", 1: "Primaria incompleta", 2: "Primaria completa",
            3: "Secundaria incompleta", 4: "Secundaria completa",
            5: "Universitaria incompleta", 6: "Universitaria completa",
            8: "Padre no declarado", 9: "Ignorado"
        }
        df["Nivedpad_cat"] = df["Nivedpad"].map(nivel_padre)

        padre_summary = []
        for prov, group in df.groupby("Provincia"):
            mode_val = group["Nivedpad_cat"].mode()[0] if not group.empty else None
            dominancia = (group["Nivedpad_cat"].value_counts(normalize=True).get(mode_val, 0)*100).round(2)
            padre_summary.append({"Provincia": prov, "Mas_comun": mode_val, "Dominancia": dominancia})
        padre_summary = pd.DataFrame(padre_summary)

        padre_mean = df.groupby("Provincia")["Nivedpad"].mean().reset_index()
        padre_summary = padre_summary.merge(padre_mean, on="Provincia", how="left")
        min_p, max_p = padre_mean["Nivedpad"].min(), padre_mean["Nivedpad"].max()
        norm_p = mcolors.Normalize(vmin=min_p, vmax=max_p)
        cmap_p = px.colors.sequential.Purples
        padre_summary["Color"] = padre_summary["Nivedpad"].apply(lambda v: cmap_p[int(norm_p(v)*(len(cmap_p)-1))])

        gdf_padre = gdf.merge(padre_summary, left_on="name", right_on="Provincia", how="left")
        m_padre = folium.Map(location=[9.75, -83.75], zoom_start=7, tiles="cartodb positron")
        folium.GeoJson(
            gdf_padre,
            style_function=lambda f: {
                "fillColor": f["properties"].get("Color", "#cccccc"),
                "color": "black",
                "weight": 1,
                "fillOpacity": 0.7,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["name", "Mas_comun", "Dominancia"],
                aliases=["Provincia", "Nivel más común", "% Dominancia"],
                localize=True,
            )
        ).add_to(m_padre)
        st_folium(m_padre, width=1000, height=600)

        padre_counts = df.groupby(["Provincia", "Nivedpad_cat"]).size().reset_index(name="Cantidad")
        fig_padre = px.bar(
            padre_counts, x="Provincia", y="Cantidad", color="Nivedpad_cat",
            color_discrete_sequence=px.colors.sequential.Purples,
            title="Distribución del nivel educativo del padre"
        )
        st.plotly_chart(fig_padre, use_container_width=True)
        st.subheader("📋 Tabla resumen del nivel educativo del padre")
        st.dataframe(padre_summary[["Provincia", "Mas_comun", "Dominancia"]])

    # ==========================================================
    # TAB 3: EDUCACIÓN DE LA MADRE (igual a tu código)
    # ==========================================================
    with tabs[2]:
        st.subheader("👩‍🎓 Nivel educativo de la madre por provincia")

        with st.expander("📘 Ver significado de los niveles educativos"):
            st.markdown("""
            **Escala de niveles educativos de la madre:**
            - 0️⃣ Ninguno  
            - 1️⃣ Primaria incompleta  
            - 2️⃣ Primaria completa  
            - 3️⃣ Secundaria incompleta  
            - 4️⃣ Secundaria completa  
            - 5️⃣ Universitaria incompleta  
            - 6️⃣ Universitaria completa  
            - 9️⃣ Ignorado  
            """)

        nivel_madre = {
            0: "Ninguno", 1: "Primaria incompleta", 2: "Primaria completa",
            3: "Secundaria incompleta", 4: "Secundaria completa",
            5: "Universitaria incompleta", 6: "Universitaria completa",
            9: "Ignorado"
        }
        df["Nivedmad_cat"] = df["Nivedmad"].map(nivel_madre)

        madre_summary = []
        for prov, group in df.groupby("Provincia"):
            mode_val = group["Nivedmad_cat"].mode()[0] if not group.empty else None
            dominancia = (group["Nivedmad_cat"].value_counts(normalize=True).get(mode_val, 0)*100).round(2)
            madre_summary.append({"Provincia": prov, "Mas_comun": mode_val, "Dominancia": dominancia})
        madre_summary = pd.DataFrame(madre_summary)

        madre_mean = df.groupby("Provincia")["Nivedmad"].mean().reset_index()
        madre_summary = madre_summary.merge(madre_mean, on="Provincia", how="left")
        min_m, max_m = madre_mean["Nivedmad"].min(), madre_mean["Nivedmad"].max()
        norm_m = mcolors.Normalize(vmin=min_m, vmax=max_m)
        cmap_m = px.colors.sequential.Reds
        madre_summary["Color"] = madre_summary["Nivedmad"].apply(lambda v: cmap_m[int(norm_m(v)*(len(cmap_m)-1))])

        gdf_madre = gdf.merge(madre_summary, left_on="name", right_on="Provincia", how="left")
        m_madre = folium.Map(location=[9.75, -83.75], zoom_start=7, tiles="cartodb positron")
        folium.GeoJson(
            gdf_madre,
            style_function=lambda f: {
                "fillColor": f["properties"].get("Color", "#cccccc"),
                "color": "black",
                "weight": 1,
                "fillOpacity": 0.7,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["name", "Mas_comun", "Dominancia"],
                aliases=["Provincia", "Nivel más común", "% Dominancia"],
                localize=True,
            )
        ).add_to(m_madre)
        st_folium(m_madre, width=1000, height=600)

        madre_counts = df.groupby(["Provincia", "Nivedmad_cat"]).size().reset_index(name="Cantidad")
        fig_madre = px.bar(
            madre_counts, x="Provincia", y="Cantidad", color="Nivedmad_cat",
            color_discrete_sequence=px.colors.sequential.Reds,
            title="Distribución del nivel educativo de la madre"
        )
        st.plotly_chart(fig_madre, use_container_width=True)
        st.subheader("📋 Tabla resumen del nivel educativo de la madre")
        st.dataframe(madre_summary[["Provincia", "Mas_comun", "Dominancia"]])

    # ==========================================================
    # TAB 4: ESTACIONALIDAD (heatmap ordenado + infografía tipo ciclo)
    # ==========================================================
    with tabs[3]:
        st.subheader("📆 Estacionalidad de nacimientos por mes y año")

        if "Mestrab" in df.columns and "Anotrab" in df.columns:
            month_names = {
                1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
                7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
            }
            df["Mes_nombre"] = df["Mestrab"].astype(int).map(month_names)

            # ---- Heatmap (Enero → Diciembre)
            pivot = df.groupby(["Anotrab", "Mes_nombre"]).size().reset_index(name="Nacimientos")
            orden = list(month_names.values())
            pivot["Mes_nombre"] = pd.Categorical(pivot["Mes_nombre"], categories=orden, ordered=True)
            pivot_wide = pivot.pivot(index="Mes_nombre", columns="Anotrab", values="Nacimientos").sort_index()

            st.markdown("### 🔥 Heatmap Año vs Mes (ordenado)")
            fig_heatmap = px.imshow(
                pivot_wide,
                color_continuous_scale="Blues",
                aspect="auto",
                labels=dict(color="Nacimientos"),
                title="Estacionalidad de nacimientos (Enero–Diciembre)"
            )
            fig_heatmap.update_layout(yaxis_title="Mes", xaxis_title="Año")
            st.plotly_chart(fig_heatmap, use_container_width=True)

            # ---- Infografía tipo “product life cycle”
            st.markdown("### 🌊 Estacionalidad (promedio mensual, estilo infográfico)")
            # promedio por mes (sobre el rango filtrado)
            monthly = df.groupby("Mes_nombre").size().reindex(orden).reset_index(name="Nacimientos")
            # Área suave
            fig_area = go.Figure()
            fig_area.add_trace(go.Scatter(
                x=monthly["Mes_nombre"], y=monthly["Nacimientos"],
                mode="lines+markers",
                line=dict(shape="spline", width=4),
                fill="tozeroy", fillcolor="rgba(0,98,255,0.25)",
                marker=dict(size=8),
                name="Promedio mensual"
            ))
            # Bandas trimestrales (Intro/Growth/Maturity/Decline) para darle look editorial
            bands = [
                ("Introducción", 0.5, 3.5, "rgba(0,123,255,0.06)"),
                ("Crecimiento", 3.5, 6.5, "rgba(0,123,255,0.04)"),
                ("Madurez", 6.5, 9.5, "rgba(0,123,255,0.06)"),
                ("Declive", 9.5, 12.5, "rgba(0,123,255,0.04)")
            ]
            for label, x0, x1, color in bands:
                fig_area.add_shape(type="rect", xref="x", yref="paper",
                                   x0=monthly["Mes_nombre"].iloc[int(x0-0.5)],
                                   x1=monthly["Mes_nombre"].iloc[min(int(x1-0.5), 11)],
                                   y0=0, y1=1, fillcolor=color, line=dict(width=0))
            # Etiquetas arriba
            fig_area.update_layout(
                title=f"Forma estacional de los nacimientos (promedio mensual • {selected_years[0]}–{selected_years[1]})",
                xaxis_title="Mes", yaxis_title="Nacimientos",
                margin=dict(l=10, r=10, t=60, b=10),
                hovermode="x unified"
            )
            st.plotly_chart(fig_area, use_container_width=True)

        else:
            st.warning("⚠️ El dataset no contiene las columnas 'Mestrab' (mes) y 'Anotrab' (año).")
# ==========================================================
# TAB 5: CORRELACIÓN 
# ==========================================================
              # ==========================================================
    # TAB 5: CORRELACIÓN
    # ==========================================================
       # ==========================================================
    # TAB 5: CORRELACIÓN 
    # ==========================================================
    with tabs[4]:
        st.subheader("🪄 Correlación nacimientos")

        if "Nivedmad" in df.columns and "Hijosten" in df.columns and "Anotrab" in df.columns:
            
            # Filtro de año específico para correlación
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("### 🎯 Filtro por año")
                selected_year_corr = st.slider(
                    "Selecciona un año específico para análisis de correlación:",
                    min_value=int(df["Anotrab"].min()),
                    max_value=int(df["Anotrab"].max()),
                    value=int(df["Anotrab"].max()),
                    step=1,
                    key="corr_year_slider"
                )
            with col2:
                st.metric("📅 Año seleccionado", selected_year_corr)
            
            # Filtrar datos por el año seleccionado
            df_year = df[df["Anotrab"] == selected_year_corr]
            
            if len(df_year) == 0:
                st.warning(f"⚠️ No hay datos disponibles para el año {selected_year_corr}")
            else:
                # ---------- PARTE 1: BURBUJAS MEJORADAS ----------
                orden_edu = [
                    "Ninguno", "Primaria incompleta", "Primaria completa",
                    "Secundaria incompleta", "Secundaria completa",
                    "Universitaria incompleta", "Universitaria completa",
                    "Ignorado"
                ]
                df_year["Nivedmad_cat"] = df_year["Nivedmad"].map({
                    0:"Ninguno",1:"Primaria incompleta",2:"Primaria completa",
                    3:"Secundaria incompleta",4:"Secundaria completa",
                    5:"Universitaria incompleta",6:"Universitaria completa",
                    9:"Ignorado"
                }).astype("category")

                corr_df = (
                    df_year.groupby("Nivedmad_cat", observed=True)
                      .agg(Promedio_hijos=("Hijosten","mean"),
                           Mediana_hijos=("Hijosten","median"),
                           Cantidad=("Hijosten","count"))
                      .reset_index()
                )
                corr_df["Nivedmad_cat"] = corr_df["Nivedmad_cat"].cat.set_categories(orden_edu, ordered=True)
                corr_df = corr_df.sort_values("Nivedmad_cat")

                prom_nacional = df_year["Hijosten"].mean()
                corr_df["Comparado_prom"] = corr_df["Promedio_hijos"].apply(
                    lambda v: "Sobre el promedio" if v >= prom_nacional else "Bajo el promedio"
                )
                # Calcular distancia del promedio para el tamaño de burbujas
                corr_df["Distancia_promedio"] = abs(corr_df["Promedio_hijos"] - prom_nacional)

                st.markdown(f"### 🎈 Burbujas (promedio de hijos por nivel educativo - Año {selected_year_corr})")
                
                # Mostrar información del año seleccionado
                total_nacimientos = len(df_year)
                st.info(f"""
                **📊 Datos del año {selected_year_corr}:**
                - **Total de nacimientos**: {total_nacimientos:,}
                - **Promedio nacional de hijos**: {prom_nacional:.2f}
                - **Mueve el slider para ver cómo cambian las correlaciones por año** 📅
                """)
                
                fig_bubble = px.scatter(
                    corr_df,
                    x="Promedio_hijos",
                    y="Nivedmad_cat",
                    size="Distancia_promedio",
                    color="Comparado_prom",
                    color_discrete_map={"Sobre el promedio":"#EF553B","Bajo el promedio":"#2CA02C"},
                    size_max=60,
                    text=corr_df["Promedio_hijos"].round(2),
                    hover_data={
                        "Nivedmad_cat": True,
                        "Promedio_hijos": ":.2f",
                        "Mediana_hijos": ":.2f",
                        "Cantidad": ":,",
                        "Comparado_prom": True
                    },
                    title=f"Relación entre educación de la madre y número de hijos - {selected_year_corr}"
                )
                # estilo: ranking horizontal + línea de referencia nacional
                fig_bubble.update_traces(
                    marker=dict(line=dict(width=1.2,color="rgba(0,0,0,0.5)")),
                    textposition="middle right",
                    selector=dict(mode="markers")
                )
                fig_bubble.update_layout(
                    yaxis_title="Nivel educativo de la madre",
                    xaxis_title="Promedio de hijos",
                    legend_title="Comparación con promedio nacional",
                    hovermode="y unified",
                    margin=dict(l=20,r=20,t=60,b=10)
                )
                fig_bubble.add_vline(x=prom_nacional, line_width=2, line_dash="dash", line_color="#636EFA",
                                     annotation_text=f"Prom. nacional {selected_year_corr}: {prom_nacional:.2f}",
                                     annotation_position="top left")
                st.plotly_chart(fig_bubble, use_container_width=True)

                st.markdown(f"#### 📋 Tabla resumen - Año {selected_year_corr}")
                st.dataframe(
                    corr_df.rename(columns={
                        "Nivedmad_cat":"Nivel educativo",
                        "Promedio_hijos":"Promedio de hijos",
                        "Mediana_hijos":"Mediana de hijos"
                    }).style.format({
                        "Promedio de hijos":"{:.2f}",
                        "Mediana de hijos":"{:.2f}",
                        "Cantidad":"{:,.0f}"
                    })
                )
        
        else:
            st.warning("⚠️ El dataset no contiene las columnas requeridas para el análisis.")

    # ==========================================================
    # TAB 6: PROYECCIONES DE NACIMIENTOS
    # ==========================================================
    with tabs[5]:
        st.subheader("🔮 Proyecciones de nacimientos (2024-2029)")
        
        # Filtro de año base para proyecciones
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### 🎯 Año base para proyecciones")
            base_year = st.slider(
                "Selecciona hasta qué año usar para calcular la tendencia:",
                min_value=int(df["Anotrab"].min()),
                max_value=int(df["Anotrab"].max()),
                value=int(df["Anotrab"].max()),
                step=1,
                key="proj_base_year_slider"
            )
        with col2:
            st.metric("📅 Año base", base_year)
        
        st.info(f"""
        **📊 ¿Qué significa esto?**
        
        - **Año base**: Se usarán datos desde el primer año hasta **{base_year}** para calcular la tendencia
        - **Proyecciones**: Se proyectará desde **{base_year + 1}** hasta **{base_year + 5}**
        - **Mueve el slider** para ver cómo cambian las proyecciones según diferentes períodos base 📅
        """)
        
        # Filtrar datos hasta el año base seleccionado
        df_base = df[df["Anotrab"] <= base_year]
        
        if len(df_base) < 3:
            st.warning(f"⚠️ Se necesitan al menos 3 años de datos para hacer proyecciones. Año base: {base_year}")
        else:
            # Análisis de tendencias históricas
            yearly_births = df_base.groupby(["Anotrab", "Sexo_nombre"]).size().reset_index(name="Nacimientos")
            
            # Preparar datos para regresión
            proyecciones = {}
            colores = {"Hombre": "#0050B8", "Mujer": "#E13C3C"}
            
            # Años futuros para proyección (desde año base + 1)
            years_actual = yearly_births["Anotrab"].unique()
            future_years = np.arange(base_year + 1, base_year + 6)
            all_years = np.concatenate([years_actual, future_years])
            
            st.markdown(f"### 📊 Análisis de tendencias y proyecciones (Base: hasta {base_year})")
            
            # Crear figura principal
            fig_projection = go.Figure()
        
        for sexo in ["Hombre", "Mujer"]:
            data_sexo = yearly_births[yearly_births["Sexo_nombre"] == sexo]
            
            # Regresión lineal
            X = data_sexo["Anotrab"].values.reshape(-1, 1)
            y = data_sexo["Nacimientos"].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Predicciones históricas (para la línea de tendencia)
            X_trend = np.array(years_actual).reshape(-1, 1)
            y_trend = model.predict(X_trend)
            
            # Predicciones futuras
            X_future = future_years.reshape(-1, 1)
            y_future = model.predict(X_future)
            
            # Datos históricos (puntos)
            fig_projection.add_trace(go.Scatter(
                x=data_sexo["Anotrab"],
                y=data_sexo["Nacimientos"],
                mode="markers+lines",
                name=f"{sexo} (Histórico)",
                marker=dict(size=8, color=colores[sexo]),
                line=dict(width=3, color=colores[sexo])
            ))
            
            # Línea de tendencia histórica
            fig_projection.add_trace(go.Scatter(
                x=years_actual,
                y=y_trend,
                mode="lines",
                name=f"Tendencia {sexo}",
                line=dict(dash="dash", width=2, color=colores[sexo]),
                opacity=0.7
            ))
            
            # Proyecciones futuras
            fig_projection.add_trace(go.Scatter(
                x=future_years,
                y=y_future,
                mode="markers+lines",
                name=f"{sexo} (Proyectado)",
                marker=dict(size=10, symbol="diamond", color=colores[sexo]),
                line=dict(width=3, color=colores[sexo], dash="dot")
            ))
            
            # Guardar datos para análisis
            proyecciones[sexo] = {
                "pendiente": model.coef_[0],
                "intercepto": model.intercept_,
                "r2": model.score(X, y),
                "proyecciones": list(zip(future_years, y_future))
            }            # Agregar banda de separación entre histórico y proyectado
            max_year_hist = base_year
            fig_projection.add_vline(
                x=max_year_hist + 0.5,
                line_dash="solid",
                line_color="gray",
                line_width=2,
                annotation_text="Proyecciones",
                annotation_position="top"
            )
            
            # Configurar diseño
            fig_projection.update_layout(
                title=f"Proyección de nacimientos por sexo ({base_year + 1}-{base_year + 5}) - Base: hasta {base_year}",
                xaxis_title="Año",
                yaxis_title="Número de nacimientos",
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=10, r=10, t=80, b=10),
                height=500
            )
        
        st.plotly_chart(fig_projection, use_container_width=True)
        
        # Análisis de estadísticas
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 👦 Estadísticas - Hombres")
            stats_h = proyecciones["Hombre"]
            st.metric("Tendencia anual", f"{stats_h['pendiente']:+.0f} nacimientos/año")
            st.metric("R² (ajuste del modelo)", f"{stats_h['r2']:.3f}")
            
            if stats_h['pendiente'] > 0:
                st.success("📈 Tendencia creciente")
            else:
                st.error("📉 Tendencia decreciente")
        
        with col2:
            st.markdown("#### 👧 Estadísticas - Mujeres")
            stats_m = proyecciones["Mujer"]
            st.metric("Tendencia anual", f"{stats_m['pendiente']:+.0f} nacimientos/año")
            st.metric("R² (ajuste del modelo)", f"{stats_m['r2']:.3f}")
            
            if stats_m['pendiente'] > 0:
                st.success("📈 Tendencia creciente")
            else:
                st.error("📉 Tendencia decreciente")            # Tabla de proyecciones detallada
            st.markdown(f"### 📅 Proyecciones detalladas por año (Base: hasta {base_year})")
            
            # Mostrar información sobre el período base
            years_used = len(years_actual)
            st.info(f"""
            **📊 Información del análisis:**
            - **Años usados para la tendencia**: {min(years_actual)} - {base_year} ({years_used} años)
            - **Años proyectados**: {base_year + 1} - {base_year + 5}
            - **R² Hombres**: {proyecciones["Hombre"]["r2"]:.3f} | **R² Mujeres**: {proyecciones["Mujer"]["r2"]:.3f}
            """)
            
            proyeccion_table = []
            for year in future_years:
                hombres_proj = next(p[1] for p in proyecciones["Hombre"]["proyecciones"] if p[0] == year)
                mujeres_proj = next(p[1] for p in proyecciones["Mujer"]["proyecciones"] if p[0] == year)
                total_proj = hombres_proj + mujeres_proj
                
                proyeccion_table.append({
                    "Año": int(year),
                    "Hombres": int(round(hombres_proj)),
                    "Mujeres": int(round(mujeres_proj)),
                    "Total": int(round(total_proj)),
                    "% Hombres": f"{(hombres_proj/total_proj)*100:.1f}%",
                    "% Mujeres": f"{(mujeres_proj/total_proj)*100:.1f}%"
                })
            
            df_proyecciones = pd.DataFrame(proyeccion_table)
            st.dataframe(df_proyecciones, use_container_width=True)
            
            # Gráfico de nacimientos proyectados por sexo
            st.markdown(f"### 👥 Nacimientos proyectados por sexo (Base: hasta {base_year})")
            
            st.info(f"""
            **📊 ¿Cómo leer este gráfico?**
            
            - **Barras azules**: Nacimientos proyectados de hombres 👦
            - **Barras rojas**: Nacimientos proyectados de mujeres 👧
            - **Números arriba**: Cantidad exacta de nacimientos proyectados
            - **Año base**: Proyecciones calculadas usando datos hasta {base_year}
            - **Mueve el slider** para ver cómo cambian las proyecciones con diferentes años base 📅
            
            📈 **Contexto biológico normal**: Suelen nacer ~105 hombres por cada 100 mujeres
            """)
        
        # Preparar datos para el gráfico de barras agrupadas
        proj_data = []
        for year in future_years:
            hombres_proj = next(p[1] for p in proyecciones["Hombre"]["proyecciones"] if p[0] == year)
            mujeres_proj = next(p[1] for p in proyecciones["Mujer"]["proyecciones"] if p[0] == year)
            
            proj_data.extend([
                {"Año": int(year), "Sexo": "Hombre", "Nacimientos": int(round(hombres_proj))},
                {"Año": int(year), "Sexo": "Mujer", "Nacimientos": int(round(mujeres_proj))}
            ])
        
        df_proj = pd.DataFrame(proj_data)
        
        # Crear gráfico de barras agrupadas
        fig_sexo = px.bar(
                df_proj,
                x="Año",
                y="Nacimientos",
                color="Sexo",
                color_discrete_map={"Hombre": "#0050B8", "Mujer": "#E13C3C"},
                barmode="group",
                text="Nacimientos",
                title=f"Proyección de nacimientos por sexo ({base_year + 1}-{base_year + 5})"
            )
        
        # Configurar el diseño
        fig_sexo.update_traces(
            texttemplate='%{text:,.0f}',
            textposition="outside"
        )
        
        fig_sexo.update_layout(
            xaxis_title="Año",
            yaxis_title="Número de nacimientos",
            legend_title="Sexo",
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_sexo, use_container_width=True)
        
        # Gráfico adicional: diferencias entre sexos
        st.markdown("### ⚖️ Diferencia entre nacimientos de hombres y mujeres")
        
        diferencias_proj = []
        for year in future_years:
            hombres_proj = next(p[1] for p in proyecciones["Hombre"]["proyecciones"] if p[0] == year)
            mujeres_proj = next(p[1] for p in proyecciones["Mujer"]["proyecciones"] if p[0] == year)
            diff = hombres_proj - mujeres_proj
            diferencias_proj.append(diff)
        
        fig_diff = go.Figure()
        
        # Barras de diferencias
        colors = ['#0050B8' if d > 0 else '#E13C3C' for d in diferencias_proj]
        fig_diff.add_trace(go.Bar(
            x=future_years,
            y=diferencias_proj,
            marker_color=colors,
            name="Diferencia (Hombres - Mujeres)",
            text=[f"{d:+.0f}" for d in diferencias_proj],
            textposition="outside"
        ))
        
        fig_diff.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
        
        fig_diff.update_layout(
                title=f"Diferencia entre nacimientos proyectados ({base_year + 1}-{base_year + 5}) - Base: hasta {base_year}",
                xaxis_title="Año",
                yaxis_title="Diferencia en nacimientos",
                showlegend=False,
                height=350
            )
        
        st.plotly_chart(fig_diff, use_container_width=True)
        
        # Interpretación automática
        st.markdown("### 🎯 Interpretación de resultados")
        
        tendencia_h = "creciente" if stats_h['pendiente'] > 0 else "decreciente"
        tendencia_m = "creciente" if stats_m['pendiente'] > 0 else "decreciente"
        
        st.info(f"""
        **Resumen de proyecciones (Base: hasta {base_year}):**
        
        • **Hombres**: Tendencia {tendencia_h} de {abs(stats_h['pendiente']):.0f} nacimientos por año
        • **Mujeres**: Tendencia {tendencia_m} de {abs(stats_m['pendiente']):.0f} nacimientos por año
        • **Confiabilidad**: Los modelos explican {stats_h['r2']*100:.1f}% (H) y {stats_m['r2']*100:.1f}% (M) de la variabilidad
        • **Proyección {base_year + 5}**: {df_proyecciones.iloc[-1]['Total']:,} nacimientos totales estimados
        • **Años base usados**: {min(years_actual)} - {base_year} ({len(years_actual)} años)
        """)
        
        if abs(stats_h['pendiente'] - stats_m['pendiente']) > 50:
            st.warning("⚠️ Se observa una diferencia significativa en las tendencias entre sexos.")
        
        st.markdown("---")
        st.markdown(f"*Nota: Las proyecciones se basan en regresión lineal usando datos de {min(years_actual)}-{base_year}. Los resultados pueden variar por factores socioeconómicos, políticas públicas y eventos externos no contemplados en el modelo. Cambia el año base para ver diferentes escenarios.*")
