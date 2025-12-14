# Nacimientos Costa Rica

Panel de análisis visual desarrollado con Streamlit para explorar cómo varían los nacimientos en Costa Rica según provincia, sexo y nivel educativo de los padres. Combina mapas interactivos con Folium, gráficos con Plotly y tablas descriptivas para identificar patrones geográficos y tendencias en un CSV cargado por el usuario.

## Características clave
- Carga interactiva: permite subir cualquier CSV con las columnas requeridas sin editar código.
- Mapa coroplético por sexo: azules indican predominio masculino, rojos predominio femenino y blanco indica paridad.
- Gráfico de línea animado para analizar tendencias anuales por sexo.
- Pestañas separadas para educación paterna y materna con tooltips, mapas de intensidad y tablas/gráficos del nivel predominante.

## Requisitos del CSV
El archivo debe incluir al menos:
- `Provocu`: código de provincia (1 San José, 2 Alajuela, 3 Cartago, 4 Heredia, 5 Guanacaste, 6 Puntarenas, 7 Limón).
- `Sexo`: 1 masculino, 2 femenino.
- `Anotrab`: año del registro.
- `Nivedpad`: nivel educativo del padre (0-6, 8, 9).
- `Nivedmad`: nivel educativo de la madre (0-6, 9).

Las columnas adicionales se mantienen; el código elimina espacios en encabezados automáticamente.

## Configuración
Opcional, crear y activar un entorno virtual:

### PowerShell
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS/Linux
```
python3 -m venv .venv
source .venv/bin/activate
```

Instalar dependencias:
```
pip install -r requirements.txt
```

## Ejecutar la app
```
streamlit run app.py
```
Abre en el navegador la URL local generada (por defecto http://localhost:8501).

## Tema visual
El tema de Streamlit ahora se define en `.streamlit/config.toml` con un estilo oscuro y acento celeste. Si quieres otro look, ajusta los valores `primaryColor`, `backgroundColor` o `secondaryBackgroundColor` en ese archivo.

## Flujo de uso recomendado
- Cargar el archivo CSV mediante el uploader.
- Ajustar el rango de años con el control deslizante.
- Navegar las pestañas:
  - Nacimientos por sexo: mapa porcentual, línea temporal y tabla por provincia.
  - Educación del padre: escala, mapa de intensidad, gráfico de barras y nivel predominante.
  - Educación de la madre: estructura equivalente para datos maternos.
- Revisar los tooltips para ver provincia, categoría dominante y diferencias porcentuales.

## Estructura del proyecto
- `app.py`: lógica principal de gráficos y mapas en Streamlit.
- `cr.json`: límites provinciales en formato GeoJSON.
- `nacimientos.csv`: dataset de ejemplo (opcional).
- `requirements.txt`: dependencias del proyecto.

## Problemas comunes
- GeoPandas en Windows: instalar Microsoft C++ Build Tools o usar ruedas precompiladas (whl o conda).
- El mapa no aparece: verificar que `cr.json` esté en el mismo directorio que `app.py` y que los nombres coincidan con el dataframe.
- Valores inválidos: confirmar que los códigos de provincia y educación usen la escala oficial.
