# ============================================================
# CABECERA
# ============================================================
# Alumno: Víctor Trilles Sánchez
# URL Streamlit Cloud: https://mda13-bc5-victor-dmzxyskuuctbcajmf98nkm.streamlit.app
# URL GitHub: https://github.com/vtrilles-code/mda13-bc5-victor.git

# ============================================================
# IMPORTS
# ============================================================
# Streamlit: framework para crear la interfaz web
# pandas: manipulación de datos tabulares
# plotly: generación de gráficos interactivos
# openai: cliente para comunicarse con la API de OpenAI
# json: para parsear la respuesta del LLM (que llega como texto JSON)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
# Modelo de OpenAI. No lo cambies.
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# >>> SYSTEM PROMPT — TU TRABAJO PRINCIPAL ESTÁ AQUÍ <<<
# -------------------------------------------------------
# El system prompt es el conjunto de instrucciones que recibe el LLM
# ANTES de la pregunta del usuario. Define cómo se comporta el modelo:
# qué sabe, qué formato debe usar, y qué hacer con preguntas inesperadas.
#
# Puedes usar estos placeholders entre llaves — se rellenan automáticamente
# con información real del dataset cuando la app arranca:
#   {fecha_min}             → primera fecha del dataset
#   {fecha_max}             → última fecha del dataset
#   {plataformas}           → lista de plataformas (Android, iOS, etc.)
#   {reason_start_values}   → valores posibles de reason_start
#   {reason_end_values}     → valores posibles de reason_end
#
# IMPORTANTE: como el prompt usa llaves para los placeholders,
# si necesitas escribir llaves literales en el texto (por ejemplo para
# mostrar un JSON de ejemplo), usa doble llave: {{ y }}
#
SYSTEM_PROMPT = """
Eres un asistente analítico especializado en explorar un historial de escucha de Spotify mediante código Python.

Tu tarea es responder a la pregunta del usuario generando ÚNICAMENTE un JSON válido con esta estructura exacta:
{{"tipo": "grafico", "codigo": "...", "interpretacion": "..."}}
o bien
{{"tipo": "fuera_de_alcance", "codigo": "", "interpretacion": "..."}}

No escribas markdown, no uses backticks, no añadas explicaciones fuera del JSON.

CONTEXTO DEL DATASET:
- Cada fila representa una reproducción.
- El dataset cubre desde {fecha_min} hasta {fecha_max}.
- Plataformas disponibles: {plataformas}.
- Valores posibles de reason_start: {reason_start_values}.
- Valores posibles de reason_end: {reason_end_values}.

COLUMNAS DISPONIBLES EN df:
- ts: timestamp de la reproducción en formato datetime
- track_name: nombre de la canción
- artist_name: nombre del artista principal
- album_name: nombre del álbum
- spotify_track_uri: identificador único de canción
- ms_played: milisegundos reproducidos
- played_minutes: minutos reproducidos
- played_hours: horas reproducidas
- platform: plataforma de escucha
- reason_start: motivo de inicio
- reason_end: motivo de fin
- shuffle: indica si shuffle estaba activado
- skipped: booleano, True si la canción fue saltada, False si no
- year: año
- month: número de mes
- month_name: nombre del mes
- year_month: mes en formato YYYY-MM
- hour: hora del día
- weekday_name: día de la semana
- is_weekend: True si es fin de semana
- semester: "S1" para enero-junio, "S2" para julio-diciembre
- season: "Winter", "Spring", "Summer", "Autumn"
- is_first_play: True si es la primera vez que aparece esa canción en el historial

OBJETIVO:
Responde preguntas sobre:
A) rankings y favoritos
B) evolución temporal
C) patrones de uso
D) comportamiento de escucha
E) comparación entre períodos

EJEMPLOS de preguntas dentro de alcance:
- artista o canción más escuchada
- top 5 o top 10 artistas/canciones
- evolución por mes
- escucha por hora o por día de la semana
- porcentaje de skips
- uso de shuffle
- comparación entre verano e invierno
- comparación entre primer y segundo semestre
- mes con más primeras escuchas

PREGUNTAS FUERA DE ALCANCE:
Devuelve tipo = "fuera_de_alcance" cuando la pregunta requiera información no presente en el dataset, por ejemplo:
- emociones, gustos subjetivos o razones psicológicas
- recomendaciones musicales
- géneros musicales si no están en df
- comparaciones con otras personas
- predicciones futuras
- cualquier dato no inferible directamente de las columnas disponibles

REGLAS PARA EL CÓDIGO:
1. El campo "codigo" debe contener código Python ejecutable.
2. El código debe crear una figura Plotly y guardarla en una variable llamada fig.
3. Solo puedes usar df, pd, px y go.
4. No uses imports.
5. No uses print, input, open, eval, exec, requests, archivos, internet ni librerías externas.
6. No modifiques la estructura general de df innecesariamente.
7. Usa títulos claros y etiquetas legibles.
8. El gráfico debe ser adecuado a la pregunta:
   - rankings: barras, preferiblemente horizontales
   - evolución temporal: líneas o barras
   - distribuciones por hora/día: barras
   - comparaciones entre grupos: barras agrupadas
9. Ordena los resultados cuando tenga sentido.
10. Limita rankings largos a un top razonable si el usuario no especifica cantidad.
11. Si el usuario pide "más escuchado", interpreta por defecto número de reproducciones, salvo que pida explícitamente minutos u horas.
12. Si una comparación temporal usa estaciones, utiliza la columna season.
13. Si la pregunta habla de descubrimientos o canciones nuevas, utiliza is_first_play.
14. Si la pregunta habla de fines de semana o entre semana, utiliza is_weekend.
15. Si la pregunta habla de tiempo escuchado, usa preferentemente played_minutes o played_hours.
16. Para rankings: primero ordena de mayor a menor y después aplica head().

REGLAS PARA LA INTERPRETACIÓN:
- "interpretacion" debe ser breve, clara y en español.
- Debe resumir el hallazgo principal sin inventar datos no presentes en el gráfico.
- No menciones código ni detalles técnicos.

GUÍA PARA CONSULTAS AMBIGUAS:
- "más escuchado" = número de reproducciones
- "tiempo de escucha" = played_minutes o played_hours
- "canciones nuevas" o "descubrimientos" = is_first_play
- "verano" = Summer, "invierno" = Winter

Antes de responder, verifica que:
- la pregunta está dentro del alcance
- el JSON es válido
- el código crea fig
- no hay texto fuera del JSON
"""


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
# Esta función se ejecuta UNA SOLA VEZ gracias a @st.cache_data.
# Lee el fichero JSON y prepara el DataFrame para que el código
# que genere el LLM sea lo más simple posible.
#
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")

    # Convertir timestamp
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    # Renombrar columnas
    df = df.rename(columns={
        "master_metadata_track_name": "track_name",
        "master_metadata_album_artist_name": "artist_name",
        "master_metadata_album_album_name": "album_name"
    })

    # Filtrar canciones válidas
    df = df[df["track_name"].notna() & df["artist_name"].notna()].copy()

    # Métricas de tiempo
    df["played_minutes"] = df["ms_played"] / 60000
    df["played_hours"] = df["ms_played"] / 3600000

    # Variables temporales
    df["year"] = df["ts"].dt.year
    df["month"] = df["ts"].dt.month
    df["month_name"] = df["ts"].dt.month_name()
    df["year_month"] = df["ts"].dt.to_period("M").astype(str)
    df["hour"] = df["ts"].dt.hour
    df["weekday_name"] = df["ts"].dt.day_name()
    df["is_weekend"] = df["ts"].dt.weekday >= 5

    # Semestre
    df["semester"] = df["month"].apply(lambda x: "S1" if x <= 6 else "S2")

    # Estación
    def get_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Autumn"

    df["season"] = df["month"].apply(get_season)

    # skipped limpio
    df["skipped"] = df["skipped"].fillna(False)

    # primera escucha
    df = df.sort_values("ts")
    first_play = df.groupby("spotify_track_uri")["ts"].idxmin()
    df["is_first_play"] = False
    df.loc[first_play, "is_first_play"] = True

    return df
   
    # ----------------------------------------------------------
    # >>> TU PREPARACIÓN DE DATOS ESTÁ AQUÍ <<<
    # ----------------------------------------------------------
    # Transforma el dataset para facilitar el trabajo del LLM.
    # Lo que hagas aquí determina qué columnas tendrá `df`,
    # y tu system prompt debe describir exactamente esas columnas.
    #
    # Cosas que podrías considerar:
    # - Convertir 'ts' de string a datetime
    # - Crear columnas derivadas (hora, día de la semana, mes...)
    # - Convertir milisegundos a unidades más legibles
    # - Renombrar columnas largas para simplificar el código generado
    # - Filtrar registros que no aportan al análisis (podcasts, etc.)
    # ----------------------------------------------------------


    # ----------------------------------------------------------
    # >>> TU PREPARACIÓN DE DATOS ESTÁ AQUÍ <<<
    # ----------------------------------------------------------

    # Convertir timestamp a datetime
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    # Renombrar columnas
    df = df.rename(columns={
        "master_metadata_track_name": "track_name",
        "master_metadata_album_artist_name": "artist_name",
        "master_metadata_album_album_name": "album_name"
    })

    # Filtrar filas sin canción o artista
    df = df[df["track_name"].notna() & df["artist_name"].notna()].copy()

    # Métricas de tiempo
    df["played_minutes"] = df["ms_played"] / 60000
    df["played_hours"] = df["ms_played"] / 3600000

    # Variables temporales
    df["year"] = df["ts"].dt.year
    df["month"] = df["ts"].dt.month
    df["month_name"] = df["ts"].dt.month_name()
    df["year_month"] = df["ts"].dt.to_period("M").astype(str)
    df["hour"] = df["ts"].dt.hour
    df["weekday_name"] = df["ts"].dt.day_name()
    df["is_weekend"] = df["ts"].dt.weekday >= 5

    # Semestre
    df["semester"] = df["month"].apply(lambda x: "S1" if x <= 6 else "S2")

    # Estación
    def get_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Autumn"

    df["season"] = df["month"].apply(get_season)

    # skipped limpio
    df["skipped"] = df["skipped"].fillna(False)

    # primera escucha
    df = df.sort_values("ts")
    first_play = df.groupby("spotify_track_uri")["ts"].idxmin()
    df["is_first_play"] = False
    df.loc[first_play, "is_first_play"] = True

    return df

def build_prompt(df):
    """
    Inyecta información dinámica del dataset en el system prompt.
    Los valores que calcules aquí reemplazan a los placeholders
    {fecha_min}, {fecha_max}, etc. dentro de SYSTEM_PROMPT.

    Si añades columnas nuevas en load_data() y quieres que el LLM
    conozca sus valores posibles, añade aquí el cálculo y un nuevo
    placeholder en SYSTEM_PROMPT.
    """
    fecha_min = df["ts"].min()
    fecha_max = df["ts"].max()
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()

    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
# Esta función envía DOS mensajes a la API de OpenAI:
# 1. El system prompt (instrucciones generales para el LLM)
# 2. La pregunta del usuario
#
# El LLM devuelve texto (que debería ser un JSON válido).
# temperature=0.2 hace que las respuestas sean más predecibles.
#
# No modifiques esta función.
#
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
# El LLM devuelve un string que debería ser un JSON con esta forma:
#
#   {"tipo": "grafico",          "codigo": "...", "interpretacion": "..."}
#   {"tipo": "fuera_de_alcance", "codigo": "",    "interpretacion": "..."}
#
# Esta función convierte ese string en un diccionario de Python.
# Si el LLM envuelve el JSON en backticks de markdown (```json...```),
# los limpia antes de parsear.
#
# No modifiques esta función.
#
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
# El LLM genera código Python como texto. Esta función lo ejecuta
# usando exec() y busca la variable `fig` que el código debe crear.
# `fig` debe ser una figura de Plotly (px o go).
#
# El código generado tiene acceso a: df, pd, px, go.
#
# No modifiques esta función.
#
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
# Toda la interfaz de usuario. No modifiques esta sección.
#

# Configuración de la página
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
# Lee la contraseña de secrets.toml. Si no coincide, no muestra la app.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

# Cargar datos y construir el prompt con información del dataset
df = load_data()
system_prompt = build_prompt(df)

# Caja de texto para la pregunta del usuario
if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    # Mostrar la pregunta en la interfaz
    with st.chat_message("user"):
        st.write(prompt)

    # Generar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                # 1. Enviar pregunta al LLM
                raw = get_response(prompt, system_prompt)

                # 2. Parsear la respuesta JSON
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    # Pregunta fuera de alcance: mostrar solo texto
                    st.write(parsed["interpretacion"])
                else:
                    # Pregunta válida: ejecutar código y mostrar gráfico
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")


# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# Responde a estas tres preguntas con tus palabras. Sé concreto
# y haz referencia a tu solución, no a generalidades.
# No superes las 30 líneas en total entre las tres respuestas.
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    ¿Cómo funciona la arquitectura de tu aplicación? ¿Qué recibe
#    el LLM? ¿Qué devuelve? ¿Dónde se ejecuta el código generado?
#    ¿Por qué el LLM no recibe los datos directamente?
#
#   La aplicación sigue una arquitectura text-to-code donde el modelo de lenguaje no recibe los datos reales, 
#     sino únicamente la descripción de su estructura mediante el system prompt. A partir de la pregunta del usuario, 
#     el LLM genera código Python que se ejecuta localmente mediante exec() sobre el dataframe previamente cargado y transformado. 
#   Esto permite mantener los datos en local y reducir costes de tokens, además de aumentar la privacidad. 
#   El modelo no interpreta directamente los datos, sino que actúa como un generador de lógica analítica que se materializa al ejecutarse el código generado dentro de la aplicación.

#
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    ¿Qué información le das al LLM y por qué? Pon un ejemplo
#    concreto de una pregunta que funciona gracias a algo específico
#    de tu prompt, y otro de una que falla o fallaría si quitases
#    una instrucción.
#
#    El system prompt define las columnas disponibles, las reglas de generación de código y el formato de salida obligatorio en JSON. 
#    Gracias a esto, preguntas como “¿Escucho más en verano o en invierno?” funcionan correctamente, ya que el prompt especifica el uso de la columna season. 
#    Sin esta instrucción, el modelo tendería a inferir estaciones a partir del mes, generando código más complejo o incorrecto. 
#    Por el contrario, si se eliminase la restricción de devolver únicamente JSON, el modelo podría incluir explicaciones en texto libre o markdown, rompiendo el proceso de parsing y la ejecución posterior del código.
#
#
# 3. EL FLUJO COMPLETO
#    Describe paso a paso qué ocurre desde que el usuario escribe
#    una pregunta hasta que ve el gráfico en pantalla.
#
#    Cuando el usuario introduce una pregunta, la aplicación envía esa pregunta junto con el system prompt al modelo mediante la API. 
#    El modelo devuelve un string con un JSON que contiene código Python y una interpretación. 
#    Este JSON se parsea y, si la pregunta está dentro de alcance, el código se ejecuta con exec() sobre el dataframe. 
#    El código genera una figura de Plotly almacenada en la variable fig, que se renderiza en la interfaz. 
#    Finalmente, se muestra la interpretación textual junto con el gráfico y el código generado.
