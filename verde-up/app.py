import atexit
import os
import json
import time
import sqlite3
import io
import logging
from threading import Lock
import httpx2
from google.genai import types
from google.genai import errors as genai_errors
from datetime import datetime
from flask import Flask, g, has_request_context, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from PIL import Image, ImageOps, ImageStat
from google import genai

load_dotenv()

app = Flask(__name__)
app.logger.setLevel(logging.INFO)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

AI_ENDPOINTS = {
    'analizar_imagen',
    'registrar_evolucion',
    'generar_guia_personalizada'
}
_AI_REQUEST_COUNT = 0
_AI_REQUEST_COUNT_LOCK = Lock()

@app.before_request
def iniciar_medicion_ia():
    global _AI_REQUEST_COUNT
    if request.endpoint not in AI_ENDPOINTS:
        return

    with _AI_REQUEST_COUNT_LOCK:
        _AI_REQUEST_COUNT += 1
        request_number = _AI_REQUEST_COUNT

    g.ai_request_number = request_number
    g.ai_request_started = time.perf_counter()
    g.ai_stage_timings = {}
    fase = 'primera' if request_number == 1 else 'subsecuente'
    app.logger.info(
        'ai_request_start id=%s fase=%s endpoint=%s',
        request_number,
        fase,
        request.endpoint
    )

@app.after_request
def finalizar_medicion_ia(response):
    inicio = getattr(g, 'ai_request_started', None)
    if inicio is None:
        return response

    duracion_ms = (time.perf_counter() - inicio) * 1000
    tiempos = g.ai_stage_timings
    tiempos['app_total'] = duracion_ms
    response.headers['X-AI-Request-ID'] = str(g.ai_request_number)
    response.headers['Server-Timing'] = ', '.join(
        f'{etapa};dur={duracion:.1f}'
        for etapa, duracion in tiempos.items()
    )
    fase = 'primera' if g.ai_request_number == 1 else 'subsecuente'
    app.logger.info(
        'ai_request_complete id=%s fase=%s endpoint=%s status=%s total_ms=%.1f',
        g.ai_request_number,
        fase,
        request.endpoint,
        response.status_code,
        duracion_ms
    )
    return response

DB_NAME = 'database.db'

def registrar_tiempo_ia(etapa, inicio, **detalles):
    duracion_ms = (time.perf_counter() - inicio) * 1000
    if has_request_context():
        request_number = getattr(g, 'ai_request_number', 'sin-id')
        tiempos = getattr(g, 'ai_stage_timings', None)
        if tiempos is not None:
            tiempos[etapa] = duracion_ms
    else:
        request_number = 'sin-request'

    detalle_texto = ' '.join(
        f'{nombre}={valor}'
        for nombre, valor in detalles.items()
    )
    app.logger.info(
        'ai_timing request=%s etapa=%s duracion_ms=%.1f %s',
        request_number,
        etapa,
        duracion_ms,
        detalle_texto
    )
    return duracion_ms

def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cultivos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT NOT NULL,
            sistema TEXT NOT NULL,
            estado TEXT NOT NULL,
            progreso INTEGER NOT NULL,
            dias_restantes INTEGER NOT NULL,
            notas TEXT,
            fecha_registro TEXT,
            foto_url TEXT
        )
    ''')

    # Migración segura por si database.db ya existía sin la columna foto_url
    try:
        cursor.execute('ALTER TABLE cultivos ADD COLUMN foto_url TEXT')
    except sqlite3.OperationalError:
        pass

    # Tabla para guardar la línea de tiempo de crecimiento de cada cultivo
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS evoluciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cultivo_id INTEGER NOT NULL,
            fecha TEXT NOT NULL,
            foto_url TEXT,
            progreso INTEGER NOT NULL,
            diagnostico TEXT NOT NULL,
            FOREIGN KEY (cultivo_id) REFERENCES cultivos (id)
        )
    ''')
    
    cursor.execute('SELECT COUNT(*) FROM cultivos')
    if cursor.fetchone()[0] == 0:
        ejemplos = [
            ('Jitomate Cherry', 'Muro Vertical (Botellas PET)', 'Saludable - Floración', 75, 14, 'Buena luz solar por la mañana. Riego cada 2 días.', '2026-11-04', ''),
            ('Lechuga Orejona', 'Hidroponía Casera (PVC)', 'Crecimiento Óptimo', 50, 20, 'Nivel de nutrientes estable. Hojas verdes sin plagas.', '2026-11-10', '')
        ]
        cursor.executemany('''
            INSERT INTO cultivos (nombre, sistema, estado, progreso, dias_restantes, notas, fecha_registro, foto_url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', ejemplos)
    
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template('index.html')

MODELO_GEMINI_PRINCIPAL = (
    os.getenv('GEMINI_MODEL', 'gemini-3.8-flash').strip()
    or 'gemini-3.8-flash'
)
MODELO_GEMINI_RESPALDO = (
    os.getenv('GEMINI_FALLBACK_MODEL', 'gemini-3.5-flash-lite').strip()
    or 'gemini-3.5-flash-lite'
)
MODELOS_GEMINI = tuple(dict.fromkeys(
    (MODELO_GEMINI_PRINCIPAL, MODELO_GEMINI_RESPALDO)
))
GEMINI_TIMEOUT_MS = 10_000
GEMINI_IMAGE_MAX_BYTES = 18 * 1024
_GEMINI_CLIENT = None
_GEMINI_CLIENT_API_KEY = None
_GEMINI_CLIENT_LOCK = Lock()
_GEMINI_MODEL_ROTATION_INDEX = 0
_GEMINI_MODEL_ROTATION_LOCK = Lock()

def crear_cliente_gemini(api_key):
    inicio = time.perf_counter()
    global _GEMINI_CLIENT, _GEMINI_CLIENT_API_KEY

    with _GEMINI_CLIENT_LOCK:
        if _GEMINI_CLIENT is None:
            _GEMINI_CLIENT = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    timeout=GEMINI_TIMEOUT_MS,
                    retry_options=types.HttpRetryOptions(attempts=1)
                )
            )
            _GEMINI_CLIENT_API_KEY = api_key
            etapa = 'client_init'
        elif _GEMINI_CLIENT_API_KEY != api_key:
            raise RuntimeError(
                'GEMINI_API_KEY cambió; reinicia Flask para aplicar la nueva clave.'
            )
        else:
            etapa = 'client_reuse'

        client = _GEMINI_CLIENT

    registrar_tiempo_ia(etapa, inicio)
    return client

def cerrar_cliente_gemini():
    global _GEMINI_CLIENT, _GEMINI_CLIENT_API_KEY
    with _GEMINI_CLIENT_LOCK:
        client = _GEMINI_CLIENT
        _GEMINI_CLIENT = None
        _GEMINI_CLIENT_API_KEY = None
    if client is not None:
        client.close()

atexit.register(cerrar_cliente_gemini)

def cargar_imagen_rgb(imagen_o_ruta):
    if isinstance(imagen_o_ruta, Image.Image):
        imagen = ImageOps.exif_transpose(imagen_o_ruta)
    else:
        with Image.open(imagen_o_ruta) as original:
            imagen = ImageOps.exif_transpose(original)
    return imagen.convert('RGB')

def preparar_imagen_ligera(imagen_o_ruta):
    """Comprime la imagen para Gemini y devuelve también el brillo medio de sus píxeles."""
    inicio = time.perf_counter()
    img = None
    try:
        img = cargar_imagen_rgb(imagen_o_ruta)
        rojo, verde, azul = ImageStat.Stat(img).mean
        luminosidad = 0.299 * rojo + 0.587 * verde + 0.114 * azul

        datos = b''
        for dimension in (400, 320, 256, 192, 128, 96, 64, 32):
            img.thumbnail((dimension, dimension), Image.Resampling.LANCZOS)
            for calidad in (65, 55, 45, 35):
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=calidad, optimize=True)
                datos = buffer.getvalue()
                if len(datos) <= GEMINI_IMAGE_MAX_BYTES:
                    break
            if len(datos) <= GEMINI_IMAGE_MAX_BYTES:
                break

        if len(datos) > GEMINI_IMAGE_MAX_BYTES:
            raise ValueError('No fue posible comprimir la imagen al tamaño máximo configurado.')

        app.logger.info(
            'ai_image_compressed bytes=%s max_bytes=%s',
            len(datos),
            GEMINI_IMAGE_MAX_BYTES
        )
        parte = types.Part.from_bytes(data=datos, mime_type='image/jpeg')
        return parte, luminosidad
    finally:
        if img is not None:
            img.close()
        registrar_tiempo_ia('image_prepare', inicio)

class GeminiTimeoutError(Exception):
    pass

def es_timeout_gemini(error):
    return (
        isinstance(error, (httpx2.TimeoutException, TimeoutError))
        or (
            isinstance(error, genai_errors.APIError)
            and error.code in (408, 504)
        )
    )

def seleccionar_modelos_gemini():
    global _GEMINI_MODEL_ROTATION_INDEX
    if not MODELOS_GEMINI:
        raise RuntimeError('No hay modelos Gemini configurados.')

    with _GEMINI_MODEL_ROTATION_LOCK:
        indice = _GEMINI_MODEL_ROTATION_INDEX % len(MODELOS_GEMINI)
        _GEMINI_MODEL_ROTATION_INDEX = (indice + 1) % len(MODELOS_GEMINI)

    return MODELOS_GEMINI[indice:] + MODELOS_GEMINI[:indice]

def generar_con_respaldo(client, ruta_imagen, prompt):
    inicio = time.perf_counter()
    modelos = seleccionar_modelos_gemini()
    registrar_tiempo_ia(
        'model_selection',
        inicio,
        discovery='disabled',
        model_count=len(modelos)
    )
    parte_img = preparar_imagen_ligera(ruta_imagen)[0] if ruta_imagen else None
    contents = [parte_img, prompt] if parte_img is not None else [prompt]
    hubo_timeout = False

    for indice, modelo in enumerate(modelos):
        inicio = time.perf_counter()
        etapa = 'generation_primary' if indice == 0 else 'generation_fallback'
        try:
            app.logger.info(
                'ai_generation_start model=%s attempt=%s',
                modelo,
                indice + 1
            )
            response = client.models.generate_content(
                model=modelo,
                contents=contents,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    temperature=0.4,
                    max_output_tokens=480
                )
            )
            if not response.text:
                raise RuntimeError(f'{modelo} devolvió una respuesta vacía.')
            registrar_tiempo_ia(
                etapa,
                inicio,
                model=modelo,
                resultado='ok'
            )
            return response.text
        except Exception as e:
            registrar_tiempo_ia(
                etapa,
                inicio,
                model=modelo,
                resultado='error'
            )
            hubo_timeout = hubo_timeout or es_timeout_gemini(e)
            app.logger.warning(
                'ai_generation_error model=%s attempt=%s error=%s',
                modelo,
                indice + 1,
                e
            )
            if indice == len(modelos) - 1:
                if hubo_timeout:
                    raise GeminiTimeoutError(
                        'La IA no pudo completar la respuesta dentro del tiempo de espera.'
                    ) from e
                raise

    raise RuntimeError('No hay modelos Gemini configurados.')

def analisis_local_respaldo(tipo_analisis, ruta_imagen):
    inicio = time.perf_counter()
    img = None
    try:
        img = cargar_imagen_rgb(ruta_imagen)
        rojo, verde, azul = ImageStat.Stat(img).mean
        luminosidad = 0.299 * rojo + 0.587 * verde + 0.114 * azul
    finally:
        if img is not None:
            img.close()
        registrar_tiempo_ia('local_image_analysis', inicio)

    if luminosidad > 145:
        nivel_luz = 'brillo alto en la foto'
        cultivos = 'jitomate cherry, fresa o albahaca, si el espacio recibe suficiente sol directo'
    elif luminosidad > 95:
        nivel_luz = 'brillo intermedio en la foto'
        cultivos = 'lechuga, cilantro o perejil, según las horas de luz disponibles'
    else:
        nivel_luz = 'brillo bajo en la foto'
        cultivos = 'hierbas tolerantes a sombra parcial; quizá se requiera luz de cultivo'

    if tipo_analisis == 'espacio':
        return (
            '### Diagnóstico local orientativo (sin IA)\n'
            f'* **Brillo promedio de la imagen:** {luminosidad:.0f}/255 ({nivel_luz}).\n'
            f'* **Cultivos a considerar:** {cultivos}.\n'
            '* **Aprovechamiento:** un sistema vertical puede ayudar a usar mejor el espacio disponible.\n'
            '* **Importante:** el brillo depende de la exposición de la cámara y no equivale a una medición de lux; '
            'confirma las horas de sol en el lugar antes de elegir cultivos.'
        )

    return (
        '### Diagnóstico local orientativo (sin IA)\n'
        f'* **Brillo promedio de la imagen:** {luminosidad:.0f}/255 ({nivel_luz}).\n'
        '* **Alcance:** el brillo de una foto no permite identificar la planta ni confirmar plagas, riego o estado de salud.\n'
        '* **Revisión sugerida:** observa el envés de las hojas, comprueba la humedad del sustrato y toma otra foto '
        'con luz natural indirecta para comparar.'
    )

def responder_analisis_local(tipo_analisis, ruta_imagen, foto_url):
    try:
        resultado = analisis_local_respaldo(tipo_analisis, ruta_imagen)
    except (OSError, ValueError, Image.DecompressionBombError):
        app.logger.exception('No se pudo analizar localmente la imagen recibida.')
        return jsonify({'error': 'No se pudo procesar la imagen. Prueba con otra fotografía.'}), 400

    return jsonify({
        'resultado': resultado,
        'foto_url': foto_url,
        'modo': 'local'
    })

# --- 1. ESCÁNER IA ---
@app.route('/api/analizar', methods=['POST'])
def analizar_imagen():
    inicio = time.perf_counter()
    archivo = request.files.get('imagen')
    registrar_tiempo_ia('upload_parse', inicio)
    if archivo is None:
        return jsonify({'error': 'No se envió ninguna imagen'}), 400
    
    tipo_analisis = request.form.get('tipo', 'espacio')
    
    if archivo.filename == '':
        return jsonify({'error': 'Archivo vacío'}), 400

    filename = secure_filename(f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{archivo.filename}")
    ruta_imagen = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    inicio = time.perf_counter()
    archivo.save(ruta_imagen)
    registrar_tiempo_ia('upload_save', inicio)
    foto_url = f"/static/uploads/{filename}"

    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key or api_key == 'tu_api_key_aqui':
        app.logger.info('ai_local_fallback reason=api_key_unavailable endpoint=analizar_imagen')
        return responder_analisis_local(tipo_analisis, ruta_imagen, foto_url)

    try:
        client = crear_cliente_gemini(api_key)
        
        if tipo_analisis == 'espacio':
            prompt = (
                "Eres un agrónomo experto en agricultura urbana e hidroponía de bajo costo para familias "
                "con poco espacio. Analiza esta foto de un espacio en casa y responde en español de forma clara, "
                "breve y estructurada con: 1) Evaluación de luz y espacio, 2) Top 3 alimentos (frutas/verduras/hierbas) "
                "ideales para cultivar ahí, 3) Cómo armar el contenedor usando materiales reciclados o económicos."
            )
        else:
            prompt = (
                "Eres un especialista en botánica y cuidado de cultivos caseros. Analiza esta planta y responde "
                "en español de forma clara y estructurada con: 1) Identificación de la planta y etapa de crecimiento, "
                "2) Diagnóstico de salud (plagas, falta de agua o luz), 3) 3 acciones prácticas y económicas para mejorar su cosecha."
            )

        texto_respuesta = generar_con_respaldo(client, ruta_imagen, prompt)
        return jsonify({'resultado': texto_respuesta, 'foto_url': foto_url})
    except Exception as e:
        app.logger.exception(
            'Gemini no disponible para analizar la imagen; usando respaldo local (%s).',
            type(e).__name__
        )
        return responder_analisis_local(tipo_analisis, ruta_imagen, foto_url)

# --- 2. OBTENER CULTIVOS E HISTORIAL ---
@app.route('/api/cultivos', methods=['GET'])
def obtener_cultivos():
    conn = get_db()
    cultivos = conn.execute('SELECT * FROM cultivos ORDER BY id DESC').fetchall()
    resultado = []
    for c in cultivos:
        item = dict(c)
        historial = conn.execute(
            'SELECT * FROM evoluciones WHERE cultivo_id = ? ORDER BY id DESC',
            (c['id'],)
        ).fetchall()
        item['historial'] = [dict(h) for h in historial]
        resultado.append(item)
    conn.close()
    return jsonify(resultado)

# --- 3. GUARDAR NUEVO CULTIVO DESDE EL ESCÁNER ---
@app.route('/api/cultivos', methods=['POST'])
def crear_cultivo():
    data = request.json
    nombre = data.get('nombre', 'Nuevo Cultivo')
    sistema = data.get('sistema', 'Muro Vertical Casero')
    notas = data.get('notas', 'Iniciado desde Escáner IA')
    foto_url = data.get('foto_url', '')
    fecha_hoy = datetime.now().strftime('%Y-%m-%d')

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO cultivos (nombre, sistema, estado, progreso, dias_restantes, notas, fecha_registro, foto_url)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (nombre, sistema, 'Recién Plantado (Día 1)', 10, 45, notas, fecha_hoy, foto_url))
    
    cultivo_id = cursor.lastrowid
    cursor.execute('''
        INSERT INTO evoluciones (cultivo_id, fecha, foto_url, progreso, diagnostico)
        VALUES (?, ?, ?, ?, ?)
    ''', (cultivo_id, fecha_hoy, foto_url, 10, f"Inicio de cultivo en {sistema}. {notas[:140]}..."))

    conn.commit()
    conn.close()
    return jsonify({'mensaje': 'Cultivo guardado en Mi Huerto', 'id': cultivo_id})

# --- 4. REGISTRAR EVOLUCIÓN CON NUEVA FOTO + IA ---
@app.route('/api/cultivos/<int:cultivo_id>/evolucion', methods=['POST'])
def registrar_evolucion(cultivo_id):
    inicio = time.perf_counter()
    archivo = request.files.get('imagen')
    registrar_tiempo_ia('upload_parse', inicio)
    if archivo is None:
        return jsonify({'error': 'Falta la fotografía de avance'}), 400

    filename = secure_filename(f"evo_{cultivo_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{archivo.filename}")
    ruta_imagen = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    inicio = time.perf_counter()
    archivo.save(ruta_imagen)
    registrar_tiempo_ia('upload_save', inicio)
    foto_url = f"/static/uploads/{filename}"
    fecha_hoy = datetime.now().strftime('%Y-%m-%d')

    conn = get_db()
    cultivo = conn.execute('SELECT * FROM cultivos WHERE id = ?', (cultivo_id,)).fetchone()
    if not cultivo:
        conn.close()
        return jsonify({'error': 'Cultivo no encontrado'}), 404

    # Valores por defecto en caso de respaldo
    nuevo_progreso = min(100, cultivo['progreso'] + 20)
    nuevo_estado = 'En Crecimiento Activo'
    dias_restantes = max(1, cultivo['dias_restantes'] - 7)
    diagnostico = 'Desarrollo foliar saludable detectado. Mantener hidratación y luz.'

    ai_timeout = False
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key and api_key != 'tu_api_key_aqui':
        try:
            client = crear_cliente_gemini(api_key)
            prompt = (
                f"Evalúa la evolución de este cultivo de '{cultivo['nombre']}' en sistema '{cultivo['sistema']}'. "
                f"Su progreso anterior era {cultivo['progreso']}%. "
                "Responde ÚNICAMENTE con un objeto JSON válido (sin bloques markdown) con estas 4 claves exactas: "
                '{"progreso": entero de 15 a 100, "estado": "frase corta de 2-3 palabras", '
                '"dias_restantes": entero estimado para cosecha, "diagnostico": "evaluación breve de máximo 2 oraciones sobre su crecimiento y cuidado"}.'
            )
            raw = generar_con_respaldo(client, ruta_imagen, prompt)
            inicio = time.perf_counter()
            limpio = raw.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip()
            datos_ia = json.loads(limpio)
            nuevo_progreso = int(datos_ia.get('progreso', nuevo_progreso))
            nuevo_estado = datos_ia.get('estado', nuevo_estado)
            dias_restantes = int(datos_ia.get('dias_restantes', dias_restantes))
            diagnostico = datos_ia.get('diagnostico', diagnostico)
            registrar_tiempo_ia('evolution_parse', inicio)
        except GeminiTimeoutError as e:
            ai_timeout = True
            app.logger.warning(
                'ai_timeout evolution_id=%s using_progress_fallback error=%s',
                cultivo_id,
                e
            )
        except Exception as e:
            app.logger.exception(
                'Error en evolución IA para cultivo %s; usando cálculo progresivo.',
                cultivo_id
            )

    inicio = time.perf_counter()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE cultivos
        SET progreso = ?, estado = ?, dias_restantes = ?, notas = ?, foto_url = ?
        WHERE id = ?
    ''', (nuevo_progreso, nuevo_estado, dias_restantes, diagnostico, foto_url, cultivo_id))

    cursor.execute('''
        INSERT INTO evoluciones (cultivo_id, fecha, foto_url, progreso, diagnostico)
        VALUES (?, ?, ?, ?, ?)
    ''', (cultivo_id, fecha_hoy, foto_url, nuevo_progreso, diagnostico))

    conn.commit()
    registrar_tiempo_ia('evolution_save', inicio)
    conn.close()
    respuesta = {'mensaje': 'Evolución registrada correctamente'}
    if ai_timeout:
        respuesta['aviso'] = (
            'La IA tardó demasiado; se guardó una estimación automática del avance.'
        )
    return jsonify(respuesta)

# --- 5. GENERADOR DE GUÍA RECICLADA PERSONALIZADA CON IA ---
@app.route('/api/guia-personalizada', methods=['POST'])
def generar_guia_personalizada():
    inicio = time.perf_counter()
    data = request.json or {}
    registrar_tiempo_ia('input_parse', inicio)
    materiales = data.get('materiales', '').strip()
    espacio = data.get('espacio', 'Pared o balcón pequeño').strip()

    if not materiales:
        return jsonify({'error': 'Escribe al menos un material reciclado que tengas disponible.'}), 400

    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key or api_key == 'tu_api_key_aqui':
        return jsonify({
            'guia': f'### Sistema Casero con {materiales}\n1. Lava y perfora orificios de drenaje.\n2. Coloca sustrato ligero y siembra **cilantro, espinaca o cebollín**.'
        })

    try:
        client = crear_cliente_gemini(api_key)
        prompt = (
            f"Eres un experto en agricultura urbana e hidroponía social de bajo costo para familias con pocos recursos. "
            f"Crea una guía práctica paso a paso en español usando estos materiales disponibles: '{materiales}' "
            f"para instalar en este espacio: '{espacio}'. "
            "Responde de forma breve y estructurada con: 1) Nombre del sistema casero, "
            "2) Alimentos recomendados y ahorro aproximado, 3) Pasos de armado (máximo 5 pasos claros), "
            "4) Cuidado y riego básico."
        )
        texto = generar_con_respaldo(client, None, prompt)
        return jsonify({'guia': texto})
    except GeminiTimeoutError as e:
        return jsonify({'error': str(e)}), 504
    except Exception as e:
        return jsonify({'error': f'Servidores ocupados, intenta de nuevo en unos segundos. Detalle: {str(e)}'}), 503

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)