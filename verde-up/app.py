import os
import json
import time
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from PIL import Image
from google import genai

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

DB_NAME = 'database.db'

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

def generar_con_respaldo(client, img, prompt):
    modelos_a_probar = ['gemini-3.8-flash']
    try:
        for m in client.models.list():
            nombre = m.name.replace('models/', '')
            acciones = getattr(m, 'supported_actions', None) or getattr(m, 'supported_generation_methods', []) or []
            if 'gemini' in nombre and 'flash' in nombre and 'generateContent' in acciones:
                if nombre not in modelos_a_probar and nombre != 'gemini-2.5-flash':
                    modelos_a_probar.append(nombre)
    except Exception:
        pass

    ultimo_error = None
    for modelo in modelos_a_probar:
        for intento in range(2):
            try:
                response = client.models.generate_content(
                    model=modelo,
                    contents=[img, prompt]
                )
                return response.text
            except Exception as e:
                ultimo_error = e
                if '503' in str(e) or 'UNAVAILABLE' in str(e):
                    time.sleep(1.5)
                else:
                    break
    raise ultimo_error

# --- 1. ESCÁNER IA ---
@app.route('/api/analizar', methods=['POST'])
def analizar_imagen():
    if 'imagen' not in request.files:
        return jsonify({'error': 'No se envió ninguna imagen'}), 400
    
    archivo = request.files['imagen']
    tipo_analisis = request.form.get('tipo', 'espacio')
    
    if archivo.filename == '':
        return jsonify({'error': 'Archivo vacío'}), 400

    filename = secure_filename(f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{archivo.filename}")
    ruta_imagen = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    archivo.save(ruta_imagen)
    foto_url = f"/static/uploads/{filename}"

    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key or api_key == 'tu_api_key_aqui':
        return jsonify({
            'resultado': '### Modo Prueba\n* Espacio ideal para **espinaca, cilantro o lechuga** en botellas recicladas.',
            'foto_url': foto_url
        })

    try:
        client = genai.Client(api_key=api_key)
        img = Image.open(ruta_imagen)
        img.thumbnail((1024, 1024))
        
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

        texto_respuesta = generar_con_respaldo(client, img, prompt)
        return jsonify({'resultado': texto_respuesta, 'foto_url': foto_url})
    
    except Exception as e:
        return jsonify({'error': f'Servidores ocupados, intenta de nuevo. Detalle: {str(e)}'}), 503

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
    if 'imagen' not in request.files:
        return jsonify({'error': 'Falta la fotografía de avance'}), 400

    archivo = request.files['imagen']
    filename = secure_filename(f"evo_{cultivo_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{archivo.filename}")
    ruta_imagen = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    archivo.save(ruta_imagen)
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

    api_key = os.getenv('GEMINI_API_KEY')
    if api_key and api_key != 'tu_api_key_aqui':
        try:
            client = genai.Client(api_key=api_key)
            img = Image.open(ruta_imagen)
            img.thumbnail((1024, 1024))
            prompt = (
                f"Evalúa la evolución de este cultivo de '{cultivo['nombre']}' en sistema '{cultivo['sistema']}'. "
                f"Su progreso anterior era {cultivo['progreso']}%. "
                "Responde ÚNICAMENTE con un objeto JSON válido (sin bloques markdown) con estas 4 claves exactas: "
                '{"progreso": entero de 15 a 100, "estado": "frase corta de 2-3 palabras", '
                '"dias_restantes": entero estimado para cosecha, "diagnostico": "evaluación breve de máximo 2 oraciones sobre su crecimiento y cuidado"}.'
            )
            raw = generar_con_respaldo(client, img, prompt)
            limpio = raw.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip()
            datos_ia = json.loads(limpio)
            nuevo_progreso = int(datos_ia.get('progreso', nuevo_progreso))
            nuevo_estado = datos_ia.get('estado', nuevo_estado)
            dias_restantes = int(datos_ia.get('dias_restantes', dias_restantes))
            diagnostico = datos_ia.get('diagnostico', diagnostico)
        except Exception as e:
            print(f"Aviso en evolución IA (usando cálculo progresivo): {e}")

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
    conn.close()
    return jsonify({'mensaje': 'Evolución registrada correctamente'})

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)