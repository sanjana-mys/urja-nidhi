"""
Urja Nidhi — Backend v4.0
====================================
New in this version:
  - Farmer registration & login (email or phone)
  - Session tokens (simple, no JWT dependency)
  - Notifications tab gated: only logged-in farmer's own alerts shown
  - Alerts sent to farmer's registered email/phone (not hardcoded)
  - Immediate alerts on threshold breach (always, even if not logged in)
  - Daily summary email/SMS at 8 AM (scheduled background thread)
  - Firebase Realtime Database integration for live sensor data
  - All previous features retained
"""

# ── Load .env FIRST ───────────────────────────────────────────────────────────
from pathlib import Path
from dotenv import load_dotenv
# Ensure we load the project's v5/.env (not whatever the current working directory is).
_BASE_DIR = Path(__file__).resolve().parents[1]  # .../v5/
load_dotenv(dotenv_path=str(_BASE_DIR / ".env"), override=True)

from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import joblib, numpy as np, pandas as pd
import os, json, threading, time, sqlite3, random, smtplib, secrets, hashlib, re
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import urllib.request
import urllib.parse
import sys

# Windows PowerShell default encoding can break emoji prints (e.g. "✅").
# Force UTF-8 output to avoid UnicodeEncodeError during startup.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── App ───────────────────────────────────────────────────────────────────────
CWD = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(CWD, "templates"))
app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(32))
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

@app.after_request
def add_no_cache_headers(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

# ── ML Model ──────────────────────────────────────────────────────────────────
try:
    model = joblib.load(os.path.join(CWD, "urja_nidhi_ensemble_model.pkl"))
    print("✅ Model loaded")
except Exception as e:
    model = None
    print(f"⚠️  Model not loaded: {e}")

try:
    with open(os.path.join(CWD, "model_feature_columns.json")) as f:
        FEATURE_COLUMNS = json.load(f)
    print(f"✅ Feature columns: {len(FEATURE_COLUMNS)}")
except Exception:
    FEATURE_COLUMNS = None
    print("⚠️  model_feature_columns.json missing — run train_model.py")

WASTE_OHE    = ["crop_residue", "mixed", "poultry_litter"]
PRETREAT_OHE = ["composted", "fermented", "raw"]
CROP_OHE     = ["groundnut", "pulses", "soybean"]

# ── SMTP Config (server-level sender account — NOT the farmer's address) ──────
SMTP_CFG = {
    "host":     os.getenv("SMTP_HOST",     "smtp.gmail.com"),
    "port":     int(os.getenv("SMTP_PORT", "587")),
    "user":     os.getenv("SMTP_USER",     ""),
    "password": os.getenv("SMTP_PASSWORD", ""),
}

# ── Twilio Config (server-level — used to send SMS to any farmer number) ──────
TWILIO_CFG = {
    "sid":          os.getenv("TWILIO_ACCOUNT_SID", ""),
    "token":        os.getenv("TWILIO_AUTH_TOKEN",  ""),
    "from_sms":     os.getenv("TWILIO_FROM",        ""),
    "from_wa":      os.getenv("TWILIO_WA_FROM",     ""),
}

# ── Gemini ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
FIREBASE_WEB_API_KEY = os.getenv("FIREBASE_WEB_API_KEY", "").strip()

# MQ5 (ADS1115) helper: convert raw 0-32767 to approximate ppm for a 0-3.3V setup.
def mq5_raw_to_ppm(methane_raw):
    try:
        if methane_raw is None:
            return None
        r = float(methane_raw)
        if r < 0:
            return None
        # MQ5 raw-to-ppm mapping is approximately banded per your reference table:
        # raw 0-3000   -> <200 ppm
        # raw 3000-8000 -> 200-1000 ppm
        # raw 8000-16000 -> 1000-3000 ppm
        # raw 16000-24000 -> 3000-6000 ppm
        # This is a piecewise-linear approximation to keep ppm-only UI consistent.
        if r <= 3000.0:
            return ((r / 3000.0) * 200.0)*14.5
        if r <= 8000.0:
            return (200.0 + (r - 3000.0) * (800.0 / 5000.0))*14.5  # 200 -> 1000
        if r <= 16000.0:
            return (1000.0 + (r - 8000.0) * (2000.0 / 8000.0))*14.5  # 1000 -> 3000
        if r <= 24000.0:
            return (3000.0 + (r - 16000.0) * (3000.0 / 8000.0))*14.5  # 3000 -> 6000
        # Extend beyond with the last slope.
        return (6000.0 + (r - 24000.0) * (3000.0 / 8000.0))*14.5
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════════════════
# ── FIREBASE CONFIG (Layer 4a — Realtime Database) ───────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# Paste your Firebase project values here (from Firebase Console →
# Project Settings → Service Accounts → Database URL, and
# Project Settings → General → Your apps → Config)
# OR set them as environment variables in your .env file.

FIREBASE_CONFIG = {
    # ── Realtime Database ─────────────────────────────────────────────────────
    # Firebase Console → Realtime Database → copy the URL shown at the top
    # Example: "https://your-project-default-rtdb.firebaseio.com"
    "database_url": os.getenv("FIREBASE_DATABASE_URL", ""),

    # ── Database path where your ESP32 writes sensor data ────────────────────
    # Example: if ESP32 writes to /digesters/DIG001/sensors, set "digesters/DIG001/sensors"
    "sensor_path": os.getenv("FIREBASE_SENSOR_PATH", "sensorData"),

    # ── Service Account credentials (for server-side read) ───────────────────
    # Firebase Console → Project Settings → Service Accounts → Generate new private key
    # Download the JSON file, place it in backend/ as firebase_credentials.json
    # OR set FIREBASE_CREDENTIALS_PATH in .env to point to a different location
    "credentials_path": os.path.join(CWD, os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase_credentials.json")),
}

# Firebase listener state
_firebase_app    = None
_firebase_active = False
_firebase_listeners = []
_last_valid_npk  = {"n": None, "p": None, "k": None}
# Merged view of the node at FIREBASE_SENSOR_PATH (put "/" + dict replaces; leaf puts merge).
_firebase_sensor_merge = {}


def _flatten_firebase_sensor_dict(data: dict) -> dict:
    """If ESP stores fields under readings/ or sensors/, merge for name lookups."""
    if not isinstance(data, dict):
        return data
    out = dict(data)
    for nest in ("readings", "sensors", "values", "data"):
        inner = out.get(nest)
        if isinstance(inner, dict):
            for k, v in inner.items():
                out.setdefault(k, v)
    return out


def _merge_firebase_listener_event(event) -> dict | None:
    """
    Firebase SSE sends a full dict at path '/' on initial sync, but per-field
    writes arrive as path '/temperature' with a scalar — those were ignored when
    we only accepted isinstance(data, dict).
    """
    global _firebase_sensor_merge
    try:
        et = getattr(event, "event_type", None)
        if et and str(et).lower() not in ("put", "patch"):
            return None
    except Exception:
        pass

    path = getattr(event, "path", "/") or "/"
    if not path.startswith("/"):
        path = "/" + path
    payload = event.data

    if path in ("/", ""):
        if isinstance(payload, dict):
            _firebase_sensor_merge.clear()
            _firebase_sensor_merge.update(payload)
        elif payload is None:
            _firebase_sensor_merge.clear()
            return None
        else:
            return None
    else:
        segments = [s for s in path.strip("/").split("/") if s]
        if isinstance(payload, dict):
            for k, v in payload.items():
                _firebase_sensor_merge[k] = v
        else:
            if segments:
                _firebase_sensor_merge[segments[-1]] = payload

    merged = _flatten_firebase_sensor_dict(_firebase_sensor_merge)
    return merged if isinstance(merged, dict) and merged else None


def _emit_sensor_update_ws(payload: dict) -> None:
    """Emit from Firebase/simulation threads; app context keeps Flask-SocketIO happy."""
    try:
        with app.app_context():
            socketio.emit("sensor_update", payload, namespace="/")
    except Exception as e:
        print(f"WebSocket sensor_update emit failed: {e}")

def init_firebase():
    """
    Initialises Firebase Admin SDK if credentials and database URL are set.
    Uses firebase-admin package. Install: pip install firebase-admin
    Falls back to simulation if Firebase is not configured.
    """
    global _firebase_app, _firebase_active
    db_url  = FIREBASE_CONFIG["database_url"].strip()
    cred_path = FIREBASE_CONFIG["credentials_path"]

    if not db_url:
        print("⚠️  Firebase: FIREBASE_DATABASE_URL not set — using simulated feed")
        return False
    if not os.path.exists(cred_path):
        print(f"⚠️  Firebase: credentials file not found at {cred_path} — using simulated feed")
        return False
    try:
        import firebase_admin
        from firebase_admin import credentials, db as firebase_db
        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_path)
            _firebase_app = firebase_admin.initialize_app(cred, {"databaseURL": db_url})
        _firebase_active = True
        print(f"✅ Firebase connected: {db_url}")
        return True
    except ImportError:
        print("⚠️  Firebase: firebase-admin not installed. Run: pip install firebase-admin")
        return False
    except Exception as e:
        print(f"⚠️  Firebase init error: {e}")
        return False

def get_firebase_auth_module():
    """
    Return firebase_admin.auth module initialised with service-account creds.
    This is separate from Realtime DB init so email/password auth can work
    even when FIREBASE_DATABASE_URL is not configured.
    """
    try:
        import firebase_admin
        from firebase_admin import credentials, auth as firebase_auth
    except Exception:
        return None

    cred_path = FIREBASE_CONFIG["credentials_path"]
    if not os.path.exists(cred_path):
        return None
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
        return firebase_auth
    except Exception:
        try:
            return firebase_auth
        except Exception:
            return None

def firebase_verify_email_password(email: str, password: str):
    """
    Verify Firebase email/password using Identity Toolkit REST API.
    Returns (firebase_uid, error_message).
    """
    if not FIREBASE_WEB_API_KEY:
        return None, "FIREBASE_WEB_API_KEY is not configured"
    payload = json.dumps({
        "email": email,
        "password": password,
        "returnSecureToken": True
    }).encode("utf-8")
    url = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?" + urllib.parse.urlencode({
        "key": FIREBASE_WEB_API_KEY
    })
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            out = json.loads(resp.read().decode("utf-8"))
            uid = out.get("localId")
            if not uid:
                return None, "Firebase login response missing user id"
            return uid, None
    except Exception:
        return None, "Invalid email/password or Firebase auth unavailable"

def firebase_sensor_listener():
    """
    Listens to Firebase Realtime Database for new sensor readings.
    When ESP32 writes a new value, this function receives it and pushes
    it through the same pipeline as the simulated feed.
    """
    try:
        from firebase_admin import db as firebase_db
        primary_path = (FIREBASE_CONFIG["sensor_path"] or "sensorData").strip("/")
        # Listen ONLY to configured live sensor node to avoid stale/history mix-ups.
        candidate_paths = [primary_path]

        def on_sensor_change(event):
            """Called by Firebase whenever data at sensor_path changes."""
            try:
                data = _merge_firebase_listener_event(event)
                if not isinstance(data, dict) or not data:
                    return
                # If this is a history bucket ({pushId: {...}}), pick the latest entry.
                if not any(k in data for k in ("temperature", "temperature_C", "temp", "ph", "pH")):
                    samples = [v for v in data.values() if isinstance(v, dict)]
                    if samples:
                        def _sample_ts(s):
                            return str(s.get("timestamp", ""))
                        data = max(samples, key=_sample_ts)
                    else:
                        return
                # Normalise field names — handle multiple naming conventions
                # Firebase may send: temperature_C, temperature, temp, etc.

                def _maybe_npk(v):
                    """Treat negative / -1 values as missing."""
                    try:
                        fv = float(v)
                        return fv if fv >= 0 else None
                    except Exception:
                        return None

                n_in = data.get("nitrogen_concentration", data.get("nitrogen_mgkg", None))
                p_in = data.get("phosphorus_concentration", data.get("phosphorus_mgkg", None))
                k_in = data.get("potassium_concentration", data.get("potassium_mgkg", None))
                n_val = _maybe_npk(n_in)
                p_val = _maybe_npk(p_in)
                k_val = _maybe_npk(k_in)

                # Reuse last valid values so the dashboard never shows -1.
                if n_val is not None:
                    _last_valid_npk["n"] = n_val
                if p_val is not None:
                    _last_valid_npk["p"] = p_val
                if k_val is not None:
                    _last_valid_npk["k"] = k_val

                reading = {
                    "digester_id": data.get("digester_id", "DIG001"),
                    "temperature": float(data.get("temperature", data.get("temperature_C", data.get("temp", 0)))),
                    "ph":          float(data.get("ph", data.get("pH", 7.0))),
                    "pressure":    float(data.get("pressure", data.get("pressure_kPa", 105.0))),
                    "gas_flow":    float(data.get("gas_flow", data.get("gasFlow", data.get("gas_flow_lh", 0)))),
                    "methane_raw": float(data.get("methane_raw", data.get("methane_ppm", 0))),
                    "nitrogen_concentration": _last_valid_npk["n"],
                    "phosphorus_concentration": _last_valid_npk["p"],
                    "potassium_concentration": _last_valid_npk["k"],
                    "timestamp":   data.get("timestamp", datetime.now().isoformat()),
                }
                reading["methane_ppm"] = mq5_raw_to_ppm(reading["methane_raw"])
                alerts = check_alerts(reading)
                # Persist to SQLite
                conn = None
                try:
                    conn = get_db()
                    conn.execute("""INSERT INTO sensor_readings
                        (digester_id,temperature,ph,pressure,gas_flow,methane_raw,methane_ppm,
                         nitrogen_concentration,phosphorus_concentration,potassium_concentration,timestamp)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                        (reading["digester_id"], reading["temperature"], reading["ph"],
                         reading["pressure"], reading["gas_flow"], reading["methane_raw"], 
                         reading["methane_ppm"],
                         reading["nitrogen_concentration"], reading["phosphorus_concentration"],
                         reading["potassium_concentration"], reading["timestamp"]))
                    conn.commit()
                except Exception as e:
                    print(f"Firebase DB insert failed: {e}")
                finally:
                    try:
                        if conn:
                            conn.close()
                    except Exception:
                        pass
                # Push to dashboard via WebSocket (thread + app context)
                _emit_sensor_update_ws({**reading, "alerts": alerts})
                # Send alerts immediately to all registered farmers
                if alerts:
                    threading.Thread(
                        target=dispatch_alerts_to_all_farmers,
                        args=(alerts,), daemon=True
                    ).start()
                print(f"📡 Firebase data: temp={reading['temperature']} pH={reading['ph']}")
            except Exception as e:
                print(f"Firebase listener error: {e}")

        # Keep registrations alive for process lifetime; otherwise updates can stop.
        global _firebase_listeners
        for p in candidate_paths:
            try:
                reg = firebase_db.reference(p).listen(on_sensor_change)
                _firebase_listeners.append(reg)
                print(f"📡 Firebase subscribed: /{p}")
            except Exception as e:
                print(f"Firebase subscribe failed for /{p}: {e}")

        # Keep this daemon thread alive so listener registrations remain reachable.
        while True:
            time.sleep(60)
    except Exception as e:
        print(f"Firebase listener failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# ── DATABASE ──────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
DB_PATH = os.path.join(CWD, "urja_nidhi.db")

def get_db():
    # Make concurrent writes from Firebase thread + HTTP handlers more resilient.
    conn = sqlite3.connect(DB_PATH, timeout=20, check_same_thread=False)
    conn.execute("PRAGMA busy_timeout = 20000")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript("""
        -- Sensor readings from IoT / Firebase
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            digester_id TEXT DEFAULT 'DIG001',
            temperature REAL, ph REAL, pressure REAL, gas_flow REAL,
            methane_raw REAL, methane_ppm REAL,
            nitrogen_concentration REAL, phosphorus_concentration REAL, potassium_concentration REAL,
            timestamp TEXT
        );

        -- ML prediction results
        CREATE TABLE IF NOT EXISTS prediction_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            digester_id TEXT DEFAULT 'DIG001',
            daily_biogas REAL, weekly_biogas REAL,
            nitrogen_level TEXT, nitrogen_pct REAL,
            compost_quality TEXT, compost_score REAL,
            money_saved REAL, energy_kwh REAL,
            urea_equivalent REAL, co2_reduction REAL,
            cooking_hours REAL, timestamp TEXT
        );

        -- Notification delivery log — per farmer
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            farmer_id INTEGER,
            channel TEXT, recipient TEXT,
            subject TEXT, message TEXT,
            status TEXT, timestamp TEXT
        );

        -- AI advisory conversations
        CREATE TABLE IF NOT EXISTS advisory_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            farmer_id INTEGER,
            question TEXT, context TEXT,
            answer TEXT, timestamp TEXT
        );

        -- Farmer accounts (email OR phone, not both required)
        CREATE TABLE IF NOT EXISTS farmers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            phone TEXT UNIQUE,
            firebase_uid TEXT,
            digester_id TEXT DEFAULT 'DIG001',
            created_at TEXT,
            last_login TEXT
        );

        -- Farmer crop preferences for personalised crop-cycle prediction
        CREATE TABLE IF NOT EXISTS farmer_crops (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            farmer_id INTEGER NOT NULL,
            crop_name TEXT NOT NULL,
            is_active INTEGER DEFAULT 1,
            created_at TEXT,
            UNIQUE(farmer_id, crop_name)
        );

        -- Active session tokens (in-memory would be faster but DB survives restarts)
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            farmer_id INTEGER NOT NULL,
            expires_at TEXT NOT NULL
        );

        -- Custom threshold limits per parameter (one row per param, system-wide)
        CREATE TABLE IF NOT EXISTS thresholds (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            param        TEXT UNIQUE NOT NULL,
            label        TEXT NOT NULL,
            unit         TEXT NOT NULL,
            low_warn     REAL,
            low_crit     REAL,
            high_warn    REAL,
            high_crit    REAL,
            sensor_type  TEXT,
            interpretation TEXT,
            enabled      INTEGER DEFAULT 1,
            updated_at   TEXT
        );
    """)
    # Schema migration for older SQLite DBs (ADD missing columns only).
    sensor_cols = {r["name"] for r in conn.execute("PRAGMA table_info(sensor_readings)").fetchall()}
    def _add_col(col: str, col_type: str):
        if col not in sensor_cols:
            conn.execute(f"ALTER TABLE sensor_readings ADD COLUMN {col} {col_type}")

    _add_col("methane_raw", "REAL")
    _add_col("methane_ppm", "REAL")
    _add_col("nitrogen_concentration", "REAL")
    _add_col("phosphorus_concentration", "REAL")
    _add_col("potassium_concentration", "REAL")
    farmer_cols = {r["name"] for r in conn.execute("PRAGMA table_info(farmers)").fetchall()}
    if "firebase_uid" not in farmer_cols:
        conn.execute("ALTER TABLE farmers ADD COLUMN firebase_uid TEXT")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_farmers_firebase_uid ON farmers(firebase_uid)")

    conn.commit(); conn.close()

init_db()

# ── Seed default thresholds (only if table is empty) ─────────────────────────
def seed_thresholds():
    """
    Seed thresholds with sensor-specific interpretations.
    Based on: DS18B20, MQ5 (via ADS1115), DFRobot pH, Pressure sensor, ADS1115 ref.
    """
    conn = get_db()
    existing = conn.execute("SELECT COUNT(*) FROM thresholds").fetchone()[0]

    # (param, label, unit, low_warn, low_crit, high_warn, high_crit, sensor_type, interpretation)
    defaults = [
            # ── DS18B20 Temperature (°C) ──────────────────────────────────────
            ("temperature", "Digester Temperature", "°C",
             20.0, 10.0, 30.0, 40.0,
             "DS18B20",
             "Optimal: 20-30°C (soil/water) or 30-40°C (biogas mesophilic)"),
            
            # ── DFRobot pH Sensor (via ADS1115) ─────────────────────────────
            ("ph", "Digester pH Level", "",
             6.0, 4.0, 8.5, 10.0,
             "DFRobot_pH",
             "Optimal: 6.8-7.5 (biogas). Soil: 6.0-7.0. Water: 6.5-8.5"),
            
            # ── Pressure (0-1.2 MPa → KPa) ──────────────────────────────────
            ("pressure", "Digester Pressure", "kPa",
             80.0, 50.0, 130.0, 150.0,
             "Pressure",
             "Normal range: 80-130 kPa. Monitor for leaks/blockages."),
            
            # ── MQ5 Methane (raw ADS1115 value: 0-32767) ──────────────────
            ("methane_raw", "Methane Sensor (Raw)", "ADC",
             8000.0, 3000.0, 16000.0, 24000.0,
             "MQ5",
             "0-200 ppm: clean | 200-1000 ppm: low | 1000-3000 ppm: moderate | 3000-6000 ppm: high | >6000 ppm: dangerous"),
            
            # ── Gas Flow Rate (L/h) ──────────────────────────────────────────
            ("gas_flow", "Gas Flow Rate", "L/h",
             1.0, 0.5, None, None,
             "Generic",
             "Minimum viable: 1 L/h. Below 0.5: check feed & mixing."),
            
            # ── Ambient Temperature (°C) ────────────────────────────────────
            ("ambient_temperature", "Ambient Temperature", "°C",
             15.0, 10.0, 40.0, 45.0,
             "DS18B20",
             "Affects digester efficiency. Keep digester insulated."),
            
            # ── Ambient Humidity (%) ────────────────────────────────────────
            ("ambient_humidity", "Ambient Humidity", "%",
             30.0, None, 85.0, 95.0,
             "Generic",
             "High humidity: check for condensation. Low: monitor evaporation."),
            
            # ── Nitrogen (ppm / % estimate) ─────────────────────────────────
            ("nitrogen_concentration", "Nitrogen Level", "mg/kg",
             1500.0, 1000.0, 3000.0, 3500.0,
             "Nutrient",
             "Low (<1500): underfed. High (>3000): overfeeding. Optimal: 1500-3000"),
            
            # ── Phosphorus (ppm / % estimate) ───────────────────────────────
            ("phosphorus_concentration", "Phosphorus Level", "mg/kg",
             420.0, 300.0, 800.0, 950.0,
             "Nutrient",
             "Low (<420): add phosphate rock. High: balanced mix needed."),
            
            # ── Potassium (ppm / % estimate from user reference) ────────────
            ("potassium_concentration", "Potassium Level", "mg/kg",
             50.0, None, 100.0, 150.0,
             "Nutrient",
             "Optimal: 50-100 mg/kg (medium). >100 = high. <50 = low."),
            
            # ── Methane % (for analyzer modules) ────────────────────────────
            ("methane_concentration", "Methane Gas %", "%",
             40.0, 30.0, 75.0, 85.0,
             "MQ5",
             "Typical biogas: 50-70% CH4. <40%: check pretreatment."),
    ]

    if existing == 0:
        for (param, label, unit, lw, lc, hw, hc, stype, interp) in defaults:
            conn.execute("""INSERT OR IGNORE INTO thresholds
                (param, label, unit, low_warn, low_crit, high_warn, high_crit, 
                 sensor_type, interpretation, enabled, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,1,?)""",
                (param, label, unit, lw, lc, hw, hc, stype, interp, datetime.now().isoformat()))
        conn.commit()
    else:
        # Backward-compatible migration:
        # Update ONLY if values still match previous seeded defaults,
        # so we don't override any custom thresholds the admin/farmer set.
        now_iso = datetime.now().isoformat()
        # Detect which columns exist in your current DB (older DBs may not have sensor_type).
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(thresholds)").fetchall()}

        # Ensure required sensor threshold rows exist (some older DBs may not have them).
        existing_params = {r["param"] for r in conn.execute("SELECT param FROM thresholds").fetchall()}
        required_params = {"temperature", "ph", "methane_raw", "pressure"}

        for (param, label, unit, lw, lc, hw, hc, stype, interp) in defaults:
            if param not in required_params:
                continue
            if param in existing_params:
                continue

            insert_cols = []
            insert_vals = []
            if "param" in cols:
                insert_cols.append("param"); insert_vals.append(param)
            if "label" in cols:
                insert_cols.append("label"); insert_vals.append(label)
            if "unit" in cols:
                insert_cols.append("unit"); insert_vals.append(unit)
            if "low_warn" in cols:
                insert_cols.append("low_warn"); insert_vals.append(lw)
            if "low_crit" in cols:
                insert_cols.append("low_crit"); insert_vals.append(lc)
            if "high_warn" in cols:
                insert_cols.append("high_warn"); insert_vals.append(hw)
            if "high_crit" in cols:
                insert_cols.append("high_crit"); insert_vals.append(hc)
            if "sensor_type" in cols:
                insert_cols.append("sensor_type"); insert_vals.append(stype)
            if "interpretation" in cols:
                insert_cols.append("interpretation"); insert_vals.append(interp)
            if "enabled" in cols:
                insert_cols.append("enabled"); insert_vals.append(1)
            if "updated_at" in cols:
                insert_cols.append("updated_at"); insert_vals.append(now_iso)

            if insert_cols:
                placeholders = ",".join(["?"] * len(insert_vals))
                sql = f"INSERT OR IGNORE INTO thresholds ({','.join(insert_cols)}) VALUES ({placeholders})"
                conn.execute(sql, tuple(insert_vals))
                existing_params.add(param)

        def exec_update(param: str, updates: dict, where_sql: str, where_params: tuple):
            # Only update columns that exist in the DB schema.
            updates = {k: v for k, v in updates.items() if k in cols}
            if not updates:
                return

            set_sql = ", ".join(f"{k}=?" for k in updates.keys())
            params = list(updates.values())

            if "updated_at" in cols:
                set_sql = set_sql + ", updated_at=?"
                params.append(now_iso)

            sql = f"UPDATE thresholds SET {set_sql} WHERE {where_sql}"
            conn.execute(sql, tuple(params + list(where_params)))

        # temperature: old (15/10, 40/43) -> new (20/10, 30/40)
        exec_update(
            "temperature",
            updates={
                "low_warn": 20.0,
                "low_crit": 10.0,
                "high_warn": 30.0,
                "high_crit": 40.0,
                "label": "Digester Temperature",
                "unit": "°C",
                "sensor_type": "DS18B20",
                "interpretation": "Optimal: 20-30°C (soil/water) or 30-40°C (biogas mesophilic)",
            },
            where_sql="param='temperature' AND low_warn=? AND low_crit=? AND high_warn=? AND high_crit=?",
            where_params=(15.0, 10.0, 40.0, 43.0),
        )

        # ph: old (6/5, 8.5/9) -> new (6/4, 8.5/10)
        exec_update(
            "ph",
            updates={
                "low_warn": 6.0,
                "low_crit": 4.0,
                "high_warn": 8.5,
                "high_crit": 10.0,
                "label": "Digester pH Level",
                "unit": "",
                "sensor_type": "DFRobot_pH",
                "interpretation": "Optimal: 6.8-7.5 (biogas). Soil: 6.0-7.0. Water: 6.5-8.5",
            },
            where_sql="param='ph' AND low_warn=? AND low_crit=? AND high_warn=? AND high_crit=?",
            where_params=(6.0, 5.0, 8.5, 9.0),
        )

        # methane_raw: old (3000/NULL, 24000/32000) -> new (8000/3000, 16000/24000)
        exec_update(
            "methane_raw",
            updates={
                "low_warn": 8000.0,
                "low_crit": 3000.0,
                "high_warn": 16000.0,
                "high_crit": 24000.0,
                "label": "Methane Sensor (Raw)",
                "unit": "ADC",
                "sensor_type": "MQ5",
                "interpretation": "0-200 ppm: clean | 200-1000 ppm: low | 1000-3000 ppm: moderate | 3000-6000 ppm: high | >6000 ppm: dangerous",
            },
            where_sql="param='methane_raw' AND low_warn=? AND low_crit IS NULL AND high_warn=? AND high_crit=?",
            where_params=(3000.0, 24000.0, 32000.0),
        )

        # Apply your required sensor threshold spec unconditionally
        # (overrides any older default values so your alerts match exactly).
        exec_update(
            "temperature",
            updates={
                "low_warn": 20.0,
                "low_crit": 10.0,
                "high_warn": 30.0,
                "high_crit": 40.0,
                "label": "Digester Temperature",
                "unit": "°C",
                "sensor_type": "DS18B20",
                "interpretation": "Optimal: 20-30°C (soil/water) or 30-40°C (biogas mesophilic)",
            },
            where_sql="param=?",
            where_params=("temperature",),
        )

        exec_update(
            "ph",
            updates={
                "low_warn": 6.0,
                "low_crit": 4.0,
                "high_warn": 8.5,
                "high_crit": 10.0,
                "label": "Digester pH Level",
                "unit": "",
                "sensor_type": "DFRobot_pH",
                "interpretation": "Optimal: 6.8-7.5 (biogas). Soil: 6.0-7.0. Water: 6.5-8.5",
            },
            where_sql="param=?",
            where_params=("ph",),
        )

        exec_update(
            "methane_raw",
            updates={
                "low_warn": 8000.0,
                "low_crit": 3000.0,
                "high_warn": 16000.0,
                "high_crit": 24000.0,
                "label": "Methane Sensor (Raw)",
                "unit": "ADC",
                "sensor_type": "MQ5",
                "interpretation": "0-200 ppm: clean | 200-1000 ppm: low | 1000-3000 ppm: moderate | 3000-6000 ppm: high | >6000 ppm: dangerous",
            },
            where_sql="param=?",
            where_params=("methane_raw",),
        )

        exec_update(
            "pressure",
            updates={
                "low_warn": 80.0,
                "low_crit": 50.0,
                "high_warn": 130.0,
                "high_crit": 150.0,
                "label": "Digester Pressure",
                "unit": "kPa",
                "sensor_type": "Pressure",
                "interpretation": "Normal range: 80-130 kPa. Monitor for leaks/blockages.",
            },
            where_sql="param=?",
            where_params=("pressure",),
        )

        conn.commit()
    conn.close()

seed_thresholds()

def get_thresholds() -> dict:
    """Return {param: threshold_row_dict} for all enabled thresholds."""
    conn = get_db()
    rows = conn.execute("SELECT * FROM thresholds WHERE enabled=1").fetchall()
    conn.close()
    return {r["param"]: dict(r) for r in rows}

# ══════════════════════════════════════════════════════════════════════════════
# ── AUTH HELPERS ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
SESSION_HOURS = 24  # session expires after 24 hours

def create_session(farmer_id: int) -> str:
    token     = secrets.token_urlsafe(32)
    expires   = (datetime.now() + timedelta(hours=SESSION_HOURS)).isoformat()
    conn      = get_db()
    conn.execute("DELETE FROM sessions WHERE farmer_id=?", (farmer_id,))  # one session per farmer
    conn.execute("INSERT INTO sessions (token,farmer_id,expires_at) VALUES (?,?,?)",
                 (token, farmer_id, expires))
    conn.commit(); conn.close()
    return token

def get_farmer_from_token(token: str):
    """Returns farmer row or None if token invalid/expired."""
    if not token: return None
    conn = get_db()
    row  = conn.execute("""
        SELECT f.* FROM farmers f
        JOIN sessions s ON s.farmer_id = f.id
        WHERE s.token=? AND s.expires_at > ?
    """, (token, datetime.now().isoformat())).fetchone()
    conn.close()
    return dict(row) if row else None

def require_login(f):
    """Decorator: returns 401 if request has no valid session token."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        token  = request.headers.get("X-Session-Token") or request.cookies.get("session_token")
        farmer = get_farmer_from_token(token)
        if not farmer:
            return jsonify({"error": "Not logged in", "code": "AUTH_REQUIRED"}), 401
        request.farmer = farmer
        return f(*args, **kwargs)
    return decorated

# ══════════════════════════════════════════════════════════════════════════════
# ── ANALYTICS ─────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def compute_analytics(daily_biogas: float) -> dict:
    w = daily_biogas * 7
    return {
        "daily_biogas_m3":      round(daily_biogas, 3),
        "weekly_biogas_m3":     round(w, 3),
        "energy_kwh":           round(w * 1.6,  2),
        "cooking_hours":        round(w * 2.8,  1),
        "urea_equivalent_kg":   round(w * 0.9,  2),
        "co2_reduction_kg":     round(w * 2.5,  2),
        "fertilizer_saved_inr": round(w * 0.9 * 22,              2),
        "energy_saved_inr":     round(w * 1.6 * 8.5,             2),
        "money_saved_inr":      round(w * 0.9 * 22 + w * 1.6 * 8.5, 2),
    }

def classify_nitrogen(n: float) -> str:
    return "High" if n >= 3000 else "Medium" if n >= 1500 else "Low"

def classify_compost(daily: float, ph: float) -> tuple:
    score = min(100, daily * 50 + (5 if 6.5 <= ph <= 7.5 else 0))
    label = ("Premium" if score >= 80 else "Good" if score >= 60
             else "Average" if score >= 40 else "Poor")
    return label, round(score, 1)

def build_feature_df(data: dict) -> pd.DataFrame:
    row = {
        "waste_quantity":               float(data.get("waste_quantity",    50)),
        "cn_ratio":                     float(data.get("cn_ratio",          25)),
        "moisture_level":               float(data.get("moisture_level",    70)),
        "temperature":                  float(data.get("temperature",       35)),
        "ph":                           float(data.get("ph",               7.0)),
        "retention_time":               float(data.get("retention_time",    25)),
        "gas_flow_rate":                float(data.get("gas_flow_rate",    3.5)),
        "methane_concentration":        float(data.get("methane_concentration", 60)),
        "ambient_temperature":          float(data.get("ambient_temperature",  28)),
        "ambient_humidity":             float(data.get("ambient_humidity",     70)),
        "nitrogen_concentration":       float(data.get("nitrogen_concentration",  2000)),
        "phosphorus_concentration":     float(data.get("phosphorus_concentration", 500)),
        "potassium_concentration":      float(data.get("potassium_concentration", 1200)),
        "microbial_activity":           float(data.get("microbial_activity", 1500000)),
        "soil_n_requirement":           float(data.get("soil_n_requirement",  30)),
        "manure_equivalent_n":          float(data.get("manure_equivalent_n", 20)),
        "external_fertilizer_required": float(data.get("external_fertilizer_required", 10)),
        "waste_collection_cost":        float(data.get("waste_collection_cost",    60)),
        "digester_operating_cost":      float(data.get("digester_operating_cost",   40)),
    }
    df = pd.DataFrame([row])
    waste    = data.get("waste_type",    "cow_dung")
    pretreat = data.get("pre_treatment", "raw")
    crop     = data.get("crop_type",     "chickpea")
    for v in WASTE_OHE:    df[f"waste_type_{v}"]    = int(waste == v)
    for v in PRETREAT_OHE: df[f"pre_treatment_{v}"] = int(pretreat == v)
    for v in CROP_OHE:     df[f"crop_type_{v}"]     = int(crop == v)
    if FEATURE_COLUMNS:
        df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return df

# Government-reference baseline data used for crop-cycle recommendations
# Sources:
# - India Meteorological Department (IMD) climatological normals and state trends
# - Soil Health Card Scheme (Govt. of India) nutrient interpretation guidance
GOVT_SOURCE_META = {
    "temperature_source": "India Meteorological Department (IMD) - climatological normals and state seasonal trends",
    "soil_source": "Soil Health Card Scheme, Department of Agriculture & Farmers Welfare, Govt. of India",
}

IMD_STATE_TEMP_BASELINES = {
    "karnataka": {"kharif": [24, 31], "rabi": [18, 30], "zaid": [27, 36]},
    "tamil nadu": {"kharif": [25, 33], "rabi": [20, 31], "zaid": [28, 37]},
    "maharashtra": {"kharif": [24, 32], "rabi": [16, 29], "zaid": [28, 38]},
    "telangana": {"kharif": [25, 34], "rabi": [18, 31], "zaid": [29, 39]},
    "andhra pradesh": {"kharif": [25, 34], "rabi": [20, 32], "zaid": [29, 38]},
}

SOIL_PROFILE_BASELINES = {
    "red_loam": {"n": 1800, "p": 450, "k": 900},
    "black_cotton": {"n": 2000, "p": 500, "k": 1200},
    "alluvial": {"n": 2200, "p": 550, "k": 1000},
    "laterite": {"n": 1600, "p": 380, "k": 800},
    "sandy_loam": {"n": 1500, "p": 420, "k": 750},
}

CROP_CYCLE_CATALOG = {
    "pulses":    {"duration_days": [95, 120], "ideal_npk": [2200, 550, 1000], "ideal_temp_c": [20, 30], "season": "rabi"},
    "soybean":   {"duration_days": [95, 115], "ideal_npk": [2500, 600, 1200], "ideal_temp_c": [22, 32], "season": "kharif"},
    "groundnut": {"duration_days": [105, 135], "ideal_npk": [2300, 520, 1250], "ideal_temp_c": [22, 33], "season": "kharif"},
    "chickpea":  {"duration_days": [95, 125], "ideal_npk": [2100, 500, 900], "ideal_temp_c": [18, 30], "season": "rabi"},
    "maize":     {"duration_days": [90, 120], "ideal_npk": [2600, 650, 1300], "ideal_temp_c": [21, 34], "season": "kharif"},
    "paddy":     {"duration_days": [110, 150], "ideal_npk": [2400, 620, 1100], "ideal_temp_c": [22, 35], "season": "kharif"},
}

DEFAULT_USER_CROPS = ["pulses", "groundnut", "soybean"]

def _latest_sensor_and_npk():
    """Return latest live sensor and NPK values from sensor_readings."""
    conn = get_db()
    sensor = conn.execute("""SELECT temperature, ph, pressure, gas_flow,
        nitrogen_concentration, phosphorus_concentration, potassium_concentration,
        timestamp
        FROM sensor_readings ORDER BY id DESC LIMIT 1""").fetchone()
    conn.close()

    sensor_d = dict(sensor) if sensor else {}
    def _f(val, default):
        try:
            return default if val is None else float(val)
        except Exception:
            return default

    n_est = _f(sensor_d.get("nitrogen_concentration"), 2000.0)
    p_est = _f(sensor_d.get("phosphorus_concentration"), 500.0)
    k_est = _f(sensor_d.get("potassium_concentration"), 1200.0)

    return {
        "temperature": float(sensor_d.get("temperature", 30.0)),
        "ph": float(sensor_d.get("ph", 7.0)),
        "pressure": float(sensor_d.get("pressure", 100.0)),
        "gas_flow": float(sensor_d.get("gas_flow", 3.0)),
        "n": float(n_est),
        "p": float(p_est),
        "k": float(k_est),
        "timestamp": sensor_d.get("timestamp", datetime.now().isoformat()),
    }

def _score_component(actual, lo, hi):
    if lo <= actual <= hi:
        return 1.0
    dist = min(abs(actual - lo), abs(actual - hi))
    span = max(1.0, hi - lo)
    return max(0.0, 1.0 - (dist / (span * 1.8)))

def build_crop_cycle_predictions(crops: list, state: str, soil_type: str, start_date: datetime = None):
    """
    Build crop-cycle predictions using:
    - Live NPK/sensor values
    - IMD state seasonal baseline temperatures
    - Soil Health Card aligned soil nutrient baselines
    """
    state_key = (state or "karnataka").strip().lower()
    soil_key = (soil_type or "black_cotton").strip().lower()
    if state_key not in IMD_STATE_TEMP_BASELINES:
        state_key = "karnataka"
    if soil_key not in SOIL_PROFILE_BASELINES:
        soil_key = "black_cotton"

    start_dt = start_date or datetime.now()
    live = _latest_sensor_and_npk()
    soil_base = SOIL_PROFILE_BASELINES[soil_key]
    out = []

    for crop in crops:
        crop_key = crop.strip().lower()
        if crop_key not in CROP_CYCLE_CATALOG:
            continue
        meta = CROP_CYCLE_CATALOG[crop_key]
        season = meta["season"]
        season_temp = IMD_STATE_TEMP_BASELINES[state_key][season]
        ideal_n, ideal_p, ideal_k = meta["ideal_npk"]
        lo_t, hi_t = meta["ideal_temp_c"]

        # Blend government baseline + live sensor values
        n_blend = round((live["n"] * 0.7) + (soil_base["n"] * 0.3), 1)
        p_blend = round((live["p"] * 0.7) + (soil_base["p"] * 0.3), 1)
        k_blend = round((live["k"] * 0.7) + (soil_base["k"] * 0.3), 1)

        n_score = _score_component(n_blend, ideal_n * 0.8, ideal_n * 1.2)
        p_score = _score_component(p_blend, ideal_p * 0.8, ideal_p * 1.2)
        k_score = _score_component(k_blend, ideal_k * 0.8, ideal_k * 1.2)
        t_score = _score_component(live["temperature"], lo_t, hi_t)
        s_score = _score_component(live["temperature"], season_temp[0], season_temp[1])
        fit = round(((n_score + p_score + k_score + t_score + s_score) / 5.0) * 100, 1)

        dur_lo, dur_hi = meta["duration_days"]
        # Better fit shortens uncertainty and slightly compresses cycle estimate
        cycle_days = int(round(((dur_lo + dur_hi) / 2.0) - ((fit - 50) / 100.0) * 6))
        cycle_days = max(dur_lo - 5, min(dur_hi + 5, cycle_days))
        harvest = (start_dt + timedelta(days=cycle_days)).strftime("%d %b %Y")

        out.append({
            "crop": crop_key,
            "start_date": start_dt.strftime("%Y-%m-%d"),
            "cycle_days": cycle_days,
            "duration_range_days": [dur_lo, dur_hi],
            "expected_harvest_date": harvest,
            "fit_score": fit,
            "season": season,
            "npk_available_mgkg": {"n": n_blend, "p": p_blend, "k": k_blend},
            "npk_required_mgkg": {"n": ideal_n, "p": ideal_p, "k": ideal_k},
            "recommendation": (
                "Good to proceed" if fit >= 75 else
                "Proceed with nutrient correction" if fit >= 55 else
                "Delay planting and improve soil/temperature fit"
            ),
            "inputs_used": {
                "live_temperature_c": live["temperature"],
                "npk_estimate_mgkg": {"n": n_blend, "p": p_blend, "k": k_blend},
                "state_temp_baseline_c": season_temp,
                "soil_profile": soil_key,
            }
        })

    return {
        "predictions": out,
        "live_context": live,
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "government_sources": GOVT_SOURCE_META,
    }

# ══════════════════════════════════════════════════════════════════════════════
# ── ALERT RULES — dynamic, driven by threshold table ─────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# ── SENSOR-SPECIFIC ADVISORY SYSTEM ───────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def get_sensor_advisory(alert: dict) -> str:
    """
    Generate sensor-specific advisory/response messages based on alert details.
    Returns actionable guidance for farmers.
    """
    param = alert.get("param", "")
    value = alert.get("value", 0)
    severity = alert.get("severity", "")
    direction = alert.get("direction", "")
    sensor_type = alert.get("sensor_type", "")
    
    advisories = {
        # ── TEMPERATURE ADVISORIES (DS18B20) ──────────────────────────────
        "temperature": {
            "critical_low": (
                "🔴 CRITICAL: Temperature too low (<10°C)\n"
                "IMMEDIATE ACTIONS:\n"
                "  1. Check insulation around digester\n"
                "  2. Verify heater is functioning (if installed)\n"
                "  3. Increase feed frequency to generate more heat\n"
                "  4. Consider adding preheated water/slurry\n"
                "⏱️  Action needed: Next 4-6 hours\n"
                "❓ Impact: Microbial activity drops dramatically below 10°C"
            ),
            "warning_low": (
                "⚠️ WARNING: Temperature below optimal (10-20°C)\n"
                "RECOMMENDED ACTIONS:\n"
                "  1. Inspect digester insulation\n"
                "  2. Increase ambient/preheating if winter\n"
                "  3. Add C:N rich feed to boost decomposition heat\n"
                "  4. Check ventilation (excess air loss lowers temp)\n"
                "⏱️  Action needed: Within 12-24 hours\n"
                "ℹ️  Optimal for soil/water: 15-25°C. For biogas: 30-40°C"
            ),
            "warning_high": (
                "⚠️ WARNING: Temperature elevated (30-40°C)\n"
                "RECOMMENDED ACTIONS:\n"
                "  1. Improve ventilation to cool digester\n"
                "  2. Check for excessive sunlight exposure\n"
                "  3. Reduce feed frequency temporarily\n"
                "  4. Monitor for accelerated gas production\n"
                "⏱️  Action needed: Monitor over 24-48 hours\n"
                "ℹ️  This may increase gas yield if controlled"
            ),
            "critical_high": (
                "🔴 CRITICAL: Temperature critical (>40°C)\n"
                "IMMEDIATE ACTIONS:\n"
                "  1. Stop adding fresh feed immediately\n"
                "  2. Increase ventilation/cooling NOW\n"
                "  3. Spray water on digester exterior if needed\n"
                "  4. Check for fire hazards near digester\n"
                "⏱️  Action needed: IMMEDIATE (within 1-2 hours)\n"
                "❌ Risk: Thermophilic conditions can kill microbes if temp oscillates"
            ),
        },
        
        # ── pH ADVISORIES (DFRobot) ──────────────────────────────────────
        "ph": {
            "critical_low": (
                "🔴 CRITICAL: pH too acidic (<4.0)\n"
                "IMMEDIATE ACTIONS:\n"
                "  1. Add alkalizing agent:\n"
                "     • Calcium carbonate (CaCO3): 5-10 kg/1000L\n"
                "     • Lime slurry: 2-5 kg/1000L\n"
                "     • Wood ash: 3-8 kg/1000L\n"
                "  2. Reduce feed rate by 50% until recovery\n"
                "  3. Increase mixing frequency\n"
                "  4. Monitor pH daily until >6.5\n"
                "⏱️  Action needed: IMMEDIATE (within 2-4 hours)\n"
                "⚡ Cause: Excessive VFA (volatile fatty acids). Check overfeeding."
            ),
            "warning_low": (
                "⚠️ WARNING: pH slightly acidic (4.0-6.0)\n"
                "RECOMMENDED ACTIONS:\n"
                "  1. Add small dose of calcium carbonate (2-3 kg/1000L)\n"
                "  2. Reduce feed frequency slightly\n"
                "  3. Check C:N ratio in feedstock\n"
                "  4. Re-measure pH in 6-12 hours\n"
                "⏱️  Action needed: Within 12 hours\n"
                "ℹ️  Trend: If pH is dropping, acid accumulation is occurring"
            ),
            "warning_high": (
                "⚠️ WARNING: pH slightly alkaline (8.5-10.0)\n"
                "RECOMMENDED ACTIONS:\n"
                "  1. Add small amount of acidifying feedstock:\n"
                "     • Vinegar: 0.5-1% by volume\n"
                "     • Whey/dairy waste: 5-10% addition\n"
                "  2. Increase feed rate gradually\n"
                "  3. Check for ammonia odors (N2 loss)\n"
                "  4. Monitor in 12-24 hours\n"
                "⏱️  Action needed: Within 24 hours\n"
                "⚠️  Ammonia loss occurs at high pH + high temp"
            ),
            "critical_high": (
                "🔴 CRITICAL: pH strongly alkaline (>10.0)\n"
                "IMMEDIATE ACTIONS:\n"
                "  1. Add acidifying material urgently:\n"
                "     • Vinegar/uric acid solution: 2-5% by volume\n"
                "     • Whey: 15-20% replacement feed\n"
                "  2. STOP adding any alkaline material\n"
                "  3. Reduce overall feed rate temporarily\n"
                "  4. Monitor PT daily until pH <8.5\n"
                "⏱️  Action needed: IMMEDIATE (within 1-2 hours)\n"
                "❌ Risk: Ammonia toxicity, process failure if not corrected"
            ),
        },
        
        # ── PRESSURE ADVISORIES ──────────────────────────────────────────
        "pressure": {
            "critical_low": (
                "🔴 CRITICAL: Pressure critically low (<50 kPa)\n"
                "IMMEDIATE ACTIONS:\n"
                "  1. Check for leaks in digester seals/valves\n"
                "  2. Inspect all connections (tighten if loose)\n"
                "  3. Test for gas leakage with soapy water\n"
                "  4. If no leaks found: may indicate blockage in outlet\n"
                "⏱️  Action needed: IMMEDIATE investigation\n"
                "⚠️  Loss of pressure = loss of gas containment"
            ),
            "warning_low": (
                "⚠️ WARNING: Pressure low (50-80 kPa)\n"
                "RECOMMENDED ACTIONS:\n"
                "  1. Inspect all seals for minor leaks\n"
                "  2. Check pressure gauge calibration\n"
                "  3. Increase feed slightly to boost gas production\n"
                "  4. Verify gas outlet is not blocked\n"
                "⏱️  Action needed: Within 12-24 hours\n"
                "ℹ️  May indicate reduced gas production (low temp? Low feed?)"
            ),
            "warning_high": (
                "⚠️ WARNING: Pressure elevated (130-150 kPa)\n"
                "RECOMMENDED ACTIONS:\n"
                "  1. Open pressure relief valve (if installed)\n"
                "  2. Increase gas utilization (more stove use)\n"
                "  3. Check for blocked outlet\n"
                "  4. Monitor for overpressure (>150 kPa)\n"
                "⏱️  Action needed: Within 24 hours\n"
                "ℹ️  Extra gas = extra energy. Use it, or vent safely."
            ),
            "critical_high": (
                "🔴 CRITICAL: Pressure critical (>150 kPa)\n"
                "IMMEDIATE ACTIONS:\n"
                "  1. RELEASE PRESSURE IMMEDIATELY via relief valve\n"
                "  2. Do NOT ignite gas until pressure normalized\n"
                "  3. Increase gas utilization rate\n"
                "  4. Check for blockages in piping\n"
                "  5. Verify digester integrity (no cracks)\n"
                "⏱️  Action needed: IMMEDIATE (within 15-30 minutes)\n"
                "🚨 Risk: Digester rupture, explosion hazard"
            ),
        },
        
        # ── METHANE/MQ5 ADVISORIES (ADS1115 raw value 0-32767) ─────────
        "methane_raw": {
            "critical_low": (
                "🔴 CRITICAL: Methane extremely low (<200 ppm)\n"
                "IMMEDIATE ACTIONS:\n"
                "  1. Check digester gas production (feed rate, mixing)\n"
                "  2. Verify methane sensor burn-in (24-48h)\n"
                "  3. Inspect MQ5 wiring and sensor saturation/blockage\n"
                "  4. Check for leaks or abnormal digester process\n"
                "⏱️  Action needed: Within 24-48 hours\n"
                "⚠️ Low methane can reduce cooking/energy output"
            ),
            "warning_low": (
                "⚠️ WARNING: Methane low (<1000 ppm)\n"
                "RECOMMENDED ACTIONS:\n"
                "  1. Check MQ5 sensor burn-in (needs 24-48h)\n"
                "  2. Verify sensor is not saturated or blocked\n"
                "  3. Check for sensor wiring/connection issues\n"
                "  4. Ensure digester is producing gas\n"
                "⏱️  Action needed: Within 24-48 hours\n"
                "ℹ️  May be normal if digester was just started"
            ),
            "warning_high": (
                "⚠️ WARNING: Methane elevated (3000–6000 ppm, ventilate if needed)\n"
                "RECOMMENDED ACTIONS:\n"
                "  1. Ensure adequate ventilation\n"
                "  2. Avoid naked flames/sparks in digester area\n"
                "  3. Use gas safely or flare excess\n"
                "  4. Monitor for leaks (gas should be contained)\n"
                "⏱️  Action needed: Monitor over 24 hours\n"
                "⚡ Good production! Use the gas for cooking/electricity."
            ),
            "critical_high": (
                "🔴 CRITICAL: Methane dangerous (>6000 ppm)\n"
                "IMMEDIATE ACTIONS:\n"
                "  1. EVACUATE digester area immediately\n"
                "  2. Stop all ignition sources\n"
                "  3. Increase ventilation drastically\n"
                "  4. Check for major gas leak/blockage\n"
                "  5. Do NOT enter digester area until below 1000 ppm\n"
                "⏱️  Action needed: IMMEDIATE evacuation\n"
                "🚨 Risk: Methane is explosive 5-15% in air. LEL zone detected."
            ),
        },
        
        # ── GAS FLOW ADVISORIES ──────────────────────────────────────────
        "gas_flow": {
            "critical_low": (
                "🔴 CRITICAL: Gas flow critically low (<0.5 L/h)\n"
                "IMMEDIATE ACTIONS:\n"
                "  1. Check digester temperature (should be >15°C)\n"
                "  2. Verify feedstock is being added\n"
                "  3. Inspect outlet for blockage/ice\n"
                "  4. Check water level in digester\n"
                "  5. Trouble: May indicate digester failure\n"
                "⏱️  Action needed: IMMEDIATE investigation\n"
                "⚠️  Minimum viable flow: 1 L/h for energy utility"
            ),
            "warning_low": (
                "⚠️ WARNING: Gas flow low (0.5-1.5 L/h)\n"
                "RECOMMENDED ACTIONS:\n"
                "  1. Check temperature (consider heating)\n"
                "  2. Increase feed rate gradually\n"
                "  3. Check mixing/agitation system\n"
                "  4. Verify pH is optimal (6.8-7.5)\n"
                "  5. Reduce system leakage if possible\n"
                "⏱️  Action needed: Within 24-48 hours\n"
                "ℹ️  Low flow = low energy output. May need process tuning."
            ),
        },
        
        # ── NITROGEN ADVISORIES ──────────────────────────────────────────
        "nitrogen_concentration": {
            "critical_low": (
                "🔴 CRITICAL: Nitrogen critically low (<1000 mg/kg)\n"
                "IMMEDIATE ACTIONS:\n"
                "  1. Add nitrogen-rich feedstock:\n"
                "     • Poultry litter: 10-20% by weight\n"
                "     • Fresh manure: 30-40%\n"
                "     • Urea: 0.5-1 kg/1000L carefully\n"
                "  2. Reduce C:N ratio temporarily\n"
                "  3. Monitor gas production recovery\n"
                "⏱️  Action needed: Within 12-24 hours\n"
                "⚠️  Nitrogen starvation → microbial shutdown"
            ),
            "warning_low": (
                "⚠️ WARNING: Nitrogen low (1000-1500 mg/kg)\n"
                "RECOMMENDED ACTIONS:\n"
                "  1. Add nitrogen source gradually:\n"
                "     • Manure: 20-30% of feed mix\n"
                "     • Crop residue: reduce proportion\n"
                "  2. Check C:N ratio (should be 20-30:1)\n"
                "  3. Monitor pressure & gas yield\n"
                "⏱️  Action needed: Within 12 hours\n"
                "ℹ️  May improve if mixing/pretreatment changed"
            ),
            "warning_high": (
                "⚠️ WARNING: Nitrogen high (3000+ mg/kg)\n"
                "RECOMMENDED ACTIONS:\n"
                "  1. Reduce manure/poultry litter in feedstock\n"
                "  2. Increase C:N ratio (add more crop residue)\n"
                "  3. Watch for ammonia odors\n"
                "  4. Monitor pH (high N + high pH → ammonia loss)\n"
                "  5. Increase dilution/water ratio\n"
                "⏱️  Action needed: Over next 48-72 hours\n"
                "⚠️  High N with high temp = ammonia volatilization"
            ),
        },
        
        # ── PHOSPHORUS ADVISORIES ────────────────────────────────────────
        "phosphorus_concentration": {
            "critical_low": (
                "🔴 CRITICAL: Phosphorus critically low (<300 mg/kg)\n"
                "IMMEDIATE ACTIONS:\n"
                "  1. Add phosphorus source:\n"
                "     • Phosphate rock: 2-5 kg/1000L\n"
                "     • Bone meal: 3-6 kg/1000L\n"
                "     • Poultry litter: 15-25%\n"
                "  2. Verify feedstock sources\n"
                "⏱️  Action needed: Within 12-24 hours\n"
                "⚠️  P deficiency → poor cell growth"
            ),
            "warning_low": (
                "⚠️ WARNING: Phosphorus low (300-420 mg/kg)\n"
                "RECOMMENDED ACTIONS:\n"
                "  1. Add phosphate-rich feed gradually\n"
                "  2. Consider phosphate rock amendment (1-2 kg/1000L)\n"
                "  3. Monitor compost output quality\n"
                "⏱️  Action needed: Within 24-48 hours"
            ),
        },
        
        # ── POTASSIUM ADVISORIES ─────────────────────────────────────────
        "potassium_concentration": {
            "critical_low": (
                "⚠️ WARNING: Potassium low (<50 mg/kg)\n"
                "RECOMMENDED ACTIONS:\n"
                "  1. Add potassium source:\n"
                "     • K-rich crop residue: increase proportion\n"
                "     • Wood ash: 2-4 kg/1000L (carefully, may raise pH)\n"
                "     • Plant waste: 10-20%\n"
                "  2. Monitor pH if adding ash\n"
                "⏱️  Action needed: Within 24-48 hours\n"
                "ℹ️  Potassium important for soil conditioning"
            ),
            "warning_high": (
                "⚠️ WARNING: Potassium high (>100 mg/kg)\n"
                "RECOMMENDED ACTIONS:\n"
                "  1. Reduce K-rich inputs (wood ash, K-fertilizer)\n"
                "  2. Balance with crop residue low in K\n"
                "  3. Monitor compost application rates\n"
                "⏱️  Action needed: Over next 7 days"
            ),
        },
    }
    
    # Get base advisory for this parameter
    param_advisories = advisories.get(param, {})
    
    # Construct key: "<severity>_<direction>"
    if severity and direction:
        advisory_key = f"{severity}_{direction}"
        return param_advisories.get(advisory_key, 
            f"Contact technical support for {param} at {severity} level.")
    
    return f"{param} requires attention. Check sensor and system status."


def check_alerts(sensor: dict, ts: str = None) -> list:
    """
    Compare sensor reading against DB thresholds.
    Returns list of alert dicts with full details for notification.
    Each alert contains: param, label, unit, value, low/high limits, severity, message, 
                        sensor_type, interpretation, advisory, timestamp.
    """
    ts      = ts or datetime.now().isoformat()
    thrs    = get_thresholds()
    alerts  = []

    for param, thr in thrs.items():
        val = sensor.get(param)
        if val is None: continue

        low_crit  = thr.get("low_crit")
        low_warn  = thr.get("low_warn")
        high_warn = thr.get("high_warn")
        high_crit = thr.get("high_crit")
        label     = thr["label"]
        unit      = thr["unit"]
        sensor_type = thr.get("sensor_type", "Generic")
        interpretation = thr.get("interpretation", "")

        # Display conversion: MQTT/UI now wants methane in ppm-only (no raw ADC anywhere).
        # Severity/direction are still computed using raw thresholds from DB.
        display_label = label
        display_unit  = unit
        display_val   = val
        if param == "methane_raw":
            display_label = "Methane (MQ5)"
            display_unit  = "ppm"
            display_val   = mq5_raw_to_ppm(val)

        severity = None
        direction = None

        if low_crit  is not None and val < low_crit:
            severity  = "critical"; direction = "low"
        elif low_warn is not None and val < low_warn:
            severity  = "warning";  direction = "low"
        elif high_crit is not None and val > high_crit:
            severity  = "critical"; direction = "high"
        elif high_warn is not None and val > high_warn:
            severity  = "warning";  direction = "high"

        if severity:
            if direction == "low":
                limit_used = low_crit if severity == "critical" else low_warn
                display_limit_used = limit_used
                if param == "methane_raw":
                    display_limit_used = mq5_raw_to_ppm(limit_used)
                msg = (f"{display_label} is too low: {display_val} {display_unit} "
                       f"(critical limit: ≥ {display_limit_used} {display_unit})")
            else:
                limit_used = high_crit if severity == "critical" else high_warn
                display_limit_used = limit_used
                if param == "methane_raw":
                    display_limit_used = mq5_raw_to_ppm(limit_used)
                msg = (f"{display_label} is too high: {display_val} {display_unit} "
                       f"(critical limit: ≤ {display_limit_used} {display_unit})")

            # Convert thresholds for display when methane is ppm-only.
            disp_low_warn = low_warn
            disp_low_crit = low_crit
            disp_high_warn = high_warn
            disp_high_crit = high_crit
            if param == "methane_raw":
                disp_low_warn  = mq5_raw_to_ppm(low_warn)  if low_warn  is not None else None
                disp_low_crit  = mq5_raw_to_ppm(low_crit)  if low_crit  is not None else None
                disp_high_warn = mq5_raw_to_ppm(high_warn) if high_warn is not None else None
                disp_high_crit = mq5_raw_to_ppm(high_crit) if high_crit is not None else None

            # Build alert with sensor-specific advisory
            alert_dict = {
                "param":         param,
                "label":         display_label,
                "unit":          display_unit,
                "value":         display_val,
                "direction":     direction,
                "severity":      severity,
                "message":       msg,
                "low_warn":      disp_low_warn,
                "low_crit":      disp_low_crit,
                "high_warn":     disp_high_warn,
                "high_crit":     disp_high_crit,
                "sensor_type":   sensor_type,
                "interpretation": interpretation,
                "timestamp":     ts,
            }
            
            # Add sensor-specific advisory
            alert_dict["advisory"] = get_sensor_advisory(alert_dict)
            
            alerts.append(alert_dict)
    
    return alerts


def build_alert_notification(alerts: list, farmer_name: str = "") -> tuple:
    """
    Build full email body and SMS string from alert list.
    Includes sensor-specific advisories, thresholds, and actionable recommendations.
    Returns (email_subject, email_body, sms_text).
    """
    ts    = datetime.now().strftime("%d %b %Y, %I:%M %p")
    crit  = [a for a in alerts if a["severity"] == "critical"]
    warn  = [a for a in alerts if a["severity"] == "warning"]

    subject = ("🚨 CRITICAL Digester Alert" if crit
               else "⚠️ Digester Warning") + f" — {ts}"

    lines = [
        f"🌿 Urja Nidhi — Sensor Alert Report",
        f"Farmer: {farmer_name or 'Registered User'}",
        f"Time  : {ts}",
        "═" * 70,
        "",
    ]
    
    # Add summary
    if crit:
        lines.append(f"🚨 {len(crit)} CRITICAL ALERT(S) — IMMEDIATE ACTION REQUIRED")
        lines.append("")
    if warn:
        lines.append(f"⚠️  {len(warn)} WARNING(S) — MONITOR & PLAN ACTION")
        lines.append("")
    
    lines.append("─" * 70)
    lines.append("")
    
    # Detailed alerts with advisories
    for i, a in enumerate(alerts, 1):
        icon = "🚨" if a["severity"] == "critical" else "⚠️"
        lines.append(f"{icon} ALERT #{i}: {a['label']}")
        lines.append(f"   Sensor Type   : {a.get('sensor_type', 'Unknown')}")
        lines.append(f"   Current Value : {a['value']} {a['unit']}")
        lines.append(f"   Status        : {a['severity'].upper()} ({a['direction'].upper()})")
        lines.append(f"   Description   : {a['message']}")
        
        # Add limits
        if a.get('low_crit')   is not None: lines.append(f"   🔴 Critical Low   : {a['low_crit']} {a['unit']}")
        if a.get('low_warn')   is not None: lines.append(f"   🟠 Warning Low    : {a['low_warn']} {a['unit']}")
        if a.get('high_warn')  is not None: lines.append(f"   🟠 Warning High   : {a['high_warn']} {a['unit']}")
        if a.get('high_crit')  is not None: lines.append(f"   🔴 Critical High  : {a['high_crit']} {a['unit']}")
        
        # Add interpretation
        if a.get('interpretation'):
            lines.append(f"   Background    : {a['interpretation']}")
        
        lines.append("")
        lines.append(f"   📋 ACTION PLAN:")
        lines.append("   " + "─" * 65)
        
        # Add sensor-specific advisory (formatted)
        advisory = a.get('advisory', 'No specific advisory available.')
        advisory_lines = advisory.split('\n')
        for adv_line in advisory_lines:
            lines.append(f"   {adv_line}")
        
        lines.append("")
        lines.append("")
    
    lines += [
        "═" * 70,
        "📲 Next Steps:",
        "  1. Review the ACTION PLAN for each alert above",
        "  2. Take immediate action for CRITICAL alerts (within 1-4 hours)",
        "  3. Monitor WARNING alerts over next 12-48 hours",
        "  4. Log your actions in the Urja Nidhi dashboard",
        "  5. Re-measure sensors after taking corrective action",
        "",
        "💡 Dashboard Link: http://localhost:5000",
        "📞 Support: Contact your Urja Nidhi coordinator",
        "",
        "─ Urja Nidhi Smart Digester Monitoring System",
    ]
    
    # Short email body: one line per alert + first advisory line.
    # (User requested shorter alerts in email.)
    short_lines = [
        f"🌿 Urja Nidhi — {subject}",
        f"Time:  {ts}",
        "",
    ]
    for a in alerts:
        adv = a.get("advisory") or ""
        adv_first = adv.splitlines()[0].strip() if adv else ""
        short_lines.append(f"- {a['label']}: {a['value']}{a['unit']} ({a['severity'].upper()})")
        if adv_first:
            short_lines.append(f"  Advice: {adv_first}")
    short_lines.append("")
    short_lines.append("Dashboard: http://localhost:5000")
    email_body = "\n".join(short_lines)

    # Compact SMS (max ~160 chars per alert suggested)
    sms_parts = [f"Urja Nidhi {ts}:"]
    for a in alerts:
        sms_parts.append(
            f"{a['label']}: {a['value']}{a['unit']} ({a['severity'].upper()}) "
            f"limit:{a.get('low_crit') or a.get('low_warn') or a.get('high_warn') or a.get('high_crit')} "
        )
    sms_text = " | ".join(sms_parts[:3])  # Limit to 3 alerts in SMS

    return subject, email_body, sms_text

# ══════════════════════════════════════════════════════════════════════════════
# ── NOTIFICATION DELIVERY ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def _notification_recently_sent(farmer_id: int, subject: str, minutes: int = 10) -> bool:
    """Avoid back-to-back repeated push notifications."""
    if not farmer_id:
        return False
    try:
        conn = get_db()
        row = conn.execute("""
            SELECT timestamp FROM notifications
            WHERE farmer_id=? AND subject=?
            ORDER BY id DESC LIMIT 1
        """, (farmer_id, subject)).fetchone()
        conn.close()
        if not row or not row["timestamp"]:
            return False
        last_ts = datetime.fromisoformat(row["timestamp"])
        return (datetime.now() - last_ts) < timedelta(minutes=minutes)
    except Exception:
        return False

def _already_sent_today(farmer_id: int, subject: str) -> bool:
    """Per-day dedupe by subject category (daily summary / threshold alerts)."""
    if not farmer_id:
        return False
    try:
        day_key = datetime.now().strftime("%Y-%m-%d")
        conn = get_db()
        row = conn.execute("""
            SELECT COUNT(*) AS c
            FROM notifications
            WHERE farmer_id=? AND subject=? AND substr(timestamp,1,10)=?
        """, (farmer_id, subject, day_key)).fetchone()
        conn.close()
        return bool(row and row["c"] > 0)
    except Exception:
        return False

def _log_notification(farmer_id, channel, recipient, subject, message, status):
    try:
        conn = get_db()
        conn.execute("""INSERT INTO notifications
            (farmer_id,channel,recipient,subject,message,status,timestamp)
            VALUES (?,?,?,?,?,?,?)""",
            (farmer_id, channel, recipient, subject, message, status,
             datetime.now().isoformat()))
        conn.commit(); conn.close()
    except Exception: pass

def _send_email(to_address: str, subject: str, body: str,
                farmer_id: int = None) -> dict:
    """Send email from server SMTP account to farmer's address."""
    if not SMTP_CFG["user"] or not to_address:
        _log_notification(farmer_id, "email", to_address, subject, body, "skipped_no_config")
        return {"success": False, "message": "Email not configured"}
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"🌿 Urja Nidhi: {subject}"
        msg["From"]    = f"Urja Nidhi <{SMTP_CFG['user']}>"
        msg["To"]      = to_address
        msg.attach(MIMEText(body, "plain", "utf-8"))
        with smtplib.SMTP(SMTP_CFG["host"], SMTP_CFG["port"]) as srv:
            srv.starttls()
            srv.login(SMTP_CFG["user"], SMTP_CFG["password"])
            srv.send_message(msg)
        _log_notification(farmer_id, "email", to_address, subject, body, "sent")
        return {"success": True, "message": f"Email sent to {to_address}"}
    except Exception as e:
        _log_notification(farmer_id, "email", to_address, subject, body, f"failed:{e}")
        return {"success": False, "message": str(e)}

def _send_sms(to_number: str, message: str, farmer_id: int = None) -> dict:
    """Send SMS via Twilio to farmer's registered number."""
    if not TWILIO_CFG["sid"] or not to_number:
        _log_notification(farmer_id, "sms", to_number, "alert", message, "skipped_no_config")
        return {"success": False, "message": "SMS not configured"}
    try:
        from twilio.rest import Client
        Client(TWILIO_CFG["sid"], TWILIO_CFG["token"]).messages.create(
            body=f"🌿 Urja Nidhi: {message}",
            from_=TWILIO_CFG["from_sms"],
            to=to_number
        )
        _log_notification(farmer_id, "sms", to_number, "alert", message, "sent")
        return {"success": True, "message": f"SMS sent to {to_number}"}
    except Exception as e:
        _log_notification(farmer_id, "sms", to_number, "alert", message, f"failed:{e}")
        return {"success": False, "message": str(e)}

def smtp_env_check() -> dict:
    """
    Check whether SMTP env fields are present (not empty/commented-out result).
    """
    required = {
        "SMTP_HOST": SMTP_CFG.get("host", ""),
        "SMTP_PORT": str(SMTP_CFG.get("port", "") or ""),
        "SMTP_USER": SMTP_CFG.get("user", ""),
        "SMTP_PASSWORD": SMTP_CFG.get("password", ""),
    }
    missing = [k for k, v in required.items() if not str(v).strip()]
    return {
        "ok": len(missing) == 0,
        "missing": missing,
    }

def get_all_farmers() -> list:
    """Return all registered farmers."""
    conn  = get_db()
    rows  = conn.execute("SELECT * FROM farmers").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def dispatch_alerts_to_all_farmers(alerts: list):
    """
    Send immediate alert to every registered farmer.
    Critical → email + SMS.  Warning → email only.
    Uses rich build_alert_notification() for detailed messages.
    """
    if not alerts: return
    farmers = get_all_farmers()
    if not farmers: return

    crit = [a for a in alerts if a["severity"] == "critical"]

    for farmer in farmers:
        fid     = farmer["id"]
        subject, email_body, sms_text = build_alert_notification(alerts, farmer.get("name",""))
        if _notification_recently_sent(fid, subject, minutes=10) or _already_sent_today(fid, subject):
            continue

        if farmer.get("email"):
            threading.Thread(target=_send_email,
                args=(farmer["email"], subject, email_body, fid),
                daemon=True).start()

        if farmer.get("phone") and crit:
            threading.Thread(target=_send_sms,
                args=(farmer["phone"], sms_text, fid),
                daemon=True).start()

def send_daily_summary():
    """
    Sends a daily summary to every registered farmer at 8 AM.
    Includes last 24h avg sensor readings and total biogas/savings.
    """
    farmers = get_all_farmers()
    if not farmers: return
    try:
        conn = get_db()
        # Last 24h sensor averages
        sensors = conn.execute("""
            SELECT AVG(temperature) as t, AVG(ph) as p,
                   AVG(gas_flow) as f, COUNT(*) as n
            FROM sensor_readings
            WHERE timestamp > datetime('now','-1 day')
        """).fetchone()
        # Last 24h predictions
        preds = conn.execute("""
            SELECT SUM(daily_biogas) as total_gas,
                   SUM(money_saved) as total_saved
            FROM prediction_results
            WHERE timestamp > datetime('now','-1 day')
        """).fetchone()
        conn.close()

        if not sensors or sensors["n"] == 0:
            return  # no data to summarise

        body = (
            f"🌿 Urja Nidhi — Daily Summary ({datetime.now().strftime('%d %b %Y')})\n\n"
            f"📊 Last 24 Hours:\n"
            f"  Avg Temperature : {round(sensors['t'] or 0, 1)}°C\n"
            f"  Avg pH          : {round(sensors['p'] or 0, 2)}\n"
            f"  Avg Gas Flow    : {round(sensors['f'] or 0, 2)} L/h\n"
            f"  Sensor Readings : {sensors['n']}\n\n"
            f"💰 Savings & Production:\n"
            f"  Biogas Produced : {round(preds['total_gas'] or 0, 2)} m³\n"
            f"  Money Saved     : ₹{round(preds['total_saved'] or 0, 2)}\n\n"
            f"Login to your dashboard for full details: http://localhost:5000"
        )

        for farmer in farmers:
            fid = farmer["id"]
            if _already_sent_today(fid, "Daily Digester Summary"):
                continue
            if farmer.get("email"):
                threading.Thread(
                    target=_send_email,
                    args=(farmer["email"], "Daily Digester Summary", body, fid),
                    daemon=True
                ).start()
            if farmer.get("phone"):
                short = (f"Urja Nidhi daily: {sensors['n']} readings, "
                         f"avg temp {round(sensors['t'] or 0,1)}°C, "
                         f"pH {round(sensors['p'] or 0,2)}, "
                         f"saved ₹{round(preds['total_saved'] or 0,0)}")
                threading.Thread(
                    target=_send_sms,
                    args=(farmer["phone"], short, fid),
                    daemon=True
                ).start()
        print(f"✅ Daily summary sent to {len(farmers)} farmer(s)")
    except Exception as e:
        print(f"Daily summary error: {e}")

def daily_summary_scheduler():
    """Background thread: fires send_daily_summary() every day at 08:00."""
    while True:
        now  = datetime.now()
        next_run = now.replace(hour=8, minute=0, second=0, microsecond=0)
        if now >= next_run:
            next_run += timedelta(days=1)
        wait = (next_run - now).total_seconds()
        time.sleep(wait)
        send_daily_summary()

# ══════════════════════════════════════════════════════════════════════════════
# ── AI ADVISORY (Gemini) ──────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
ADVISORY_SYSTEM_PROMPT = (
    "You are Urja Nidhi's farm advisor — an expert in biogas production, "
    "anaerobic digestion, organic manure, and legume crop farming in rural "
    "Karnataka and Tamil Nadu. Give clear, practical, actionable advice to "
    "small-scale Indian farmers. Keep answers under 200 words, give specific "
    "numbers and timelines, end with one practical next step. "
    "If farmer writes in Kannada or Hindi, reply in that language."
)

def get_ai_advisory(question: str, sensor_context: dict = None) -> str:
    key = GEMINI_API_KEY.strip()
    if not key or not key.startswith("AIzaSy"):
        return _rule_based_advisory(question, sensor_context)
    try:
        import urllib.request
        context_str = ""
        if sensor_context:
            context_str = f"\n\nCurrent digester readings: {json.dumps(sensor_context)}"
        payload = json.dumps({
            "contents": [{"parts": [{"text": ADVISORY_SYSTEM_PROMPT + "\n\n" + question + context_str}]}],
            "generationConfig": {"maxOutputTokens": 512, "temperature": 0.4}
        }).encode("utf-8")
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={key}"
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=20) as resp:
            result = json.loads(resp.read())
            return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print(f"⚠️  Gemini API error: {e}")
        return _rule_based_advisory(question, sensor_context)

def _extract_numeric_context(question: str) -> dict:
    """
    Parse typed numeric readings from user text.
    Supports phrases like: "pH 6.4", "temperature is 32", "pressure weak 65 kpa",
    "methane 1200 ppm", "N 1800 P 500 K 80".
    """
    q = question.lower()
    parsed = {}

    def _pick(patterns):
        for p in patterns:
            m = re.search(p, q)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    continue
        return None

    ph = _pick([r"\bph\s*(?:is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)"])
    temp = _pick([r"\b(?:temp|temperature)\s*(?:is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)"])
    pressure = _pick([r"\bpressure\s*(?:is|=|:|weak|low|high)?\s*([0-9]+(?:\.[0-9]+)?)"])
    methane = _pick([r"\bmethane\s*(?:is|=|:|level)?\s*([0-9]+(?:\.[0-9]+)?)"])
    n = _pick([r"\bn\s*(?:is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)", r"\bnitrogen\s*(?:is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)"])
    p = _pick([r"\bp\s*(?:is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)", r"\bphosphorus\s*(?:is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)"])
    k = _pick([r"\bk\s*(?:is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)", r"\bpotassium\s*(?:is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)"])

    if ph is not None: parsed["ph"] = ph
    if temp is not None: parsed["temperature"] = temp
    if pressure is not None: parsed["pressure"] = pressure
    if methane is not None: parsed["methane_ppm"] = methane
    if n is not None: parsed["nitrogen_concentration"] = n
    if p is not None: parsed["phosphorus_concentration"] = p
    if k is not None: parsed["potassium_concentration"] = k
    return parsed

def _is_urja_related(question: str) -> bool:
    q = question.lower()
    markers = [
        "urja", "biogas", "digester", "methane", "slurry", "manure", "compost",
        "ph", "temperature", "pressure", "gas", "npk", "nitrogen", "phosphorus",
        "potassium", "fertilizer", "farm advisor", "cow dung", "poultry"
    ]
    return any(m in q for m in markers)

def _rule_based_advisory(question: str, ctx: dict = None) -> str:
    q   = question.lower()
    ctx = dict(ctx or {})
    ctx.update(_extract_numeric_context(question))
    ph  = float(ctx.get("ph", 7.0))
    tmp = float(ctx.get("temperature", 35))
    pressure = ctx.get("pressure")
    methane_ppm = ctx.get("methane_ppm")
    n_val = ctx.get("nitrogen_concentration")
    p_val = ctx.get("phosphorus_concentration")
    k_val = ctx.get("potassium_concentration")

    # Slurry intent must win before generic waste/feedstock intent.
    if any(w in q for w in ["slurry", "apply slurry", "slurry application", "slurry use"]):
        return (
            "For slurry application: dilute digested slurry 1:1 with water, apply near root zone "
            "in evening/early morning, and avoid flooding immediately after. Typical dose: 2000–3000 L/acre "
            "split over 2-3 rounds in the crop cycle."
        )
    if any(w in q for w in ["pressure", "weak pressure", "low pressure", "high pressure"]):
        if pressure is not None:
            p = float(pressure)
            if p < 80:
                return f"Pressure is {p} kPa (low). Check for leaks, slurry dilution, and inlet feeding consistency. Target range is 80-130 kPa."
            if p > 130:
                return f"Pressure is {p} kPa (high). Open gas usage line, inspect outlet blockage, and release pressure safely. Target is 80-130 kPa."
        return "Pressure guidance: below 80 kPa suggests leak/low gas; above 130 kPa suggests blockage or trapped gas. Share the latest kPa value for exact steps."
    if any(w in q for w in ["methane", "ch4"]):
        if methane_ppm is not None:
            m = float(methane_ppm)
            if m < 1000:
                return f"Methane is {m:.0f} ppm (low). Improve feed quality, keep pH 6.8-7.2, maintain 35-40°C, and avoid sudden feed changes."
            if m > 3000:
                return f"Methane is {m:.0f} ppm (high). Ventilation and flame-safety checks are important; verify pressure and pipeline integrity."
        return "Methane guidance: aim roughly 1000-3000 ppm on this setup. Share your latest methane reading for targeted action."
    if any(w in q for w in ["ph","acid","alkaline","sour"]):
        if ph < 6.5: return f"Your pH is {ph} — too acidic. Add 100–200g slaked lime (chuna) in water. Check again in 24 hours. Ideal: 6.8–7.2."
        if ph > 7.8: return f"Your pH is {ph} — too high. Reduce waste input by 30% for 2 days. Target 6.8–7.2."
        return "Ideal pH is 6.8–7.2. Test daily with pH strip."
    if any(w in q for w in ["temp","cold","heat","garam","thanda"]):
        if tmp < 30: return f"Temperature is {tmp}°C — too cold. Cover digester with black plastic sheet. Bacteria work best at 35–40°C."
        return "Keep digester at 35–40°C for maximum gas. Insulate in winter."
    if any(w in q for w in ["fertilizer","urea","manure","npk","nitrogen","khad"]):
        if n_val is not None or p_val is not None or k_val is not None:
            return (
                f"Using your NPK context (N={n_val if n_val is not None else '—'}, "
                f"P={p_val if p_val is not None else '—'}, K={k_val if k_val is not None else '—'}), "
                "apply slurry in split doses and reduce chemical urea gradually over 2-3 weeks."
            )
        return "Digested slurry = excellent fertilizer. 1 m³ biogas → 0.9 kg urea equivalent (saves ₹22/kg). Dilute 1:1 with water before applying."
    # Feedstock / waste selection (cow dung vs poultry litter)
    if any(w in q for w in ["cow dung", "cow manure", "dung", "poultry litter", "poultry", "waste", "feedstock", "slurry", "manure"]):
        if "poultry" in q:
            return (
                "For biogas, start mainly with cow dung (best stable microbes). "
                "Poultry litter can work, but it needs careful handling: pre-compost/ferment it 24–72h, "
                "use only ~10–20% of daily feed at first (rest cow dung + water), and keep pH ~6.8–7.2. "
                "Reduce poultry fraction immediately if smell/ammonia increases."
            )
        return (
            "Best waste for biogas (stable digestion): cow dung. "
            "Mix with water to make slurry (~1:1 to 1:1.5), feed daily, and aim for 35–40°C. "
            "Start with small batches and monitor pH every day for the first week."
        )
    if any(w in q for w in ["gas","biogas","production","low","less","kam"]):
        return "Low gas? Check: (1) pH 6.8–7.2, (2) Temp 35–40°C, (3) Feed 50–60 kg daily, (4) No pipe leaks."
    if any(w in q for w in ["save","money","cost","profit","paisa"]):
        return "2 m³ digester with 50 kg dung/day saves ₹800–1100/week. Annual saving: ₹40,000–55,000. Payback in under 6 months."
    if any(w in q for w in ["smell","odour","stink"]):
        return "Bad smell = too much protein or high pH. Reduce poultry input. Check pH — if >7.8 reduce feeding 2 days."
    if _is_urja_related(question):
        return "I can help better with exact readings. Please share pH, temperature, pressure, methane, or NPK values with your question."
    return "Please rephrase your question with Urja Nidhi context (biogas, digester readings, methane, slurry, or fertilizer use)."

# ══════════════════════════════════════════════════════════════════════════════
# ── ROUTES ────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    return render_template("urja-nidhi-dashboard.html")

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":            "online",
        "model_loaded":      model is not None,
        "firebase_active":   _firebase_active,
        "advisory_enabled":  GEMINI_API_KEY.startswith("AIzaSy"),
        "timestamp":         datetime.now().isoformat(),
        "version":           "4.0.0"
    })

# ── AUTH: Register ────────────────────────────────────────────────────────────
@app.route("/api/auth/register", methods=["POST"])
def register():
    """
    Register a new farmer with Firebase email+password and local profile.
    Body: { "name": "...", "email": "...", "phone": "...", "password": "......" }
    """
    data  = request.get_json(force=True)
    name  = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower() or None
    phone = (data.get("phone") or "").strip() or None
    password = (data.get("password") or "").strip()

    if not name:
        return jsonify({"error": "Name is required"}), 400
    if not email:
        return jsonify({"error": "Email is required for website access"}), 400
    if not phone:
        return jsonify({"error": "Phone number is required for notifications"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    firebase_auth = get_firebase_auth_module()
    if not firebase_auth:
        return jsonify({"error": "Firebase credentials are not configured on server"}), 500

    created_uid = None
    try:
        fb_user = firebase_auth.create_user(email=email, password=password, display_name=name)
        created_uid = fb_user.uid
        conn = get_db()
        conn.execute("""INSERT INTO farmers (name,email,phone,firebase_uid,created_at)
            VALUES (?,?,?,?,?)""",
            (name, email, phone, created_uid, datetime.now().isoformat()))
        conn.commit()
        farmer_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()

        token = create_session(farmer_id)
        # Welcome message
        if email:
            threading.Thread(target=_send_email, daemon=True,
                args=(email, "Welcome to Urja Nidhi 🌿",
                      f"Namaskara {name}!\n\nYour Urja Nidhi account is active.\n"
                      f"You will receive digester alerts at this address.\n\n"
                      f"Login at: http://localhost:5000",
                      farmer_id)).start()
        return jsonify({
            "success": True,
            "token":   token,
            "farmer":  {"id": farmer_id, "name": name, "email": email, "phone": phone}
        })
    except sqlite3.IntegrityError:
        if created_uid:
            try:
                firebase_auth.delete_user(created_uid)
            except Exception:
                pass
        # Return a precise conflict reason to avoid confusing "new email" cases.
        try:
            conn = get_db()
            by_email = conn.execute("SELECT id FROM farmers WHERE email=?", (email,)).fetchone() if email else None
            by_phone = conn.execute("SELECT id FROM farmers WHERE phone=?", (phone,)).fetchone() if phone else None
            conn.close()
            if by_email:
                return jsonify({"error": "Email already registered. Please login."}), 409
            if by_phone:
                return jsonify({"error": "Phone number already registered. Use a different phone or login."}), 409
        except Exception:
            pass
        return jsonify({"error": "Registration conflict. Email, phone, or account mapping already exists."}), 409
    except Exception as e:
        if created_uid and firebase_auth:
            try:
                firebase_auth.delete_user(created_uid)
            except Exception:
                pass
        return jsonify({"error": str(e)}), 400

# ── AUTH: Login ───────────────────────────────────────────────────────────────
@app.route("/api/auth/login", methods=["POST"])
def login():
    """
    Login with Firebase email+password, then issue local session token.
    Body: { "email": "...", "password": "..." }
    Returns a session token.
    """
    data  = request.get_json(force=True)
    email = (data.get("email") or "").strip().lower() or None
    password = (data.get("password") or "").strip()
    if not email:
        return jsonify({"error": "Email is required for website access"}), 400
    if not password:
        return jsonify({"error": "Password is required"}), 400

    firebase_uid, fb_err = firebase_verify_email_password(email, password)
    if fb_err:
        return jsonify({"error": fb_err}), 401

    conn = get_db()
    farmer = conn.execute("SELECT * FROM farmers WHERE email=?", (email,)).fetchone()
    conn.close()

    if not farmer:
        return jsonify({"error": "No account found. Please register first."}), 404

    farmer = dict(farmer)
    if not farmer.get("firebase_uid"):
        return jsonify({"error": "Account not linked to Firebase. Please register again."}), 401
    if farmer.get("firebase_uid") != firebase_uid:
        return jsonify({"error": "Firebase account mismatch for this email"}), 401
    # Update last_login
    conn = get_db()
    conn.execute("UPDATE farmers SET last_login=? WHERE id=?",
                 (datetime.now().isoformat(), farmer["id"]))
    conn.commit(); conn.close()

    token = create_session(farmer["id"])
    return jsonify({
        "success": True,
        "token":   token,
        "farmer":  {
            "id":    farmer["id"],
            "name":  farmer["name"],
            "email": farmer["email"],
            "phone": farmer["phone"],
        }
    })

# ── AUTH: Logout ──────────────────────────────────────────────────────────────
@app.route("/api/auth/logout", methods=["POST"])
def logout():
    token = request.headers.get("X-Session-Token") or request.cookies.get("session_token")
    if token:
        conn = get_db()
        conn.execute("DELETE FROM sessions WHERE token=?", (token,))
        conn.commit(); conn.close()
    return jsonify({"success": True})

# ── AUTH: Check session ───────────────────────────────────────────────────────
@app.route("/api/auth/me", methods=["GET"])
def me():
    token  = request.headers.get("X-Session-Token") or request.cookies.get("session_token")
    farmer = get_farmer_from_token(token)
    if not farmer:
        return jsonify({"logged_in": False}), 200
    return jsonify({"logged_in": True, "farmer": {
        "id":    farmer["id"],
        "name":  farmer["name"],
        "email": farmer["email"],
        "phone": farmer["phone"],
    }})

# ── Farmer crops (personalized crop list) ────────────────────────────────────
@app.route("/api/crops", methods=["GET"])
@require_login
def get_crops():
    try:
        conn = get_db()
        rows = conn.execute("""SELECT crop_name FROM farmer_crops
            WHERE farmer_id=? AND is_active=1 ORDER BY crop_name""",
            (request.farmer["id"],)).fetchall()
        conn.close()
        crops = [r["crop_name"] for r in rows] or DEFAULT_USER_CROPS
        return jsonify({"crops": crops})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/crops", methods=["POST"])
@require_login
def add_crop():
    data = request.get_json(force=True)
    crop = (data.get("crop_name") or "").strip().lower()
    if not crop:
        return jsonify({"error": "crop_name is required"}), 400
    try:
        conn = get_db()
        conn.execute("""INSERT INTO farmer_crops (farmer_id,crop_name,is_active,created_at)
            VALUES (?,?,1,?)
            ON CONFLICT(farmer_id,crop_name) DO UPDATE SET is_active=1""",
            (request.farmer["id"], crop, datetime.now().isoformat()))
        conn.commit(); conn.close()
        return jsonify({"success": True, "crop_name": crop})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/crops/<crop_name>", methods=["DELETE"])
@require_login
def remove_crop(crop_name):
    try:
        conn = get_db()
        conn.execute("""UPDATE farmer_crops SET is_active=0
            WHERE farmer_id=? AND crop_name=?""",
            (request.farmer["id"], crop_name.strip().lower()))
        conn.commit(); conn.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ── Crop cycle prediction (NPK + govt weather/soil references) ───────────────
@app.route("/api/crop-cycle/predict", methods=["POST"])
@require_login
def crop_cycle_predict():
    data = request.get_json(force=True)
    crops = data.get("crops") or []
    state = (data.get("state") or "Karnataka").strip()
    soil  = (data.get("soil_type") or "black_cotton").strip()
    start_date_raw = (data.get("start_date") or "").strip()
    if not isinstance(crops, list) or not crops:
        return jsonify({"error": "Provide at least one crop"}), 400
    try:
        start_dt = None
        if start_date_raw:
            try:
                start_dt = datetime.strptime(start_date_raw, "%Y-%m-%d")
            except ValueError:
                return jsonify({"error": "start_date must be YYYY-MM-DD"}), 400
        base = build_crop_cycle_predictions(crops, state, soil, start_dt)

        # AI narrative layer (Gemini if available, rule-based fallback otherwise)
        ai_prompt = (
            "You are an agronomy advisor. Summarise crop-cycle suitability "
            "for Indian farmers in <=120 words, include 1 action step.\n"
            f"State: {state}, Soil: {soil}, Predictions: {json.dumps(base['predictions'])}"
        )
        ai_text = get_ai_advisory(ai_prompt, {
            "temperature": base["live_context"]["temperature"],
            "ph": base["live_context"]["ph"],
            "nitrogen": base["live_context"]["n"],
        })
        return jsonify({
            **base,
            "state": state,
            "soil_type": soil,
            "start_date": base.get("start_date"),
            "ai_summary": ai_text,
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 400

# ── AI Prediction (Layers 5+6+7) ─────────────────────────────────────────────
@app.route("/api/predict/auto", methods=["POST", "GET"])
def predict_auto():
    """
    Run prediction using latest sensor data from database.
    No manual input required — auto-fetches from sensor_readings.
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503
    try:
        conn = get_db()
        row  = conn.execute("""SELECT digester_id, temperature, ph, pressure, gas_flow,
            nitrogen_concentration, phosphorus_concentration, potassium_concentration
            FROM sensor_readings ORDER BY id DESC LIMIT 1""").fetchone()
        conn.close()

        if not row:
            return jsonify({"error": "No sensor data yet. Use Live Sensors → Manual Entry or wait for IoT feed."}), 400

        # Build payload from latest sensor + sensible defaults
        # sqlite3.Row supports dict-style indexing; `.get()` may not exist.
        digester_id = row["digester_id"] if row and "digester_id" in row.keys() else "DIG001"
        data = {
            "waste_quantity": 50, "cn_ratio": 25, "moisture_level": 70,
            "temperature": float(row["temperature"] or 36),
            "ph": float(row["ph"] or 7.0),
            "retention_time": 25, "gas_flow_rate": float(row["gas_flow"] or 3.5),
            "methane_concentration": 60, "ambient_temperature": 28, "ambient_humidity": 70,
            # Use live N/P/K from sensor_readings (not hardcoded defaults).
            "nitrogen_concentration": float(row["nitrogen_concentration"] or 2000),
            "phosphorus_concentration": float(row["phosphorus_concentration"] or 500),
            "potassium_concentration": float(row["potassium_concentration"] or 1200),
            "microbial_activity": 1500000,
            "soil_n_requirement": 30, "manure_equivalent_n": 20,
            "external_fertilizer_required": 10, "waste_collection_cost": 60,
            "digester_operating_cost": 40,
            "waste_type": "cow_dung", "pre_treatment": "raw", "crop_type": "chickpea",
            "digester_id": digester_id,
        }
        feat  = build_feature_df(data)
        raw   = float(model.predict(feat)[0])
        daily = max(0.001, raw)
        ana   = compute_analytics(daily)
        ph    = data["ph"]
        n_conc = data["nitrogen_concentration"]
        p_conc = data["phosphorus_concentration"]
        k_conc = data["potassium_concentration"]
        comp_label, comp_score = classify_compost(daily, ph)
        alerts = check_alerts({"ph": ph, "temperature": data["temperature"], "gas_flow": data["gas_flow_rate"]})
        result = {
            "daily_biogas_m3":  ana["daily_biogas_m3"],
            "weekly_biogas_m3": ana["weekly_biogas_m3"],
            "nitrogen_level":   classify_nitrogen(n_conc),
            "nitrogen_pct":     round(n_conc / 10000, 3),
            # Expose live NPK so the UI can display correct values.
            "npk": {"n": n_conc, "p": p_conc, "k": k_conc},
            "nitrogen_concentration": n_conc,
            "phosphorus_concentration": p_conc,
            "potassium_concentration": k_conc,
            "compost_quality":  comp_label,
            "compost_score":    comp_score,
            "analytics":        ana,
            "alerts":           alerts,
            "timestamp":        datetime.now().isoformat(),
            "source":           "auto",
        }
        conn = get_db()
        conn.execute("""INSERT INTO prediction_results
            (digester_id,daily_biogas,weekly_biogas,nitrogen_level,nitrogen_pct,
             compost_quality,compost_score,money_saved,energy_kwh,urea_equivalent,
             co2_reduction,cooking_hours,timestamp) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (digester_id, ana["daily_biogas_m3"], ana["weekly_biogas_m3"],
             result["nitrogen_level"], result["nitrogen_pct"], comp_label, comp_score,
             ana["money_saved_inr"], ana["energy_kwh"], ana["urea_equivalent_kg"],
             ana["co2_reduction_kg"], ana["cooking_hours"], result["timestamp"]))
        conn.commit(); conn.close()
        socketio.emit("prediction_update", result)
        if alerts:
            threading.Thread(target=dispatch_alerts_to_all_farmers, args=(alerts,), daemon=True).start()
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 400

@app.route("/api/predict/latest", methods=["GET"])
def predict_latest():
    """Return the latest prediction from database (for dashboard/reports)."""
    try:
        conn = get_db()
        row  = conn.execute("""SELECT daily_biogas, weekly_biogas, nitrogen_level, nitrogen_pct,
            compost_quality, compost_score, money_saved, energy_kwh,
            urea_equivalent, co2_reduction, cooking_hours, timestamp
            FROM prediction_results ORDER BY id DESC LIMIT 1""").fetchone()
        sensor = conn.execute("""SELECT nitrogen_concentration, phosphorus_concentration, potassium_concentration
            FROM sensor_readings ORDER BY id DESC LIMIT 1""").fetchone()
        conn.close()
        if not row:
            return jsonify({"exists": False})
        r = dict(row)
        daily = r.get("daily_biogas") or 0
        r["daily_biogas_m3"] = daily
        r["analytics"] = compute_analytics(daily)
        if sensor:
            sensor = dict(sensor)
            n_conc = sensor.get("nitrogen_concentration")
            p_conc = sensor.get("phosphorus_concentration")
            k_conc = sensor.get("potassium_concentration")
            r["npk"] = {
                "n": float(n_conc) if n_conc is not None else 2000.0,
                "p": float(p_conc) if p_conc is not None else 500.0,
                "k": float(k_conc) if k_conc is not None else 1200.0,
            }
            r["nitrogen_concentration"] = r["npk"]["n"]
            r["phosphorus_concentration"] = r["npk"]["p"]
            r["potassium_concentration"] = r["npk"]["k"]
        r["exists"] = True
        r["alerts"] = []
        return jsonify(r)
    except Exception as e:
        return jsonify({"error": str(e), "exists": False}), 400

@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503
    try:
        data  = request.get_json(force=True)
        feat  = build_feature_df(data)
        raw   = float(model.predict(feat)[0])
        daily = max(0.001, raw)
        ana   = compute_analytics(daily)
        ph    = float(data.get("ph", 7.0))
        n_conc = float(data.get("nitrogen_concentration", 2000))
        comp_label, comp_score = classify_compost(daily, ph)
        alerts = check_alerts({"ph": ph,
                                "temperature": float(data.get("temperature", 35)),
                                "gas_flow":    float(data.get("gas_flow_rate", 3.5))})
        result = {
            "daily_biogas_m3":  ana["daily_biogas_m3"],
            "weekly_biogas_m3": ana["weekly_biogas_m3"],
            "nitrogen_level":   classify_nitrogen(n_conc),
            "nitrogen_pct":     round(n_conc / 10000, 3),
            "compost_quality":  comp_label,
            "compost_score":    comp_score,
            "analytics":        ana,
            "alerts":           alerts,
            "timestamp":        datetime.now().isoformat(),
        }
        conn = get_db()
        conn.execute("""INSERT INTO prediction_results
            (digester_id,daily_biogas,weekly_biogas,nitrogen_level,nitrogen_pct,
             compost_quality,compost_score,money_saved,energy_kwh,urea_equivalent,
             co2_reduction,cooking_hours,timestamp) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (data.get("digester_id","DIG001"), ana["daily_biogas_m3"],
             ana["weekly_biogas_m3"], result["nitrogen_level"], result["nitrogen_pct"],
             comp_label, comp_score, ana["money_saved_inr"], ana["energy_kwh"],
             ana["urea_equivalent_kg"], ana["co2_reduction_kg"],
             ana["cooking_hours"], result["timestamp"]))
        conn.commit(); conn.close()
        socketio.emit("prediction_update", result)
        # Dispatch alerts immediately to all registered farmers
        if alerts:
            threading.Thread(target=dispatch_alerts_to_all_farmers,
                             args=(alerts,), daemon=True).start()
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 400

# ── IoT Ingestion (Layer 2) ───────────────────────────────────────────────────
@app.route("/api/sensor", methods=["POST"])
def ingest_sensor():
    try:
        data = request.get_json(force=True)
        # Required fields
        for f in ["temperature", "ph", "pressure", "gas_flow"]:
            if f not in data:
                return jsonify({"error": f"Missing: {f}"}), 400
        
        ts     = data.get("timestamp", datetime.now().isoformat())
        # Handle field name variants
        temp = data.get("temperature", data.get("temperature_C", 30))
        pres = data.get("pressure", data.get("pressure_kPa", 100))
        
        sensor_reading = {
            "digester_id": data.get("digester_id", "DIG001"),
            "temperature": float(temp),
            "ph": float(data.get("ph", 7.0)),
            "pressure": float(pres),
            "gas_flow": float(data.get("gas_flow", 0)),
            "methane_raw": float(data.get("methane_raw", data.get("methane_ppm", 0))),
            "nitrogen_concentration": float(data.get("nitrogen_concentration", data.get("nitrogen_mgkg", 2000))),
            "phosphorus_concentration": float(data.get("phosphorus_concentration", data.get("phosphorus_mgkg", 500))),
            "potassium_concentration": float(data.get("potassium_concentration", data.get("potassium_mgkg", 1200))),
        }
        sensor_reading["methane_ppm"] = mq5_raw_to_ppm(sensor_reading["methane_raw"])
        
        alerts = check_alerts(sensor_reading)
        conn   = get_db()
        conn.execute("""INSERT INTO sensor_readings
            (digester_id,temperature,ph,pressure,gas_flow,methane_raw,methane_ppm,
             nitrogen_concentration,phosphorus_concentration,potassium_concentration,timestamp)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (sensor_reading["digester_id"], sensor_reading["temperature"],
             sensor_reading["ph"], sensor_reading["pressure"], sensor_reading["gas_flow"],
             sensor_reading["methane_raw"], sensor_reading["methane_ppm"],
             sensor_reading["nitrogen_concentration"], sensor_reading["phosphorus_concentration"],
             sensor_reading["potassium_concentration"], ts))
        conn.commit(); conn.close()
        _emit_sensor_update_ws({**sensor_reading, "alerts": alerts, "timestamp": ts})
        if alerts:
            threading.Thread(target=dispatch_alerts_to_all_farmers,
                             args=(alerts,), daemon=True).start()
        return jsonify({"status": "ingested", "alerts": alerts, "timestamp": ts})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ── Analytics ─────────────────────────────────────────────────────────────────
@app.route("/api/analytics", methods=["GET"])
def analytics_summary():
    try:
        conn    = get_db()
        rows    = conn.execute("""SELECT daily_biogas,weekly_biogas,money_saved,energy_kwh,
            urea_equivalent,co2_reduction,nitrogen_level,compost_quality,timestamp
            FROM prediction_results ORDER BY id DESC LIMIT 30""").fetchall()
        conn.close()
        history = [dict(r) for r in rows]
        totals  = {
            "total_money_saved_inr":  round(sum(r["money_saved"]   or 0 for r in history), 2),
            "total_biogas_m3":        round(sum(r["weekly_biogas"] or 0 for r in history), 3),
            "total_co2_reduction_kg": round(sum(r["co2_reduction"] or 0 for r in history), 2),
            "avg_daily_biogas_m3":    round(sum(r["daily_biogas"]  or 0 for r in history) / max(len(history),1), 3),
        } if history else {}
        return jsonify({"history": history, "totals": totals})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/sensors/history", methods=["GET"])
def sensor_history():
    limit = request.args.get("limit", 50, type=int)
    try:
        conn = get_db()
        rows = conn.execute("""SELECT temperature,ph,pressure,gas_flow,methane_raw,methane_ppm,
            nitrogen_concentration,phosphorus_concentration,potassium_concentration,timestamp
            FROM sensor_readings ORDER BY id DESC LIMIT ?""", (limit,)).fetchall()
        conn.close()
        return jsonify([dict(r) for r in reversed(rows)])
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ── Thresholds — GET all ──────────────────────────────────────────────────────
@app.route("/api/thresholds", methods=["GET"])
def get_thresholds_api():
    try:
        conn = get_db()
        rows = conn.execute("SELECT * FROM thresholds ORDER BY id").fetchall()
        conn.close()
        return jsonify([dict(r) for r in rows])
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ── Thresholds — UPDATE one param ─────────────────────────────────────────────
@app.route("/api/thresholds/<param>", methods=["PUT"])
@require_login
def update_threshold(param):
    """
    Update threshold limits for a single sensor parameter.
    Body: { "low_warn": 30, "low_crit": null, "high_warn": 40, "high_crit": 42, "enabled": 1 }
    Any field can be null to disable that limit.
    """
    data = request.get_json(force=True)
    allowed = ["low_warn", "low_crit", "high_warn", "high_crit", "enabled"]
    try:
        conn = get_db()
        # Check param exists
        existing = conn.execute("SELECT id FROM thresholds WHERE param=?", (param,)).fetchone()
        if not existing:
            conn.close()
            return jsonify({"error": f"Unknown parameter: {param}"}), 404
        # Build update
        sets  = ", ".join(f"{k}=?" for k in allowed if k in data)
        vals  = [data[k] for k in allowed if k in data]
        sets += ", updated_at=?"
        vals += [datetime.now().isoformat(), param]
        conn.execute(f"UPDATE thresholds SET {sets} WHERE param=?", vals)
        conn.commit()
        updated = conn.execute("SELECT * FROM thresholds WHERE param=?", (param,)).fetchone()
        conn.close()
        return jsonify(dict(updated))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ── Thresholds — RESET all to defaults ───────────────────────────────────────
@app.route("/api/thresholds/reset", methods=["POST"])
@require_login
def reset_thresholds():
    try:
        conn = get_db()
        conn.execute("DELETE FROM thresholds")
        conn.commit()
        conn.close()
        seed_thresholds()
        return jsonify({"status": "reset to defaults"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ── PDF Report ────────────────────────────────────────────────────────────────
@app.route("/api/report/pdf", methods=["GET"])
@require_login
def generate_pdf_report():
    """
    Generate a plain-text report as downloadable .txt file.
    For a real PDF, replace with reportlab or weasyprint.
    Returns Content-Disposition: attachment so browser downloads it.
    """
    farmer = request.farmer
    days   = request.args.get("days", 7, type=int)
    report_type = (request.args.get("type", "digester") or "digester").lower()
    # Support download buttons for multiple report categories.
    if report_type in ("weekly", "week"):
        days = 7
        report_title = "WEEKLY REPORT"
    elif report_type in ("monthly", "month"):
        days = 30
        report_title = "MONTHLY REPORT"
    elif report_type in ("daily", "daily_summary", "daily-summary"):
        days = 1
        report_title = "DAILY SUMMARY"
    elif report_type in ("predictions", "prediction_alerts", "prediction-alerts"):
        report_title = "PREDICTION REPORT"
    elif report_type in ("thresholds", "threshold_alerts", "threshold-alerts"):
        report_title = "THRESHOLD ALERT REPORT"
    else:
        report_title = "DIGESTER REPORT"
    try:
        conn = get_db()
        # Sensor averages
        sens = conn.execute("""
            SELECT AVG(temperature) t, AVG(ph) p, AVG(pressure) pr,
                   AVG(gas_flow) f, COUNT(*) n, MIN(timestamp) from_ts, MAX(timestamp) to_ts
            FROM sensor_readings
            WHERE timestamp > datetime('now', ? || ' days')
        """, (f"-{days}",)).fetchone()
        # Prediction totals
        preds = conn.execute("""
            SELECT SUM(daily_biogas) total_gas, SUM(money_saved) total_saved,
                   SUM(energy_kwh) total_kwh, SUM(co2_reduction) total_co2,
                   COUNT(*) n
            FROM prediction_results
            WHERE timestamp > datetime('now', ? || ' days')
        """, (f"-{days}",)).fetchone()
        # Alert count (varies by report type)
        if report_type in ("thresholds", "threshold_alerts", "threshold-alerts"):
            alerts_count = conn.execute("""
                SELECT COUNT(*) n FROM notifications
                WHERE farmer_id=?
                  AND timestamp > datetime('now', ? || ' days')
                  AND (
                        message LIKE '%(CRITICAL)%'
                     OR message LIKE '%(WARNING)%'
                     OR subject LIKE '%Digester Alert%'
                     OR subject='alert'
                  )
                  AND subject NOT LIKE '%Daily Digester Summary%'
            """, (farmer["id"], f"-{days}")).fetchone()
            recent_threshold_notifs = conn.execute("""
                SELECT channel, timestamp, subject, message FROM notifications
                WHERE farmer_id=?
                  AND timestamp > datetime('now', ? || ' days')
                  AND (
                        message LIKE '%(CRITICAL)%'
                     OR message LIKE '%(WARNING)%'
                     OR subject LIKE '%Digester Alert%'
                     OR subject='alert'
                  )
                ORDER BY id DESC
                LIMIT 8
            """, (farmer["id"], f"-{days}")).fetchall()
        elif report_type in ("daily", "daily_summary", "daily-summary"):
            alerts_count = conn.execute("""
                SELECT COUNT(*) n FROM notifications
                WHERE farmer_id=?
                  AND timestamp > datetime('now', ? || ' days')
                  AND subject LIKE '%Daily Digester Summary%'
            """, (farmer["id"], f"-{days}")).fetchone()
            recent_threshold_notifs = conn.execute("""
                SELECT channel, timestamp, subject, message FROM notifications
                WHERE farmer_id=?
                  AND timestamp > datetime('now', ? || ' days')
                  AND subject LIKE '%Daily Digester Summary%'
                ORDER BY id DESC
                LIMIT 3
            """, (farmer["id"], f"-{days}")).fetchall()
        else:
            alerts_count = conn.execute("""
                SELECT COUNT(*) n FROM notifications
                WHERE farmer_id=? AND timestamp > datetime('now', ? || ' days')
            """, (farmer["id"], f"-{days}")).fetchone()
            recent_threshold_notifs = []
        # Recent predictions
        recent = conn.execute("""
            SELECT daily_biogas,weekly_biogas,money_saved,energy_kwh,
                   compost_quality,nitrogen_level,timestamp
            FROM prediction_results ORDER BY id DESC LIMIT 10
        """).fetchall()
        conn.close()

        now = datetime.now().strftime("%d %b %Y, %I:%M %p")
        L   = []
        L.append("=" * 60)
        L.append(f"        URJA NIDHI — {report_title}")
        L.append("=" * 60)
        L.append(f"  Farmer  : {farmer['name']}")
        L.append(f"  Email   : {farmer.get('email') or '—'}")
        L.append(f"  Phone   : {farmer.get('phone') or '—'}")
        L.append(f"  Period  : Last {days} days")
        L.append(f"  Generated: {now}")
        L.append("")
        L.append("─" * 60)
        L.append("  SENSOR AVERAGES")
        L.append("─" * 60)
        if sens and sens["n"]:
            L.append(f"  Readings collected : {sens['n']}")
            L.append(f"  Avg Temperature    : {round(sens['t'] or 0, 1)} °C")
            L.append(f"  Avg pH             : {round(sens['p'] or 0, 2)}")
            L.append(f"  Avg Pressure       : {round(sens['pr'] or 0, 1)} kPa")
            L.append(f"  Avg Gas Flow       : {round(sens['f'] or 0, 2)} L/h")
        else:
            L.append("  No sensor data in this period.")
        L.append("")
        L.append("─" * 60)
        L.append("  PRODUCTION & SAVINGS")
        L.append("─" * 60)
        if preds and preds["n"]:
            L.append(f"  Predictions run    : {preds['n']}")
            L.append(f"  Total Biogas       : {round(preds['total_gas']  or 0, 2)} m³")
            L.append(f"  Total Energy       : {round(preds['total_kwh']  or 0, 2)} kWh")
            L.append(f"  Total CO₂ Saved    : {round(preds['total_co2']  or 0, 2)} kg")
            L.append(f"  Total Money Saved  : ₹{round(preds['total_saved'] or 0, 2)}")
        else:
            L.append("  No predictions in this period.")
        L.append("")
        L.append("─" * 60)
        L.append("  ALERTS SENT TO YOUR ACCOUNT")
        L.append("─" * 60)
        L.append(f"  Total alerts : {alerts_count['n'] if alerts_count else 0}")
        L.append("")
        if report_type in ("thresholds", "threshold_alerts", "threshold-alerts") and recent_threshold_notifs:
            L.append("─" * 60)
            L.append("  RECENT THRESHOLD ALERT MESSAGES (latest 8)")
            L.append("─" * 60)
            for n in recent_threshold_notifs:
                ts2 = n["timestamp"][:16].replace("T", " ")
                msg = (n["message"] or "").replace("\n", " ").strip()
                L.append(f"  {ts2} | {msg[:120]}")
            L.append("")
        elif report_type in ("daily", "daily_summary", "daily-summary") and recent_threshold_notifs:
            L.append("─" * 60)
            L.append("  RECENT DAILY SUMMARY NOTIFICATIONS")
            L.append("─" * 60)
            for n in recent_threshold_notifs:
                ts2 = n["timestamp"][:16].replace("T", " ")
                msg = (n["message"] or "").replace("\n", " ").strip()
                L.append(f"  {ts2} | {msg[:120]}")
            L.append("")
        L.append("─" * 60)
        L.append("  RECENT PREDICTIONS (last 10)")
        L.append("─" * 60)
        L.append(f"  {'Date':<18} {'Biogas m³':>9} {'kWh':>7} {'Saved ₹':>10} {'Compost':<10} {'N Level'}")
        L.append("  " + "-" * 58)
        for r in recent:
            ts = r["timestamp"][:16].replace("T", " ")
            L.append(f"  {ts:<18} {str(r['daily_biogas'] or '—'):>9} "
                     f"{str(r['energy_kwh'] or '—'):>7} "
                     f"{('₹'+str(round(r['money_saved'] or 0))):>10} "
                     f"{(r['compost_quality'] or '—'):<10} {r['nitrogen_level'] or '—'}")
        L.append("")
        L.append("=" * 60)
        L.append("  Urja Nidhi — Energy Treasure Platform")
        L.append("  Generated automatically. For support contact your admin.")
        L.append("=" * 60)

        report_text = "\n".join(L)
        filename    = f"urja_nidhi_report_{farmer['name'].replace(' ','_')}_{days}d.txt"

        from flask import Response
        return Response(
            report_text,
            mimetype="text/plain",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 400
@app.route("/api/notifications", methods=["GET"])
@require_login
def get_notifications():
    """Returns only THIS farmer's notification history."""
    limit = request.args.get("limit", 30, type=int)
    try:
        conn = get_db()
        rows = conn.execute("""SELECT channel,recipient,subject,message,status,timestamp
            FROM notifications WHERE farmer_id=?
            ORDER BY id DESC LIMIT ?""",
            (request.farmer["id"], limit)).fetchall()
        conn.close()
        return jsonify([dict(r) for r in rows])
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/notify/test", methods=["POST"])
@require_login
def notify_test():
    """Send a test notification to the logged-in farmer's own contact."""
    farmer  = request.farmer
    channel = request.get_json(force=True).get("channel", "email")
    msg     = "Test alert from Urja Nidhi — your digester monitoring is active."
    if channel == "email" and farmer.get("email"):
        result = _send_email(farmer["email"], "Test Notification", msg, farmer["id"])
    elif channel == "sms" and farmer.get("phone"):
        result = _send_sms(farmer["phone"], msg, farmer["id"])
    else:
        return jsonify({"error": f"No {channel} address registered for your account"}), 400
    return jsonify(result)

@app.route("/api/notify/test-email", methods=["POST"])
@require_login
def notify_test_email():
    """
    Send a dedicated test email and report SMTP env readiness.
    """
    farmer = request.farmer
    env_status = smtp_env_check()
    if not env_status["ok"]:
        return jsonify({
            "success": False,
            "error": "SMTP configuration is incomplete in .env",
            "missing_env": env_status["missing"],
            "env_ready": env_status["ok"],
        }), 400

    to_email = (request.get_json(silent=True) or {}).get("email") or farmer.get("email")
    if not to_email:
        return jsonify({"success": False, "error": "No email found for this account"}), 400

    body = (
        "This is a test email from Urja Nidhi.\n\n"
        "If you received this, SMTP configuration is working correctly.\n"
        f"Time: {datetime.now().isoformat()}\n"
    )
    result = _send_email(to_email, "SMTP Test Email", body, farmer["id"])
    return jsonify({
        "success": bool(result.get("success")),
        "message": result.get("message"),
        "env_ready": env_status["ok"],
        "missing_env": env_status["missing"],
        "to": to_email,
    }), (200 if result.get("success") else 400)

# ── Daily summary manual trigger (for testing) ────────────────────────────────
@app.route("/api/notify/daily-summary", methods=["POST"])
@require_login
def trigger_daily_summary():
    threading.Thread(target=send_daily_summary, daemon=True).start()
    return jsonify({"status": "Daily summary dispatched to all farmers"})

# ── AI Advisory ───────────────────────────────────────────────────────────────
@app.route("/api/advisory", methods=["POST"])
def advisory():
    data     = request.get_json(force=True)
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "question field is required"}), 400
    ctx = data.get("sensor_context", {})
    if not isinstance(ctx, dict):
        ctx = {}
    # Typed readings in the question should override generic live context.
    ctx.update(_extract_numeric_context(question))
    answer = get_ai_advisory(question, ctx)

    # Get farmer_id if logged in (advisory works even without login)
    token    = request.headers.get("X-Session-Token") or request.cookies.get("session_token")
    farmer   = get_farmer_from_token(token)
    fid      = farmer["id"] if farmer else None

    try:
        conn = get_db()
        conn.execute("""INSERT INTO advisory_history
            (farmer_id,question,context,answer,timestamp) VALUES (?,?,?,?,?)""",
            (fid, question, json.dumps(ctx), answer, datetime.now().isoformat()))
        conn.commit(); conn.close()
    except Exception: pass

    socketio.emit("advisory_response", {
        "question": question, "answer": answer,
        "timestamp": datetime.now().isoformat()
    })
    return jsonify({"question": question, "answer": answer,
                    "timestamp": datetime.now().isoformat()})

@app.route("/api/advisory/history", methods=["GET"])
def advisory_history():
    token   = request.headers.get("X-Session-Token") or request.cookies.get("session_token")
    farmer  = get_farmer_from_token(token)
    limit   = request.args.get("limit", 10, type=int)
    try:
        conn = get_db()
        if farmer:
            rows = conn.execute("""SELECT question,answer,timestamp FROM advisory_history
                WHERE farmer_id=? ORDER BY id DESC LIMIT ?""",
                (farmer["id"], limit)).fetchall()
        else:
            rows = conn.execute("""SELECT question,answer,timestamp FROM advisory_history
                WHERE farmer_id IS NULL ORDER BY id DESC LIMIT ?""", (limit,)).fetchall()
        conn.close()
        return jsonify([dict(r) for r in rows])
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ── Simulated IoT (replaced by Firebase listener when configured) ─────────────
_base = {"temperature": 36.5, "ph": 7.1, "pressure": 105.0, "gas_flow": 3.8}

def simulate_iot_feed():
    """Runs only if Firebase is not active."""
    while True:
        time.sleep(5)
        if _firebase_active: break   # stop once Firebase takes over
        reading = {
            "digester_id": "DIG001",
            "temperature": round(_base["temperature"] + random.uniform(-1.5, 1.5), 2),
            "ph":          round(_base["ph"]          + random.uniform(-0.25, 0.25), 2),
            "pressure":    round(_base["pressure"]    + random.uniform(-8, 8), 1),
            "gas_flow":    round(max(0.1, _base["gas_flow"] + random.uniform(-0.6, 0.6)), 2),
            "methane_raw": round(random.uniform(5000, 18000), 1),
            "methane_ppm": None,
            "nitrogen_concentration": round(random.uniform(1500, 3000), 1),
            "phosphorus_concentration": round(random.uniform(420, 800), 1),
            "potassium_concentration": round(random.uniform(50, 100), 1),
            "timestamp":   datetime.now().isoformat(),
        }
        reading["methane_ppm"] = mq5_raw_to_ppm(reading["methane_raw"])
        alerts = check_alerts(reading)
        try:
            conn = get_db()
            conn.execute("""INSERT INTO sensor_readings
                (digester_id,temperature,ph,pressure,gas_flow,methane_raw,methane_ppm,
                 nitrogen_concentration,phosphorus_concentration,potassium_concentration,timestamp)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (reading["digester_id"], reading["temperature"], reading["ph"],
                 reading["pressure"], reading["gas_flow"], reading["methane_raw"],
                 reading["methane_ppm"], reading["nitrogen_concentration"],
                 reading["phosphorus_concentration"], reading["potassium_concentration"],
                 reading["timestamp"]))
            conn.commit(); conn.close()
        except Exception: pass
        _emit_sensor_update_ws({**reading, "alerts": alerts})
        if alerts:
            threading.Thread(target=dispatch_alerts_to_all_farmers,
                             args=(alerts,), daemon=True).start()

# ── WebSocket ─────────────────────────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    emit("status", {
        "message":        "Connected to Urja Nidhi v4.0",
        "model_loaded":   model is not None,
        "firebase_active": _firebase_active,
        "advisory_ready": GEMINI_API_KEY.startswith("AIzaSy"),
    })

@socketio.on("disconnect")
def on_disconnect(): pass

# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Try Firebase first; fall back to simulation
    firebase_ready = init_firebase()
    if firebase_ready:
        threading.Thread(target=firebase_sensor_listener, daemon=True).start()
        print("📡 Firebase listener started")
    else:
        threading.Thread(target=simulate_iot_feed, daemon=True).start()
        print("🔄 Simulated IoT feed started")

    # Daily summary scheduler
    threading.Thread(target=daily_summary_scheduler, daemon=True).start()
    print("⏰ Daily summary scheduler started (fires at 08:00)")

    print("=" * 60)
    print("  🌿  URJA NIDHI v4.0  |  http://localhost:5000")
    print("=" * 60)
    print(f"  Model:    {'✅ Loaded'  if model          else '⚠️  Missing — run train_model.py'}")
    print(f"  Firebase: {'✅ Active'  if firebase_ready  else '⚠️  Not configured — using simulation'}")
    print(f"  Advisory: {'✅ Gemini'  if GEMINI_API_KEY.startswith('AIzaSy') else '⚡ Rule-based fallback'}")
    print(f"  Email:    {'✅ Ready'   if SMTP_CFG['user']    else '⚠️  SMTP not configured'}")
    print(f"  SMS:      {'✅ Ready'   if TWILIO_CFG['sid']   else '⚠️  Twilio not configured'}")
    print("=" * 60)
    socketio.run(app, debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
                 host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)
