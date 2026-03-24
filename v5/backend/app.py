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
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import joblib, numpy as np, pandas as pd
import os, json, threading, time, sqlite3, random, smtplib, secrets, hashlib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ── App ───────────────────────────────────────────────────────────────────────
CWD = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(CWD, "templates"))
app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(32))
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

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
    "sensor_path": os.getenv("FIREBASE_SENSOR_PATH", "digesters/DIG001/sensors"),

    # ── Service Account credentials (for server-side read) ───────────────────
    # Firebase Console → Project Settings → Service Accounts → Generate new private key
    # Download the JSON file, place it in backend/ as firebase_credentials.json
    # OR set FIREBASE_CREDENTIALS_PATH in .env to point to a different location
    "credentials_path": os.getenv(
        "FIREBASE_CREDENTIALS_PATH",
        os.path.join(CWD, "firebase_credentials.json")
    ),
}

# Firebase listener state
_firebase_app    = None
_firebase_active = False

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

def firebase_sensor_listener():
    """
    Listens to Firebase Realtime Database for new sensor readings.
    When ESP32 writes a new value, this function receives it and pushes
    it through the same pipeline as the simulated feed.
    """
    try:
        from firebase_admin import db as firebase_db
        path = FIREBASE_CONFIG["sensor_path"]
        ref  = firebase_db.reference(path)

        def on_sensor_change(event):
            """Called by Firebase whenever data at sensor_path changes."""
            try:
                data = event.data
                if not isinstance(data, dict): return
                # Normalise field names (ESP32 may send camelCase or snake_case)
                reading = {
                    "digester_id": data.get("digester_id", "DIG001"),
                    "temperature": float(data.get("temperature", data.get("temp", 0))),
                    "ph":          float(data.get("ph", data.get("pH", 7.0))),
                    "pressure":    float(data.get("pressure", 105.0)),
                    "gas_flow":    float(data.get("gas_flow", data.get("gasFlow", 0))),
                    "timestamp":   data.get("timestamp", datetime.now().isoformat()),
                }
                alerts = check_alerts(reading)
                # Persist to SQLite
                try:
                    conn = get_db()
                    conn.execute("""INSERT INTO sensor_readings
                        (digester_id,temperature,ph,pressure,gas_flow,timestamp)
                        VALUES (?,?,?,?,?,?)""",
                        (reading["digester_id"], reading["temperature"], reading["ph"],
                         reading["pressure"], reading["gas_flow"], reading["timestamp"]))
                    conn.commit(); conn.close()
                except Exception: pass
                # Push to dashboard via WebSocket
                socketio.emit("sensor_update", {**reading, "alerts": alerts})
                # Send alerts immediately to all registered farmers
                if alerts:
                    threading.Thread(
                        target=dispatch_alerts_to_all_farmers,
                        args=(alerts,), daemon=True
                    ).start()
                print(f"📡 Firebase data: temp={reading['temperature']} pH={reading['ph']}")
            except Exception as e:
                print(f"Firebase listener error: {e}")

        # Start listening (this blocks, so it runs in its own thread)
        ref.listen(on_sensor_change)
    except Exception as e:
        print(f"Firebase listener failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# ── DATABASE ──────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
DB_PATH = os.path.join(CWD, "urja_nidhi.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        -- Sensor readings from IoT / Firebase
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            digester_id TEXT DEFAULT 'DIG001',
            temperature REAL, ph REAL, pressure REAL, gas_flow REAL,
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
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            param     TEXT UNIQUE NOT NULL,
            label     TEXT NOT NULL,
            unit      TEXT NOT NULL,
            low_warn  REAL,
            low_crit  REAL,
            high_warn REAL,
            high_crit REAL,
            enabled   INTEGER DEFAULT 1,
            updated_at TEXT
        );
    """)
    conn.commit(); conn.close()

init_db()

# ── Seed default thresholds (only if table is empty) ─────────────────────────
def seed_thresholds():
    conn = get_db()
    existing = conn.execute("SELECT COUNT(*) FROM thresholds").fetchone()[0]
    if existing == 0:
        defaults = [
            ("temperature",          "Temperature",   "°C",  30.0, None, 40.0,  42.0),
            ("ph",                   "pH Level",      "",     6.8,  6.5,  7.5,   7.8),
            ("pressure",             "Pressure",      "kPa", 80.0, None, 130.0, 150.0),
            ("gas_flow",             "Gas Flow",      "L/h",  1.0,  0.5,  None,  None),
            ("methane_concentration","Methane",       "%",   50.0, 40.0,  75.0,  80.0),
            ("ambient_temperature",  "Ambient Temp",  "°C",  15.0, None,  40.0,  45.0),
            ("ambient_humidity",     "Humidity",      "%",   30.0, None,  85.0,  95.0),
        ]
        for (param, label, unit, lw, lc, hw, hc) in defaults:
            conn.execute("""INSERT OR IGNORE INTO thresholds
                (param,label,unit,low_warn,low_crit,high_warn,high_crit,enabled,updated_at)
                VALUES (?,?,?,?,?,?,?,1,?)""",
                (param, label, unit, lw, lc, hw, hc, datetime.now().isoformat()))
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
    """Return latest live sensor and best-available NPK values."""
    conn = get_db()
    sensor = conn.execute("""SELECT temperature, ph, pressure, gas_flow, timestamp
        FROM sensor_readings ORDER BY id DESC LIMIT 1""").fetchone()
    pred = conn.execute("""SELECT nitrogen_pct, nitrogen_level, timestamp
        FROM prediction_results ORDER BY id DESC LIMIT 1""").fetchone()
    conn.close()

    sensor_d = dict(sensor) if sensor else {}
    if pred and pred["nitrogen_pct"] is not None:
        n_est = float(pred["nitrogen_pct"]) * 10000
    else:
        n_est = 2000.0

    return {
        "temperature": float(sensor_d.get("temperature", 30.0)),
        "ph": float(sensor_d.get("ph", 7.0)),
        "pressure": float(sensor_d.get("pressure", 100.0)),
        "gas_flow": float(sensor_d.get("gas_flow", 3.0)),
        "n": float(n_est),
        "p": 500.0,
        "k": 1200.0,
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
def check_alerts(sensor: dict, ts: str = None) -> list:
    """
    Compare sensor reading against DB thresholds.
    Returns list of alert dicts with full details for notification.
    Each alert contains: param, label, unit, value, low/high limits, severity, message, timestamp.
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
                msg = (f"{label} is too low: {val}{unit} "
                       f"(limit: ≥ {limit_used}{unit})")
            else:
                limit_used = high_crit if severity == "critical" else high_warn
                msg = (f"{label} is too high: {val}{unit} "
                       f"(limit: ≤ {limit_used}{unit})")

            alerts.append({
                "param":      param,
                "label":      label,
                "unit":       unit,
                "value":      val,
                "direction":  direction,
                "severity":   severity,
                "message":    msg,
                "low_warn":   low_warn,
                "low_crit":   low_crit,
                "high_warn":  high_warn,
                "high_crit":  high_crit,
                "timestamp":  ts,
            })
    return alerts


def build_alert_notification(alerts: list, farmer_name: str = "") -> tuple:
    """
    Build a full email body and short SMS string from a list of alert dicts.
    Returns (email_subject, email_body, sms_text).
    """
    ts    = datetime.now().strftime("%d %b %Y, %I:%M %p")
    crit  = [a for a in alerts if a["severity"] == "critical"]
    warn  = [a for a in alerts if a["severity"] == "warning"]

    subject = ("🚨 CRITICAL Digester Alert" if crit
               else "⚠️ Digester Warning") + f" — {ts}"

    lines = [
        f"🌿 Urja Nidhi — Digester Alert",
        f"Farmer: {farmer_name}",
        f"Time  : {ts}",
        "─" * 45,
        "",
    ]
    for a in alerts:
        icon = "🚨" if a["severity"] == "critical" else "⚠️"
        lines.append(f"{icon}  {a['label']}")
        lines.append(f"    Current value : {a['value']} {a['unit']}")
        if a["low_crit"]  is not None: lines.append(f"    Critical low  : {a['low_crit']} {a['unit']}")
        if a["low_warn"]  is not None: lines.append(f"    Warning low   : {a['low_warn']} {a['unit']}")
        if a["high_warn"] is not None: lines.append(f"    Warning high  : {a['high_warn']} {a['unit']}")
        if a["high_crit"] is not None: lines.append(f"    Critical high : {a['high_crit']} {a['unit']}")
        lines.append(f"    Status        : {a['severity'].upper()} — {a['message']}")
        lines.append("")

    lines += ["─" * 45,
              "Login to your dashboard for details: http://localhost:5000",
              "— Urja Nidhi Team"]
    email_body = "\n".join(lines)

    # Short SMS — max ~160 chars per alert
    sms_parts = [f"Urja Nidhi Alert {ts}:"]
    for a in alerts:
        sms_parts.append(f"{a['label']}: {a['value']}{a['unit']} ({a['severity'].upper()})")
    sms_text = " | ".join(sms_parts)

    return subject, email_body, sms_text

# ══════════════════════════════════════════════════════════════════════════════
# ── NOTIFICATION DELIVERY ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
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
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=20) as resp:
            result = json.loads(resp.read())
            return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return _rule_based_advisory(question, sensor_context) + f"\n\n(Gemini note: {e})"

def _rule_based_advisory(question: str, ctx: dict = None) -> str:
    q   = question.lower()
    ctx = ctx or {}
    ph  = ctx.get("ph",  7.0)
    tmp = ctx.get("temperature", 35)
    if any(w in q for w in ["ph","acid","alkaline","sour"]):
        if ph < 6.5: return f"Your pH is {ph} — too acidic. Add 100–200g slaked lime (chuna) in water. Check again in 24 hours. Ideal: 6.8–7.2."
        if ph > 7.8: return f"Your pH is {ph} — too high. Reduce waste input by 30% for 2 days. Target 6.8–7.2."
        return "Ideal pH is 6.8–7.2. Test daily with pH strip."
    if any(w in q for w in ["temp","cold","heat","garam","thanda"]):
        if tmp < 30: return f"Temperature is {tmp}°C — too cold. Cover digester with black plastic sheet. Bacteria work best at 35–40°C."
        return "Keep digester at 35–40°C for maximum gas. Insulate in winter."
    if any(w in q for w in ["fertilizer","urea","manure","npk","nitrogen","khad"]):
        return "Digested slurry = excellent fertilizer. 1 m³ biogas → 0.9 kg urea equivalent (saves ₹22/kg). Dilute 1:1 with water before applying."
    if any(w in q for w in ["gas","biogas","production","low","less","kam"]):
        return "Low gas? Check: (1) pH 6.8–7.2, (2) Temp 35–40°C, (3) Feed 50–60 kg daily, (4) No pipe leaks."
    if any(w in q for w in ["save","money","cost","profit","paisa"]):
        return "2 m³ digester with 50 kg dung/day saves ₹800–1100/week. Annual saving: ₹40,000–55,000. Payback in under 6 months."
    if any(w in q for w in ["smell","odour","stink"]):
        return "Bad smell = too much protein or high pH. Reduce poultry input. Check pH — if >7.8 reduce feeding 2 days."
    return "Namaskara! Ask me about biogas production, digester maintenance, manure quality, or crop nutrition."

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
    Register a new farmer with name + (email or phone or both).
    Body: { "name": "Rajesh Kumar", "email": "...", "phone": "+91..." }
    At least one of email/phone is required.
    """
    data  = request.get_json(force=True)
    name  = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower() or None
    phone = (data.get("phone") or "").strip() or None

    if not name:
        return jsonify({"error": "Name is required"}), 400
    if not email:
        return jsonify({"error": "Email is required for website access"}), 400
    if not phone:
        return jsonify({"error": "Phone number is required for notifications"}), 400

    try:
        conn = get_db()
        conn.execute("""INSERT INTO farmers (name,email,phone,created_at)
            VALUES (?,?,?,?)""",
            (name, email, phone, datetime.now().isoformat()))
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
        return jsonify({"error": "Email or phone already registered"}), 409
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ── AUTH: Login ───────────────────────────────────────────────────────────────
@app.route("/api/auth/login", methods=["POST"])
def login():
    """
    Login with email OR phone.
    Body: { "email": "..." }  OR  { "phone": "+91..." }
    Returns a session token.
    """
    data  = request.get_json(force=True)
    email = (data.get("email") or "").strip().lower() or None
    if not email:
        return jsonify({"error": "Email is required for website access"}), 400

    conn = get_db()
    farmer = conn.execute("SELECT * FROM farmers WHERE email=?", (email,)).fetchone()
    conn.close()

    if not farmer:
        return jsonify({"error": "No account found. Please register first."}), 404

    farmer = dict(farmer)
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
        row  = conn.execute("""SELECT temperature, ph, pressure, gas_flow
            FROM sensor_readings ORDER BY id DESC LIMIT 1""").fetchone()
        conn.close()

        if not row:
            return jsonify({"error": "No sensor data yet. Use Live Sensors → Manual Entry or wait for IoT feed."}), 400

        # Build payload from latest sensor + sensible defaults
        data = {
            "waste_quantity": 50, "cn_ratio": 25, "moisture_level": 70,
            "temperature": float(row["temperature"] or 36),
            "ph": float(row["ph"] or 7.0),
            "retention_time": 25, "gas_flow_rate": float(row["gas_flow"] or 3.5),
            "methane_concentration": 60, "ambient_temperature": 28, "ambient_humidity": 70,
            "nitrogen_concentration": 2000, "phosphorus_concentration": 500,
            "potassium_concentration": 1200, "microbial_activity": 1500000,
            "soil_n_requirement": 30, "manure_equivalent_n": 20,
            "external_fertilizer_required": 10, "waste_collection_cost": 60,
            "digester_operating_cost": 40,
            "waste_type": "cow_dung", "pre_treatment": "raw", "crop_type": "chickpea",
            "digester_id": "DIG001",
        }
        feat  = build_feature_df(data)
        raw   = float(model.predict(feat)[0])
        daily = max(0.001, raw)
        ana   = compute_analytics(daily)
        ph    = data["ph"]
        n_conc = data["nitrogen_concentration"]
        comp_label, comp_score = classify_compost(daily, ph)
        alerts = check_alerts({"ph": ph, "temperature": data["temperature"], "gas_flow": data["gas_flow_rate"]})
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
            "source":           "auto",
        }
        conn = get_db()
        conn.execute("""INSERT INTO prediction_results
            (digester_id,daily_biogas,weekly_biogas,nitrogen_level,nitrogen_pct,
             compost_quality,compost_score,money_saved,energy_kwh,urea_equivalent,
             co2_reduction,cooking_hours,timestamp) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            ("DIG001", ana["daily_biogas_m3"], ana["weekly_biogas_m3"],
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
        conn.close()
        if not row:
            return jsonify({"exists": False})
        r = dict(row)
        daily = r.get("daily_biogas") or 0
        r["daily_biogas_m3"] = daily
        r["analytics"] = compute_analytics(daily)
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
        for f in ["temperature", "ph", "pressure", "gas_flow"]:
            if f not in data:
                return jsonify({"error": f"Missing: {f}"}), 400
        ts     = data.get("timestamp", datetime.now().isoformat())
        alerts = check_alerts(data)
        conn   = get_db()
        conn.execute("""INSERT INTO sensor_readings
            (digester_id,temperature,ph,pressure,gas_flow,timestamp) VALUES (?,?,?,?,?,?)""",
            (data.get("digester_id","DIG001"), data["temperature"],
             data["ph"], data["pressure"], data["gas_flow"], ts))
        conn.commit(); conn.close()
        socketio.emit("sensor_update", {**data, "alerts": alerts, "timestamp": ts})
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
        rows = conn.execute("""SELECT temperature,ph,pressure,gas_flow,timestamp
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
        # Alert count
        alerts_count = conn.execute("""
            SELECT COUNT(*) n FROM notifications
            WHERE farmer_id=? AND timestamp > datetime('now', ? || ' days')
        """, (farmer["id"], f"-{days}")).fetchone()
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
        L.append("        URJA NIDHI — DIGESTER REPORT")
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
    ctx    = data.get("sensor_context", {})
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
            "timestamp":   datetime.now().isoformat(),
        }
        alerts = check_alerts(reading)
        try:
            conn = get_db()
            conn.execute("""INSERT INTO sensor_readings
                (digester_id,temperature,ph,pressure,gas_flow,timestamp) VALUES (?,?,?,?,?,?)""",
                (reading["digester_id"], reading["temperature"], reading["ph"],
                 reading["pressure"], reading["gas_flow"], reading["timestamp"]))
            conn.commit(); conn.close()
        except Exception: pass
        socketio.emit("sensor_update", {**reading, "alerts": alerts})
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
