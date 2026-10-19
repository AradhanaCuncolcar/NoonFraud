"""Central configuration: paths, feature lists, model registry and the Noon colour palette."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

# ----------------------------------------------------------------------------- brand palette
NOON_YELLOW = "#FEEE00"
BLACK = "#0B0B0C"
PANEL = "#17171A"
PANEL_2 = "#1F1F24"
GRID = "#2B2B31"
TEXT = "#F2F2F2"
MUTED = "#9A9AA2"
FRAUD_RED = "#FF5A5F"
SAFE_GREEN = "#2EC4B6"
CYAN = "#3DD9EB"
PINK = "#FF7AB6"
ORANGE = "#FFA94D"
PURPLE = "#B197FC"

# ----------------------------------------------------------------------------- schema
NUMERIC = ["order_amount", "account_age_days", "hour_of_day", "num_orders_last_7d",
           "delivery_distance_km", "prev_refunds"]
BINARY = ["discount_applied", "device_change", "location_mismatch"]
CATEGORICAL = ["payment_method"]
PAYMENTS = ["card", "wallet", "cod", "applepay"]
FEATURES = ["order_amount", "account_age_days", "hour_of_day", "num_orders_last_7d",
            "delivery_distance_km", "discount_applied", "prev_refunds", "device_change",
            "location_mismatch", "payment_method"]
TARGET = "is_fraud"

FEATURE_LABELS = {
    "order_amount": "Order amount (AED)",
    "account_age_days": "Account age (days)",
    "hour_of_day": "Hour of day",
    "num_orders_last_7d": "Orders in last 7 days",
    "delivery_distance_km": "Delivery distance (km)",
    "discount_applied": "Discount applied",
    "prev_refunds": "Previous refunds",
    "device_change": "New / changed device",
    "location_mismatch": "Billing ≠ delivery location",
    "payment_method": "Payment method",
}

# ----------------------------------------------------------------------------- registries
MODELS = {
    "N1": {"label": "N1 · Final model (my legit + Noon GAN fraud)", "file": "N1.joblib"},
    "S2": {"label": "S2 · Phase-1 model (my seed + my GAN fraud)", "file": "S2.joblib"},
    "NoonB": {"label": "Noon B · Noon's benchmark (real + Noon GAN)", "file": "NoonB.joblib"},
}

EVAL_SETS = {
    "referee": {"label": "Noon real referee set (20,000 orders · 65 fraud)", "file": "referee_real_test.csv"},
    "seed_test": {"label": "My synthetic seed test (40,000 orders · 200 fraud)", "file": "seed_test.csv"},
}

FRAUD_SOURCES = {
    "real": {"label": "Real fraud (Noon referee)", "color": NOON_YELLOW},
    "noon_gan": {"label": "Noon GAN fraud", "color": CYAN},
    "my_gan": {"label": "My GAN fraud", "color": PINK},
    "sim": {"label": "GAN Lab simulation", "color": FRAUD_RED},
    "real_legit": {"label": "Real legit (Noon referee)", "color": "#6C6C75"},
}

DEFAULT_THRESHOLD = 0.50

PRESETS = {
    "Night-time takeover (3 am, new device + mismatch)": dict(
        order_amount=47.0, account_age_days=95, hour_of_day=3, num_orders_last_7d=4,
        delivery_distance_km=2.8, payment_method="card", discount_applied=0, prev_refunds=0,
        device_change=1, location_mismatch=1),
    "Daytime refund scammer (7 am, 2 refunds, mismatch)": dict(
        order_amount=127.0, account_age_days=273, hour_of_day=7, num_orders_last_7d=1,
        delivery_distance_km=1.8, payment_method="card", discount_applied=1, prev_refunds=2,
        device_change=1, location_mismatch=1),
    "Ordinary lunch order": dict(
        order_amount=70.0, account_age_days=202, hour_of_day=14, num_orders_last_7d=2,
        delivery_distance_km=1.5, payment_method="card", discount_applied=0, prev_refunds=1,
        device_change=0, location_mismatch=0),
    "New account, 1 am, big basket (no device flags)": dict(
        order_amount=240.0, account_age_days=5, hour_of_day=1, num_orders_last_7d=2,
        delivery_distance_km=6.0, payment_method="wallet", discount_applied=1, prev_refunds=0,
        device_change=0, location_mismatch=0),
}
