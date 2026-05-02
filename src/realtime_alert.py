# src/realtime_alert.py
"""
Real-time Network Intrusion Detection Alert System.
TV5: Best Model Deployment - Suricata-style Alert Generation.
Simulates real-time traffic analysis and generates alerts for detected attacks.
"""
import joblib
import pandas as pd
import numpy as np
import time
import os
import sys
import argparse
from datetime import datetime

# ANSI Color codes
class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

# Force utf-8 encoding for Windows and enable ANSI colors
if sys.platform == 'win32':
    os.system('') # Kích hoạt hỗ trợ ANSI trên Windows 10+
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.config import MODELS_DIR, DATA_PROCESSED_DIR, SELECTED_FEATURES

# Label mapping for CICIDS2017
LABEL_MAP = {
    0: 'BENIGN',
    1: 'Bot',
    2: 'DDoS',
    3: 'DoS GoldenEye',
    4: 'DoS Hulk',
    5: 'DoS Slowhttptest',
    6: 'DoS slowloris',
    7: 'FTP-Patator',
    8: 'Heartbleed',
    9: 'Infiltration',
    10: 'PortScan',
    11: 'SSH-Patator',
    12: 'Web Attack - Brute Force',
    13: 'Web Attack - Sql Injection',
    14: 'Web Attack - XSS'
}

# Severity levels
SEVERITY_MAP = {
    'BENIGN': 'INFO',
    'Bot': 'HIGH',
    'DDoS': 'CRITICAL',
    'DoS GoldenEye': 'HIGH',
    'DoS Hulk': 'HIGH',
    'DoS Slowhttptest': 'MEDIUM',
    'DoS slowloris': 'MEDIUM',
    'FTP-Patator': 'HIGH',
    'Heartbleed': 'CRITICAL',
    'Infiltration': 'HIGH',
    'PortScan': 'MEDIUM',
    'SSH-Patator': 'HIGH',
    'Web Attack - Brute Force': 'MEDIUM',
    'Web Attack - Sql Injection': 'HIGH',
    'Web Attack - XSS': 'MEDIUM'
}


def load_best_model():
    """Load Random Forest model (best model)."""
    model_path = os.path.join(MODELS_DIR, 'random_forest.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}.\n"
            "Please run: python src/models/random_forest.py"
        )
    model = joblib.load(model_path)
    print(f"[OK] Loaded model: Random Forest (Best Model)")
    return model


def load_scaler():
    """Load StandardScaler for feature preprocessing."""
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"[OK] Loaded scaler")
        return scaler
    else:
        print("[WARN] Scaler not found, using raw features")
        return None


def preprocess_single_flow(row, scaler):
    """Preprocess a single network flow row."""
    features = []
    for feature in SELECTED_FEATURES:
        if feature in row:
            try:
                val = float(row[feature])
                if np.isnan(val) or np.isinf(val):
                    val = 0.0
                features.append(val)
            except (ValueError, TypeError):
                features.append(0.0)
        else:
            features.append(0.0)
    
    if scaler:
        features = scaler.transform([features])[0]
    
    return np.array(features).reshape(1, -1)


def get_flow_info(row):
    """Extract key info from flow row for alert display."""
    info = {}
    for col in ['Destination Port', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts']:
        if col in row and pd.notna(row.get(col)):
            info[col] = row[col]
    return info


def generate_suricata_alert(prediction, flow_info, confidence=0.95):
    """Generate Suricata-style alert log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    label = LABEL_MAP.get(prediction, f"Unknown({prediction})")
    severity = SEVERITY_MAP.get(label, 'MEDIUM')
    
    # Build alert
    alert = []
    alert.append("")
    alert.append(Color.RED + "=" * 70 + Color.END)
    alert.append(f"{Color.RED}{Color.BOLD}[ALERT] {timestamp}{Color.END}")
    alert.append(Color.RED + "=" * 70 + Color.END)
    alert.append(f"{Color.RED}{Color.BOLD}[**] Intrusion Detected: {label} [**]{Color.END}")
    alert.append(f"{Color.CYAN}[Classification: Network Intrusion Detection]{Color.END}")
    alert.append(f"{Color.YELLOW}[Priority: {severity}]{Color.END}")
    alert.append("")
    
    if flow_info:
        alert.append("Flow Information:")
        for key, val in flow_info.items():
            alert.append(f"  {key}: {val}")
        alert.append("")
    
    alert.append(f"  Prediction: {Color.BOLD}{label}{Color.END}")
    alert.append(f"  Confidence: {Color.YELLOW}{confidence:.2%}{Color.END}")
    alert.append(f"  Action: {Color.RED}ALERT - Logging to console{Color.END}")
    alert.append(Color.RED + "=" * 70 + Color.END)
    alert.append("")
    
    return "\n".join(alert)


def simulate_realtime_traffic(csv_path=None, delay=0.5, max_flows=None):
    """Simulate real-time traffic by reading flows from CSV one by one."""
    model = load_best_model()
    scaler = load_scaler()
    
    if csv_path is None:
        csv_path = os.path.join(DATA_PROCESSED_DIR, 'final_data.csv')
    
    if not os.path.exists(csv_path):
        print(f"[ERROR] Test file not found: {csv_path}")
        print("Please provide a valid CSV file path or run balancing.py first.")
        return
    
    df = pd.read_csv(csv_path)
    total_flows = len(df)
    
    if max_flows:
        df = df.head(max_flows)
    
    print("")
    print("=" * 70)
    print(" REAL-TIME NETWORK INTRUSION DETECTION SYSTEM (NIDS)")
    print("=" * 70)
    print(f"  Model: Random Forest Classifier (Best Model)")
    print(f"  Dataset: {os.path.basename(csv_path)}")
    print(f"  Total flows: {len(df):,}")
    print(f"  Delay between flows: {delay}s")
    print(f"  Features: {len(SELECTED_FEATURES)}")
    print("-" * 70)
    print(f"  Legend:")
    print(f"    {Color.RED}[ALERT] = Attack detected{Color.END}")
    print(f"    {Color.GREEN}[OK]    = Normal traffic (BENIGN){Color.END}")
    print("-" * 70)
    print(f"{Color.BOLD}Press Ctrl+C to stop{Color.END}\n")
    
    stats = {'total': 0, 'benign': 0, 'attack': 0, 'alerts': 0}
    attack_breakdown = {}
    
    try:
        for idx, (_, row) in enumerate(df.iterrows()):
            stats['total'] += 1
            
            # Preprocess and predict
            X = preprocess_single_flow(row.to_dict(), scaler)
            prediction = model.predict(X)[0]
            
            # Get prediction probability if available
            confidence = 0.95
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(X)[0]
                    confidence = max(proba)
                except:
                    pass
            
            label = LABEL_MAP.get(prediction, f"Class_{prediction}")
            is_attack = label != 'BENIGN'
            
            if is_attack:
                stats['attack'] += 1
                stats['alerts'] += 1
                attack_breakdown[label] = attack_breakdown.get(label, 0) + 1
                
                flow_info = get_flow_info(row)
                alert_msg = generate_suricata_alert(prediction, flow_info, confidence)
                print(alert_msg)
            else:
                stats['benign'] += 1
                if stats['total'] <= 10 or stats['total'] % 50 == 0:
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"{Color.GREEN}[OK]{Color.END} {ts} - Flow #{stats['total']:5d} - BENIGN")
            
            # Progress update every 100 flows
            if stats['total'] % 100 == 0:
                print(f"\n{Color.YELLOW}[STATS]{Color.END} Processed: {stats['total']}/{len(df)} | "
                      f"Benign: {stats['benign']} | Attacks: {stats['attack']} | "
                      f"Alert Rate: {Color.BOLD}{stats['alerts']/stats['total']*100:.1f}%{Color.END}\n")
            
            time.sleep(delay)
            
            # Check max flows limit
            if max_flows and stats['total'] >= max_flows:
                break
    
    except KeyboardInterrupt:
        print("\n")
    
    # Final statistics
    print("\n" + "=" * 70)
    print(" REAL-TIME DETECTION COMPLETE")
    print("=" * 70)
    print(f"\n  Final Statistics:")
    print(f"  - Total flows processed: {stats['total']:,}")
    print(f"  - Normal traffic (BENIGN): {stats['benign']:,} ({stats['benign']/stats['total']*100:.1f}%)")
    print(f"  - Attacks detected: {stats['attack']:,} ({stats['attack']/stats['total']*100:.1f}%)")
    print(f"  - Total alerts generated: {stats['alerts']}")
    
    if attack_breakdown:
        print(f"\n  Attack Breakdown:")
        for attack, count in sorted(attack_breakdown.items(), key=lambda x: -x[1]):
            print(f"    - {attack}: {count} ({count/stats['total']*100:.1f}%)")
    
    print("\n" + "=" * 70)


def batch_mode(csv_path=None, sample_size=500):
    """Quick batch detection mode for demonstration."""
    model = load_best_model()
    scaler = load_scaler()
    
    if csv_path is None:
        csv_path = os.path.join(DATA_PROCESSED_DIR, 'final_data.csv')
    
    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    print(f"\n[INFO] Batch mode: processing {len(df)} flows...")
    
    X_list = []
    for _, row in df.iterrows():
        X_list.append(preprocess_single_flow(row.to_dict(), scaler)[0])
    
    X_batch = np.array(X_list)
    predictions = model.predict(X_batch)
    
    # Get probabilities
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_batch)
        confidences = probabilities.max(axis=1)
    else:
        confidences = [0.95] * len(predictions)
    
    # Count results
    unique, counts = np.unique(predictions, return_counts=True)
    
    print("\n" + "=" * 70)
    print(" BATCH DETECTION RESULTS")
    print("=" * 70)
    print(f"\n  Processed: {len(df)} flows\n")
    
    for label_id, count in zip(unique, counts):
        label = LABEL_MAP.get(label_id, f"Class_{label_id}")
        pct = count / len(predictions) * 100
        if label != 'BENIGN':
            symbol = f"{Color.RED}[ALERT]{Color.END}"
        else:
            symbol = f"{Color.GREEN}[OK]{Color.END}"
        print(f"  {symbol} {label}: {count} ({pct:.1f}%)")
    
    # Show sample alerts
    alerts_issued = 0
    print("\n" + "-" * 70)
    print(" SAMPLE ALERTS:")
    print("-" * 70)
    
    for i, (pred, conf) in enumerate(zip(predictions, confidences)):
        label = LABEL_MAP.get(pred, f"Class_{pred}")
        if label != 'BENIGN' and alerts_issued < 5:
            print(f"\n{Color.RED}{Color.BOLD}[ALERT] Intrusion Detected: {label}{Color.END}")
            print(f"        Confidence: {Color.YELLOW}{conf:.2%}{Color.END}")
            print(f"        Sample index: {i}")
            alerts_issued += 1
    
    if alerts_issued == 0:
        print("\n  No attacks detected in this sample.")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Real-time Network Intrusion Detection')
    parser.add_argument('--csv', type=str, default=None,
                       help='Path to CSV file for simulation')
    parser.add_argument('--batch', action='store_true',
                       help='Run in batch mode (faster, for demonstration)')
    parser.add_argument('--delay', type=float, default=0.3,
                       help='Delay between flows in seconds (default: 0.3)')
    parser.add_argument('--max', type=int, default=None,
                       help='Maximum number of flows to process')
    parser.add_argument('--sample', type=int, default=500,
                       help='Sample size for batch mode (default: 500)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print(" REAL-TIME NETWORK INTRUSION DETECTION SYSTEM")
    print(" Based on CICIDS2017 Dataset")
    print("=" * 70)
    
    if args.batch:
        batch_mode(csv_path=args.csv, sample_size=args.sample)
    else:
        simulate_realtime_traffic(
            csv_path=args.csv,
            delay=args.delay,
            max_flows=args.max
        )


if __name__ == "__main__":
    main()
