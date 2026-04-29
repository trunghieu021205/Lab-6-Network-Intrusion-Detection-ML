"""
TV5: Real-time Network Intrusion Detection Alert System
Network Intrusion Detection System

Mo phong he thong canh bao xam nhap mang truc tuyen
- Load Random Forest model (best model)
- Doc tung dong tu CSV de mo phong realtime
- Hien thi canh bao khi phat hien tan cong
- Co che batch processing de xu ly nhieu flow cung luc

Cach su dung:
    python src/realtime_alert.py
    python src/realtime_alert.py --csv data/processed/final_data.csv --delay 0.1 --limit 50
"""
import os
import sys
import time
import argparse
import joblib
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.config import MODELS_DIR, DATA_PROCESSED_DIR, SELECTED_FEATURES


# Mau label mapping (dua tren CIC-IDS2017)
# Label = 0: BENIGN (luong binh thuong)
# Label = 1: ATTACK (tan cong)
LABEL_MAP = {
    0: 'BENIGN',
    1: 'ATTACK'
}


def load_model():
    """Load Random Forest model (best model)"""
    model_path = os.path.join(MODELS_DIR, 'random_forest.pkl')
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        print("Vui long train Random Forest truoc bang: python src/models/random_forest.py")
        sys.exit(1)
    model = joblib.load(model_path)
    print(f"[OK] Loaded model: Random Forest (F1-Score = 0.9811)")
    return model


def load_scaler():
    """Load StandardScaler da duoc TV2 luu"""
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"[OK] Loaded scaler for preprocessing")
        return scaler
    return None


def preprocess_row(row_dict):
    """Tien xu ly 1 dong du lieu"""
    features = []
    for feature in SELECTED_FEATURES:
        if feature in row_dict:
            val = row_dict[feature]
            if pd.isna(val) or np.isinf(val):
                features.append(0.0)
            else:
                features.append(float(val))
        else:
            features.append(0.0)
    return features


def simulate_realtime(csv_path, model, scaler, delay=0.5, limit=None, verbose=False):
    """
    Mo phong real-time intrusion detection

    Args:
        csv_path: Duong dan file CSV chua du lieu can kiem tra
        model: Da train san
        scaler: Scaler de tien xu ly (co the la None)
        delay: Thoi gian cho giua cac flow (giay)
        limit: So luong flow toi da can xu ly (None = tat ca)
        verbose: Hien thi tat ca hay chi hien thi canh bao
    """
    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)

    total = len(df)
    print("\n" + "=" * 60)
    print("REAL-TIME NETWORK INTRUSION DETECTION SYSTEM")
    print("=" * 60)
    print(f"Model      : Random Forest (Best Model)")
    print(f"Data file  : {os.path.basename(csv_path)}")
    print(f"Total flows: {total}")
    print(f"Delay      : {delay}s per flow")
    print(f"Mode       : {'Verbose' if verbose else 'Alert-only'}")
    print("=" * 60)
    print("Dang cho khoi dong...")
    time.sleep(1)
    print("\nBat dau theo doi...\n")

    alert_count = 0
    normal_count = 0
    attack_classes = set()

    start_time = time.time()

    try:
        for idx, row in df.iterrows():
            # Tien xu ly
            features = preprocess_row(row.to_dict())
            if scaler:
                features = scaler.transform([features])[0]

            # Du doan
            prediction = model.predict([features])[0]
            proba = model.predict_proba([features])[0]
            confidence = max(proba) * 100

            # Xac dinh loai
            is_attack = prediction != 0
            label_str = LABEL_MAP.get(prediction, f"Class_{prediction}")

            timestamp = time.strftime("%H:%M:%S")

            if is_attack:
                alert_count += 1
                attack_classes.add(label_str)
                print(f"[ALERT] {timestamp} | Flow #{idx:4d} | {label_str:10s} | Conf: {confidence:5.1f}%")
                if verbose:
                    print(f"        Features preview: {features[:3]}...")
                print("-" * 60)
            else:
                normal_count += 1
                if not verbose:
                    # Hien thi tien do nhe
                    if (idx + 1) % 50 == 0 or idx == 0:
                        pct = (idx + 1) / total * 100
                        print(f"  [.] Da xu ly {idx+1}/{total} flows ({pct:.1f}%) | Canh bao: {alert_count}")

            # Real-time delay
            if delay > 0:
                time.sleep(delay)

    except KeyboardInterrupt:
        print("\n\n[STOP] Da dung boi nguoi dung (Ctrl+C)")

    elapsed = time.time() - start_time

    # Thong ke cuoi cung
    print("\n" + "=" * 60)
    print("KET QUA THEO DOI")
    print("=" * 60)
    print(f"  Tong flow da xu ly  : {total}")
    print(f"  Luong binh thuong   : {normal_count} ({normal_count/total*100:.1f}%)")
    print(f"  Luong tan cong      : {alert_count} ({alert_count/total*100:.1f}%)")
    print(f"  Thoi gian chay       : {elapsed:.1f}s")
    print(f"  Toc do xu ly         : {total/elapsed:.1f} flows/s")
    if attack_classes:
        print(f"  Loai tan cong phat hien: {', '.join(attack_classes)}")
    print("=" * 60)

    if alert_count > 0:
        print("\n[CANH BAO] Phat hien {0} luong xam nhap! Can kiem tra ngay.".format(alert_count))
    else:
        print("\n[OK] Khong phat hien xam nhap nao trong {0} luong.".format(total))


def batch_processing(csv_path, model, scaler, batch_size=100):
    """
    Che do xu ly hang loat (nhanh hon, phu hop khi can kiem tra nhieu data)

    Args:
        csv_path: Duong dan file CSV
        model: Da train san
        scaler: Scaler (co the la None)
        batch_size: Kich thuoc moi batch
    """
    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    total = len(df)
    total_batches = (total + batch_size - 1) // batch_size

    print("\n" + "=" * 60)
    print("BATCH PROCESSING MODE")
    print("=" * 60)
    print(f"Total flows : {total}")
    print(f"Batch size  : {batch_size}")
    print(f"Total batches: {total_batches}")
    print("=" * 60 + "\n")

    all_predictions = []
    start_time = time.time()

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch_df = df.iloc[start:end]

        # Tien xu ly batch
        features_list = [preprocess_row(row.to_dict()) for _, row in batch_df.iterrows()]
        features_array = np.array(features_list)

        if scaler:
            features_array = scaler.transform(features_array)

        # Du doan batch
        predictions = model.predict(features_array)
        all_predictions.extend(predictions)

        # Thong ke batch
        batch_alerts = sum(predictions != 0)
        print(f"  Batch {batch_idx+1}/{total_batches}: xu ly {end}/{total} flows | "
              f"Alerts: {batch_alerts}/{len(predictions)}")

    elapsed = time.time() - start_time

    # Tong hop ket qua
    predictions = np.array(all_predictions)
    alert_count = int(sum(predictions != 0))
    normal_count = int(total - alert_count)

    print("\n" + "=" * 60)
    print("KET QUA BATCH PROCESSING")
    print("=" * 60)
    print(f"  Tong flow da xu ly  : {total}")
    print(f"  Luong binh thuong   : {normal_count} ({normal_count/total*100:.1f}%)")
    print(f"  Luong tan cong      : {alert_count} ({alert_count/total*100:.1f}%)")
    print(f"  Thoi gian xu ly     : {elapsed:.2f}s")
    print(f"  Toc do xu ly        : {total/elapsed:.1f} flows/s")
    print("=" * 60)

    if alert_count > 0:
        print(f"\n[CANH BAO] Phat hien {alert_count} luong xam nhap! Can kiem tra ngay.")
    else:
        print("\n[OK] Tat ca luong deu binh thuong.")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time Network Intrusion Detection Alert System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Vi du su dung:
  python src/realtime_alert.py                          # Mo phong realtime, 0.5s/dong
  python src/realtime_alert.py --delay 0.1              # Nhanh hon
  python src/realtime_alert.py --limit 100              # Chi xu ly 100 dong
  python src/realtime_alert.py --verbose                 # Hien thi moi flow
  python src/realtime_alert.py --batch                   # Xu ly hang loat (nhanh)
  python src/realtime_alert.py --csv data/dummy/dummy_data.csv
        """
    )
    parser.add_argument('--csv', default=None,
                        help='Duong dan file CSV (mac dinh: data/processed/final_data.csv)')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Thoi gian cho giua cac flow, giay (mac dinh: 0.5)')
    parser.add_argument('--limit', type=int, default=None,
                        help='So luong flow toi da xu ly (mac dinh: tat ca)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Hien thi moi luong (ke ca binh thuong)')
    parser.add_argument('--batch', '-b', action='store_true',
                        help='Su dung che do xu ly hang loat (nhanh hon)')

    args = parser.parse_args()

    # Duong dan file mac dinh
    if args.csv is None:
        csv_path = os.path.join(DATA_PROCESSED_DIR, 'final_data.csv')
    else:
        csv_path = args.csv
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), csv_path)

    # Kiem tra file
    if not os.path.exists(csv_path):
        # Thu voi duong dan tuyet doi
        csv_path_alt = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            args.csv if args.csv else 'data/processed/final_data.csv'
        )
        if os.path.exists(csv_path_alt):
            csv_path = csv_path_alt
        else:
            print(f"[ERROR] File not found: {csv_path}")
            print("Vui long kiem tra duong dan file.")
            sys.exit(1)

    # Load model va scaler
    model = load_model()
    scaler = load_scaler()

    # Chay che do phu hop
    if args.batch:
        batch_processing(csv_path, model, scaler, batch_size=100)
    else:
        simulate_realtime(csv_path, model, scaler,
                          delay=args.delay, limit=args.limit, verbose=args.verbose)


if __name__ == "__main__":
    main()
