from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)

# Load model
model_path = 'model/best_model.pkl'
model = joblib.load(model_path)

# Setup path CSV
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)
CSV_PATH = os.path.join(DATA_DIR, 'user_predictions.csv')

# Halaman
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediksi')
def predic_page():
    return render_template('predic.html')

@app.route('/visualisasi')
def visualisasi_page():
    return render_template('visualisasi.html')

@app.route('/tentang')
def tentang_page():
    return render_template('tentang.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Urutan kolom yang sesuai dengan pelatihan model
        feature_order = [
            'usia_tua',
            'jenis_kelamin_wanita',
            'merokok_pasif',
            'bekerja_ya',
            'rumah_tangga_ya',
            'aktivitas_begadang_ya',
            'aktivitas_olahraga_sering',
            'asuransi_tidak',
            'penyakit_bawaan_tidak'
        ]

        # Ambil nilai sesuai urutan kolom
        input_values = [data[feature] for feature in feature_order]

        # Prediksi
        input_array = np.array(input_values).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0][1]

        # Simpan hasil prediksi
        row = {**data, "prediction": int(prediction), "probability": float(probability), "timestamp": datetime.now()}
        df_row = pd.DataFrame([row])

        if os.path.exists(CSV_PATH):
            df_row.to_csv(CSV_PATH, mode='a', header=False, index=False)
        else:
            df_row.to_csv(CSV_PATH, mode='w', header=True, index=False)

        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability)
        })

    except Exception as e:
        print("ERROR:", e)  # Debugging lebih mudah
        return jsonify({"error": str(e)})

# Endpoint visualisasi pie chart
@app.route('/visualisasi-data')
def visualisasi_data():
    try:
        df = pd.read_csv(CSV_PATH)
        total_berisiko = df['prediction'].sum()
        total_tidak_berisiko = len(df) - total_berisiko

        return jsonify({
            "berisiko": int(total_berisiko),
            "tidak_berisiko": int(total_tidak_berisiko)
        })
    except Exception as e:
        return jsonify({"error": str(e)})

# Endpoint visualisasi bar chart usia
@app.route('/visualisasi-usia')
def visualisasi_usia():
    try:
        df = pd.read_csv(CSV_PATH)

        # Pastikan kolom usia_tua dan prediction ada
        if 'usia_tua' in df.columns and 'prediction' in df.columns:
            usia_prediksi = df.groupby(['usia_tua', 'prediction']).size().unstack(fill_value=0)
            usia_prediksi = usia_prediksi.rename(index={0: 'Muda', 1: 'Tua'})

            data = {
                "labels": usia_prediksi.index.tolist(),
                "berisiko": usia_prediksi[1].tolist() if 1 in usia_prediksi.columns else [0] * len(usia_prediksi),
                "tidak_berisiko": usia_prediksi[0].tolist() if 0 in usia_prediksi.columns else [0] * len(usia_prediksi)
            }
            return jsonify(data)
        else:
            return jsonify({"error": "Kolom usia_tua atau prediction tidak ditemukan."})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/riwayat-prediksi')
def riwayat_prediksi():
    try:
        df = pd.read_csv(CSV_PATH)

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

        kolom_urutan = [
            'usia_tua', 'jenis_kelamin_wanita', 'merokok_pasif', 'bekerja_ya',
            'rumah_tangga_ya', 'aktivitas_begadang_ya', 'aktivitas_olahraga_sering',
            'asuransi_tidak', 'penyakit_bawaan_tidak', 'prediction', 'probability', 'timestamp'
        ]

        kolom_ada = [col for col in kolom_urutan if col in df.columns]
        df = df[kolom_ada]

         # Tambahkan kolom 'no' secara otomatis
        df.insert(0, 'no', range(1, len(df) + 1))

        # Pastikan hanya ambil kolom yang tersedia
        df = df[[col for col in kolom_urutan if col in df.columns]]

        return jsonify(df.to_dict(orient='records'))

    except Exception as e:
        return jsonify({"error": str(e)})

    
@app.route('/visualisasi-kategori/<kolom>')
def visualisasi_kategori(kolom):
    try:
        df = pd.read_csv(CSV_PATH)

        if kolom not in df.columns:
            return jsonify({"error": f"Kolom '{kolom}' tidak ditemukan dalam data."})

        # Mapping label deskriptif
        mapping_dict = {
            'usia_tua': {0: 'Muda', 1: 'Tua'},
            'jenis_kelamin_wanita': {0: 'Pria', 1: 'Wanita'},
            'merokok_pasif': {0: 'Tidak', 1: 'Ya'},
            'bekerja_ya': {0: 'Tidak', 1: 'Ya'},
            'rumah_tangga_ya': {0: 'Tidak', 1: 'Ya'},
            'aktivitas_begadang_ya': {0: 'Tidak', 1: 'Ya'},
            'aktivitas_olahraga_sering': {0: 'Jarang', 1: 'Sering'},
            'asuransi_tidak': {0: 'Ya', 1: 'Tidak'},
            'penyakit_bawaan_tidak': {0: 'Ya', 1: 'Tidak'}
        }

        # Pastikan kolom prediction ada
        if 'prediction' not in df.columns:
            return jsonify({"error": "Kolom prediction tidak ditemukan dalam data."})

        # Terapkan mapping jika kolom cocok
        if kolom in mapping_dict:
            df[kolom] = df[kolom].map(mapping_dict[kolom])

        grouped = df.groupby([kolom, 'prediction']).size().unstack(fill_value=0)

        # Ambil data
        labels = grouped.index.astype(str).tolist()
        berisiko = grouped[1].tolist() if 1 in grouped.columns else [0] * len(labels)
        tidak_berisiko = grouped[0].tolist() if 0 in grouped.columns else [0] * len(labels)

        return jsonify({
            "labels": labels,
            "berisiko": berisiko,
            "tidak_berisiko": tidak_berisiko
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# Jalankan Flask
if __name__ == '__main__':
    app.run(debug=True)
