import streamlit as st
import librosa
import numpy as np
import pickle
import scipy.signal
import pandas as pd
import os
from datetime import datetime
from tensorflow.keras.models import load_model

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Speech Emotion AI", page_icon="🎙️", layout="centered")
st.title("🎙️ Deteksi Emosi Suara AI")
st.write("Upload rekaman suara, dan AI akan mendeteksi emosi dari suara tersebut.")

# --- FUNGSI MEMUAT MODEL ---
@st.cache_resource
def load_ai_components():
    model = load_model('model_multitask.h5')
    with open('le_emosi.pkl', 'rb') as f:
        le_emo = pickle.load(f)
    with open('le_intensitas.pkl', 'rb') as f:
        le_int = pickle.load(f)
    return model, le_emo, le_int

model, le_emo, le_int = load_ai_components()

# --- FUNGSI ANALISIS TIMELINE ---
def deteksi_timeline_emosi(audio_data, sample_rate, model, le_emo, le_int, window_size=2.0, hop_size=0.5):
    durasi_total = librosa.get_duration(y=audio_data, sr=sample_rate)
    samples_per_window = int(window_size * sample_rate)
    samples_per_hop = int(hop_size * sample_rate)
    
    prediksi_mentah_emosi = []
    prediksi_mentah_intensitas = []
    waktu_frame = []
    
    for start_idx in range(0, len(audio_data) - samples_per_window + 1, samples_per_hop):
        potongan_audio = audio_data[start_idx : start_idx + samples_per_window]
        mfccs = librosa.feature.mfcc(y=potongan_audio, sr=sample_rate, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        fitur = np.expand_dims(mfccs_mean, axis=0)
        fitur = np.expand_dims(fitur, axis=2)
        
        pred_emo_prob, pred_int_prob = model.predict(fitur, verbose=0)
        prediksi_mentah_emosi.append(np.argmax(pred_emo_prob, axis=1)[0])
        prediksi_mentah_intensitas.append(np.argmax(pred_int_prob, axis=1)[0])
        waktu_frame.append(start_idx / sample_rate)

    if not prediksi_mentah_emosi:
         return []

    kernel_size = 5 
    if len(prediksi_mentah_emosi) >= kernel_size:
        prediksi_halus_emosi = scipy.signal.medfilt(prediksi_mentah_emosi, kernel_size)
        prediksi_halus_intensitas = scipy.signal.medfilt(prediksi_mentah_intensitas, kernel_size)
    else:
        prediksi_halus_emosi = np.array(prediksi_mentah_emosi)
        prediksi_halus_intensitas = np.array(prediksi_mentah_intensitas)

    timeline = []
    emosi_sebelumnya = None
    intensitas_sebelumnya = None
    waktu_mulai_emosi = 0.0
    
    for i in range(len(waktu_frame)):
        emo_sekarang = le_emo.inverse_transform([int(prediksi_halus_emosi[i])])[0].upper()
        int_sekarang = le_int.inverse_transform([int(prediksi_halus_intensitas[i])])[0].upper()
        
        if emo_sekarang != emosi_sebelumnya or int_sekarang != intensitas_sebelumnya:
            if emosi_sebelumnya is not None:
                timeline.append({'start': waktu_mulai_emosi, 'end': waktu_frame[i], 'emosi': emosi_sebelumnya, 'intensitas': intensitas_sebelumnya})
            emosi_sebelumnya = emo_sekarang
            intensitas_sebelumnya = int_sekarang
            waktu_mulai_emosi = waktu_frame[i]
            
    if emosi_sebelumnya is not None:
        timeline.append({'start': waktu_mulai_emosi, 'end': durasi_total, 'emosi': emosi_sebelumnya, 'intensitas': intensitas_sebelumnya})
        
    return timeline

# --- FUNGSI MENYIMPAN FEEDBACK ---
def simpan_feedback(nama_file, emosi_benar, intensitas_benar):
    file_csv = 'database_pembelajaran_baru.csv'
    waktu_sekarang = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_baru = pd.DataFrame([{
        'Waktu_Input': waktu_sekarang,
        'Nama_File': nama_file,
        'Label_Emosi_Koreksi': emosi_benar,
        'Label_Intensitas_Koreksi': intensitas_benar
    }])
    if os.path.exists(file_csv):
        data_baru.to_csv(file_csv, mode='a', header=False, index=False)
    else:
        data_baru.to_csv(file_csv, mode='w', header=True, index=False)

# --- UI: AREA UPLOAD ---
st.write("---")
uploaded_file = st.file_uploader("📂 Pilih file audio (.wav)", type=['wav'], label_visibility="collapsed")

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("🧠 Analisis Audio Sekarang", use_container_width=True, type="primary"):
        with st.spinner("Memproses audio..."):
            try:
                audio_data, sample_rate = librosa.load(uploaded_file, sr=None)
                durasi_total = librosa.get_duration(y=audio_data, sr=sample_rate)
                
                # PREDIKSI UTAMA
                mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
                mfccs_mean = np.mean(mfccs.T, axis=0)
                fitur = np.expand_dims(np.expand_dims(mfccs_mean, axis=0), axis=2)
                
                prediksi_emosi, prediksi_intensitas = model.predict(fitur, verbose=0)
                
                st.session_state.hasil_emosi = le_emo.inverse_transform([np.argmax(prediksi_emosi)])[0]
                yakin_emosi = np.max(prediksi_emosi) * 100
                st.session_state.hasil_int = le_int.inverse_transform([np.argmax(prediksi_intensitas)])[0]
                yakin_int = np.max(prediksi_intensitas) * 100
                
                st.session_state.audio_data = audio_data
                st.session_state.sample_rate = sample_rate
                st.session_state.durasi_total = durasi_total
                st.session_state.analisis_selesai = True
                
            except Exception as e:
                st.error(f"Gagal memproses file: {e}")

# --- UI: AREA HASIL ANALISIS ---
if st.session_state.get('analisis_selesai', False):
    st.write("### Hasil Analisis")
    
    # 1. TAMPILAN DEFAULT (Hanya kesimpulan utama, bersih dan jelas)
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="🎭 Emosi Utama", value=st.session_state.hasil_emosi.upper())
    with col2:
        st.metric(label="⚡ Intensitas", value=st.session_state.hasil_int.upper())
    
    st.write("---")
    
    # 2. FITUR ADVANCED: TIMELINE (Disembunyikan dalam Expander)
    with st.expander("📊 Lihat Detail Timeline Perubahan Emosi"):
        if st.session_state.durasi_total >= 3.0:
            st.caption("AI membedah perubahan emosi dari detik ke detik.")
            hasil_timeline = deteksi_timeline_emosi(st.session_state.audio_data, st.session_state.sample_rate, model, le_emo, le_int)
            if hasil_timeline:
                for kejadian in hasil_timeline:
                    st.markdown(f"**`{kejadian['start']:04.1f}s - {kejadian['end']:04.1f}s`** &nbsp; ➔ &nbsp; **{kejadian['emosi']}** ({kejadian['intensitas']})")
        else:
            st.info("Audio terlalu pendek (minimal 3 detik) untuk memunculkan grafik timeline.")

    # 3. FITUR ADVANCED: FEEDBACK (Disembunyikan dalam Expander)
    with st.expander("🛠️ AI Salah Tebak? Berikan Koreksi"):
        st.caption("Bantu kami mengumpulkan data logat baru agar AI semakin cerdas.")
        with st.form("form_koreksi", border=False):
            koreksi_emo = st.selectbox("Emosi Seharusnya:", le_emo.classes_)
            koreksi_int = st.selectbox("Intensitas Seharusnya:", le_int.classes_)
            if st.form_submit_button("Simpan Koreksi"):
                simpan_feedback(uploaded_file.name, koreksi_emo, koreksi_int)
                st.success("Koreksi berhasil disimpan ke database!")