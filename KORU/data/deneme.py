import pandas as pd
from spacetrack import SpaceTrackClient
import os

# --- AYARLAR ---
# Space-Track.org kullanıcı adı ve şifresi
USER = "mahmutdonali19@gmail.com"
PASS = "MahmutDnl_27109"

# Türk Uyduları NORAD ID Listesi
TURKISH_SATS_IDS = [47306, 49574, 60232, 56214, 41875, 39038]

# Kayıt Klasörü
OUTPUT_PATH = "c:/Users/mhmd/Desktop/hackathon2/data/"

def verileri_ayri_kaydet():
    # Klasör yoksa oluştur
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        
    # Space-Track istemcisini başlat
    st = SpaceTrackClient(identity=USER, password=PASS)
    print("🚀 Space-Track baglantisi kuruldu...")

    try:
        # 1. BÖLÜM: TÜRK UYDULARI
        print("🛰️ Turk uydulari verileri cekiliyor...")
        turk_verileri = st.tle_latest(norad_cat_id=TURKISH_SATS_IDS, ordinal=1, format='json')
        
        turk_liste = []
        for v in turk_verileri:
            turk_liste.append({
                "Obje_Adi": v['OBJECT_NAME'],
                "TLE_Satir_1": v['TLE_LINE1'],
                "TLE_Satir_2": v['TLE_LINE2']
            })
            
        df_turk = pd.DataFrame(turk_liste)
        df_turk.to_csv(os.path.join(OUTPUT_PATH, "turk_uydulari.csv"), index=False, encoding="utf-8")
        print(f"✅ Turk uydulari kaydedildi: {len(df_turk)} obje.")

        # 2. BÖLÜM: UZAY ÇÖPLERİ (DEBRIS) - 2000 ADET LİMİT
        print("🧹 2000 adet cop verisi cekiliyor...")
        cop_verileri = st.tle_latest(object_type='debris', ordinal=1, limit=2000, format='json')
        
        cop_liste = []
        for v in cop_verileri:
            cop_liste.append({
                "Obje_Adi": v['OBJECT_NAME'],
                "TLE_Satir_1": v['TLE_LINE1'],
                "TLE_Satir_2": v['TLE_LINE2']
            })
            
        df_cop = pd.DataFrame(cop_liste)
        df_cop.to_csv(os.path.join(OUTPUT_PATH, "uzay_copleri.csv"), index=False, encoding="utf-8")
        print(f"✅ Uzay copleri kaydedildi: {len(df_cop)} obje.")

        print(f"\n🚀 ISLEM TAMAMLANDI! Dosyalar '{OUTPUT_PATH}' klasorunde.")

    except Exception as e:
        print(f"❌ Hata olustu: {e}")

if __name__ == "__main__":
    verileri_ayri_kaydet()