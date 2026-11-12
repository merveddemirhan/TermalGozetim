"""Rapordaki tüm Python deneylerini (3 tohum x 10 yapılandırma) 2 paralel süreçte çalıştırır.
Kullanım: python calistir.py   (her koşu ~3 dk; sonuçlar sonuclar/ klasörüne)"""
import json, subprocess, sys
from multiprocessing import Pool
DENEYLER = [
    ("final", {}),                                   # referans: köşegen sabit sorumluluk
    ("dinamik", {"sabit_sorumluluk": False}),
    ("ortyas", {"bolge_yasi": "ort", "sabit_sorumluluk": False, "maske": False}),
    ("maskesiz", {"maske": False}),
    ("gamma090", {"gamma": 0.90}),
    ("gamma099", {"gamma": 0.99}),
    ("bekleme06", {"bekleme_cezasi": 0.6}),
    ("kendibolge", {"kendi_bolgem": True}),          # köşegen + yalnız kendi bölgelerini tara
    ("bolgebazli", {"bolge_bazli": True}),            # bölge bazlı sabit sorumluluk
    ("bolgekendi", {"bolge_bazli": True, "kendi_bolgem": True}),
]
def calis(arg):
    etiket, ajan, t = arg
    r = subprocess.run([sys.executable, "deney.py", "egit", "--tohum", str(t), "--etiket", etiket,
                        "--bolum", "5000", "--test", "2000", "--ajan", json.dumps(ajan)], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr[-300:], flush=True)
if __name__ == "__main__":
    with Pool(2) as p:
        p.map(calis, [(e, a, t) for e, a in DENEYLER for t in (1, 2, 3)], chunksize=1)
    print("BITTI", flush=True)
