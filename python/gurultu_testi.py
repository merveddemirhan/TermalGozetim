"""Sensör gürültüsü gürbüzlük testi: gürültüsüz ortamda eğitilmiş Q ajanları (3 tohum) ve
devriye, artan termal gürültü / sahte okuma seviyelerinde değerlendirilir."""
import json, os, pickle, sys
import numpy as np
from ptz_gozetim_env import PTZGozetimEnv
from politikalar import QAjani, Devriye
from deney import degerlendir, KLASOR

SEVIYELER = [("yok", 0.0, 0.0), ("orta", 2.0, 0.005), ("yuksek", 4.0, 0.01)]
TEST = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

sonuc = []
for ad, g, s in SEVIYELER:
    kw = dict(termal_gurultu=g, sahte_okuma=s)
    for t in (1, 2, 3):
        env = PTZGozetimEnv(**kw)
        ajan = QAjani(env)
        ajan.Q = pickle.load(open(os.path.join(KLASOR, f"final_t{t}.pkl"), "rb"))
        d = degerlendir(ajan, TEST, **kw)
        sonuc.append(dict(seviye=ad, gurultu=g, sahte=s, politika="Q", tohum=t, test=d))
        print(f"[{ad}] Q t{t}: yangın/bölüm {d['yangin']:.4f} ödül {d['odul']:.1f} doğrulanan {d['dogrulanan']:.2f}", flush=True)
    d = degerlendir(Devriye, TEST, **kw)
    sonuc.append(dict(seviye=ad, gurultu=g, sahte=s, politika="Devriye", test=d))
    print(f"[{ad}] Devriye: yangın/bölüm {d['yangin']:.4f} ödül {d['odul']:.1f} doğrulanan {d['dogrulanan']:.2f}", flush=True)
json.dump(sonuc, open(os.path.join(KLASOR, "gurultu.json"), "w"), indent=1)
