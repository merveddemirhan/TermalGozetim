"""
Deney betiği: eğitim (çok tohumlu), değerlendirme, baseline karşılaştırması,
sensör gürültüsü gürbüzlük testi ve hiperparametre analizi.

Kullanım:
  python deney.py egit   --tohum 1 --bolum 3000 [--param deger ...]
  python deney.py hepsi            (rapordaki tüm deneyler; sonuçlar sonuclar/ klasörüne)
"""
import argparse, json, os, pickle, time
import numpy as np
from ptz_gozetim_env import PTZGozetimEnv
from politikalar import QAjani, Devriye, Rastgele

KLASOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sonuclar")


def bolum_oyna(env, pol, tohum):
    obs, info = env.reset(seed=tohum); pol.sifirla()
    toplam, bitti = 0.0, False
    while not bitti:
        a = pol.eylem(obs)
        obs2, r, term, trunc, info = env.step(a)
        bitti = term or trunc
        pol.gozlemle(obs, a, info, obs2, bitti)
        obs, toplam = obs2, toplam + r
    ist = {k: v for k, v in info.items() if k not in ("kamera_odul", "yanginlar")}
    ist["odul"] = toplam
    return ist


def ozet(liste):
    d = {k: float(np.mean([x[k] for x in liste])) for k in liste[0]}
    d["dogrulama_sicakligi"] = (sum(x["dogrulama_sicaklik_toplam"] for x in liste) /
                                max(1, sum(x["dogrulanan"] for x in liste)))
    d["yangin_toplam"] = int(sum(x["yangin"] for x in liste)); d["bolum"] = len(liste)
    return d


def degerlendir(pol, bolum=1000, tohum0=10**6, **env_kw):
    env = PTZGozetimEnv(**env_kw)
    if isinstance(pol, type): pol = pol(env)
    else: pol.env = env
    if hasattr(pol, "ogren"): pol.ogren = False
    return ozet([bolum_oyna(env, pol, tohum0 + b) for b in range(bolum)])


def egit(tohum=1, bolum=3000, pencere=100, env_kw=None, **ajan_kw):
    env = PTZGozetimEnv(**(env_kw or {}))
    ajan = QAjani(env, tohum=tohum, **ajan_kw)
    egri, tampon = [], []
    for b in range(bolum):
        tampon.append(bolum_oyna(env, ajan, tohum * 10**5 + b))
        if (b + 1) % pencere == 0:
            o = ozet(tampon); tampon = []
            egri.append(dict(bolum=b + 1, karar=ajan.karar, eps=ajan.eps, odul=o["odul"], yangin=o["yangin"],
                             dogrulanan=o["dogrulanan"], dogrulama_sicakligi=o["dogrulama_sicakligi"]))
    return ajan, egri


# ---------------------------------------------------------------------- #
def hepsi(bolum=3000, test=1000, tohumlar=(1, 2, 3)):
    os.makedirs(KLASOR, exist_ok=True)
    sonuc = {}
    t0 = time.time()

    # 1) Final ajan: 3 tohumla eğitim + açgözlü değerlendirme
    sonuc["final"] = []
    for t in tohumlar:
        ajan, egri = egit(t, bolum)
        d = degerlendir(ajan, test)
        sonuc["final"].append(dict(tohum=t, egri=egri, test=d))
        with open(os.path.join(KLASOR, f"qtablo_t{t}.pkl"), "wb") as f: pickle.dump(ajan.Q, f)
        print(f"[final t{t}] yangın/bölüm {d['yangin']:.4f} ödül {d['odul']:.1f} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(sonuc, open(os.path.join(KLASOR, "sonuclar.json"), "w"), indent=1)
    return sonuc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("komut", choices=["egit", "hepsi", "baseline"])
    ap.add_argument("--tohum", type=int, default=1)
    ap.add_argument("--bolum", type=int, default=3000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--etiket", default="")
    ap.add_argument("--ajan", default="{}", help="QAjani parametreleri, JSON")
    ap.add_argument("--ortam", default="{}", help="Ortam parametreleri, JSON (eğitim ve test)")
    ap.add_argument("--test_ortam", default=None, help="Sadece test ortamı parametreleri, JSON")
    a = ap.parse_args()
    os.makedirs(KLASOR, exist_ok=True)
    if a.komut == "egit":
        ajan_kw, env_kw = json.loads(a.ajan), json.loads(a.ortam)
        test_kw = json.loads(a.test_ortam) if a.test_ortam else env_kw
        t0 = time.time()
        ajan, egri = egit(a.tohum, a.bolum, env_kw=env_kw, **ajan_kw)
        d = degerlendir(ajan, a.test, **test_kw)
        ad = f"{a.etiket or 'q'}_t{a.tohum}"
        json.dump(dict(etiket=a.etiket, tohum=a.tohum, ajan=ajan_kw, ortam=env_kw, egri=egri, test=d),
                  open(os.path.join(KLASOR, ad + ".json"), "w"), indent=1)
        with open(os.path.join(KLASOR, ad + ".pkl"), "wb") as f: pickle.dump(ajan.Q, f)
        print(f"[{ad}] {time.time()-t0:.0f}s | yangın/bölüm {d['yangin']:.4f} ({d['yangin_toplam']}) | ödül {d['odul']:.1f} | "
              f"doğrulanan {d['dogrulanan']:.2f} | doğr.sıc {d['dogrulama_sicakligi']:.1f} | "
              f"teşhis B/K/G {d['yangin_bilinen']*d['bolum']:.0f}/{d['yangin_kacirilan']*d['bolum']:.0f}/{d['yangin_gorulmeyen']*d['bolum']:.0f}", flush=True)
    elif a.komut == "baseline":
        env_kw = json.loads(a.ortam)
        for ad, P in [("devriye", Devriye), ("rastgele", Rastgele)]:
            d = degerlendir(P, a.test, **env_kw)
            json.dump(dict(etiket=f"{ad}{a.etiket}", ortam=env_kw, test=d), open(os.path.join(KLASOR, f"{ad}{a.etiket}.json"), "w"), indent=1)
            print(f"[{ad}{a.etiket}] yangın/bölüm {d['yangin']:.4f} | ödül {d['odul']:.1f} | doğrulanan {d['dogrulanan']:.2f} | "
                  f"doğr.sıc {d['dogrulama_sicakligi']:.1f} | elenen {d['yanlis_elenen']:.2f}/{d['yanlis_olusan']:.2f}", flush=True)
    else:
        hepsi(a.bolum, a.test)
