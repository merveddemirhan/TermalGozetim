"""Rapor grafikleri: 3 tohumlu öğrenme eğrileri, ablasyon/hiperparametre tablosu, gürültü testi."""
import json, os, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from deney import KLASOR

def yukle(etiket):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(KLASOR, f"{etiket}_t*.json")))]

def egri_ciz(etiketler, ad, basliklar=None):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    renkler = ["#1f4e8c", "#d0742c", "#3a9a5b", "#8a4fa0", "#b03a3a", "#6b6b6b"]
    for e, rk in zip(etiketler, renkler):
        kosular = yukle(e)
        if not kosular: continue
        x = [p["bolum"] for p in kosular[0]["egri"]]
        for a, m in zip(ax, ["odul", "yangin", "dogrulama_sicakligi"]):
            Y = np.array([[p[m] for p in k["egri"]] for k in kosular])
            ort, sd = Y.mean(0), Y.std(0)
            a.plot(x, ort, color=rk, lw=2, label=f"{(basliklar or {}).get(e, e)} (n={len(kosular)})")
            a.fill_between(x, ort - sd, ort + sd, color=rk, alpha=.18)
    for a, t in zip(ax, ["Bölüm ödülü", "Yangın / bölüm", "Doğrulama sıcaklığı (°C)"]):
        a.set_title(t); a.set_xlabel("Eğitim bölümü"); a.grid(alpha=.3)
    ax[1].set_ylim(bottom=0); ax[0].legend(fontsize=8)
    fig.suptitle("Öğrenme eğrileri (100 bölümlük pencere; çizgi = tohum ortalaması, gölge = ±1 std)")
    fig.tight_layout(); fig.savefig(os.path.join(KLASOR, ad), dpi=150); plt.close(fig)

def tablo(etiketler, basliklar):
    satirlar = []
    for e in etiketler:
        k = yukle(e)
        if not k: continue
        T = [x["test"] for x in k]
        f = lambda m: (np.mean([t[m] for t in T]), np.std([t[m] for t in T]))
        y_top = sum(t["yangin_toplam"] for t in T); b_top = sum(t["bolum"] for t in T)
        satirlar.append(dict(deney=basliklar.get(e, e), n=len(k), yangin_1000=1000 * y_top / b_top,
                             yangin_tohum=[round(1000 * t["yangin"], 1) for t in T],
                             odul=f("odul"), dogrulanan=f("dogrulanan"), sicaklik=f("dogrulama_sicakligi"),
                             gorulmeyen=sum(t["yangin_gorulmeyen"] * t["bolum"] for t in T),
                             bilinen=sum(t["yangin_bilinen"] * t["bolum"] for t in T)))
    return satirlar

if __name__ == "__main__":
    B = {"final": "Final (γ=0.95, sabit sorumluluk, en eski hücre, maske)", "dinamik": "Dinamik sorumluluk",
         "ortyas": "İlk tasarım (ortalama yaş, dinamik, maskesiz)", "maskesiz": "Maskesiz",
         "gamma090": "γ = 0.90", "gamma099": "γ = 0.99", "bekleme06": "Bekleme cezası 0.6"}
    egri_ciz(["final"], "egri_final.png", B)
    egri_ciz(["final", "dinamik", "ortyas"], "egri_tasarim.png", B)
    egri_ciz(["final", "gamma090", "gamma099", "bekleme06"], "egri_hiperparametre.png", B)
    S = tablo(list(B), B)
    json.dump(S, open(os.path.join(KLASOR, "ablasyon_tablosu.json"), "w"), indent=1, ensure_ascii=False)
    for s in S:
        print(f"{s['deney'][:45]:45s} n={s['n']} yangın/1000={s['yangin_1000']:6.2f} {s['yangin_tohum']} "
              f"ödül={s['odul'][0]:5.1f}±{s['odul'][1]:.1f} doğr={s['dogrulanan'][0]:.2f} sıc={s['sicaklik'][0]:.1f} "
              f"gör.={s['gorulmeyen']:.0f} bil.={s['bilinen']:.0f}")
