"""Unity öğrenme eğrileri (docs/egriler/*.csv): köşegen ve bölge bazlı sorumluluk, 3 tohum."""
import csv, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

KOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")

def oku(ad):
    with open(os.path.join(KOK, "egriler", ad + ".csv")) as f:
        r = list(csv.DictReader(f, delimiter=";"))
    return {k: np.array([float(x[k]) for x in r]) for k in r[0]}

def duzle(y, w=5):
    return np.convolve(np.pad(y, (w // 2, w - 1 - w // 2), mode="edge"), np.ones(w) / w, mode="valid")

GRUP = [("Köşegen sorumluluk", ["egri_t1", "egri_t2", "egri_t3"]),
        ("Bölge bazlı sorumluluk (final)", ["egri_bolge_t1", "egri_bolge_t2", "egri_bolge_t3"])]
OLC = [("yangin", "Yangın / bölüm"), ("odul", "Bölüm ödülü"),
       ("dogrulama_sicakligi", "Doğrulama sıcaklığı (°C)"), ("cakisma", "Çakışan hücre / adım")]
RENK = ["#2a6fdb", "#e0782b", "#2a9d5c"]

fig, ax = plt.subplots(2, 4, figsize=(17, 7.2), sharex="col")
for i, (baslik, dosyalar) in enumerate(GRUP):
    for d in dosyalar:
        t = int(d[-1]) - 1
        e = oku(d); x = e["toplam_bolum"]
        for j, (k, et) in enumerate(OLC):
            a = ax[i, j]
            a.plot(x, e[k], color=RENK[t], alpha=0.18, lw=0.8)
            a.plot(x, duzle(e[k]), color=RENK[t], lw=1.6, label=f"tohum {t+1}")
    for j, (k, et) in enumerate(OLC):
        a = ax[i, j]; a.set_title(f"{baslik}\n{et}" if j == 0 else et, fontsize=10, loc="left")
        a.grid(alpha=0.3); a.axvline(480, color="gray", ls=":", lw=1)
        if i == 1: a.set_xlabel("Bölüm (8 ortam toplamı)")
    ax[i, 0].legend(fontsize=8)
fig.suptitle("Unity eğitimi — 80 bölümlük pencereler, kalın çizgi 5 pencerelik ortalama; noktalı çizgi ε = 0,05", fontsize=11)
fig.tight_layout()
cikti = os.path.join(KOK, "gorseller", "ogrenme_egrisi_unity_3tohum.png")
fig.savefig(cikti, dpi=130); print(cikti)
