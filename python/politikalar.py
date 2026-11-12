"""
Politikalar: Rastgele, Devriye (klasik kural tabanlı baseline) ve Tablo Q-öğrenme ajanı.
Unity'deki PTZKameraAjani.cs ile aynı mantık.

TABLO Q-ÖĞRENME (sinir ağı yok)
  Durum (36)  = aciliyet (0..3) x en eski sorumlu bölge (0..8)
                 aciliyet: sorumlu olunan en sıcak bilinen anomali  0 yok, 1 <10, 2 <18, 3 >=18 °C
                 en eski bölge: bölgedeki en uzun süredir bakılmayan hücrenin yaşı en büyük olan bölge
  Eylem (10)  = 0..8 "o bölgenin merkezine Pan/Tilt + GENİŞ zoom (termal tarama)"
                9    "en sıcak sorumlu anomaliye Pan/Tilt + YAKIN zoom (optik doğrulama)"
                (hedef komutları = makro eylem; kamera hedefe her adım 1 hücre ilerler)
  Öğrenme     = yarı-Markov (SMDP) Q-öğrenme:
                Q(s,a) <- Q(s,a) + alfa [ R_tau + gamma^tau max_a' Q(s',a') - Q(s,a) ]
  Sorumluluk  = sabit: saha köşegenle ikiye bölünür (Kamera_A sol üst, Kamera_B sağ alt)
  Maskeler    = anomali yokken 'doğrula' geçersiz; kameranın zaten bulunduğu bölge geçersiz
  Yerel ödül  = kameranın kendi ödülü + kendi alanındaki yangın / kör nokta / bekleyen alarm cezası
                + taranan kör noktalar için küçük ödül
"""
import numpy as np

MERKEZ = [1, 4, 6]
BOLGE_MERKEZ = [(r, c) for r in MERKEZ for c in MERKEZ]


def bolge(i, j):
    return (0 if i < 3 else 1 if i < 6 else 2) * 3 + (0 if j < 3 else 1 if j < 6 else 2)


def sahip_haritasi(N, K, sabit=True, bolge_bazli=False):
    """Sabit sorumluluk. Köşegen: sol-üst yarı kamera 0, sağ-alt yarı kamera 1.
    Bölge bazlı: B0-B3 kamera 0, B4-B8 kamera 1 (her hücre, onu tarayan bölgenin sahibine ait)."""
    ii, jj = np.indices((N, N))
    if bolge_bazli:
        return np.array([[0 if bolge(i, j) <= 3 else 1 for j in range(N)] for i in range(N)])
    return np.where((ii + jj < N - 1) | ((ii + jj == N - 1) & (ii < N // 2)), 0, 1)


def hedefe_komut(r, c, hr, hc, zoom):
    pan = 1 if hc < c else 2 if hc > c else 0
    tilt = 1 if hr < r else 2 if hr > r else 0
    return [pan, tilt, zoom]


# ---------------------------------------------------------------------- #
class Rastgele:
    def __init__(self, env, tohum=0): self.env, self.rng = env, np.random.default_rng(tohum)
    def sifirla(self): pass
    def eylem(self, obs, info=None): return self.rng.integers([3, 3, 2] * self.env.K)
    def gozlemle(self, *a, **k): pass


class Devriye:
    """Klasik kural: her kamera kendi dikey şeridinde; bilinen en sıcak anomaliye gider ve
    yanına varınca yakın zoomla doğrular; anomali yoksa şeridindeki en uzun süredir
    bakılmayan hücreye gider (eşitlikte en yakın)."""
    def __init__(self, env): self.env = env
    def sifirla(self): pass
    def gozlemle(self, *a, **k): pass

    def eylem(self, obs, info=None):
        N, K = self.env.N, self.env.K
        yas, bil, kam = obs["yas"], obs["anomali"], obs["kameralar"]
        out = []
        for k in range(K):
            r, c, _ = kam[k]
            s0, s1 = k * N // K, (k + 1) * N // K
            serit = bil[:, s0:s1]
            if serit.max() > 0:
                ti, tj = np.unravel_index(np.argmax(serit), serit.shape); tj += s0
                z = 1 if max(abs(ti - r), abs(tj - c)) <= 1 else 0
            else:
                en, ti, tj = -1, r, c
                for i in range(N):
                    for j in range(s0, s1):
                        m = max(abs(i - r), abs(j - c))
                        if yas[i, j] > en or (yas[i, j] == en and m < enm):
                            en, enm, ti, tj = yas[i, j], m, i, j
                z = 0
            out += hedefe_komut(r, c, ti, tj, z)
        return np.array(out)


# ---------------------------------------------------------------------- #
class QAjani:
    DURUM, EYLEM = 36, 10

    def __init__(self, env, gamma=0.95, alfa_min=0.01, eps_bas=1.0, eps_son=0.05, eps_karar=60000,
                 bekleme_cezasi=0.3, tarama_odulu=0.1, maks_sure=12, bolge_yasi="maks",
                 sabit_sorumluluk=True, maske=True, hedef_iptal=False, kendi_bolgem=False, bolge_bazli=False, tohum=0):
        self.env = env
        self.gamma, self.alfa_min = gamma, alfa_min
        self.eps_bas, self.eps_son, self.eps_karar = eps_bas, eps_son, eps_karar
        self.bekleme, self.tarama, self.maks_sure = bekleme_cezasi, tarama_odulu, maks_sure
        self.bolge_yasi, self.sabit, self.maske, self.hedef_iptal = bolge_yasi, sabit_sorumluluk, maske, hedef_iptal
        self.kendi_bolgem = kendi_bolgem
        self.rng = np.random.default_rng(tohum)
        self.Q = np.zeros((self.DURUM, self.EYLEM)); self.n = np.zeros((self.DURUM, self.EYLEM), int)
        self.karar = 0; self.ogren = True
        N = env.N
        self.sahip = sahip_haritasi(N, env.K, bolge_bazli=bolge_bazli)
        self.bolge_hucre = np.array([[bolge(i, j) for j in range(N)] for i in range(N)])
        self.sifirla()

    @property
    def eps(self):
        if not self.ogren: return 0.0
        return max(self.eps_son, self.eps_bas - (self.eps_bas - self.eps_son) * self.karar / self.eps_karar)

    def sifirla(self):
        K = self.env.K
        self.s = [-1] * K; self.a = [-1] * K; self.biriken = [0.0] * K; self.sure = [0] * K
        self.hedef = [None] * K; self.hzoom = [0] * K; self.son_odul = np.zeros(K)

    # --- sorumluluk ---
    def sorumlu(self, k, kam):
        if self.sabit: return self.sahip == k
        N = self.env.N; ii, jj = np.indices((N, N))
        d = np.stack([np.maximum(abs(ii - kk[0]), abs(jj - kk[1])) for kk in kam])
        return np.argmax(d == d.min(0), axis=0) == k

    def aciliyet(self, k, obs):
        bil = np.where(self.sorumlu(k, obs["kameralar"]), obs["anomali"], 0)
        if bil.max() <= 0: return 0, None
        h = np.unravel_index(np.argmax(bil), bil.shape); T = bil[h]
        return (1 if T < 10 else 2 if T < 18 else 3), (int(h[0]), int(h[1]))

    def durum(self, k, obs):
        ac, _ = self.aciliyet(k, obs)
        yas = np.minimum(obs["yas"], 50)
        m = self.sorumlu(k, obs["kameralar"])
        en, enb = -1, None
        for b in range(9):
            if not m[BOLGE_MERKEZ[b]]: continue
            h = yas[self.bolge_hucre == b]
            v = h.max() if self.bolge_yasi == "maks" else h.mean()
            if v > en: en, enb = v, b
        if enb is None: enb = bolge(*obs["kameralar"][k][:2])
        return ac * 9 + enb

    def yasak(self, k, obs):
        ys = set()
        r, c = obs["kameralar"][k][:2]
        if self.maske:
            for b in range(9):
                if BOLGE_MERKEZ[b] == (r, c): ys.add(b)
        if self.kendi_bolgem:   # başka kameranın sorumlu olduğu bölgeleri tarama hedefi yapma
            m = self.sorumlu(k, obs["kameralar"])
            for b in range(9):
                if not m[BOLGE_MERKEZ[b]]: ys.add(b)
        return ys

    def gecerli(self, s): return 10 if s // 9 > 0 else 9

    def maksQ(self, s, ys):
        q = self.Q[s, :self.gecerli(s)].copy()
        for y in ys: q[y] = -np.inf
        return q.max()

    def sec(self, s, ys):
        g = self.gecerli(s)
        if self.rng.random() < self.eps:
            while True:
                a = int(self.rng.integers(g))
                if a not in ys: return a
        q = self.Q[s, :g] + self.rng.random(g) * 1e-6
        for y in ys: q[y] = -np.inf
        return int(np.argmax(q))

    def guncelle(self, s, a, hedef):
        if not self.ogren: return
        self.n[s, a] += 1
        alfa = max(self.alfa_min, 1.0 / self.n[s, a] ** 0.7)
        self.Q[s, a] += alfa * (hedef - self.Q[s, a])

    # --- her adım ---
    def eylem(self, obs, info=None):
        out = []
        for k in range(self.env.K):
            r, c, z = obs["kameralar"][k]
            if self.s[k] >= 0:
                self.biriken[k] += self.gamma ** self.sure[k] * self.son_odul[k]; self.sure[k] += 1
            vardi = self.s[k] >= 0 and (r, c) == self.hedef[k] and (self.hzoom[k] == 0 or z == 1)
            ac, h = self.aciliyet(k, obs)
            kesinti = self.s[k] >= 0 and self.a[k] < 9 and ac > 0
            hedef_kalmadi = self.hedef_iptal and self.s[k] >= 0 and self.a[k] == 9 and ac == 0
            if self.s[k] < 0 or vardi or kesinti or hedef_kalmadi or self.sure[k] >= self.maks_sure:
                s2, ys = self.durum(k, obs), self.yasak(k, obs)
                if self.s[k] >= 0:
                    self.guncelle(self.s[k], self.a[k], self.biriken[k] + self.gamma ** self.sure[k] * self.maksQ(s2, ys))
                self.s[k] = s2; self.a[k] = self.sec(s2, ys)
                if self.ogren: self.karar += 1
                self.biriken[k] = 0.0; self.sure[k] = 0
                if self.a[k] < 9: self.hedef[k], self.hzoom[k] = BOLGE_MERKEZ[self.a[k]], 0
                else: self.hedef[k], self.hzoom[k] = h, 1
            elif self.a[k] == 9 and h is not None:
                self.hedef[k] = h
            hr, hc = self.hedef[k]
            bir = max(abs(hr - r), abs(hc - c)) <= 1
            out += hedefe_komut(r, c, hr, hc, 1 if (self.hzoom[k] == 1 and bir) else 0)
            # taranan kör noktalar için küçük yerel ödül (taramadan önceki yaşlarla)
            self._tarama_on = getattr(self, "_tarama_on", {})
        self._onceki_obs = obs; self._komut = out
        return np.array(out)

    def gozlemle(self, onceki_obs, komut, info, obs, bitti):
        """Adım sonrası yerel ödülleri hesapla (Unity: qOdul)."""
        env = self.env; N = env.N
        kam_odul = info["kamera_odul"].copy()
        for k in range(env.K):
            m = self.sorumlu(k, obs["kameralar"])
            r, c, z = obs["kameralar"][k]
            if z == 0:   # tarama ödülü: taranan hücrelerin taramadan önceki (normalize) yaşı
                y0 = np.minimum(onceki_obs["yas"][max(0, r-1):r+2, max(0, c-1):c+2], 50) / 50
                kam_odul[k] += self.tarama * y0.sum()
            for (i, j, _) in info["yanginlar"]:
                if m[i, j]: kam_odul[k] -= env.ceza_yangin
            kam_odul[k] -= env.ceza_kor_nokta * np.sum(np.minimum(obs["yas"][m], 50) / 50) / (N * N)
            kam_odul[k] -= self.bekleme * np.sum(obs["anomali"][m]) / env.T_max
        self.son_odul = kam_odul
        if bitti:   # bölüm sonu: terminal güncelleme
            for k in range(env.K):
                if self.s[k] >= 0:
                    self.guncelle(self.s[k], self.a[k], self.biriken[k] + self.gamma ** self.sure[k] * self.son_odul[k])
            self.sifirla()
