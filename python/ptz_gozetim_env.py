"""
Ödev 20 - Bi-spektral PTZ kameralarla aktif yangın/anomali gözetimi
=====================================================================
gymnasium.Env uyumlu simülasyon ortamı (reset / step).
Unity simülasyonundaki (SahaYoneticisi.cs + OrtamKontrolcusu.cs + PTZKameraAjani.cs)
modelin birebir Python karşılığıdır; ödül ve fizik parametreleri aynıdır.

SAHA
  N x N ızgara. Her hücre sahadaki bir bölge (ör. kömür yığınının bir parçası).

STOKASTİK SÜREÇLER (geçiş dinamiği P)
  * Gerçek ısınma (kendiliğinden yanma): her adım p_tutusma olasılıkla boş bir hücre 1 °C'den
    ısınmaya başlar ve her adım +1 °C artar. T_max'a ulaşırsa doğrulanmadan YANGIN çıkar.
  * Yanlış ısı kaynağı (araç motoru, güneşte ısınmış metal): her adım p_yanlis olasılıkla
    sabit sıcaklıklı (6-15 °C) bir kaynak belirir, 20-60 adım sonra kaybolur. Tehlikesizdir.
  * Termal fark etme olasılığı: P(fark) = min(1, T / T_algi). Soğuk anomali kolay kaçırılır.
  * (Gürbüzlük testi) Termal sensör gürültüsü: okumaya N(0, termal_gurultu²) eklenir;
    boş hücre sahte_okuma olasılıkla 2-6 °C'lik sahte sıcak nokta gibi görünür.

BI-SPEKTRAL KAMERA
  Her kamera aynı kafada termal + optik sensör taşır; zoom ikisinin görüş alanını birlikte belirler.
  * GENİŞ açı: termal sensör 3x3 alanı tarar ve ısıyı görür, ama gerçek ısınmayı yanlış
    kaynaktan AYIRT EDEMEZ (optik çözünürlük yetersiz).
  * YAKIN zoom: tek hücre, yüksek çözünürlük. Optik sensör gerçek ısınmayı DOĞRULAR
    (müdahale ekibi soğutur) ya da yanlış alarmı ELER.

DURUM (gözlem)  - ödevdeki üç bileşen
  yas      (N,N): her hücrenin son görülmeden beri geçen adım sayısı (belirsizlik yaşı)
  anomali  (N,N): termal sensörün bildiği anomali sıcaklık tahmini (0 = bilinen anomali yok)
  kameralar(K,3): her kameranın bakış açısı (satır, sütun) ve zoom'u
EYLEM
  Her kamera için [pan, tilt, zoom]: pan 0=dur 1=sol 2=sağ, tilt 0=dur 1=yukarı 2=aşağı,
  zoom 0=geniş 1=yakın.  -> MultiDiscrete([3,3,2] * K)
ÖDÜL (takım ödülü, her adım)
  + odul_dogrulama * (1 - T/T_max)   gerçek ısınma doğrulandığında (erken = büyük)
  + odul_ilk_tespit                   yeni fark edilen her termal anomali için
  + odul_yanlis_eleme                 optik ile elenen her yanlış alarm için
  - ceza_motor                        hareket eden / zoom değiştiren her kamera için
  - ceza_yangin                       çıkan her yangın için
  - ceza_kor_nokta * ortalama(min(yas,50)/50)   kör noktalar için
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PTZGozetimEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, N=8, K=2, max_adim=200,
                 p_tutusma=0.02, T_max=30, T_algi=8.0,
                 yanlis_kaynaklar=True, p_yanlis=0.02, yanlis_T=(6, 15), yanlis_omur=(20, 60),
                 termal_gurultu=0.0, sahte_okuma=0.0,
                 odul_dogrulama=10.0, odul_ilk_tespit=1.0, odul_yanlis_eleme=0.5,
                 ceza_motor=0.05, ceza_yangin=20.0, ceza_kor_nokta=0.5, yas_tavan=50.0,
                 baslangic=None, render_mode=None):
        super().__init__()
        self.N, self.K, self.max_adim = N, K, max_adim
        self.p_tutusma, self.T_max, self.T_algi = p_tutusma, T_max, T_algi
        self.yanlis_kaynaklar, self.p_yanlis = yanlis_kaynaklar, p_yanlis
        self.yanlis_T, self.yanlis_omur = yanlis_T, yanlis_omur
        self.termal_gurultu, self.sahte_okuma = termal_gurultu, sahte_okuma
        self.odul_dogrulama, self.odul_ilk_tespit, self.odul_yanlis_eleme = odul_dogrulama, odul_ilk_tespit, odul_yanlis_eleme
        self.ceza_motor, self.ceza_yangin, self.ceza_kor_nokta, self.yas_tavan = ceza_motor, ceza_yangin, ceza_kor_nokta, yas_tavan
        # Unity'deki gibi: Kamera_A sol üst köşe, Kamera_B sağ alt köşe
        self.baslangic = baslangic or [(0, 0), (N - 1, N - 1), (0, N - 1), (N - 1, 0)][:K]
        self.render_mode = render_mode

        self.action_space = spaces.MultiDiscrete([3, 3, 2] * K)
        self.observation_space = spaces.Dict({
            "yas": spaces.Box(0.0, float(max_adim), (N, N), np.float32),
            "anomali": spaces.Box(0.0, float(T_max + 10), (N, N), np.float32),
            "kameralar": spaces.Box(0, N - 1, (K, 3), np.int64),
        })

    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        N = self.N
        self.yas = np.zeros((N, N))
        self.sicaklik = np.zeros((N, N))          # GERÇEK ısınma (gizli)
        self.yanlis = np.zeros((N, N))            # yanlış kaynak sıcaklığı (gizli)
        self.yanlis_kalan = np.zeros((N, N), int)
        self.elendi = np.zeros((N, N), bool)      # optik ile elenmiş yanlış kaynak
        self.bilinen = np.zeros((N, N))           # termal sensörün bildiği anomali tahmini
        self.tespit_T = np.zeros((N, N)); self.tespit_t = np.zeros((N, N), int)
        self.kam = np.array([[r, c, 0] for r, c in self.baslangic], dtype=np.int64)
        self.t = 0
        self.ist = dict(yangin=0, dogrulanan=0, dogrulama_sicaklik_toplam=0.0, yanlis_olusan=0,
                        yanlis_elenen=0, motor=0, yangin_bilinen=0, yangin_kacirilan=0, yangin_gorulmeyen=0)
        return self._gozlem(), {}

    def _gozlem(self):
        return {"yas": self.yas.astype(np.float32), "anomali": self.bilinen.astype(np.float32),
                "kameralar": self.kam.copy()}

    def _bos(self, i, j):
        return self.sicaklik[i, j] == 0 and self.yanlis[i, j] == 0

    # ------------------------------------------------------------------ #
    def _genis_tara(self, ci, cj):
        """Termal sensör 3x3 alanı tarar. Yeni fark edilen anomali sayısını döndürür."""
        yeni = 0
        rng = self.np_random
        for i in range(max(0, ci - 1), min(self.N, ci + 2)):
            for j in range(max(0, cj - 1), min(self.N, cj + 2)):
                self.yas[i, j] = 0
                g = self.sicaklik[i, j] if self.sicaklik[i, j] > 0 else \
                    (self.yanlis[i, j] if self.yanlis[i, j] > 0 and not self.elendi[i, j] else 0.0)
                if g > 0 and self.termal_gurultu > 0:
                    g = max(0.5, g + rng.normal() * self.termal_gurultu)
                elif g == 0 and self.sahte_okuma > 0 and rng.random() < self.sahte_okuma:
                    g = rng.uniform(2, 6)
                if g > 0:
                    if rng.random() < min(1.0, g / self.T_algi):
                        if self.bilinen[i, j] == 0:
                            yeni += 1
                            self.tespit_T[i, j], self.tespit_t[i, j] = g, self.t
                        self.bilinen[i, j] = g
                else:
                    self.bilinen[i, j] = 0.0
        return yeni

    def _yakin_dogrula(self, i, j):
        """Optik yakın inceleme. >0: doğrulanan sıcaklık, -1: yanlış alarm elendi, 0: bir şey yok."""
        self.yas[i, j] = 0
        self.bilinen[i, j] = 0
        T = self.sicaklik[i, j]
        if T > 0:
            self.sicaklik[i, j] = 0
            self.ist["dogrulanan"] += 1; self.ist["dogrulama_sicaklik_toplam"] += T
            return T
        if self.yanlis[i, j] > 0 and not self.elendi[i, j]:
            self.elendi[i, j] = True; self.ist["yanlis_elenen"] += 1
            return -1.0
        return 0.0

    # ------------------------------------------------------------------ #
    def step(self, action):
        a = np.asarray(action).reshape(self.K, 3)
        kamera_odul = np.zeros(self.K)            # her kameranın kendi eylemlerinden aldığı ödül
        # 1) Pan / Tilt / Zoom komutları ve algılama
        for k in range(self.K):
            pan, tilt, zoom = a[k]
            r, c, z = self.kam[k]
            nr = int(np.clip(r + (-1 if tilt == 1 else 1 if tilt == 2 else 0), 0, self.N - 1))
            nc = int(np.clip(c + (-1 if pan == 1 else 1 if pan == 2 else 0), 0, self.N - 1))
            if nr != r or nc != c or zoom != z:
                kamera_odul[k] -= self.ceza_motor; self.ist["motor"] += 1
            self.kam[k] = (nr, nc, zoom)
            if zoom == 0:
                kamera_odul[k] += self.odul_ilk_tespit * self._genis_tara(nr, nc)
            else:
                T = self._yakin_dogrula(nr, nc)
                if T > 0: kamera_odul[k] += self.odul_dogrulama * (1 - T / self.T_max)
                elif T < 0: kamera_odul[k] += self.odul_yanlis_eleme

        # 2) Dünya bir adım ilerler
        rng = self.np_random
        self.t += 1
        self.yas += 1
        yanginlar = []
        isinan = self.sicaklik > 0
        self.sicaklik[isinan] += 1
        bil_gercek = isinan & (self.bilinen > 0)
        self.bilinen[bil_gercek] += 1
        diger = (~isinan) & (self.bilinen > 0)   # sistem türünü bilmez: tahminini artırır
        self.bilinen[diger] = np.minimum(self.bilinen[diger] + 1, self.T_max)
        for i, j in np.argwhere(self.sicaklik >= self.T_max):
            tur = 0 if self.bilinen[i, j] > 0 else (1 if self.yas[i, j] < self.sicaklik[i, j] - 1 else 2)
            self.ist[["yangin_bilinen", "yangin_kacirilan", "yangin_gorulmeyen"][tur]] += 1
            yanginlar.append((int(i), int(j), tur))
            self.sicaklik[i, j] = 0; self.bilinen[i, j] = 0
        self.ist["yangin"] += len(yanginlar)
        omur = self.yanlis > 0
        self.yanlis_kalan[omur] -= 1
        biten = omur & (self.yanlis_kalan <= 0)
        self.yanlis[biten] = 0; self.elendi[biten] = False
        if rng.random() < self.p_tutusma:
            for _ in range(50):
                i, j = rng.integers(self.N, size=2)
                if self._bos(i, j): self.sicaklik[i, j] = 1; break
        if self.yanlis_kaynaklar and rng.random() < self.p_yanlis:
            for _ in range(50):
                i, j = rng.integers(self.N, size=2)
                if self._bos(i, j):
                    self.yanlis[i, j] = rng.uniform(*self.yanlis_T)
                    self.yanlis_kalan[i, j] = rng.integers(self.yanlis_omur[0], self.yanlis_omur[1] + 1)
                    self.elendi[i, j] = False; self.ist["yanlis_olusan"] += 1
                    break

        kor = np.mean(np.minimum(self.yas, self.yas_tavan) / self.yas_tavan)
        odul = kamera_odul.sum() - self.ceza_yangin * len(yanginlar) - self.ceza_kor_nokta * kor
        kesildi = self.t >= self.max_adim
        info = dict(self.ist, kamera_odul=kamera_odul, yanginlar=yanginlar)
        return self._gozlem(), float(odul), False, kesildi, info

    # ------------------------------------------------------------------ #
    def render(self):
        """ANSI: . boş, h ısınma, f yanlış kaynak, büyük harf = bilinen, A/B kamera."""
        sat = []
        for i in range(self.N):
            s = ""
            for j in range(self.N):
                ch = "h" if self.sicaklik[i, j] > 0 else ("f" if self.yanlis[i, j] > 0 else ".")
                if self.bilinen[i, j] > 0: ch = ch.upper() if ch != "." else "?"
                for k in range(self.K):
                    if self.kam[k][0] == i and self.kam[k][1] == j: ch = "AB"[k] if self.kam[k][2] == 0 else "ab"[k]
                s += ch + " "
            sat.append(s)
        return "\n".join(sat)


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env
    env = PTZGozetimEnv()
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    toplam = 0
    for _ in range(200):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        toplam += r
    print(env.render()); print("rastgele bölüm ödülü:", round(toplam, 1), "| yangın:", info["yangin"])
