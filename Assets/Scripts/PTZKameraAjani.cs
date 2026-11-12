using UnityEngine;
using UnityEngine.InputSystem;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

/// <summary>
/// ADIM 7.2 - PTZ kamera ajanı.
/// MDP ile birebir eşleşme:
///   Durum  -> CollectObservations   (kameranın "kendi gözünden" haritalar)
///   Eylem  -> OnActionReceived      (Pan, Tilt, Zoom komutları)
///   Ödül   -> kontrolcu.OdulVer(...) (tespit, doğrulama, motor cezası)
/// Zamanı (dünyanın ilerlemesi, bölüm sonu) OrtamKontrolcusu yönetir.
/// </summary>
public class PTZKameraAjani : Agent
{
    [Header("Bağlantılar (OrtamKontrolcusu otomatik doldurur)")]
    public SahaYoneticisi saha;
    public OrtamKontrolcusu kontrolcu;

    [Header("Başlangıçta bakılan hücre")]
    public int baslangicSatir = 0;
    public int baslangicSutun = 0;

    [Header("Ödül ayarları")]
    public float odulDogrulama = 10f;
    public float odulIlkTespit = 1f;
    public float cezaMotor = 0.05f;
    [Tooltip("Optik doğrulamayla bir yanlış alarmı elemenin ödülü (gerçek ısınmayı doğrulamaktan çok daha az).")]
    public float odulYanlisEleme = 0.5f;
    [Tooltip("Yeni fark edilen her sıcak nokta için ek ödül = katsayı * (1 - T/Tmax). 0 = kapalı.")]
    public float erkenTespitBonusu = 0f;
    [Tooltip("Doğrulama ödülünün üssü: 1 = doğrusal (varsayılan), 2 = karesel (geç doğrulama çok daha az ödül).")]
    public float dogrulamaUssu = 1f;

    [Header("Güvenlik kalkanı (sadece değerlendirme/kullanım için, eğitimde KAPALI)")]
    [Tooltip("Açık: eşiği aşan bilinen sıcak noktaya en yakın kamera RL kararını bırakıp doğrulamaya gider.")]
    public bool guvenlikKalkani = false;
    public float kalkanEsigi = 18f;

    [Header("Ödül şekillendirme (Adım 9)")]
    [Tooltip("Bilinen bir sıcak noktaya yaklaştıkça küçük ödül, uzaklaştıkça küçük ceza.")]
    public bool sekillendirme = true;
    public float sekillendirmeKatsayisi = 0.2f;
    public float gamma = 0.99f; // eğitim ayar dosyasındaki gamma ile aynı olmalı

    public enum ElleKontrol { Klavye, Devriye, Rastgele, QOgrenme }

    [Header("Q-öğrenme (Heuristic Modu = QOgrenme iken)")]
    [Tooltip("Taranan (yaşı sıfırlanan) her hücre için ödül katsayısı.")]
    public float qTaramaOdulu = 0.1f;
    [Tooltip("Bir eylem (hedef komutu) en fazla kaç adım sürebilir.")]
    public int qMaksSure = 12;
    [Tooltip("Açıksa 'en eski bölge' bölgedeki EN UZUN süredir bakılmayan hücreye göre seçilir (tek bir unutulmuş köşe de fark edilir). Kapalıysa bölge ortalaması kullanılır. Değiştirirsen Q tablosu sıfırdan eğitilmeli.")]
    public bool qBolgeYasiMaks = true;
    [Tooltip("Açıksa kamera, merkezinde zaten durduğu bölgeyi 'tara' hedefi olarak seçemez. Böyle bir eylem hiç hareket üretmez ve açgözlü politikada kameranın yerinde donmasına yol açabilir.")]
    public bool qBulunduguBolgeyiMaskele = true;

    /// <summary>Kamera bir bölge merkezindeyse o bölgenin 'tara' eylemi (yasak), değilse -1.</summary>
    int QYasakEylem()
    {
        if (!qBulunduguBolgeyiMaskele) return -1;
        for (int b = 0; b < 9; b++)
        { var m = BolgeMerkezi(b); if (m.x == Satir && m.y == Sutun) return b; }
        return -1;
    }

    // SMDP Q-öğrenme durumu
    [HideInInspector] public float qOdul;   // son adımda bu kameranın aldığı (yerel) ödül
    int qDurum = -1, qEylem = -1, qSure;
    float qBirikim;
    Vector2Int qHedef; int qHedefZoom;
    /// <summary>Politika değiştirildiğinde yarım kalan Q makro eylemini temizler (sunum için).</summary>
    public void QSifirla() { qDurum = qEylem = -1; qBirikim = 0f; qOdul = 0f; qSure = 0; }
    public string QEylemAdi => qEylem < 0 ? "-" : QTablosu.EylemAdlari[qEylem];
    public int QAciliyet => Aciliyet(out _);

    static readonly int[] BOLGE_MERKEZ = { 1, 4, 6 };
    [Header("Heuristic modu (Behavior Type = Heuristic Only iken)")]
    [Tooltip("Klavye: ok tuşları + Boşluk. Devriye: kural tabanlı kıyas politikası.")]
    public ElleKontrol heuristicModu = ElleKontrol.Klavye;

    [Header("Görsel")]
    public float direkYuksekligi = 8f;
    public Color gorusRengi = Color.cyan;

    public int Satir { get; private set; }
    public int Sutun { get; private set; }
    public int Zoom { get; private set; }   // 0 = geniş, 1 = yakın

    /// <summary>(i,j) hücresi şu an bu kameranın görüş alanında mı? (geniş = 3x3, yakın = 1 hücre)</summary>
    public bool GorusAlanindaMi(int i, int j)
    {
        int r = Zoom == 0 ? 1 : 0;
        return Mathf.Abs(i - Satir) <= r && Mathf.Abs(j - Sutun) <= r;
    }

    const float YAS_TAVAN = 50f;
    float oncekiPotansiyel;
    Transform kafa;
    LineRenderer gorusAlani, isin;

    // ------------------------------------------------------------------
    public override void Initialize()
    {
        GorselleriOlustur();
    }

    public override void OnEpisodeBegin()
    {
        Satir = baslangicSatir;
        Sutun = baslangicSutun;
        Zoom = 0;
        oncekiPotansiyel = (saha != null && saha.bilinen != null) ? Potansiyel() : 0f;

        // Q-öğrenme: bölüm bitti -> son eylemi terminal hedefle güncelle
        if (qDurum >= 0)
        {
            qBirikim += Mathf.Pow(QTablosu.gamma, qSure) * qOdul;
            QTablosu.Guncelle(qDurum, qEylem, qBirikim);
        }
        qDurum = qEylem = -1; qBirikim = 0f; qOdul = 0f; qSure = 0;
    }

    // ------------------------------------------------------------------
    // Q-ÖĞRENME: özet durum, hedef komutları ve SMDP güncellemesi
    int Bolge(int i, int j)
    {
        int r = i < 3 ? 0 : i < 6 ? 1 : 2, c = j < 3 ? 0 : j < 6 ? 1 : 2; // bölge sınırları
        return r * 3 + c;
    }
    Vector2Int BolgeMerkezi(int b) => new Vector2Int(BOLGE_MERKEZ[b / 3], BOLGE_MERKEZ[b % 3]);

    /// <summary>Sorumlu olduğum en sıcak bilinen anomali ve aciliyet seviyesi (0 = yok, 1 &lt;10, 2 &lt;18, 3 ≥18).</summary>
    int Aciliyet(out Vector2Int hedef)
    {
        hedef = new Vector2Int(Satir, Sutun); float enSicak = 0f;
        for (int i = 0; i < saha.N; i++)
            for (int j = 0; j < saha.N; j++)
                if (saha.bilinen[i, j] > enSicak && kontrolcu.EnYakinKameraMi(this, i, j))
                { enSicak = saha.bilinen[i, j]; hedef = new Vector2Int(i, j); }
        return enSicak <= 0f ? 0 : enSicak < 10f ? 1 : enSicak < 18f ? 2 : 3;
    }

    /// <summary>Durum = aciliyet x (sorumlu olduğum bölgeler arasında ortalama yaşı en büyük olan bölge).</summary>
    int QDurumu()
    {
        int aciliyet = Aciliyet(out _);
        float[] toplam = new float[9]; int[] sayi = new int[9];
        for (int i = 0; i < saha.N; i++)
            for (int j = 0; j < saha.N; j++)
            {
                int b = Bolge(i, j); float y = Mathf.Min(saha.yas[i, j], 50f);
                if (qBolgeYasiMaks) toplam[b] = Mathf.Max(toplam[b], y); else toplam[b] += y;
                sayi[b]++;
            }
        int enYasli = -1; float enBuyuk = -1f;
        for (int b = 0; b < 9; b++)
        {
            var m = BolgeMerkezi(b);
            if (!kontrolcu.EnYakinKameraMi(this, m.x, m.y)) continue; // sadece benim bölgelerim
            float ort = qBolgeYasiMaks ? toplam[b] : toplam[b] / Mathf.Max(1, sayi[b]);
            if (ort > enBuyuk) { enBuyuk = ort; enYasli = b; }
        }
        if (enYasli < 0) enYasli = Bolge(Satir, Sutun);
        return aciliyet * 9 + enYasli;
    }

    void QEylemi(ActionSegment<int> d)
    {
        // 1) Son adımın ödülünü bu eylemin birikimine ekle
        if (qDurum >= 0)
        {
            qBirikim += Mathf.Pow(QTablosu.gamma, qSure) * qOdul;
            qSure++;
        }
        qOdul = 0f;

        // 2) Eylem (hedef komutu) bitti mi? Hedefe vardı, yeni anomali çıktı (kesinti) ya da süre doldu
        bool vardi = qDurum >= 0 && Satir == qHedef.x && Sutun == qHedef.y && (qHedefZoom == 0 || Zoom == 1);
        bool kesinti = qDurum >= 0 && qEylem < 9 && Aciliyet(out _) > 0;
        // Doğrulamaya giderken hedef anomali artık bilinmiyorsa (başka yoldan çözüldü / kayboldu)
        // boş bir hücreye yakın zoom yapma: eylemi bitir ve yeniden karar ver.
        bool hedefKalmadi = qDurum >= 0 && qEylem == 9 && Aciliyet(out _) == 0;
        if (qDurum < 0 || vardi || kesinti || hedefKalmadi || qSure >= qMaksSure)
        {
            int yeniDurum = QDurumu();
            if (qDurum >= 0)
                QTablosu.Guncelle(qDurum, qEylem,
                    qBirikim + Mathf.Pow(QTablosu.gamma, qSure) * QTablosu.MaksQ(yeniDurum, QYasakEylem()));
            qDurum = yeniDurum;
            qEylem = QTablosu.Sec(qDurum, QYasakEylem());
            if (QTablosu.ogrenmeAktif) QTablosu.kararSayisi++;
            qBirikim = 0f; qSure = 0;
            if (qEylem < 9) { qHedef = BolgeMerkezi(qEylem); qHedefZoom = 0; }
            else { Aciliyet(out qHedef); qHedefZoom = 1; }
        }
        else if (qEylem == 9)
        {
            Aciliyet(out var h); // doğrulanacak anomali hâlâ benimse hedefi güncel tut
            if (saha.bilinen[h.x, h.y] > 0f) qHedef = h;
        }

        // 3) Hedef komutunu motor komutuna çevir: her adımda hedefe doğru 1 hücre (çapraz dahil)
        d[0] = qHedef.y < Sutun ? 1 : qHedef.y > Sutun ? 2 : 0;
        d[1] = qHedef.x < Satir ? 1 : qHedef.x > Satir ? 2 : 0;
        bool birAdimKaldi = Mathf.Max(Mathf.Abs(qHedef.x - Satir), Mathf.Abs(qHedef.y - Sutun)) <= 1;
        d[2] = (qHedefZoom == 1 && birAdimKaldi) ? 1 : 0; // yolda termal (geniş), varınca optik (yakın)
    }

    /// <summary>
    /// POTANSİYEL: Bilinen en yakın sıcak noktaya ne kadar yakınım?
    /// Uzaklık, çapraz hareket de 1 adım sayıldığı için Chebyshev uzaklığı: max(|dy|, |dx|).
    /// Bilinen anomali yoksa 0. Varsa katsayı * (N-1 - uzaklık): üstündeyken en büyük.
    /// </summary>
    float Potansiyel()
    {
        int enYakin = int.MaxValue;
        for (int i = 0; i < saha.N; i++)
            for (int j = 0; j < saha.N; j++)
                // Sadece BENİM sorumluluğumdaki (bana en yakın olan) sıcak noktalar:
                // böylece iki kamera aynı noktaya çekilmez.
                if (saha.bilinen[i, j] > 0f &&
                    (!kontrolcu.sorumluKameraSekillendirme || kontrolcu.EnYakinKameraMi(this, i, j)))
                {
                    int d = Mathf.Max(Mathf.Abs(i - Satir), Mathf.Abs(j - Sutun));
                    if (d < enYakin) enYakin = d;
                }
        if (enYakin == int.MaxValue) return 0f;
        return sekillendirmeKatsayisi * (saha.N - 1 - enYakin);
    }

    /// <summary>
    /// DURUM: Haritaları kameranın kendi konumuna göre ortalanmış olarak veriyoruz
    /// ("benim 2 sağımda sıcak bir nokta var" gibi). Her hücre için 3 sayı:
    ///   belirsizlik yaşı (saha dışıysa -1), bilinen anomali, başka kamera bakıyor mu.
    /// Boyut: 3 * (2N-1)^2 + 1   (N=8 için 3*15*15+1 = 676)
    /// </summary>
    public override void CollectObservations(VectorSensor sensor)
    {
        if (saha == null || kontrolcu == null || saha.yas == null)
        {
            // Ortam henüz bağlanmadı ya da gizlendi (ör. Kontrol Paneli diğer ortamları kapattığında):
            // gözlem boyutu bozulmasın diye sıfır gözlem ver.
            int boyut = GetComponent<Unity.MLAgents.Policies.BehaviorParameters>().BrainParameters.VectorObservationSize;
            for (int i = 0; i < boyut; i++) sensor.AddObservation(0f);
            return;
        }
        int N = saha.N, R = N - 1;
        for (int di = -R; di <= R; di++)
            for (int dj = -R; dj <= R; dj++)
            {
                int i = Satir + di, j = Sutun + dj;
                if (i < 0 || i >= N || j < 0 || j >= N)
                {
                    sensor.AddObservation(-1f); // saha dışı
                    sensor.AddObservation(0f);
                    sensor.AddObservation(0f);
                    continue;
                }
                sensor.AddObservation(Mathf.Min(saha.yas[i, j], YAS_TAVAN) / YAS_TAVAN);
                sensor.AddObservation(saha.bilinen[i, j] / saha.Tmax);
                sensor.AddObservation(kontrolcu.BaskaKameraBakiyorMu(this, i, j) ? 1f : 0f);
            }
        sensor.AddObservation(Zoom);
    }

    /// <summary>
    /// EYLEM: 3 ayrık dal
    ///   dal 0 = Pan  : 0 dur, 1 sol, 2 sağ
    ///   dal 1 = Tilt : 0 dur, 1 yukarı (kuzey), 2 aşağı (güney)
    ///   dal 2 = Zoom : 0 geniş, 1 yakın
    /// </summary>
    public override void OnActionReceived(ActionBuffers actions)
    {
        var d = actions.DiscreteActions;
        int pan = d[0], tilt = d[1], zoom = d[2];

        // GÜVENLİK KALKANI: tehlikeli sıcaklığa ulaşmış bilinen bir sıcak noktanın sorumlusuysam,
        // RL kararını geçersiz kıl ve doğrudan doğrulamaya git.
        if (guvenlikKalkani && KalkanHedefi(out int ki, out int kj))
        {
            pan = kj < Sutun ? 1 : kj > Sutun ? 2 : 0;
            tilt = ki < Satir ? 1 : ki > Satir ? 2 : 0;
            zoom = Mathf.Max(Mathf.Abs(ki - Satir), Mathf.Abs(kj - Sutun)) <= 1 ? 1 : 0;
            OrtamKontrolcusu.kalkanMudahalesi++;
        }

        int dj = pan == 1 ? -1 : pan == 2 ? 1 : 0;
        int di = tilt == 1 ? -1 : tilt == 2 ? 1 : 0;
        int yeniSatir = Mathf.Clamp(Satir + di, 0, saha.N - 1);
        int yeniSutun = Mathf.Clamp(Sutun + dj, 0, saha.N - 1);
        int yeniZoom = zoom;

        // Motor cezası: kamera kıpırdadıysa ya da zoom değiştiyse
        if (yeniSatir != Satir || yeniSutun != Sutun || yeniZoom != Zoom)
            kontrolcu.OdulVer(this, -cezaMotor);

        Satir = yeniSatir; Sutun = yeniSutun; Zoom = yeniZoom;

        // Algılama ve ödül
        if (Zoom == 0)
        {
            if (heuristicModu == ElleKontrol.QOgrenme) // Q-öğrenme: taranan kör noktalar için yerel ödül
            {
                float temizlenen = 0f;
                for (int pi = -1; pi <= 1; pi++)
                    for (int pj = -1; pj <= 1; pj++)
                    {
                        int i = Satir + pi, j = Sutun + pj;
                        if (i >= 0 && i < saha.N && j >= 0 && j < saha.N)
                            temizlenen += Mathf.Min(saha.yas[i, j], 50f) / 50f;
                    }
                qOdul += qTaramaOdulu * temizlenen;
            }
            int yeni = saha.GenisTara(Satir, Sutun, out float erkenlik);
            if (yeni > 0)
            {
                kontrolcu.OdulVer(this, odulIlkTespit * yeni);
                // Erken tespit bonusu: soğukken fark etmek daha değerli.
                // Yardımcı ödül -> "gerçek" bölüm ödülüne SAYILMAZ (adil kıyas için).
                if (erkenTespitBonusu > 0f)
                    kontrolcu.SekillendirmeOdulu(this, erkenTespitBonusu * erkenlik);
            }
        }
        else
        {
            float T = saha.YakinDogrula(Satir, Sutun);
            if (T < 0f)
            {
                // OPTİK doğrulama yanlış alarmı eledi: tehlike yok ama belirsizlik giderildi
                kontrolcu.OdulVer(this, odulYanlisEleme);
            }
            else if (T > 0f)
            {
                float erkenlik = 1f - T / saha.Tmax;
                float temel = odulDogrulama * erkenlik;              // erken = büyük ödül (doğrusal)
                kontrolcu.OdulVer(this, temel);
                // B planı: ajan ödülü erkenlik^us olarak görür (us > 1 -> geç doğrulama çok daha az değerli).
                // Fark yardımcı ödül olarak verilir; "gerçek" ödül doğrusal kalır (adil kıyas için).
                if (dogrulamaUssu != 1f)
                    kontrolcu.SekillendirmeOdulu(this, odulDogrulama * Mathf.Pow(erkenlik, dogrulamaUssu) - temel);
            }
        }

        // Potansiyel tabanlı ödül şekillendirme: F = γ·Φ(yeni) − Φ(eski)
        // Bölüm boyunca toplamı neredeyse sıfırlanır, bu yüzden "en iyi davranış" değişmez;
        // sadece kameraya doğru yöne gittiğini her adımda fısıldar.
        if (sekillendirme)
        {
            float yeni = Potansiyel();
            kontrolcu.SekillendirmeOdulu(this, gamma * yeni - oncekiPotansiyel);
            oncekiPotansiyel = yeni;
        }
    }

    /// <summary>Kalkan hedefi: bilinen sıcaklığı eşiği aşmış ve bana en yakın olan en sıcak nokta.</summary>
    bool KalkanHedefi(out int ti, out int tj)
    {
        ti = tj = -1;
        float enSicak = kalkanEsigi;
        for (int i = 0; i < saha.N; i++)
            for (int j = 0; j < saha.N; j++)
                if (saha.bilinen[i, j] >= enSicak && kontrolcu.EnYakinKameraMi(this, i, j))
                { enSicak = saha.bilinen[i, j]; ti = i; tj = j; }
        return ti >= 0;
    }

    /// <summary>Elle test: ok tuşları / WASD hareket, Boşluk basılı = yakın zoom.</summary>
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var d = actionsOut.DiscreteActions;
        d[0] = 0; d[1] = 0; d[2] = 0;
        if (heuristicModu == ElleKontrol.Devriye) { DevriyeEylemi(d); return; }
        if (heuristicModu == ElleKontrol.QOgrenme) { QEylemi(d); return; }
        if (heuristicModu == ElleKontrol.Rastgele) // kıyas: tamamen rastgele kamera
        {
            d[0] = Random.Range(0, 3); d[1] = Random.Range(0, 3); d[2] = Random.Range(0, 2);
            return;
        }
        var k = Keyboard.current;
        if (k == null) return;
        if (k.leftArrowKey.isPressed || k.aKey.isPressed) d[0] = 1;
        else if (k.rightArrowKey.isPressed || k.dKey.isPressed) d[0] = 2;
        if (k.upArrowKey.isPressed || k.wKey.isPressed) d[1] = 1;
        else if (k.downArrowKey.isPressed || k.sKey.isPressed) d[1] = 2;
        if (k.spaceKey.isPressed) d[2] = 1;
    }

    /// <summary>
    /// KURAL TABANLI DEVRİYE (kıyas politikası, Python'daki "sezgisel" ile aynı mantık):
    ///   - Saha sütunlara göre kameralar arasında şeritlere bölünür (iş bölümü elle verilir).
    ///   - Şeridimde bilinen bir sıcak nokta varsa en sıcağına git; 1 adım kalınca yakın zoom yap.
    ///   - Yoksa şeridimdeki en uzun süredir bakılmayan hücreye geniş açıyla git.
    /// Kullanılan bilgi ajanın gözlemiyle aynı: yaş, bilinen anomaliler, kendi konumu.
    /// </summary>
    void DevriyeEylemi(ActionSegment<int> d)
    {
        int N = saha.N;
        int K = Mathf.Max(1, kontrolcu.kameralar.Count);
        int idx = Mathf.Max(0, kontrolcu.kameralar.IndexOf(this));
        int s0 = idx * N / K, s1 = (idx + 1) * N / K; // şerit: [s0, s1) sütunları

        int ti = -1, tj = -1;
        float enSicak = 0f;
        for (int i = 0; i < N; i++)
            for (int j = s0; j < s1; j++)
                if (saha.bilinen[i, j] > enSicak) { enSicak = saha.bilinen[i, j]; ti = i; tj = j; }

        bool dogrula = false;
        if (ti >= 0)
        {
            dogrula = Mathf.Max(Mathf.Abs(ti - Satir), Mathf.Abs(tj - Sutun)) <= 1;
        }
        else
        {
            float enYasli = -1f; int enYakin = int.MaxValue;
            for (int i = 0; i < N; i++)
                for (int j = s0; j < s1; j++)
                {
                    float y = saha.yas[i, j];
                    int mesafe = Mathf.Max(Mathf.Abs(i - Satir), Mathf.Abs(j - Sutun));
                    if (y > enYasli || (y == enYasli && mesafe < enYakin))
                    { enYasli = y; enYakin = mesafe; ti = i; tj = j; }
                }
        }

        d[0] = tj < Sutun ? 1 : tj > Sutun ? 2 : 0; // pan
        d[1] = ti < Satir ? 1 : ti > Satir ? 2 : 0; // tilt
        d[2] = dogrula ? 1 : 0;                      // zoom
    }

    // ------------------------------------------------------------------
    // Görsel kısım: direk, dönen kafa, görüş alanı çerçevesi. Öğrenmeye etkisi yok.
    void GorselleriOlustur()
    {
        var direk = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        direk.name = "Direk";
        direk.transform.SetParent(transform, false);
        direk.transform.localPosition = new Vector3(0f, direkYuksekligi / 2f, 0f);
        direk.transform.localScale = new Vector3(0.4f, direkYuksekligi / 2f, 0.4f);
        Destroy(direk.GetComponent<Collider>());

        var k = GameObject.CreatePrimitive(PrimitiveType.Cube);
        k.name = "Kafa";
        k.transform.SetParent(transform, false);
        k.transform.localPosition = new Vector3(0f, direkYuksekligi + 0.5f, 0f);
        k.transform.localScale = new Vector3(0.8f, 0.6f, 1.4f);
        Destroy(k.GetComponent<Collider>());
        kafa = k.transform;

        var mercek = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        mercek.name = "Mercek";
        mercek.transform.SetParent(kafa, false);
        mercek.transform.localPosition = new Vector3(0f, 0f, 0.55f);
        mercek.transform.localRotation = Quaternion.Euler(90f, 0f, 0f);
        mercek.transform.localScale = new Vector3(0.5f, 0.1f, 0.5f);
        mercek.GetComponent<Renderer>().material.color = Color.black;
        Destroy(mercek.GetComponent<Collider>());

        var malzeme = new Material(Shader.Find("Sprites/Default"));
        gorusAlani = CizgiOlustur("GorusAlani", malzeme, 0.25f, gorusRengi, 4);
        gorusAlani.loop = true;
        isin = CizgiOlustur("Isin", malzeme, 0.05f, new Color(gorusRengi.r, gorusRengi.g, gorusRengi.b, 0.5f), 2);
    }

    LineRenderer CizgiOlustur(string ad, Material m, float kalinlik, Color renk, int nokta)
    {
        var go = new GameObject(ad);
        go.transform.SetParent(transform, false);
        var lr = go.AddComponent<LineRenderer>();
        lr.material = m;
        lr.widthMultiplier = kalinlik;
        lr.startColor = lr.endColor = renk;
        lr.positionCount = nokta;
        lr.useWorldSpace = true;
        return lr;
    }

    void Update()
    {
        if (saha == null || kafa == null || saha.yas == null) return;

        // Pan/Tilt: kafa hedef hücreye doğru yumuşakça döner
        Vector3 hedef = saha.HucreMerkezi(Satir, Sutun);
        Quaternion q = Quaternion.LookRotation(hedef - kafa.position);
        kafa.rotation = Quaternion.Slerp(kafa.rotation, q, 10f * Time.unscaledDeltaTime); // görsel: yavaş oynatmada da akıcı

        // Görüş alanı: geniş = 3x3 (saha kenarında kırpılır), yakın = 1 hücre
        int r = Zoom == 0 ? 1 : 0, N = saha.N;
        Vector3 a = saha.HucreMerkezi(Mathf.Max(0, Satir - r), Mathf.Max(0, Sutun - r));
        Vector3 b = saha.HucreMerkezi(Mathf.Min(N - 1, Satir + r), Mathf.Min(N - 1, Sutun + r));
        float h = saha.hucreBoyu * 0.5f, y = saha.transform.position.y + 0.4f;
        float x0 = Mathf.Min(a.x, b.x) - h, x1 = Mathf.Max(a.x, b.x) + h;
        float z0 = Mathf.Min(a.z, b.z) - h, z1 = Mathf.Max(a.z, b.z) + h;
        // Termal (geniş) ve optik (yakın) modlar görsel olarak ayrışsın: optikte çerçeve beyaz ve kalın
        gorusAlani.widthMultiplier = Zoom == 1 ? 0.45f : 0.25f;
        gorusAlani.startColor = gorusAlani.endColor = Zoom == 1 ? Color.white : gorusRengi;
        gorusAlani.SetPosition(0, new Vector3(x0, y, z0));
        gorusAlani.SetPosition(1, new Vector3(x1, y, z0));
        gorusAlani.SetPosition(2, new Vector3(x1, y, z1));
        gorusAlani.SetPosition(3, new Vector3(x0, y, z1));
        isin.SetPosition(0, kafa.position);
        isin.SetPosition(1, hedef + Vector3.up * 0.4f);
    }
}
