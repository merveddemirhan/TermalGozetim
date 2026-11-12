using UnityEngine;

/// <summary>
/// Saha, ısınma fiziği ve BI-SPEKTRAL algılama modeli.
///
/// TERMAL sensör (geniş açı, 3x3): sıcak görünen her şeyi fark eder. İki tür sıcak kaynak vardır:
///   1) GERÇEK ısınma (kendiliğinden yanma): sıcaklığı her adım artar, Tmax'a ulaşınca YANGIN olur.
///   2) YANLIŞ sıcak kaynak (güneşte ısınmış metal, çalışan araç/makine, egzoz...):
///      sıcak görünür ama tehlikesizdir; sıcaklığı sabittir ve bir süre sonra kaybolur.
///   Termal sensör bu ikisini AYIRT EDEMEZ: ikisi de "bilinen anomali" olarak görünür.
///
/// OPTİK sensör (yakın zoom, 1 hücre, yüksek çözünürlük): görüntüden kaynağın ne olduğunu ayırt eder.
///   - Gerçek ısınma ise -> DOĞRULANIR, müdahale ekibi gönderilir (ödül, erken = büyük).
///   - Yanlış kaynak ise -> YANLIŞ ALARM ELENİR (küçük ödül), bir daha alarm üretmez.
/// </summary>
public class SahaYoneticisi : MonoBehaviour
{
    [Header("Saha")]
    public int N = 8;
    public float hucreBoyu = 4f;

    [Header("Isınma fiziği")]
    [Range(0f, 1f)] public float pTutusma = 0.04f; // her adımda yeni ısınma başlama olasılığı
    public int Tmax = 20;                           // bu sıcaklığa ulaşan hücre yangına döner
    [Tooltip("Eğitimde bazı bölümleri önceden ısınmış hücrelerle başlatır (keşifli başlangıç). Değerlendirmede etkisizdir.")]
    public bool kesifliBaslangic = false;
    [Range(0f, 1f)] public float kesifliBaslangicOlasiligi = 0.5f;
    public float Talgi = 8f;                        // geniş açıdan kesin görülme sıcaklığı

    [Header("Sensör gürültüsü (gürbüzlük testi)")]
    [Tooltip("Termal sensörün sıcaklık okumasına eklenen Gauss gürültüsünün standart sapması (°C). Fark etme olasılığını ve tahmini sıcaklığı (aciliyeti) bozar.")]
    public float termalGurultu = 0f;
    [Tooltip("Termal taramada boş bir hücrenin her okumada sahte sıcak nokta (2-6 °C) olarak görünme olasılığı. Optik doğrulama bunları eler.")]
    [Range(0f, 0.05f)] public float termalSahteOkuma = 0f;
    [HideInInspector] public int sahteOkumaSayisi;   // bu bölümde oluşan sahte okuma (sensör hatası) sayısı

    [Header("Yanlış alarm kaynakları (bi-spektral doğrulama)")]
    [Tooltip("Açık: termal kamerada sıcak görünen ama tehlikesiz kaynaklar (araç, güneşte ısınmış metal) oluşur.")]
    public bool yanlisKaynaklar = false;
    [Range(0f, 1f)] public float pYanlis = 0.02f;            // her adımda yeni yanlış kaynak olasılığı
    public Vector2 yanlisSicaklikAraligi = new Vector2(6f, 15f); // termal kamerada görünen sabit sıcaklık
    public Vector2Int yanlisOmurAraligi = new Vector2Int(20, 60); // kaç adım sonra kaybolur

    [Header("Test (kamera yokken sahayı kendi kendine çalıştırır)")]
    public bool testModu = true;
    public float testAdimSuresi = 0.25f;
    public bool yanginLogla = false; // her yangını Console'a yaz (çok yavaşlatır)

    // ---- Durum dizileri ----
    [HideInInspector] public float[,] sicaklik;      // GERÇEK ısınmanın (gizli) sıcaklığı, 0 = yok
    [HideInInspector] public float[,] yas;           // son görülmeden beri geçen adım
    [HideInInspector] public float[,] bilinen;       // termal kameraların fark ettiği anomali tahmini
    [HideInInspector] public float[,] yanlisSicaklik;// YANLIŞ kaynağın termal görünümü (gizli), 0 = yok
    [HideInInspector] public int[,] yanlisOmur;      // yanlış kaynağın kalan ömrü
    [HideInInspector] public bool[,] elendi;         // optik olarak "yanlış alarm" diye elendi mi

    [Header("İstatistik (sadece izlemek için)")]
    public int yanginSayisi;
    // Yangın teşhisi: yangına dönen anomali neden yakalanamadı?
    public int yanginBilinen;     // termalde görülmüştü ama optik doğrulama yetişmedi
    public int yanginKacirilan;   // ısınırken üstünden geçildi ama termal sensör fark etmedi (düşük sıcaklık)
    public int yanginGorulmeyen;  // ısınma başladığından beri o hücreye hiç bakılmadı (kapsama boşluğu)

    // Ayrıntılı yangın teşhisi: anomali ilk fark edildiğinde sıcaklığı ve zamanı saklanır
    [HideInInspector] public float[,] tespitT;
    [HideInInspector] public int[,] tespitAdim;
    [HideInInspector] public int adimSayaci;
    public struct YanginBilgi
    {
        public int i, j, tur;          // tur: 0 bilinen-doğrulanamadı, 1 bakıldı-fark edilmedi, 2 hiç bakılmadı
        public float tespitSicakligi;  // ilk fark edildiği andaki sıcaklık (bilinen için)
        public int tespittenBeri;      // fark edildikten yangına kadar geçen adım
        public int digerBilinen, digerGercek; // o anda bilinen başka anomali sayısı / bunların gerçek olanları
    }
    [HideInInspector] public System.Collections.Generic.List<YanginBilgi> sonYanginBilgi =
        new System.Collections.Generic.List<YanginBilgi>();
    public int dogrulananSayisi;
    public float dogrulamaSicaklikToplami; // doğrulanan hücrelerin o anki sıcaklıklarının toplamı
    public int yanlisToplam;               // bu bölümde oluşan yanlış kaynak sayısı
    public int yanlisElenen;               // optik doğrulamayla elenen yanlış alarm sayısı

    Renderer[,] hucreler;
    int[,] efektSayac; // kısa süreli renk efekti
    int[,] efektTip;   // 1 = yangın, 2 = doğrulama, 3 = yanlış alarm elendi
    float testSayac;

    static readonly Color komurRengi = new Color(0.15f, 0.15f, 0.15f);
    static readonly Color sicakRengi = new Color(1f, 0.2f, 0.05f);
    static readonly Color yanlisRengi = new Color(0.3f, 0.5f, 1f);   // yanlış kaynak (sadece biz görürüz)
    static readonly Color yanginRengi = new Color(1f, 0.9f, 0.1f);
    static readonly Color dogrulamaRengi = new Color(0.1f, 0.9f, 0.3f);
    static readonly Color elemeRengi = new Color(1f, 1f, 1f);        // yanlış alarm elendi: beyaz

    // ------------------------------------------------------------------
    void Awake()
    {
        HucreleriOlustur();
        SahayiSifirla();
    }

    void HucreleriOlustur()
    {
        hucreler = new Renderer[N, N];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
            {
                var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
                go.name = $"Hucre_{i}_{j}";
                go.transform.SetParent(transform, false);
                go.transform.localPosition = HucreYerel(i, j);
                go.transform.localScale = new Vector3(hucreBoyu * 0.95f, 0.5f, hucreBoyu * 0.95f);
                hucreler[i, j] = go.GetComponent<Renderer>();
            }
    }

    // Satır i kuzeyden güneye, sütun j batıdan doğuya gider; saha merkezi Saha nesnesinin konumu.
    Vector3 HucreYerel(int i, int j) =>
        new Vector3((j - (N - 1) / 2f) * hucreBoyu, 0f, ((N - 1) / 2f - i) * hucreBoyu);

    public Vector3 HucreMerkezi(int i, int j) => transform.TransformPoint(HucreYerel(i, j));

    /// <summary>Yeni bölüm: her şeyi sıfırla.</summary>
    public void SahayiSifirla()
    {
        sicaklik = new float[N, N];
        yas = new float[N, N];
        bilinen = new float[N, N];
        yanlisSicaklik = new float[N, N];
        yanlisOmur = new int[N, N];
        elendi = new bool[N, N];
        efektSayac = new int[N, N];
        efektTip = new int[N, N];
        yanginSayisi = 0;
        yanginBilinen = yanginKacirilan = yanginGorulmeyen = 0;
        sahteOkumaSayisi = 0;
        tespitT = new float[N, N];
        tespitAdim = new int[N, N];
        adimSayaci = 0;

        // Keşifli başlangıç (exploring starts): sadece EĞİTİMDE, bazı bölümler önceden ısınmış
        // 1-2 hücreyle başlar. Böylece ajan nadir görülen "sıcak anomali var" durumlarını daha sık
        // yaşar ve bu durumlardaki Q değerlerini sağlam öğrenir. Değerlendirmede (öğrenme kapalı) etkisi yoktur.
        if (kesifliBaslangic && QTablosu.ogrenmeAktif && Random.value < kesifliBaslangicOlasiligi)
        {
            int adet = Random.Range(1, 3);
            for (int k = 0; k < adet; k++)
            {
                int i = Random.Range(0, N), j = Random.Range(0, N);
                sicaklik[i, j] = Random.Range(3, 23);
            }
        }
        dogrulananSayisi = 0;
        dogrulamaSicaklikToplami = 0f;
        yanlisToplam = 0;
        yanlisElenen = 0;
    }

    bool BosMu(int i, int j) => sicaklik[i, j] == 0f && yanlisSicaklik[i, j] == 0f;

    /// <summary>
    /// Dünyayı bir adım ilerletir: yaşlanma, ısınma, yangın, yanlış kaynakların ömrü, yeni kaynaklar.
    /// Bu adımda çıkan yangın sayısını döndürür.
    /// </summary>
    [HideInInspector] public System.Collections.Generic.List<Vector2Int> sonYanginlar =
        new System.Collections.Generic.List<Vector2Int>(); // bu adımda yangın çıkan hücreler

    public int AdimIlerle()
    {
        int yeniYangin = 0;
        sonYanginlar.Clear();
        sonYanginBilgi.Clear();
        adimSayaci++;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
            {
                yas[i, j] += 1f;
                if (efektSayac[i, j] > 0) efektSayac[i, j]--;

                if (sicaklik[i, j] > 0f)
                {
                    sicaklik[i, j] += 1f;
                    if (bilinen[i, j] > 0f) bilinen[i, j] += 1f;

                    if (sicaklik[i, j] >= Tmax) // doğrulanmadan yangına döndü
                    {
                        yeniYangin++;
                        int tur = bilinen[i, j] > 0f ? 0 : (yas[i, j] < sicaklik[i, j] - 1f ? 1 : 2);
                        if (tur == 0) yanginBilinen++; else if (tur == 1) yanginKacirilan++; else yanginGorulmeyen++;
                        var bilgi = new YanginBilgi { i = i, j = j, tur = tur,
                            tespitSicakligi = tur == 0 ? tespitT[i, j] : 0f,
                            tespittenBeri = tur == 0 ? adimSayaci - tespitAdim[i, j] : 0 };
                        for (int a = 0; a < N; a++)
                            for (int b = 0; b < N; b++)
                                if ((a != i || b != j) && bilinen[a, b] > 0f)
                                { bilgi.digerBilinen++; if (sicaklik[a, b] > 0f) bilgi.digerGercek++; }
                        sonYanginBilgi.Add(bilgi);
                        sonYanginlar.Add(new Vector2Int(i, j));
                        sicaklik[i, j] = 0f;
                        bilinen[i, j] = 0f;
                        efektSayac[i, j] = 6; efektTip[i, j] = 1;
                        if (yanginLogla) Debug.Log($"YANGIN! Hücre ({i},{j})");
                    }
                }
                else if (bilinen[i, j] > 0f)
                {
                    // Sistem termal görüntüden kaynağın türünü bilemez: her bilinen anomalinin
                    // ısındığını varsayarak tahminini artırır (yanlış kaynaklar için de).
                    bilinen[i, j] = Mathf.Min(bilinen[i, j] + 1f, Tmax);
                }

                if (yanlisSicaklik[i, j] > 0f && --yanlisOmur[i, j] <= 0)
                {
                    // Yanlış kaynak kayboldu (araç gitti, güneş çekildi). Sistem bunu ancak
                    // tekrar baktığında anlar; bilinen tahmin bir sonraki taramaya kadar kalır.
                    yanlisSicaklik[i, j] = 0f;
                    elendi[i, j] = false;
                }
            }

        if (Random.value < pTutusma) // rastgele boş bir hücre ısınmaya başlar
        {
            for (int deneme = 0; deneme < 50; deneme++)
            {
                int i = Random.Range(0, N), j = Random.Range(0, N);
                if (BosMu(i, j)) { sicaklik[i, j] = 1f; break; }
            }
        }

        if (yanlisKaynaklar && Random.value < pYanlis) // rastgele bir yerde yanlış sıcak kaynak belirir
        {
            for (int deneme = 0; deneme < 50; deneme++)
            {
                int i = Random.Range(0, N), j = Random.Range(0, N);
                if (BosMu(i, j))
                {
                    yanlisSicaklik[i, j] = Random.Range(yanlisSicaklikAraligi.x, yanlisSicaklikAraligi.y);
                    yanlisOmur[i, j] = Random.Range(yanlisOmurAraligi.x, yanlisOmurAraligi.y + 1);
                    elendi[i, j] = false;
                    yanlisToplam++;
                    break;
                }
            }
        }

        yanginSayisi += yeniYangin;
        return yeniYangin;
    }

    static float Gauss() // Box-Muller
    {
        float u1 = 1f - Random.value, u2 = Random.value;
        return Mathf.Sqrt(-2f * Mathf.Log(u1)) * Mathf.Cos(2f * Mathf.PI * u2);
    }

    /// <summary>GENİŞ zoom (TERMAL): 3x3 alanı tarar. Yeni fark edilen anomali sayısını döndürür.</summary>
    public int GenisTara(int ci, int cj) => GenisTara(ci, cj, out _);

    /// <summary>
    /// Aynı tarama; ek olarak yeni fark edilen her anomali için (1 - T/Tmax) toplamını verir.
    /// Termal sensör gerçek ısınma ile yanlış kaynağı AYIRT ETMEZ: ikisi de anomali olarak görünür.
    /// </summary>
    public int GenisTara(int ci, int cj, out float erkenlikToplami)
    {
        erkenlikToplami = 0f;
        int yeniTespit = 0;
        for (int di = -1; di <= 1; di++)
            for (int dj = -1; dj <= 1; dj++)
            {
                int i = ci + di, j = cj + dj;
                if (i < 0 || i >= N || j < 0 || j >= N) continue;

                yas[i, j] = 0f;

                // Termal kamerada görünen sıcaklık: gerçek ısınma ya da (elenmemiş) yanlış kaynak
                float gorunen = sicaklik[i, j] > 0f ? sicaklik[i, j]
                              : (yanlisSicaklik[i, j] > 0f && !elendi[i, j]) ? yanlisSicaklik[i, j]
                              : 0f;

                // Sensör gürültüsü: sıcak bir şey varsa okuma gürültülü; boş hücre bazen sahte sıcak nokta gibi görünür
                if (gorunen > 0f && termalGurultu > 0f)
                    gorunen = Mathf.Max(0.5f, gorunen + Gauss() * termalGurultu);
                else if (gorunen == 0f && termalSahteOkuma > 0f && Random.value < termalSahteOkuma)
                { gorunen = Random.Range(2f, 6f); sahteOkumaSayisi++; }

                if (gorunen > 0f)
                {
                    float p = Mathf.Min(1f, gorunen / Talgi);
                    if (Random.value < p)
                    {
                        if (bilinen[i, j] == 0f)
                        {
                            tespitT[i, j] = gorunen; tespitAdim[i, j] = adimSayaci;
                            yeniTespit++;
                            erkenlikToplami += 1f - gorunen / Tmax;
                        }
                        bilinen[i, j] = gorunen;
                    }
                }
                else bilinen[i, j] = 0f; // sıcak bir şey yok: eski tahmini temizle
            }
        return yeniTespit;
    }

    /// <summary>
    /// YAKIN zoom (OPTİK): tek hücreyi yüksek çözünürlükte inceler.
    ///   Gerçek ısınma -> doğrulanır, o anki sıcaklık döndürülür (> 0).
    ///   Yanlış kaynak -> yanlış alarm elenir, -1 döndürülür.
    ///   Hiçbir şey yok -> 0 döndürülür.
    /// </summary>
    public float YakinDogrula(int i, int j)
    {
        yas[i, j] = 0f;
        bilinen[i, j] = 0f;
        float T = sicaklik[i, j];
        if (T > 0f)
        {
            sicaklik[i, j] = 0f; // müdahale ekibi soğuttu
            dogrulananSayisi++;
            dogrulamaSicaklikToplami += T;
            efektSayac[i, j] = 6; efektTip[i, j] = 2;
            return T;
        }
        if (yanlisSicaklik[i, j] > 0f && !elendi[i, j])
        {
            elendi[i, j] = true; // optik görüntü: tehlike yok, bir daha alarm üretme
            yanlisElenen++;
            efektSayac[i, j] = 6; efektTip[i, j] = 3;
            return -1f;
        }
        return 0f;
    }

    /// <summary>Kör nokta cezası için: 0..1 arası ortalama normalize yaş.</summary>
    public float OrtalamaNormalYas(float tavan = 50f)
    {
        float toplam = 0f;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                toplam += Mathf.Min(yas[i, j], tavan) / tavan;
        return toplam / (N * N);
    }

    // ------------------------------------------------------------------
    void Update()
    {
        if (!testModu) return;
        testSayac += Time.deltaTime;
        if (testSayac >= testAdimSuresi)
        {
            testSayac = 0f;
            AdimIlerle();
        }
    }

    void LateUpdate() // her karede hücre renklerini güncelle (sadece görsel)
    {
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
            {
                Color c;
                if (efektSayac[i, j] > 0)
                    c = efektTip[i, j] == 1 ? yanginRengi : efektTip[i, j] == 2 ? dogrulamaRengi : elemeRengi;
                else if (sicaklik[i, j] > 0f)
                    c = Color.Lerp(komurRengi, sicakRengi, sicaklik[i, j] / Tmax);
                else if (yanlisSicaklik[i, j] > 0f)
                    c = Color.Lerp(komurRengi, yanlisRengi, elendi[i, j] ? 0.25f : 0.8f); // elendi: soluk mavi
                else
                    c = komurRengi;
                hucreler[i, j].material.color = c;

                // Kameraların FARK ETTİĞİ (bilinen) anomaliler yükseltilerek gösterilir.
                var s = hucreler[i, j].transform.localScale;
                s.y = bilinen[i, j] > 0f ? 2f : 0.5f;
                hucreler[i, j].transform.localScale = s;
            }
    }
}
