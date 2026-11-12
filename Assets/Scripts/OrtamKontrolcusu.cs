using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;

/// <summary>
/// ADIM 7.2 - Ortam kontrolcüsü: zamanı ve bölümleri yönetir.
/// Her MDP adımında sırasıyla:
///   1) Dünya bir adım ilerler (ısınma, yangın)  -> yangın ve kör nokta cezaları
///   2) Bölüm süresi dolduysa bölümü bitirir ve sahayı sıfırlar
///   3) Bütün kameralardan karar ister (kameralar OnActionReceived'da hareket edip bakar)
/// </summary>
public class OrtamKontrolcusu : MonoBehaviour
{
    public SahaYoneticisi saha;
    public List<PTZKameraAjani> kameralar = new List<PTZKameraAjani>();

    [Header("Zaman")]
    public int maxAdim = 200;      // bir bölümdeki MDP adımı (Python'daki max_adim)
    public int kararAraligi = 10;  // kaç fizik karesinde bir MDP adımı (elle test: 10, eğitim: 1)

    [Header("Ödül")]
    public float cezaYangin = 20f;
    public float cezaKorNokta = 0.5f;
    [Tooltip("Kapalı: PPO için her kameraya ayrı ödül. Açık: MA-POCA için takım ödülü.")]
    public bool takimOdulu = false;
    [Tooltip("Görüş alanları çakışan her hücre için adım başına ceza (yardımcı ödül, gerçek ödüle sayılmaz).")]
    public float cezaCakisma = 0.02f;
    [Tooltip("Açık: gözlemin 4. katmanı öteki kameranın TÜM görüş alanını gösterir (Adım 12+). " +
             "Kapalı: sadece merkez hücresini gösterir (poca_iki_kamera modeli bununla eğitildi).")]
    public bool gorusAlaniGozlemi = true;
    [Tooltip("Açık: 'yaklaşıyorsun' ipucu sadece sıcak noktaya en yakın kameraya verilir (Adım 12+). " +
             "Kapalı: her kameraya verilir (poca_iki_kamera modeli bununla eğitildi).")]
    public bool sorumluKameraSekillendirme = true;

    [Header("Değerlendirme (eğitim DIŞINDA kullanılır)")]
    [Tooltip("Python bağlı değilken simülasyon hızı. 10 = 10 kat hızlı.")]
    public float zamanOlcegi = 1f;
    [Tooltip("Kaç bölümde bir Console'a ortalama sonuç yazılacağı (8 ortamın toplamı).")]
    public int raporAraligi = 80;

    // 8 ortamın sonuçlarını birlikte toplamak için ortak (static) sayaçlar
    static int topBolum, topYangin, topDogrulanan;
    static float topOdul, topSicaklik, topCakisma;
    static int topYanlisToplam, topYanlisElenen;
    static int topYBilinen, topYKacirilan, topYGorulmeyen; // yangın teşhisi
    // Pencere (son raporAraligi bölüm) ortalamaları: öğrenme eğrisi için
    static int pBolum, pYangin, pDogrulanan;
    static float pOdul, pSicaklik, pCakisma;
    public static int kalkanMudahalesi; // güvenlik kalkanının RL kararını geçersiz kıldığı adım sayısı
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    static void SayaclariSifirla()
    {
        topBolum = topYangin = topDogrulanan = kalkanMudahalesi = 0;
        topYanlisToplam = topYanlisElenen = 0;
        topYBilinen = topYKacirilan = topYGorulmeyen = 0;
        pBolum = pYangin = pDogrulanan = 0; pOdul = pSicaklik = pCakisma = 0f;
        topOdul = topSicaklik = topCakisma = 0f;
    }
    float bolumCakisma; // bu bölümde adım adım toplanan çakışan hücre sayısı

    [Header("İzleme")]
    public int adim;
    public float bolumOdulu;

    SimpleMultiAgentGroup grup;
    int fizikSayac;

    void Start()
    {
        saha.testModu = false; // artık zamanı biz yönetiyoruz
        if (!Academy.Instance.IsCommunicatorOn) Time.timeScale = zamanOlcegi; // eğitimde ML-Agents ayarlar
        grup = new SimpleMultiAgentGroup();
        foreach (var k in kameralar)
        {
            k.saha = saha;
            k.kontrolcu = this;
            if (takimOdulu) grup.RegisterAgent(k);
        }
        BolumuSifirla();
    }

    void FixedUpdate()
    {
        fizikSayac++;
        if (fizikSayac % kararAraligi != 0) return;

        // 1) Dünya ilerler
        int yangin = saha.AdimIlerle();
        if (yanginDetayLog) YanginDetayiYaz();
        OdulVerHerkese(-cezaYangin * yangin - cezaKorNokta * saha.OrtalamaNormalYas());
        QYerelOdulleri();

        // Çakışmayı her durumda ölç (ceza kapalı olsa bile) -> raporda "üst üste binme" ölçütü
        int cakisan = kameralar.Count > 1 ? CakisanHucreSayisi() : 0;
        bolumCakisma += cakisan;

        // Çakışma cezası: aynı yere iki kameranın bakması boşa emek.
        // "Gerçek" bölüm ödülüne SAYILMAZ (devriye ile adil kıyas için), sadece öğrenmeye yardım eder.
        if (cezaCakisma > 0f && kameralar.Count > 1)
        {
            float c = -cezaCakisma * cakisan;
            if (takimOdulu) grup.AddGroupReward(c);
            else foreach (var k in kameralar) k.AddReward(c);
        }

        // 2) Bölüm sonu
        adim++;
        if (adim >= maxAdim)
        {
            // TensorBoard'a "gerçek" başarı ölçütlerini yaz (şekillendirme ödülü HARİÇ).
            // Böylece farklı ödül ayarlarıyla yapılan eğitimleri adil karşılaştırabiliriz.
            var stats = Academy.Instance.StatsRecorder;
            stats.Add("Ortam/Gercek Bolum Odulu", bolumOdulu);
            stats.Add("Ortam/Yangin Sayisi", saha.yanginSayisi);
            stats.Add("Ortam/Dogrulanan Sayisi", saha.dogrulananSayisi);
            if (saha.dogrulananSayisi > 0)
                stats.Add("Ortam/Dogrulama Sicakligi", saha.dogrulamaSicaklikToplami / saha.dogrulananSayisi);
            stats.Add("Ortam/Cakisan Hucre", bolumCakisma / maxAdim);
            if (saha.yanlisKaynaklar)
            {
                stats.Add("Ortam/Yanlis Alarm Elenen", saha.yanlisElenen);
                topYanlisToplam += saha.yanlisToplam;
                topYanlisElenen += saha.yanlisElenen;
            }

            // Eğitim dışı değerlendirme özeti (Console)
            topBolum++; topOdul += bolumOdulu;
            topYangin += saha.yanginSayisi; topDogrulanan += saha.dogrulananSayisi;
            topSicaklik += saha.dogrulamaSicaklikToplami;
            topYBilinen += saha.yanginBilinen; topYKacirilan += saha.yanginKacirilan; topYGorulmeyen += saha.yanginGorulmeyen;
            topCakisma += bolumCakisma / maxAdim;
            pBolum++; pOdul += bolumOdulu; pYangin += saha.yanginSayisi; pDogrulanan += saha.dogrulananSayisi;
            pSicaklik += saha.dogrulamaSicaklikToplami; pCakisma += bolumCakisma / maxAdim;
            if (raporAraligi > 0 && pBolum >= raporAraligi)
            {
                float ps = pDogrulanan > 0 ? pSicaklik / pDogrulanan : 0f;
                string satir = $"{topBolum};{QTablosu.kararSayisi};{QTablosu.Epsilon:F3};{pOdul / pBolum:F2};" +
                               $"{(float)pYangin / pBolum:F3};{(float)pDogrulanan / pBolum:F2};{ps:F2};{pCakisma / pBolum:F3}";
                Debug.Log($"[PENCERE son {pBolum} bölüm] ödül {pOdul / pBolum:F1} | yangın {(float)pYangin / pBolum:F2} | " +
                          $"doğrulanan {(float)pDogrulanan / pBolum:F2} | doğrulama sıcaklığı {ps:F1} | " +
                          $"Q: ε={QTablosu.Epsilon:F2}, karar={QTablosu.kararSayisi}");
                if (QOgrenmeYoneticisi.EgriYolu != null)
                    System.IO.File.AppendAllText(QOgrenmeYoneticisi.EgriYolu, satir.Replace(',', '.') + "\n");
                pBolum = pYangin = pDogrulanan = 0; pOdul = pSicaklik = pCakisma = 0f;
            }
            if (raporAraligi > 0 && topBolum % raporAraligi == 0)
                Debug.Log($"[DEĞERLENDİRME] {topBolum} bölüm | ort. ödül {topOdul / topBolum:F1} | " +
                          $"ort. yangın {(float)topYangin / topBolum:F2} | ort. doğrulanan {(float)topDogrulanan / topBolum:F2} | " +
                          $"ort. doğrulama sıcaklığı {(topDogrulanan > 0 ? topSicaklik / topDogrulanan : 0f):F1} / {saha.Tmax} | " +
                          $"ort. çakışan hücre {topCakisma / topBolum:F2} | " +
                          $"kalkan müdahalesi/bölüm {(float)kalkanMudahalesi / topBolum:F2}" +
                          (saha.yanlisKaynaklar
                              ? $" | yanlış alarm: oluşan {(float)topYanlisToplam / topBolum:F2}, elenen {(float)topYanlisElenen / topBolum:F2}"
                              : "") +
                          $" | yangın teşhisi (toplam): bilinen-doğrulanamadı {topYBilinen}, bakıldı-fark edilmedi {topYKacirilan}, hiç bakılmadı {topYGorulmeyen}");

            if (takimOdulu) grup.GroupEpisodeInterrupted();
            else foreach (var k in kameralar) k.EpisodeInterrupted();
            BolumuSifirla();
            return;
        }

        // 3) Kameralardan karar iste
        foreach (var k in kameralar) k.RequestDecision();
    }

    [Header("Sorumluluk")]
    [Tooltip("Açıksa her kameranın sorumlu olduğu hücreler SABİT (saha köşegenle ikiye bölünür). Kapalıysa her hücre o an en yakın kameranın sorumluluğundadır (dinamik). Değiştirirsen Q tablosu sıfırdan eğitilmeli.")]
    public bool sabitSorumluluk = false;
    [Tooltip("Sabit sorumlulukta saha köşegen yerine BÖLGELERE göre bölünür: bir hücreden, o hücreyi tarayan bölgenin sahibi sorumludur. Köşelerde 'taranan ama sahibinin görmediği' hücre kalmaz. Değiştirirsen Q tablosu sıfırdan eğitilmeli.")]
    public bool bolgeBazliSorumluluk = false;

    [Header("Teşhis")]
    [Tooltip("Her yangında, nedenini ve o anda kameraların ne yaptığını Console'a yazar.")]
    public bool yanginDetayLog = true;

    void YanginDetayiYaz()
    {
        foreach (var y in saha.sonYanginBilgi)
        {
            string tur = y.tur == 0 ? "BİLİNEN-doğrulanamadı" : y.tur == 1 ? "bakıldı-fark edilmedi" : "hiç bakılmadı";
            var sb = new System.Text.StringBuilder();
            sb.Append($"[YANGIN] {name} hücre ({y.i},{y.j}) | {tur}");
            if (y.tur == 0)
                sb.Append($" | ilk fark edildiğinde {y.tespitSicakligi:F0}°C, {y.tespittenBeri} adım önce");
            sb.Append($" | o anda bilinen başka anomali {y.digerBilinen} (gerçek {y.digerGercek})");
            foreach (var k in kameralar)
            {
                int d = Mathf.Max(Mathf.Abs(k.Satir - y.i), Mathf.Abs(k.Sutun - y.j));
                bool sorumlu = EnYakinKameraMi(k, y.i, y.j);
                sb.Append($" | {k.name}: uzaklık {d}, {(sorumlu ? "SORUMLU" : "sorumlu değil")}, aciliyet {k.QAciliyet}, eylem {k.QEylemAdi}");
            }
            Debug.Log(sb.ToString());
        }
    }

    void BolumuSifirla()
    {
        adim = 0;
        bolumOdulu = 0f;
        bolumCakisma = 0f;
        saha.SahayiSifirla();
    }

    [Header("Q-öğrenme yerel ödülleri")]
    [Tooltip("Doğrulanmayı bekleyen (bilinen) her alarm için adım başına ceza katsayısı.")]
    public float qBeklemeCezasi = 0.3f;

    /// <summary>
    /// Q-öğrenme kameraları için YEREL ödül: her kamera sadece kendi sorumluluk alanındaki
    /// (kendisine en yakın hücrelerdeki) yangından, kör noktadan ve bekleyen alarmdan sorumlu tutulur.
    /// </summary>
    void QYerelOdulleri()
    {
        foreach (var k in kameralar)
        {
            if (k.heuristicModu != PTZKameraAjani.ElleKontrol.QOgrenme) continue;
            float yerel = 0f, kor = 0f, bekleyen = 0f;
            foreach (var y in saha.sonYanginlar)
                if (EnYakinKameraMi(k, y.x, y.y)) yerel -= cezaYangin;
            for (int i = 0; i < saha.N; i++)
                for (int j = 0; j < saha.N; j++)
                    if (EnYakinKameraMi(k, i, j))
                    {
                        kor += Mathf.Min(saha.yas[i, j], 50f) / 50f;
                        bekleyen += saha.bilinen[i, j];
                    }
            yerel -= cezaKorNokta * kor / (saha.N * saha.N);
            yerel -= qBeklemeCezasi * bekleyen / saha.Tmax;
            k.qOdul += yerel;
        }
    }

    // ---- Ödül dağıtımı ----
    public void OdulVer(PTZKameraAjani kamera, float r)
    {
        kamera.qOdul += r; // Q-öğrenme için kameranın kendi (algılama, doğrulama, motor) ödülü
        if (takimOdulu) grup.AddGroupReward(r);
        else kamera.AddReward(r);
        bolumOdulu += r;
    }

    /// <summary>
    /// Şekillendirme ödülü: "gerçek" bölüm ödülüne SAYILMAZ.
    /// Takım modunda bile BİREYSEL verilir: "sıcak noktaya yaklaşan sensin" ipucu
    /// kameranın kendi hareketine ait. MA-POCA bireysel ve takım ödüllerini birlikte kullanır.
    /// </summary>
    public void SekillendirmeOdulu(PTZKameraAjani kamera, float r)
    {
        kamera.AddReward(r);
    }

    public void OdulVerHerkese(float r)
    {
        if (takimOdulu) grup.AddGroupReward(r);
        else foreach (var k in kameralar) k.AddReward(r);
        bolumOdulu += r;
    }

    // ---- Kameraların gözlemi için ----
    /// <summary>Gözlemin 4. katmanı: (i,j) başka bir kameranın GÖRÜŞ ALANINDA mı? (sadece merkezi değil)</summary>
    public bool BaskaKameraBakiyorMu(PTZKameraAjani ben, int i, int j)
    {
        foreach (var k in kameralar)
        {
            if (k == ben) continue;
            if (gorusAlaniGozlemi ? k.GorusAlanindaMi(i, j) : (k.Satir == i && k.Sutun == j)) return true;
        }
        return false;
    }

    /// <summary>
    /// (i,j) sıcak noktasına en yakın kamera "ben" miyim? Eşitlikte listede önce gelen kazanır.
    /// Şekillendirme ipucu sadece sorumlu kameraya verilir -> iki kamera aynı noktaya koşmaz.
    /// </summary>
    public bool EnYakinKameraMi(PTZKameraAjani ben, int i, int j)
    {
        if (sabitSorumluluk && kameralar.Count == 2)
        {
            // Sabit bölge paylaşımı: köşegenin sol-üst yarısı listedeki 1. kameranın (Kamera_A),
            // sağ-alt yarısı 2. kameranın (Kamera_B). Köşegen hücreleri ikiye bölünür.
            int sahip;
            if (bolgeBazliSorumluluk)
            {
                // Her hücre, onu tarayan bölgenin sahibine aittir: B0-B3 Kamera_A, B4-B8 Kamera_B (33 / 31 hücre)
                int b = (i < 3 ? 0 : i < 6 ? 1 : 2) * 3 + (j < 3 ? 0 : j < 6 ? 1 : 2);
                sahip = b <= 3 ? 0 : 1;
            }
            else sahip = (i + j < saha.N - 1 || (i + j == saha.N - 1 && i < saha.N / 2)) ? 0 : 1;
            return kameralar.IndexOf(ben) == sahip;
        }
        int benimMesafem = Mathf.Max(Mathf.Abs(i - ben.Satir), Mathf.Abs(j - ben.Sutun));
        foreach (var k in kameralar)
        {
            if (k == ben) continue;
            int m = Mathf.Max(Mathf.Abs(i - k.Satir), Mathf.Abs(j - k.Sutun));
            if (m < benimMesafem) return false;
            if (m == benimMesafem && kameralar.IndexOf(k) < kameralar.IndexOf(ben)) return false;
        }
        return true;
    }

    /// <summary>Kameraların görüş alanlarının üst üste bindiği hücre sayısı.</summary>
    int CakisanHucreSayisi()
    {
        int cakisma = 0;
        for (int i = 0; i < saha.N; i++)
            for (int j = 0; j < saha.N; j++)
            {
                int kac = 0;
                foreach (var k in kameralar) if (k.GorusAlanindaMi(i, j)) kac++;
                if (kac > 1) cakisma += kac - 1;
            }
        return cakisma;
    }
}
