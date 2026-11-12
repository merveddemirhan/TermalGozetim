using System.IO;
using UnityEngine;

/// <summary>
/// Sahnedeki tek bir nesneye eklenir: Q-öğrenme ayarları, tablonun kaydedilmesi/yüklenmesi ve öğrenme eğrisi CSV'si.
/// </summary>
public class QOgrenmeYoneticisi : MonoBehaviour
{
    [Header("Q-öğrenme ayarları")]
    public bool ogrenmeAktif = true;     // kapalı: tablo değişmez, epsilon = 0 (değerlendirme)
    public float gamma = 0.95f;
    public float alfaMin = 0.01f;
    public float epsilonBaslangic = 1f;
    public float epsilonSon = 0.05f;
    public int epsilonKararSayisi = 60000;
    [Tooltip("Rastgele tohum (random seed). 0 = her çalıştırmada farklı. Kararlılık analizi için 1, 2, 3 gibi farklı tohumlarla ayrı ayrı eğit.")]
    public int tohum = 0;

    [Header("Dosyalar (proje klasöründeki QTablolari klasörüne)")]
    public string tabloDosyasi = "qtablo.bin";
    public bool baslangictaYukle = true;
    public float otomatikKayitSaniye = 60f;
    public string egriDosyasi = "ogrenme_egrisi.csv";

    public static string EgriYolu;
    float sayac;

    string Klasor => Path.Combine(Application.dataPath, "..", "QTablolari");
    string TabloYolu => Path.Combine(Klasor, tabloDosyasi);

    void Awake()
    {
        if (tohum != 0) { Random.InitState(tohum); Debug.Log($"[Q] Rastgele tohum: {tohum}"); }
        QTablosu.ogrenmeAktif = ogrenmeAktif;
        QTablosu.gamma = gamma;
        QTablosu.alfaMin = alfaMin;
        QTablosu.epsBas = epsilonBaslangic;
        QTablosu.epsSon = epsilonSon;
        QTablosu.epsKarar = epsilonKararSayisi;
        if (baslangictaYukle && QTablosu.Yukle(TabloYolu))
            Debug.Log($"[Q] Tablo yüklendi: {TabloYolu} (karar sayısı {QTablosu.kararSayisi})");
        else if (!ogrenmeAktif)
            Debug.LogError($"[Q] DİKKAT: Öğrenme kapalı ama tablo yüklenemedi! Değerlendirme BOŞ tabloyla (eğitilmemiş ajanla) yapılıyor. " +
                           $"Aranan dosya: {TabloYolu} | Baslangicta Yukle: {baslangictaYukle}");
        else
            Debug.Log($"[Q] Yeni (boş) tablo ile başlanıyor: {TabloYolu}");

        Directory.CreateDirectory(Klasor);
        EgriYolu = ogrenmeAktif ? Path.Combine(Klasor, egriDosyasi) : null;
        if (EgriYolu != null && !File.Exists(EgriYolu))
            File.WriteAllText(EgriYolu, "toplam_bolum;karar;epsilon;odul;yangin;dogrulanan;dogrulama_sicakligi;cakisma\n");
    }

    void Update()
    {
        if (!ogrenmeAktif) return;
        sayac += Time.unscaledDeltaTime;
        if (sayac >= otomatikKayitSaniye) { sayac = 0f; Kaydet(); }
    }

    void OnApplicationQuit() => Kaydet();
    void OnDisable() => Kaydet();

    public void Kaydet()
    {
        if (!ogrenmeAktif) return;
        QTablosu.Kaydet(TabloYolu);
        QTablosu.MetinOlarakKaydet(Path.Combine(Klasor, "qtablo_okunur.csv"));
    }
}
