using System.Collections.Generic;
using System.IO;
using UnityEngine;

/// <summary>
/// TABLO TABANLI Q-ÖĞRENME (sinir ağı yok).
/// Bütün kameralar tek bir Q tablosunu paylaşır. Durum ve eylem sayısı küçük olduğu için
/// tablo bellekte bir sözlük olarak tutulur ve diske kaydedilir.
///
///   Durum (36):  aciliyet (0..3) x en yaşlı sorumlu bölge (0..8)
///   Eylem (10):  0..8 = o bölgenin merkezine Pan/Tilt hedefi + GENİŞ zoom (termal tarama)
///                9    = sorumlu olduğum en sıcak bilinen anomaliye Pan/Tilt hedefi + YAKIN zoom (optik doğrulama)
///   Güncelleme:  yarı-Markov (SMDP) Q-öğrenme
///                Q(s,a) <- Q(s,a) + alfa * [ R + gamma^tau * max_a' Q(s',a') - Q(s,a) ]
///                R = eylem süresince toplanan indirimli ödül, tau = eylemin sürdüğü adım sayısı
/// </summary>
public static class QTablosu
{
    public const int EYLEM = 10;
    public const int DURUM = 36;

    public static float[,] Q = new float[DURUM, EYLEM];
    public static int[,] ziyaret = new int[DURUM, EYLEM];
    public static long kararSayisi;

    // Ayarlar (QOgrenmeYoneticisi Inspector'dan doldurur)
    public static float gamma = 0.95f;
    public static float alfaMin = 0.01f;
    public static float epsBas = 1f, epsSon = 0.05f;
    public static long epsKarar = 60000;   // epsilon bu kadar kararda epsBas'tan epsSon'a iner
    public static bool ogrenmeAktif = true;

    public static float Epsilon =>
        ogrenmeAktif ? Mathf.Max(epsSon, epsBas - (epsBas - epsSon) * kararSayisi / (float)epsKarar) : 0f;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    public static void Sifirla()
    {
        Q = new float[DURUM, EYLEM];
        ziyaret = new int[DURUM, EYLEM];
        kararSayisi = 0;
    }

    public static int GecerliEylemSayisi(int durum) => durum / 9 > 0 ? 10 : 9; // anomali yoksa "doğrula" yok

    /// <param name="yasak">Bu durumda seçilemeyen eylem (ör. kameranın zaten bulunduğu bölge), yoksa -1.</param>
    public static float MaksQ(int s, int yasak = -1)
    {
        int g = GecerliEylemSayisi(s); float m = float.NegativeInfinity;
        for (int a = 0; a < g; a++) if (a != yasak) m = Mathf.Max(m, Q[s, a]);
        return m;
    }

    public static int Sec(int s, int yasak = -1)
    {
        int g = GecerliEylemSayisi(s);
        if (Random.value < Epsilon)
        {
            int r = Random.Range(0, g);
            while (r == yasak) r = Random.Range(0, g);
            return r;
        }
        int en = 0; float m = float.NegativeInfinity;
        for (int a = 0; a < g; a++)
        {
            if (a == yasak) continue;
            float v = Q[s, a] + Random.value * 1e-5f; // eşitlikte rastgele seç
            if (v > m) { m = v; en = a; }
        }
        return en;
    }

    public static void Guncelle(int s, int a, float hedef)
    {
        if (!ogrenmeAktif) return;
        int n = ++ziyaret[s, a];
        float alfa = Mathf.Max(alfaMin, 1f / Mathf.Pow(n, 0.7f)); // ziyaretle azalan öğrenme hızı
        Q[s, a] += alfa * (hedef - Q[s, a]);
    }

    // ---- Disk ----
    public static void Kaydet(string yol)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(yol));
        using var w = new BinaryWriter(File.Open(yol, FileMode.Create));
        w.Write(kararSayisi);
        for (int s = 0; s < DURUM; s++)
            for (int a = 0; a < EYLEM; a++) { w.Write(Q[s, a]); w.Write(ziyaret[s, a]); }
    }

    public static bool Yukle(string yol)
    {
        if (!File.Exists(yol)) return false;
        using var r = new BinaryReader(File.Open(yol, FileMode.Open));
        kararSayisi = r.ReadInt64();
        for (int s = 0; s < DURUM; s++)
            for (int a = 0; a < EYLEM; a++) { Q[s, a] = r.ReadSingle(); ziyaret[s, a] = r.ReadInt32(); }
        return true;
    }

    /// <summary>Raporda kullanmak için tabloyu okunur metin olarak yazar.</summary>
    public static void MetinOlarakKaydet(string yol)
    {
        using var w = new StreamWriter(yol);
        w.WriteLine("aciliyet;en_yasli_bolge;en_iyi_eylem;" + string.Join(";", EylemAdlari));
        for (int s = 0; s < DURUM; s++)
        {
            int g = GecerliEylemSayisi(s), en = 0;
            for (int a = 1; a < g; a++) if (Q[s, a] > Q[s, en]) en = a;
            var deger = new List<string>();
            for (int a = 0; a < EYLEM; a++) deger.Add(a < g ? Q[s, a].ToString("F2") : "-");
            w.WriteLine($"{s / 9};{s % 9};{EylemAdlari[en]};" + string.Join(";", deger));
        }
    }

    public static readonly string[] EylemAdlari =
        { "tara_B0", "tara_B1", "tara_B2", "tara_B3", "tara_B4", "tara_B5", "tara_B6", "tara_B7", "tara_B8", "dogrula" };
}
