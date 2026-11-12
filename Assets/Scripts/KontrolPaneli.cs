using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

/// <summary>
/// KONTROL PANELİ: Sahnedeki boş bir nesneye eklenir. Eğitimi/öğrenmeyi etkilemez, sadece gösterim içindir.
///  * Tek bir ortama odaklanır (diğer 7 ortamı gizler), ana kamerayı ona çevirir.
///  * Ekranda canlı bilgi paneli (HUD): bölüm istatistikleri, kameraların modu ve kararı, renk açıklaması.
///  * Klavye ile canlı kontrol:
///      1 / 2 / 3 : politika  -> Q-öğrenme / Devriye (kural) / Rastgele
///      K         : güvenlik kalkanı aç/kapa
///      G         : termal sensör gürültüsü  -> yok / orta / yüksek
///      + / -     : simülasyon hızı
///      Boşluk    : duraklat / devam
///      H         : paneli gizle / göster
/// </summary>
public class KontrolPaneli : MonoBehaviour
{
    [Header("Odak")]
    [Tooltip("Gösterilecek ortamın kontrolcüsü. Boşsa sahnedeki ilk ortam seçilir.")]
    public OrtamKontrolcusu odakOrtam;
    public bool digerOrtamlariGizle = true;
    public bool kamerayiYerlestir = true;
    [Tooltip("Kamera yüksekliği ve geriye uzaklığı (saha boyutuna göre çarpan).")]
    public float kameraYukseklik = 1.25f, kameraGeri = 0.75f;

    [Header("Başlangıç")]
    [Tooltip("Saniyede kaç MDP adımı oynatılsın (1 adım = kameraların bir kararı + dünyanın bir adımı).")]
    public float baslangicHizi = 2f;
    public bool paneliGoster = true;

    static readonly float[] HIZLAR = { 0.5f, 1f, 2f, 4f, 8f, 15f, 30f, 60f }; // adım / saniye
    static readonly (string ad, float g, float s)[] GURULTU =
        { ("yok", 0f, 0f), ("orta (σ=2 °C, sahte %0,5)", 2f, 0.005f), ("yüksek (σ=4 °C, sahte %1)", 4f, 0.01f) };

    float hiz; bool duraklat; int gurultuSeviye;
    SahaYoneticisi saha;

    // Bölüm istatistikleri (sahne başından beri, sadece odak ortam)
    int sonAdim, bolumSayisi, topYangin, topDogrulanan, topElenen, topYanlis;
    float topSicaklik;
    string sonBolumOzeti = "-";
    // bir önceki karedeki sayaçlar (bölüm bittiğinde sıfırlanmadan önceki son değerler)
    int oYangin, oDogrulanan, oElenen, oYanlis; float oSicaklik, oOdul;

    GUIStyle baslik, metin, kucuk, kutu;
    Texture2D bosDoku;

    System.Collections.IEnumerator Start()
    {
        if (odakOrtam == null) odakOrtam = FindFirstObjectByType<OrtamKontrolcusu>();
        if (odakOrtam == null) { Debug.LogError("[Panel] Sahnede OrtamKontrolcusu bulunamadı."); enabled = false; yield break; }
        saha = odakOrtam.saha;

        if (digerOrtamlariGizle)
            foreach (var o in FindObjectsByType<OrtamKontrolcusu>(FindObjectsSortMode.None))
                if (o != odakOrtam) KokNesne(o).SetActive(false);

        if (kamerayiYerlestir && Camera.main != null && saha != null)
        {
            Vector3 a = saha.HucreMerkezi(0, 0), b = saha.HucreMerkezi(saha.N - 1, saha.N - 1);
            Vector3 merkez = (a + b) / 2f;
            float boy = saha.N * saha.hucreBoyu;
            Camera.main.transform.position = merkez + new Vector3(0f, boy * kameraYukseklik, -boy * kameraGeri);
            Camera.main.transform.LookAt(merkez);
        }
        hiz = baslangicHizi;
        // OrtamKontrolcusu.Start zaman ölçeğini kendi ayarına çeker; birkaç kare bekleyip bizimkini uygula
        yield return null; yield return null;
        HiziUygula();
    }

    GameObject KokNesne(OrtamKontrolcusu o)
    {
        // Ortam prefab'ının kökü: kontrolcünün en üstteki ebeveyni
        Transform t = o.transform;
        while (t.parent != null) t = t.parent;
        return t.gameObject;
    }

    // Bir MDP adımı = kararAraligi fizik adımı. Saniyede 'hiz' adım için gereken zaman ölçeği:
    void HiziUygula() => Time.timeScale = duraklat ? 0f : Mathf.Min(100f, hiz * Time.fixedDeltaTime * Mathf.Max(1, odakOrtam.kararAraligi));

    void PolitikaAyarla(PTZKameraAjani.ElleKontrol mod)
    {
        foreach (var k in odakOrtam.kameralar) { k.heuristicModu = mod; k.QSifirla(); }
    }

    void Update()
    {
        var kb = Keyboard.current;
        if (kb != null)
        {
            if (kb.digit1Key.wasPressedThisFrame) PolitikaAyarla(PTZKameraAjani.ElleKontrol.QOgrenme);
            if (kb.digit2Key.wasPressedThisFrame) PolitikaAyarla(PTZKameraAjani.ElleKontrol.Devriye);
            if (kb.digit3Key.wasPressedThisFrame) PolitikaAyarla(PTZKameraAjani.ElleKontrol.Rastgele);
            if (kb.kKey.wasPressedThisFrame)
                foreach (var k in odakOrtam.kameralar) k.guvenlikKalkani = !k.guvenlikKalkani;
            if (kb.gKey.wasPressedThisFrame)
            {
                gurultuSeviye = (gurultuSeviye + 1) % GURULTU.Length;
                saha.termalGurultu = GURULTU[gurultuSeviye].g;
                saha.termalSahteOkuma = GURULTU[gurultuSeviye].s;
            }
            if (kb.equalsKey.wasPressedThisFrame || kb.numpadPlusKey.wasPressedThisFrame) HizDegistir(+1);
            if (kb.minusKey.wasPressedThisFrame || kb.numpadMinusKey.wasPressedThisFrame) HizDegistir(-1);
            if (kb.spaceKey.wasPressedThisFrame) { duraklat = !duraklat; HiziUygula(); }
            if (kb.hKey.wasPressedThisFrame) paneliGoster = !paneliGoster;
        }

        // Bölüm bitti mi? (adım sayacı başa döndü)
        if (odakOrtam.adim < sonAdim)
        {
            bolumSayisi++;
            topYangin += oYangin; topDogrulanan += oDogrulanan; topSicaklik += oSicaklik;
            topElenen += oElenen; topYanlis += oYanlis;
            sonBolumOzeti = $"yangın {oYangin} | doğrulanan {oDogrulanan} | elenen yanlış alarm {oElenen}/{oYanlis} | ödül {oOdul:F1}";
        }
        sonAdim = odakOrtam.adim;
        oYangin = saha.yanginSayisi; oDogrulanan = saha.dogrulananSayisi; oSicaklik = saha.dogrulamaSicaklikToplami;
        oElenen = saha.yanlisElenen; oYanlis = saha.yanlisToplam; oOdul = odakOrtam.bolumOdulu;
    }

    // Paneldeki yazı Saha'daki GERÇEK değerleri gösterir (Inspector'dan elle değiştirilmiş olsa bile)
    string GurultuMetni()
    {
        if (saha.termalGurultu <= 0f && saha.termalSahteOkuma <= 0f) return "yok";
        return $"<color=#ff8080>σ={saha.termalGurultu:0.#} °C, sahte okuma %{100f * saha.termalSahteOkuma:0.#}</color>";
    }

    void HizDegistir(int yon)
    {
        int i = System.Array.IndexOf(HIZLAR, hiz);
        if (i < 0) i = 2;
        hiz = HIZLAR[Mathf.Clamp(i + yon, 0, HIZLAR.Length - 1)];
        HiziUygula();
    }

    // ------------------------------------------------------------------ HUD
    void Stiller()
    {
        if (baslik != null) return;
        bosDoku = new Texture2D(1, 1); bosDoku.SetPixel(0, 0, Color.white); bosDoku.Apply();
        baslik = new GUIStyle(GUI.skin.label) { fontSize = 22, fontStyle = FontStyle.Bold, richText = true };
        metin = new GUIStyle(GUI.skin.label) { fontSize = 17, richText = true, wordWrap = true };
        kucuk = new GUIStyle(GUI.skin.label) { fontSize = 15, richText = true, wordWrap = true };
        kutu = new GUIStyle(GUI.skin.box);
        var arka = new Texture2D(1, 1); arka.SetPixel(0, 0, new Color(0f, 0f, 0f, 0.72f)); arka.Apply();
        kutu.normal.background = arka;
    }

    void Renk(Rect r, Color c) { var e = GUI.color; GUI.color = c; GUI.DrawTexture(r, bosDoku); GUI.color = e; }

    void OnGUI()
    {
        if (!paneliGoster || odakOrtam == null) return;
        Stiller();
        float olcek = Screen.height / 1080f;
        GUI.matrix = Matrix4x4.Scale(new Vector3(olcek, olcek, 1f));
        float W = 540f, x = 15f, y = 15f;

        // --- Sol panel: durum ---
        GUI.Box(new Rect(x - 5, y - 5, W, 640), GUIContent.none, kutu);
        GUILayout.BeginArea(new Rect(x, y, W - 15, 630));
        GUILayout.Label("Bi-spektral PTZ kameralarla aktif yangın gözetimi", baslik);
        var k0 = odakOrtam.kameralar.Count > 0 ? odakOrtam.kameralar[0] : null;
        string pol = k0 == null ? "-" : k0.heuristicModu == PTZKameraAjani.ElleKontrol.QOgrenme ? "<color=#7fd6ff>Tablo Q-öğrenme (öğrenilmiş)</color>"
                   : k0.heuristicModu == PTZKameraAjani.ElleKontrol.Devriye ? "<color=#ffc060>Devriye (klasik kural)</color>"
                   : k0.heuristicModu == PTZKameraAjani.ElleKontrol.Rastgele ? "<color=#bbbbbb>Rastgele</color>" : k0.heuristicModu.ToString();
        GUILayout.Label($"Politika: <b>{pol}</b>", metin);
        GUILayout.Label($"Güvenlik kalkanı: <b>{(k0 != null && k0.guvenlikKalkani ? "açık" : "kapalı")}</b>   |   " +
                        $"Sensör gürültüsü: <b>{GurultuMetni()}</b>", kucuk);
        GUILayout.Label($"Hız: <b>{(duraklat ? "DURAKLATILDI" : hiz + " adım/sn")}</b>   |   Adım <b>{odakOrtam.adim}</b> / {odakOrtam.maxAdim}", kucuk);
        GUILayout.Space(8);

        GUILayout.Label("<b>Bu bölüm</b>", metin);
        float ort = saha.dogrulananSayisi > 0 ? saha.dogrulamaSicaklikToplami / saha.dogrulananSayisi : 0f;
        GUILayout.Label($"Yangın: <b><color=#ffe34d>{saha.yanginSayisi}</color></b>    Doğrulanan ısınma: <b><color=#5df07f>{saha.dogrulananSayisi}</color></b>" +
                        $"  (ort. {ort:F1} °C)", metin);
        GUILayout.Label($"Yanlış alarm: oluşan {saha.yanlisToplam}, optikle elenen <b>{saha.yanlisElenen}</b>    Ödül: <b>{odakOrtam.bolumOdulu:F1}</b>", metin);
        GUILayout.Label($"Önceki bölüm: {sonBolumOzeti}", kucuk);
        GUILayout.Space(6);
        if (bolumSayisi > 0)
        {
            GUILayout.Label($"<b>Panel açıldığından beri ({bolumSayisi} bölüm)</b>", metin);
            GUILayout.Label($"Yangın / bölüm: <b>{(float)topYangin / bolumSayisi:F2}</b>   Doğrulanan / bölüm: <b>{(float)topDogrulanan / bolumSayisi:F2}</b>   " +
                            $"Ort. doğrulama: <b>{(topDogrulanan > 0 ? topSicaklik / topDogrulanan : 0f):F1} °C</b>   " +
                            $"Eleme: <b>{(topYanlis > 0 ? 100f * topElenen / topYanlis : 0f):F0}%</b>", kucuk);
            GUILayout.Space(6);
        }

        GUILayout.Label("<b>Kameralar</b>", metin);
        foreach (var k in odakOrtam.kameralar)
        {
            string mod = k.Zoom == 0 ? "<color=#ff9a5c>TERMAL tarama (geniş, 3x3)</color>" : "<color=#ffffff>OPTİK doğrulama (yakın, 1 hücre)</color>";
            string ac = k.QAciliyet == 0 ? "yok" : k.QAciliyet == 1 ? "düşük (<10 °C)" : k.QAciliyet == 2 ? "orta (10-18 °C)" : "<color=#ff5050>kritik (≥18 °C)</color>";
            string karar = k.heuristicModu == PTZKameraAjani.ElleKontrol.QOgrenme ? $"   Q kararı: <b>{EylemMetni(k.QEylemAdi)}</b>" : "";
            GUILayout.Label($"<b>{k.name}</b> ({k.Satir},{k.Sutun}) · {mod}\n   Sorumlu alanında alarm: {ac}{karar}", kucuk);
        }
        GUILayout.EndArea();

        // --- Sağ alt: renk açıklaması + tuşlar ---
        float sagX = Screen.width / olcek - 420f, sagY = Screen.height / olcek - 520f;
        GUI.Box(new Rect(sagX - 5, sagY - 5, 410, 510), GUIContent.none, kutu);
        GUILayout.BeginArea(new Rect(sagX, sagY, 400, 500));
        GUILayout.Label("<b>Renkler</b>", metin);
        Satir(new Color(0.15f, 0.15f, 0.15f), "Normal saha (kömür yığını)");
        Satir(new Color(1f, 0.35f, 0.1f), "Gerçek ısınma (koyulaştıkça sıcak)");
        Satir(new Color(0.3f, 0.5f, 1f), "Yanlış ısı kaynağı (araç, güneşte metal)*");
        Satir(new Color(0.75f, 0.75f, 0.75f), "Yükselmiş hücre: termalde fark edildi");
        Satir(new Color(0.1f, 0.9f, 0.3f), "Yeşil flaş: optikle doğrulandı");
        Satir(Color.white, "Beyaz flaş: yanlış alarm elendi");
        Satir(new Color(1f, 0.9f, 0.1f), "Sarı flaş: YANGIN (kaçırıldı)");
        Satir(Color.cyan, "Renkli çerçeve: termal görüş alanı");
        GUILayout.Label("<size=11>* Kameralar gerçek ile yanlış kaynağı ayırt edemez; renkler sadece izleyici için.</size>", kucuk);
        GUILayout.Space(6);
        GUILayout.Label("<b>Tuşlar</b>  1 Q-öğrenme · 2 Devriye · 3 Rastgele\nK kalkan · G gürültü · +/- hız · Boşluk duraklat · H panel", kucuk);
        GUILayout.EndArea();
    }

    void Satir(Color c, string ad)
    {
        GUILayout.BeginHorizontal();
        var r = GUILayoutUtility.GetRect(18, 18, GUILayout.Width(18), GUILayout.Height(18));
        Renk(new Rect(r.x, r.y + 2, 16, 16), c);
        GUILayout.Label(ad, kucuk);
        GUILayout.EndHorizontal();
    }

    static string EylemMetni(string e)
    {
        if (e == "dogrula") return "anomaliye dön + yakın zoom (doğrula)";
        if (e.StartsWith("tara_B")) return $"bölge {e.Substring(6)} merkezine dön + geniş tara";
        return e;
    }
}
