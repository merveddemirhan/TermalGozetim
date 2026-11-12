import pickle, numpy as np
from ptz_gozetim_env import PTZGozetimEnv
from politikalar import QAjani
def ayak(r, c, z, N=8):
    m = np.zeros((N, N), bool)
    if z: m[r, c] = True
    else: m[max(0,r-1):r+2, max(0,c-1):c+2] = True
    return m
for etiket, kw in [("final", {}), ("kendibolge", {"kendi_bolgem": True})]:
    cak, ikili, adim = 0, 0, 0; Y = 0; D = 0; B = 0
    for t in (1, 2, 3):
        env = PTZGozetimEnv(); a = QAjani(env, **kw); a.Q = pickle.load(open(f"sonuclar/{etiket}_t{t}.pkl", "rb")); a.ogren = False
        for b in range(400):
            obs, _ = env.reset(seed=10**6 + b); a.sifirla(); bitti = False
            while not bitti:
                act = a.eylem(obs); obs2, r, te, tr, info = env.step(act); bitti = te or tr
                a.gozlemle(obs, act, info, obs2, bitti)
                k = obs2["kameralar"]; A = ayak(*k[0]); Bm = ayak(*k[1])
                cak += (A & Bm).sum(); adim += 1
                # ikisi birden aynı gerçek ısınmaya 1 hücre yakın mı?
                for i, j in np.argwhere(env.sicaklik > 0):
                    if all(max(abs(kk[0]-i), abs(kk[1]-j)) <= 1 for kk in k): ikili += 1
                obs = obs2
            Y += info["yangin"]; D += info["dogrulanan"]; B += 1
    print(f"{etiket:11s}: çakışan hücre/adım {cak/adim:.3f} | iki kamera aynı ısınmanın yanında (adım/bölüm) {ikili/B:.2f} | yangın/1000 {1000*Y/B:.2f} | doğrulanan {D/B:.2f}")
