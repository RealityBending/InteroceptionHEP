import os

import neurokit2 as nk
import numpy as np
import pandas as pd

files = [f for f in os.listdir("../data/") if f.startswith("data_hep")]
df = pd.concat([pd.read_csv(f"../data/{f}", engine="c") for f in files])


# persub = df.groupby(["Condition", "time"]).median(numeric_only=True).reset_index()
# persub.groupby('Condition').plot(x='time', y='AF7')

# Feature =====================================================================
df = df.drop(columns=["ECG", "PPG_Muse", "RSP"])
window = 0.12
dfall = []

for p in df.Participant.unique():
    print(p)
    for c in ["RestingState", "HCT"]:
        data = df[(df.Participant == p) & (df.Condition == c)]

        if len(data) == 0:
            continue

        feat = []
        for e in data.epoch.unique():
            d = data[data.epoch == e].reset_index(drop=True)

            # Time frequency
            d = pd.concat(
                [
                    d,
                    nk.signal_power(
                        d.AF7.values,
                        frequency_band=[(10, 15), (15, 30), (10, 30)],
                        sampling_rate=100,
                        continuous=True,
                        normalize=False,
                    ).add_prefix("AF7_"),
                    nk.signal_power(
                        d.AF8.values,
                        frequency_band=[(10, 15), (15, 30), (10, 30)],
                        sampling_rate=100,
                        continuous=True,
                        normalize=False,
                    ).add_prefix("AF8_"),
                ],
                axis=1,
            )

            for i, tmax in enumerate(d.time.values[12::]):
                _feat = {"time": tmax - window / 2, "tmin": tmax - window, "tmax": tmax}

                # ERP -----------------------------------------------------------
                w = d[(d.time >= _feat["tmin"]) & (d.time < tmax)]

                _feat["AF7_ERP_Mean"] = w.AF7.mean()
                _feat["AF8_ERP_Mean"] = w.AF8.mean()
                _feat["AF7_ERP_Median"] = w.AF7.median()
                _feat["AF8_ERP_Median"] = w.AF8.median()

                # Time/Frequency ------------------------------------------------
                _feat["AF7_TF_1015Mean"] = w["AF7_10.00-15.00Hz"].mean()
                _feat["AF8_TF_1015Mean"] = w["AF8_10.00-15.00Hz"].mean()
                _feat["AF7_TF_1530Mean"] = w["AF7_15.00-30.00Hz"].mean()
                _feat["AF8_TF_1530Mean"] = w["AF8_15.00-30.00Hz"].mean()
                _feat["AF7_TF_1030Mean"] = w["AF7_10.00-30.00Hz"].mean()
                _feat["AF8_TF_1030Mean"] = w["AF8_10.00-30.00Hz"].mean()
                # _feat["AF7_TF_1015Median"] = w["AF7_10.00-15.00Hz"].median()
                # _feat["AF8_TF_1015Median"] = w["AF8_10.00-15.00Hz"].median()
                # _feat["AF7_TF_1530Median"] = w["AF7_15.00-30.00Hz"].median()
                # _feat["AF8_TF_1530Median"] = w["AF8_15.00-30.00Hz"].median()

                # Complexity ----------------------------------------------------
                _feat["AF7_Fractal_LL"], _ = nk.fractal_linelength(w.AF7.values)
                _feat["AF8_Fractal_LL"], _ = nk.fractal_linelength(w.AF8.values)
                _feat["AF7_Fractal_KFD"], _ = nk.fractal_katz(w.AF7.values)
                _feat["AF8_Fractal_KFD"], _ = nk.fractal_katz(w.AF8.values)
                _feat["AF7_Fractal_SFD"], _ = nk.fractal_sevcik(w.AF7.values)
                _feat["AF8_Fractal_SFD"], _ = nk.fractal_sevcik(w.AF8.values)
                _feat["AF7_Fractal_PFDsign"], _ = nk.fractal_petrosian(w.AF7.values, symbolize="C")
                _feat["AF8_Fractal_PFDsign"], _ = nk.fractal_petrosian(w.AF8.values, symbolize="C")
                _feat["AF7_Fractal_PFDmean"], _ = nk.fractal_petrosian(w.AF7.values, symbolize="A")
                _feat["AF8_Fractal_PFDmean"], _ = nk.fractal_petrosian(w.AF8.values, symbolize="A")

                _feat["AF7_Entropy_Hjorth"], _ = nk.complexity_hjorth(w.AF7.values)
                _feat["AF8_Entropy_Hjorth"], _ = nk.complexity_hjorth(w.AF8.values)
                # _feat["AF7_Entropy_DiffEn"], _ = nk.entropy_differential(w.AF7.values)
                # _feat["AF8_Entropy_DiffEn"], _ = nk.entropy_differential(w.AF8.values)
                _feat["AF7_Entropy_AttEn"], _ = nk.entropy_attention(w.AF7)
                _feat["AF8_Entropy_AttEn"], _ = nk.entropy_attention(w.AF8)
                _feat["AF7_Entropy_Delay"], _ = nk.complexity_delay(
                    w.AF7.values,
                    delay_max=4,
                    method="fraser1986",
                    algorithm="first local minimum (corrected)",
                )
                _feat["AF8_Entropy_Delay"], _ = nk.complexity_delay(
                    w.AF8.values,
                    delay_max=4,
                    method="fraser1986",
                    algorithm="first local minimum (corrected)",
                )
                _feat["AF7_Entropy_SVDEn"], _ = nk.entropy_svd(w.AF7.values, delay=2, dimension=2)
                _feat["AF8_Entropy_SVDEn"], _ = nk.entropy_svd(w.AF8.values, delay=2, dimension=2)

                feat.append(pd.DataFrame(_feat, index=[0]))

        # Concat
        feat = pd.concat(feat).groupby(["time"]).mean().reset_index()
        feat["Participant"] = p
        feat["Condition"] = c

        dfall.append(feat)

dfall = pd.concat(dfall)
dfall.to_csv("../data/data_features.csv", index=False)

print("Done!")
