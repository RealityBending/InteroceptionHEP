import io
import os
import time

import autoreject
import matplotlib.pyplot as plt
import mne
import neurokit2 as nk
import numpy as np
import pandas as pd
import PIL
import pyllusion as ill
import requests
import scipy.stats

mne.set_log_level(verbose="WARNING")

# Convenience functions ======================================================================
# Download the load_physio() function
exec(
    requests.get(
        "https://raw.githubusercontent.com/RealityBending/InteroceptionPrimals/main/analysis/func_load_physio.py"
    ).text
)


def qc_eeg(raw, sub, plots=[]):
    """Quality control (QC) of EEG."""
    ch_names = ["AF7", "AF8", "TP9", "TP10"]

    # Make fig
    fig, ax = plt.subplot_mosaic(
        [["A", "right"], ["B", "right"], ["C", "right"], ["D", "right"]],
        figsize=(10, 6),
    )
    fig.suptitle(f"{sub}")

    # Traces
    df = raw.to_data_frame()
    df.index = df["time"] / 60
    df[ch_names].plot(
        ax=[ax[k] for k in ["A", "B", "C", "D"]], subplots=True, linewidth=0.5
    )
    [ax[k].get_legend().remove() for k in ["A", "B", "C", "D"]]

    # PSD
    psd = raw.compute_psd(picks="eeg", n_fft=256 * 20, fmax=80).to_data_frame()
    psd.plot(x="freq", y=ch_names, ax=ax["right"], logy=True)

    # Add legend
    ax["right"].legend(loc="upper right")

    # Resize
    fig.set_size_inches(fig.get_size_inches() * 0.4)

    img = nk.fig2img(fig)

    # To image
    plots.append(img)
    plt.close("all")
    return plots


def analyze_hep(raw, events):
    out = {}  # Initialize info dictionary

    # From Zaccaro et al. (preprint): Epochs were not baseline-corrected (Petzschner et al., 2019).
    # This decision was made to exclude confounding the evoked signal by ECG waves that precede
    # the R-peak (P and Q-waves).

    epochs = mne.Epochs(
        raw,
        events,
        tmin=-0.40,
        tmax=0.80,
        detrend=None,
        decim=20,  # Downsample to 100 Hz
        verbose=False,
        preload=True,
    )
    # Remove epochs with missing data
    epochs = epochs.drop([np.isnan(e).any() for e in epochs])

    # Autoreject
    try:
        original_epochs = epochs.copy()
        ar = autoreject.AutoReject(verbose=False, picks="eeg")
        epochs = ar.fit_transform(original_epochs)
        out["autoreject_log"] = ar.get_reject_log(original_epochs.copy().pick("eeg"))
        # out["autoreject_log"].plot("horizontal", aspect="auto")
        # out["autoreject_log"].plot_epochs(original_epochs.copy().pick("eeg"))
    except IndexError:
        out["autoreject_log"] = None

    # Save data
    out["epochs"] = epochs
    data = epochs.copy().to_data_frame()
    out["df"] = data[["time", "epoch", "AF7", "AF8", "PPG_Muse", "ECG", "RSP"]]

    # # Compute HEP Features
    # hep = epochs.average(
    #     picks=["AF7", "AF8", "ECG", "PPG_Muse"],
    #     method=lambda x: np.nanmean(x, axis=0),
    # ).to_data_frame()
    # hep.index = hep["time"]
    # out["df"] = hep.copy().decimate(2)  # Futher decimate

    # hep1 = hep[0.2:0.4][["AF7", "AF8"]]
    # hep2 = hep[0.4:0.6][["AF7", "AF8"]]
    # rez = pd.DataFrame(
    #     {
    #         "HEP_Amplitude_200_400_EEG": [hep1.mean(axis=1).mean()],
    #         "HEP_Amplitude_400_600_EEG": [hep2.mean(axis=1).mean()],
    #     }
    # )

    # for ch in hep1.columns:
    #     rez[f"HEP_Amplitude_200_400_{ch}"] = hep1[ch].mean()
    #     rez[f"HEP_Amplitude_400_600_{ch}"] = hep2[ch].mean()

    # Compute Time-frequency Power
    # "We set the frequency range at 1 Hz intervals from 5 Hz to 20 Hz, excluding frequencies below 5 Hz to reduce the influence of
    # slow-varying artifacts, aligning with approaches used in previous studies (Park et al., 2018, Kern et al., 2013)" (Lee et al., 2024)
    freqs = np.logspace(*np.log10([5, 40]), num=40)  # freqs of interest (log-spaced)
    n_cycles = freqs / 2.0  # different number of cycle per frequency
    power, itc = mne.time_frequency.tfr_morlet(
        epochs,
        picks=["AF7", "AF8"],
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=True,
    )

    # power.plot()
    # itc.plot()

    out["timefrequency"] = power
    out["itc"] = itc

    return out


def qc_hep(epochs, reject_log, sub, plots=[]):
    """Quality control (QC) of HEPs."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{sub}")

    if reject_log is not None:
        reject_log.plot("horizontal", aspect="auto", ax=ax[0], show=False)

    hep = epochs.average(picks=["AF7", "AF8"], method=lambda x: np.nanmean(x, axis=0))
    hep.to_data_frame().plot(x="time", y=["AF7", "AF8"], ax=ax[1])

    # To image
    plots.append(nk.fig2img(fig))
    plt.close("all")
    return plots


def qc_heo(power, itc, sub, plots=[]):
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))

    power.plot(
        baseline=None,
        mode="logratio",
        axes=ax[0],
        vlim=(0, None),
        show=False,
    )
    ax[0][0].set_title("Power (AF7)")
    ax[0][1].set_title("Power (AF8)")

    itc.plot(
        title="Inter-Trial coherence",
        vlim=(0, None),
        cmap="CMRmap",
        axes=ax[1],
        show=False,
    )
    ax[1][0].set_title("Inter-Trial coherence (AF7)")
    ax[1][1].set_title("Inter-Trial coherence (AF8)")

    fig.tight_layout()
    fig.suptitle(f"{sub}")

    plots.append(nk.fig2img(fig))
    plt.close("all")
    return plots


# Variables ==================================================================================
# Change the path to your local data folder.
# The data can be downloaded from OpenNeuro (TODO).
path = "C:/Users/domma/Box/Data/InteroceptionPrimals/Reality Bending Lab - InteroceptionPrimals/"
# path = "C:/Users/dmm56/Box/Data/InteroceptionPrimals/Reality Bending Lab - InteroceptionPrimals/"

# Get participant list
meta = pd.read_csv(path + "participants.tsv", sep="\t")

# Initialize variables
df = pd.DataFrame()
qc = {
    "rs_eeg": [],
    "hct_eeg": [],
    "rs_hep": [],
    "rs_heo": [],
    "hct_hep": [],
    "hct_heo": [],
}

# sub = "sub-19"
# Loop through participants ==================================================================
for i, sub in enumerate(meta["participant_id"].values[0::]):
    # Print progress and comments
    print(f"{i}: {sub}")
    print("  * " + meta[meta["participant_id"] == sub]["Comments"].values[0])

    if sub in ["sub-86"]:  # No RS data
        continue

    # Load data
    rs, hct = load_physio(path, sub)  # Function loaded from script at URL

    # Resting State ==========================================================================

    # Preprocessing --------------------------------------------------------------------------
    print("  - RS")

    rs, _ = mne.set_eeg_reference(rs, ["TP9", "TP10"])
    rs = rs.notch_filter(np.arange(50, 251, 50), picks="eeg")
    rs = rs.filter(1, 40, picks="eeg")

    # QC
    qc["rs_eeg"] = qc_eeg(rs, sub, plots=qc["rs_eeg"])

    # Preprocess physio
    bio, info = nk.bio_process(
        ecg=rs["ECG"][0][0],
        sampling_rate=rs.info["sfreq"],
    )

    # Find R-peaks
    events, _ = nk.events_to_mne(bio["ECG_R_Peaks"].values.nonzero()[0])

    # Analyze HEP -------------------------------------------------------------------------
    hep = analyze_hep(rs, events)
    hep["df"]["Participant"] = sub
    hep["df"]["Condition"] = "RestingState"

    # QC
    qc["rs_hep"] = qc_hep(hep["epochs"], hep["autoreject_log"], sub, plots=qc["rs_hep"])
    qc["rs_heo"] = qc_heo(hep["timefrequency"], hep["itc"], sub, plots=qc["rs_heo"])

    # Save data
    df = pd.concat([df, hep["df"]], axis=0)

    # Heartbeat Counting Task (HCT) ===========================================================

    # Preprocessing --------------------------------------------------------------------------
    print("  - HCT")

    if sub in ["sub-03"]:
        continue

    hct, _ = mne.set_eeg_reference(hct, ["TP9", "TP10"])
    hct = hct.notch_filter(np.arange(50, 251, 50), picks="eeg")
    hct = hct.filter(1, 40, picks="eeg")

    # QC
    qc["hct_eeg"] = qc_eeg(hct, sub, plots=qc["hct_eeg"])
    # Preprocess physio
    ecg, _ = nk.bio_process(
        ecg=hct["ECG"][0][0], ppg=hct["PPG_Muse"][0][0], sampling_rate=hct.info["sfreq"]
    )

    # Find events (again as data was cropped) and epoch ---------------------------------------
    events = nk.events_find(
        hct["PHOTO"][0][0], threshold_keep="below", duration_min=15000
    )
    assert len(events["onset"]) == 6  # Check that there are 6 epochs (the 6 intervals)

    # Find R-peaks
    beats = ecg["ECG_R_Peaks"].values.nonzero()[0]
    intervals = [[o, o + d] for o, d in zip(events["onset"], events["duration"])]
    for i_b, b in enumerate(beats):
        # If it's not in any interval, remove it
        if not any([b >= j[0] and b <= j[1] for j in intervals]):
            beats[i_b] = 0
    beats = beats[beats != 0]

    # Analyze HEP -------------------------------------------------------------------------
    hep = analyze_hep(raw=hct, events=nk.events_to_mne(beats)[0])
    hep["df"]["Participant"] = sub
    hep["df"]["Condition"] = "HCT"

    # QC
    qc["hct_hep"] = qc_hep(
        hep["epochs"], hep["autoreject_log"], sub, plots=qc["hct_hep"]
    )
    qc["hct_heo"] = qc_heo(hep["timefrequency"], hep["itc"], sub, plots=qc["hct_heo"])

    # Concat
    df = pd.concat([df, hep["df"]], axis=0)

    # Save data
    if i in [49, 99, len(meta["participant_id"].values) - 1]:
        print(f"**SAVING DATA** ({i})")
        # Clean up and Save data
        for j in range(13):
            df[df["Participant"].str.contains(f"sub-{j}[0-4]$")].to_csv(
                f"../data/data_hep{j+1}a.csv", index=False
            )
            df[df["Participant"].str.contains(f"sub-{j}[5-9]$")].to_csv(
                f"../data/data_hep{j+1}b.csv", index=False
            )

        # Save figures
        ill.image_mosaic(qc["rs_eeg"], ncols=5, nrows="auto").save(
            f"signals/rs_eeg_{i+1}.png"
        )
        ill.image_mosaic(qc["rs_hep"], ncols=5, nrows="auto").save(
            f"signals/rs_hep_{i+1}.png"
        )
        ill.image_mosaic(qc["rs_heo"], ncols=5, nrows="auto").save(
            f"signals/rs_heo_{i+1}.png"
        )
        ill.image_mosaic(qc["hct_eeg"], ncols=5, nrows="auto").save(
            f"signals/hct_eeg_{i+1}.png"
        )
        ill.image_mosaic(qc["hct_hep"], ncols=5, nrows="auto").save(
            f"signals/hct_hep_{i+1}.png"
        )
        ill.image_mosaic(qc["hct_heo"], ncols=5, nrows="auto").save(
            f"signals/hct_heo_{i+1}.png"
        )

        qc = {
            "rs_eeg": [],
            "rs_hep": [],
            "rs_heo": [],
            "hct_eeg": [],
            "hct_hep": [],
            "hct_heo": [],
        }


print("Done!")
