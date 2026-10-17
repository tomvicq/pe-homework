"""Génère `comptage-velos.csv` : comptages journaliers (fictifs) d'un compteur
vélo parisien du 1er janvier 2023 au 31 décembre 2024.

Le jeu de données est entièrement simulé (graine fixée) et contient, volontairement :

* une tendance à la hausse (~ +12 % par an),
* une saisonnalité annuelle (maximum début juillet, creux en hiver) et un creux d'août,
* un cycle hebdomadaire (week-end plus bas),
* des creux les jours fériés « fixes » (1er janvier, 1er mai, 25 décembre),
* du bruit,
* 8 jours sans mesure (valeur vide),
* 3 lignes dupliquées,
* 2 valeurs aberrantes (un 0 de panne, une valeur ×10),
* des lignes partiellement mélangées (le fichier n'est pas trié par date).

Usage : python generate.py   (écrit comptage-velos.csv dans le dossier du script)
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
rng = np.random.default_rng(SEED)

dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
n = len(dates)
t = np.arange(n)

# tendance : 2500 vélos/jour au départ, +12 % par an environ
base = 2500 * (1 + 0.12 * t / 365)

# saisonnalité annuelle : maximum vers le 1er juillet (jour ~182), amplitude ±35 %
doy = dates.dayofyear.to_numpy()
saison = 1 + 0.35 * np.cos(2 * np.pi * (doy - 182) / 365.25)

# creux d'août (vacances) : du 1er au 20 août, -30 %
aout = np.asarray((dates.month == 8) & (dates.day <= 20))
vacances = np.where(aout, 0.7, 1.0)

# cycle hebdomadaire : lundi=0 … dimanche=6
poids_jour = np.array([1.0, 1.05, 1.05, 1.0, 0.95, 0.6, 0.5])
semaine = poids_jour[dates.dayofweek.to_numpy()]

# jours fériés fixes : -60 %
ferie = np.asarray(
    ((dates.month == 1) & (dates.day == 1))
    | ((dates.month == 5) & (dates.day == 1))
    | ((dates.month == 12) & (dates.day == 25))
)
feries = np.where(ferie, 0.4, 1.0)

# bruit multiplicatif ~ 8 %
bruit = rng.normal(1.0, 0.08, size=n)

comptage = np.round(base * saison * vacances * semaine * feries * bruit).astype(float)
comptage = np.clip(comptage, 50, None)

df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "comptage": comptage})

# 8 jours sans mesure
manquants = rng.choice(n, size=8, replace=False)
df.loc[manquants, "comptage"] = np.nan

# 2 valeurs aberrantes : une panne (0) et un capteur fou (×10)
candidats = np.setdiff1d(np.arange(n), manquants)
i_zero, i_dix = rng.choice(candidats, size=2, replace=False)
df.loc[i_zero, "comptage"] = 0
df.loc[i_dix, "comptage"] = df.loc[i_dix, "comptage"] * 10

# 3 lignes dupliquées (recopiées telles quelles)
doublons = rng.choice(candidats, size=3, replace=False)
df = pd.concat([df, df.loc[doublons]], ignore_index=True)

# mélange partiel : on permute entre elles une soixantaine de lignes
idx = rng.choice(len(df), size=60, replace=False)
df.iloc[idx] = df.iloc[rng.permutation(idx)].to_numpy()

# comptages entiers dans le fichier (les NaN restent vides)
df["comptage"] = df["comptage"].astype("Int64")

out = Path(__file__).resolve().parent / "comptage-velos.csv"
df.to_csv(out, sep=";", index=False)
print(f"{out} : {len(df)} lignes")
print("jours manquants :", sorted(dates[manquants].strftime("%Y-%m-%d")))
print("zéro :", dates[i_zero].date(), " x10 :", dates[i_dix].date())
print("doublons :", sorted(dates[doublons].strftime("%Y-%m-%d")))
