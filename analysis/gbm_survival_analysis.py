# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Survival in glioblastoma: who lives longer, and can we predict it for an individual patient?
#
# **Data:** CFB-GBM collection (The Cancer Imaging Archive), clinical, treatment, RANO and PyRadiomics tables.
#
# **The two questions from the physician**
# 1. Which characteristics best predict *longer* survival?
# 2. Can we build a model that gives an **individualized survival prediction** from clinical characteristics, diagnosis and/or radiomic imaging features?
#
# __KEY_FINDINGS__

# %% [markdown]
# ## 0. Setup

# %%
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from IPython.display import display, Markdown

from lifelines import CoxPHFitter, CoxTimeVaryingFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test
from statsmodels.stats.multitest import multipletests
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import PowerTransformer
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

DATA = Path("..")                 # the four .tsv files live in the project root
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
SEED = 2026
N_REPEATS = 10                    # repeats of 5-fold cross-validation
GRID = np.arange(0, 240.5, 0.5)   # weeks at which survival curves are evaluated

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.precision", 3)

# Chart style: fixed categorical order, recessive axes, 2px lines
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, MUTED, GRIDC, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "lines.linewidth": 2, "legend.frameon": False, "legend.fontsize": 9,
    "font.family": "sans-serif", "figure.dpi": 110,
})

def savefig(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=160, bbox_inches="tight")

# %% [markdown]
# ## 1. Data assembly
#
# | File | Content | Rows |
# |---|---|---|
# | clinical | survival (weeks from t0 MRI to death), age, WHO performance status (PS), sex, height, weight, WHO classification edition | 264 patients |
# | treatment | radiotherapy delay, dose, number of fractions | 264 (70 with no treatment information) |
# | RANO | tumour volume at t0/t1/t2 and RANO response category | 165 patients |
# | PyRadiomics | 107 features per scan (shape, first-order intensity, 5 texture families) for 7 MRI sequences at t0/t1/t2 | 2,424 scans, 261 patients |
#
# **Design decisions**
# * **Time zero is the t0 MRI.** The survival column is measured from that scan, so a prediction model must use only information available at t0.
# * **Every patient has an observed death time**: there is no censoring indicator and no missing survival. This is a complete-follow-up cohort (event = 1 for all). That simplifies validation, because calibration can be checked against observed proportions directly.
# * **Radiotherapy regimen** is defined by dose per fraction, which reflects the *intended* schedule: conventional (1.8–2 Gy/fraction, e.g. 60 Gy/30 fx) vs hypofractionated (>2.5 Gy/fraction, e.g. 40.05 Gy/15 fx). Patients without treatment records form an explicit "Unknown" category instead of being dropped, so no one is excluded.
# * **Tumour volume** comes from the t0 GTV shape features. It matches the RANO `size_t0` almost perfectly (r > 0.9999) and is available for 261 vs 165 patients.

# %%
clin = pd.read_csv(DATA / "CFB-GBM_clinical_data_v03_20260619.tsv", sep="\t")
trt = pd.read_csv(DATA / "CFB-GBM_treatment_data_v02_20260129.tsv", sep="\t")
rano = pd.read_csv(DATA / "CFB-GBM_rano_criteria_v04_20260909.tsv", sep="\t")
rad = pd.read_csv(DATA / "CFB_GBM_features_extraction_pyradiomics_v03_20260619.tsv", sep="\t", low_memory=False)

df = (clin.merge(trt, on="id_patient", how="left")
          .merge(rano, on="id_patient", how="left")
          .rename(columns={"survival (weeks)": "time", "age_at_t0 (years)": "age",
                           "who_performance_status": "ps", "height (cm)": "height",
                           "weight (kg)": "weight", "delay_t0_to_radiotherapy (weeks)": "rt_delay",
                           "dose (Gy)": "dose", "fractions_number": "fx"})
          .set_index("id_patient"))
df["event"] = 1
df["male"] = (df.gender == "M").astype(int)
df["bmi"] = df.weight / (df.height / 100) ** 2
df["who2021"] = (df.who_guideline == 2021).astype(int)
df["ps_missing"] = df.ps.isna()
df["ps"] = df.ps.fillna(df.ps.mode()[0])          # 1 patient; imputed with the mode (PS 1)
df["ps_cat"] = df.ps.map({0: "0-1", 1: "0-1", 2: "2", 3: "3"})
dose_per_fx = df.dose / df.fx
df["regimen"] = np.select([df.dose.isna(), dose_per_fx > 2.5], ["Unknown", "Hypofractionated"], "Conventional")

FEATS = [c for c in rad.columns if c.startswith("original_")]
SHAPE = [c for c in FEATS if "_shape_" in c]
NONSHAPE = [c for c in FEATS if "_shape_" not in c]
t0 = rad[rad.Temporality == "t0"]
t1gd = t0[t0.Sequence == "t1gd"].set_index("Patient")[FEATS]
flair = t0[t0.Sequence == "flair"].set_index("Patient")[NONSHAPE]
df["vol"] = t1gd.original_shape_MeshVolume.reindex(df.index) / 1000        # cm^3
df["log2vol"] = np.log2(df.vol)

df["age10"] = (df.age - 65) / 10
df["ps2"] = (df.ps == 2).astype(int); df["ps3"] = (df.ps == 3).astype(int)
df["hypo"] = (df.regimen == "Hypofractionated").astype(int); df["unk"] = (df.regimen == "Unknown").astype(int)
df["lv"] = df.log2vol - np.log2(30)   # centred at 30 cm^3

print(f"Patients: {len(df)} | deaths observed: {df.event.sum()} | with t0 T1-Gd radiomics: {df.vol.notna().sum()} "
      f"| with t0 FLAIR radiomics: {df.index.isin(flair.index).sum()} | with RANO: {df['size_t0 (cm3)'].notna().sum()}")
print("Check: shape features identical across sequences of the same scan ->",
      bool((t0.groupby("Patient")[SHAPE].nunique() <= 1).all().all()))
print("Check: radiomic volume vs RANO size_t0, r =",
      round(np.corrcoef(df.dropna(subset=["vol", "size_t0 (cm3)"])[["vol", "size_t0 (cm3)"]].T)[0, 1], 6))

# %% [markdown]
# ### Radiomic data available at baseline (t0)
# T1-Gd (contrast-enhanced T1) and FLAIR are available for almost everyone at t0. The other sequences have substantial gaps, so they are not used for modelling.
# The features were extracted from images that had already been intensity-standardized (image-wide mean ≈ 0), with 1 mm isotropic resampling and bin width 5. Intensity and texture values are therefore broadly comparable across patients.

# %%
avail = pd.crosstab(rad.Sequence, rad.Temporality).loc[["t1gd", "flair", "t2star", "t1eg", "adc", "t2tse", "t1tse"]]
avail.index = ["T1-Gd", "FLAIR", "T2*", "T1-EG", "ADC", "T2-TSE", "T1-TSE"]
display(avail.rename_axis("Sequence \\ timepoint"))

# %% [markdown]
# ## 2. Cohort description

# %%
def iqr(s):
    return f"{s.median():.0f} [{s.quantile(.25):.0f}–{s.quantile(.75):.0f}]"

def npct(mask):
    return f"{mask.sum()} ({100 * mask.mean():.0f}%)"

df["long"] = df.time >= 52
groups = {"All patients": df, "Died < 1 year": df[~df.long], "Survived ≥ 1 year": df[df.long]}
rows = {
    "N": lambda g: f"{len(g)}",
    "Survival, weeks": lambda g: iqr(g.time),
    "Age, years": lambda g: iqr(g.age),
    "Male sex": lambda g: npct(g.male == 1),
    "WHO PS 0–1": lambda g: npct(g.ps <= 1),
    "WHO PS 2": lambda g: npct(g.ps == 2),
    "WHO PS 3": lambda g: npct(g.ps == 3),
    "BMI, kg/m²": lambda g: iqr(g.bmi),
    "WHO 2021 classification": lambda g: npct(g.who2021 == 1),
    "RT conventional": lambda g: npct(g.regimen == "Conventional"),
    "RT hypofractionated": lambda g: npct(g.regimen == "Hypofractionated"),
    "RT unknown": lambda g: npct(g.regimen == "Unknown"),
    "Tumour volume, cm³": lambda g: iqr(g.vol.dropna()),
}
t1 = pd.DataFrame({k: {r: f(g) for r, f in rows.items()} for k, g in groups.items()})
a, b = df[~df.long], df[df.long]
p = {
    "Age, years": stats.mannwhitneyu(a.age, b.age).pvalue,
    "Male sex": stats.chi2_contingency(pd.crosstab(df.long, df.male))[1],
    "WHO PS 0–1": stats.chi2_contingency(pd.crosstab(df.long, df.ps_cat))[1],
    "BMI, kg/m²": stats.mannwhitneyu(a.bmi, b.bmi).pvalue,
    "WHO 2021 classification": stats.chi2_contingency(pd.crosstab(df.long, df.who2021))[1],
    "RT conventional": stats.chi2_contingency(pd.crosstab(df.long, df.regimen))[1],
    "Tumour volume, cm³": stats.mannwhitneyu(a.vol.dropna(), b.vol.dropna()).pvalue,
}
t1["p"] = pd.Series(p).map(lambda x: f"{x:.3f}" if x >= 0.001 else "<0.001")
t1["p"] = t1["p"].fillna("")
t1.to_csv(OUT / "table1.csv")
display(t1)

# %%
km = KaplanMeierFitter().fit(df.time, df.event)
ci = km.confidence_interval_survival_function_
med_ci = km.median_survival_time_
from lifelines.utils import median_survival_times
mci = median_survival_times(km.confidence_interval_)
print(f"Median overall survival: {med_ci:.0f} weeks (95% CI {mci.iloc[0, 0]:.0f}–{mci.iloc[0, 1]:.0f}) "
      f"≈ {med_ci / 4.345:.1f} months")
for t in [26, 52, 104]:
    print(f"  Alive at {t:>3} weeks ({t / 52:.1f} y): {km.predict(t):.1%}")
print(f"Missing data: performance status {int(df.ps_missing.sum())} (imputed), treatment record {int(df.dose.isna().sum())} "
      f"(kept as 'Unknown' regimen), baseline imaging {int(df.vol.isna().sum())}")

# %% [markdown]
# ## 3. Which factors are associated with longer survival?
# ### 3.1 Kaplan–Meier curves

# %%
df["age_grp"] = pd.cut(df.age, [0, 59, 69, 74, 120], labels=["<60", "60–69", "70–74", "≥75"])
df["vol_grp"] = pd.qcut(df.vol, 4, labels=[f"Q{i}" for i in range(1, 5)])
vq = pd.qcut(df.vol, 4).cat.categories
vol_labels = {f"Q{i+1}": f"{iv.left:.0f}–{iv.right:.0f} cm³" for i, iv in enumerate(vq)}
panels = [
    ("ps_cat", "WHO performance status", {"0-1": "PS 0–1", "2": "PS 2", "3": "PS 3"}),
    ("regimen", "Radiotherapy regimen", None),
    ("age_grp", "Age group (years)", None),
    ("vol_grp", "Tumour volume (quartiles)", vol_labels),
    ("who2021", "WHO classification edition", {0: "2016", 1: "2021"}),
    ("gender", "Sex", {"F": "Female", "M": "Male"}),
]
fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), sharey=True)
logrank = {}
for ax, (var, title, labels) in zip(axes.ravel(), panels):
    sub = df.dropna(subset=[var])
    levels = [l for l in (sub[var].cat.categories if hasattr(sub[var], "cat") else sorted(sub[var].unique()))]
    if var == "regimen":
        levels = ["Conventional", "Hypofractionated", "Unknown"]
    for i, lev in enumerate(levels):
        g = sub[sub[var] == lev]
        k = KaplanMeierFitter().fit(g.time, g.event)
        lab = labels[lev] if labels else str(lev)
        k.plot_survival_function(ax=ax, ci_show=False, color=C[i], label=f"{lab} (n={len(g)}, median {k.median_survival_time_:.0f} wk)")
    pv = multivariate_logrank_test(sub.time, sub[var], sub.event).p_value
    logrank[title] = pv
    ax.set_title(f"{title}\nlog-rank p = {pv:.2g}", loc="left")
    ax.set_xlabel("Weeks since t0 MRI"); ax.set_ylabel("Proportion alive")
    ax.set_xlim(0, 200); ax.set_xticks([0, 26, 52, 78, 104, 130, 156, 182])
    ax.legend(loc="upper right", fontsize=8)
fig.tight_layout(); savefig(fig, "fig1_km_by_factor"); plt.show()

# %% [markdown]
# ### 3.2 Univariable Cox regression
# Hazard ratio (HR) > 1 means a higher death rate, i.e. **shorter** survival; HR < 1 means **longer** survival.

# %%
def cox_rows(cols, data, label=None):
    d = data[["time", "event"] + cols].dropna()
    s = CoxPHFitter().fit(d, "time", "event").summary
    return pd.DataFrame({"n": len(d), "HR": s["exp(coef)"], "95% CI low": s["exp(coef) lower 95%"],
                         "95% CI high": s["exp(coef) upper 95%"], "p": s["p"]})

df["bmi5"] = df.bmi / 5
uni_spec = [
    (["age10"], {"age10": "Age (per 10 years)"}),
    (["male"], {"male": "Male sex"}),
    (["bmi5"], {"bmi5": "BMI (per 5 kg/m²)"}),
    (["ps2", "ps3"], {"ps2": "WHO PS 2 (vs 0–1)", "ps3": "WHO PS 3 (vs 0–1)"}),
    (["who2021"], {"who2021": "WHO 2021 classification (vs 2016)"}),
    (["hypo", "unk"], {"hypo": "Hypofractionated RT (vs conventional)", "unk": "RT unknown (vs conventional)"}),
    (["log2vol"], {"log2vol": "Tumour volume (per doubling)"}),
    (["rt_delay"], {"rt_delay": "Delay to RT start (per week)"}),
]
uni = pd.concat([cox_rows(c, df).rename(index=lab) for c, lab in uni_spec])
uni.to_csv(OUT / "table2_univariable_cox.csv")
display(uni.style.format({"HR": "{:.2f}", "95% CI low": "{:.2f}", "95% CI high": "{:.2f}", "p": "{:.3g}"}))

# %% [markdown]
# ### 3.3 Multivariable Cox model
# All candidate clinical predictors are entered together. Sex and BMI are dropped because they show no association, univariable or adjusted (checked below).
# **Model A** uses simple linear terms, which are easy to read as HRs.
# **Model B** lets age and log-volume act non-linearly through restricted cubic splines (3 knots at the 10th/50th/90th percentiles). Model B is the basis of the prediction tool.

# %%
def rcs(x, knots):
    """Harrell's restricted cubic spline: returns the k-2 non-linear basis columns."""
    x = np.asarray(x, float); k = np.asarray(knots, float)
    p = lambda u: np.clip(u, 0, None) ** 3
    cols = [(p(x - k[j]) - p(x - k[-2]) * (k[-1] - k[j]) / (k[-1] - k[-2])
             + p(x - k[-1]) * (k[-2] - k[j]) / (k[-1] - k[-2])) / (k[-1] - k[0]) ** 2
            for j in range(len(k) - 2)]
    return np.column_stack(cols)

class ClinicalDesign:
    """Builds the clinical design matrix; spline knots are learned from the training data only."""
    BASE = ["age10", "ps2", "ps3", "who2021", "hypo", "unk"]

    def __init__(self, volume=True, spline=True):
        self.volume, self.spline = volume, spline

    def fit(self, d):
        self.age_k = np.percentile(d.age, [10, 50, 90])
        self.lv_k = np.percentile(d.lv, [10, 50, 90])
        return self

    def transform(self, d):
        X = d[self.BASE].astype(float).copy()
        if self.spline:
            X["age_nl"] = rcs(d.age, self.age_k)[:, 0] / 10
        if self.volume:
            X["lv"] = d.lv
            if self.spline:
                X["lv_nl"] = rcs(d.lv, self.lv_k)[:, 0]
        return X

LABELS = {"age10": "Age (per 10 years)", "ps2": "WHO PS 2 (vs 0–1)", "ps3": "WHO PS 3 (vs 0–1)",
          "who2021": "WHO 2021 classification", "hypo": "Hypofractionated RT", "unk": "RT unknown",
          "lv": "Tumour volume (per doubling)", "age_nl": "Age, non-linear term", "lv_nl": "Volume, non-linear term"}

dm = df.dropna(subset=["vol"]).copy()        # 261 patients with baseline imaging
desA = ClinicalDesign(volume=True, spline=False).fit(dm)
desB = ClinicalDesign(volume=True, spline=True).fit(dm)
XA, XB = desA.transform(dm), desB.transform(dm)
coxA = CoxPHFitter().fit(XA.assign(time=dm.time, event=dm.event), "time", "event")
coxB = CoxPHFitter().fit(XB.assign(time=dm.time, event=dm.event), "time", "event")

tabA = coxA.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
tabA.columns = ["HR", "95% CI low", "95% CI high", "p"]
tabA = tabA.rename(index=LABELS)
tabA.to_csv(OUT / "table3_multivariable_cox.csv")
display(Markdown(f"**Model A (linear terms), n = {len(dm)}, apparent C-index = {coxA.concordance_index_:.3f}**"))
display(tabA.style.format({"HR": "{:.2f}", "95% CI low": "{:.2f}", "95% CI high": "{:.2f}", "p": "{:.3g}"}))

lr = 2 * (coxB.log_likelihood_ - coxA.log_likelihood_)
print(f"Model B vs A: likelihood-ratio χ² = {lr:.1f} on 2 df, p = {stats.chi2.sf(lr, 2):.3f}; "
      f"AIC {coxA.AIC_partial_:.1f} → {coxB.AIC_partial_:.1f}; apparent C {coxA.concordance_index_:.3f} → {coxB.concordance_index_:.3f}")
for v in ["age_nl", "lv_nl"]:
    print(f"  Non-linearity test, {LABELS[v]}: p = {coxB.summary.loc[v, 'p']:.3f}")

extra = CoxPHFitter().fit(XA.assign(male=dm.male, bmi5=dm.bmi5, time=dm.time, event=dm.event), "time", "event").summary
print("Adjusted for the model A covariates: male HR {:.2f} (p={:.2f}); BMI per 5 HR {:.2f} (p={:.2f})".format(
    extra.loc["male", "exp(coef)"], extra.loc["male", "p"], extra.loc["bmi5", "exp(coef)"], extra.loc["bmi5", "p"]))

# %% [markdown]
# ### 3.4 How strong is each predictor? Shape of the age and tumour-volume effects
# Left: the drop in model likelihood when each predictor (all its terms) is removed from Model B. This is a likelihood-ratio χ², shown minus its degrees of freedom so predictors with more terms don't get extra credit.
# Middle and right: the adjusted HR across the range of age and tumour volume, with 95% pointwise CIs.

# %%
groups_terms = {"Age": ["age10", "age_nl"], "Performance status": ["ps2", "ps3"],
                "RT regimen": ["hypo", "unk"], "WHO 2021 classification": ["who2021"],
                "Tumour volume": ["lv", "lv_nl"]}
imp = {}
for name, terms in groups_terms.items():
    red = CoxPHFitter().fit(XB.drop(columns=terms).assign(time=dm.time, event=dm.event), "time", "event")
    chi = 2 * (coxB.log_likelihood_ - red.log_likelihood_)
    imp[name] = dict(chi2=chi, df=len(terms), p=stats.chi2.sf(chi, len(terms)))
imp = pd.DataFrame(imp).T.sort_values("chi2")
imp["chi2 - df"] = imp.chi2 - imp.df

def partial_effect(var_terms, xs, make_row, ref_row):
    """HR (with 95% CI) of each row in xs relative to the reference, from Model B."""
    beta = coxB.params_[var_terms].values
    V = coxB.variance_matrix_.loc[var_terms, var_terms].values
    D = np.array([make_row(x) for x in xs]) - np.array(make_row(ref_row))
    lp = D @ beta; se = np.sqrt(np.einsum("ij,jk,ik->i", D, V, D))
    return np.exp(lp), np.exp(lp - 1.96 * se), np.exp(lp + 1.96 * se)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.1, 1, 1]})
ax = axes[0]
ax.barh(imp.index, imp["chi2 - df"], color=C[0], height=0.55)
for i, (n, r) in enumerate(imp.iterrows()):
    ax.text(r["chi2 - df"] + 0.3, i, f"p = {r.p:.3f}" if r.p >= 0.001 else "p < 0.001", va="center", fontsize=8, color=INK2)
ax.set_xlabel("Likelihood-ratio χ² − df  (higher = more important)"); ax.set_title("Relative importance (Model B)", loc="left")
ax.grid(axis="y", visible=False)

ages = np.linspace(dm.age.quantile(.02), dm.age.quantile(.98), 100)
hr, lo, hi = partial_effect(["age10", "age_nl"], ages, lambda a: [(a - 65) / 10, rcs([a], desB.age_k)[0, 0] / 10], 65)
ax = axes[1]; ax.fill_between(ages, lo, hi, color=C[0], alpha=0.15, lw=0); ax.plot(ages, hr, color=C[0])
ax.axhline(1, color=AXIS, lw=1); ax.set_yscale("log"); ax.set_yticks([0.5, 1, 2, 3]); ax.set_yticklabels(["0.5", "1", "2", "3"])
ax.set_xlabel("Age (years)"); ax.set_ylabel("Adjusted HR (vs age 65)"); ax.set_title("Age effect", loc="left")

vols = np.exp(np.linspace(np.log(dm.vol.quantile(.02)), np.log(dm.vol.quantile(.98)), 100))
lvs = np.log2(vols) - np.log2(30)
hr, lo, hi = partial_effect(["lv", "lv_nl"], lvs, lambda v: [v, rcs([v], desB.lv_k)[0, 0]], 0.0)
ax = axes[2]; ax.fill_between(vols, lo, hi, color=C[0], alpha=0.15, lw=0); ax.plot(vols, hr, color=C[0])
ax.axhline(1, color=AXIS, lw=1); ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([3, 10, 30, 60, 120]); ax.set_xticklabels(["3", "10", "30", "60", "120"])
ax.set_yticks([0.5, 1, 2, 3]); ax.set_yticklabels(["0.5", "1", "2", "3"])
ax.set_xlabel("Tumour volume at t0 (cm³, log scale)"); ax.set_ylabel("Adjusted HR (vs 30 cm³)"); ax.set_title("Tumour-volume effect", loc="left")
fig.tight_layout(); savefig(fig, "fig2_importance_and_shapes"); plt.show()
display(imp[["chi2", "df", "p"]].iloc[::-1].style.format({"chi2": "{:.1f}", "df": "{:.0f}", "p": "{:.2g}"}))

# %% [markdown]
# ### 3.5 Does each effect stay constant over time? (proportional-hazards check)
# The Cox model assumes each HR is constant over follow-up. Schoenfeld-residual tests flag several variables, so we estimate each HR separately for the **first 6 months** (0–26 weeks) and **after 6 months**, using a time-varying-coefficient Cox model.

# %%
ph = proportional_hazard_test(coxA, XA.assign(time=dm.time, event=dm.event), time_transform="rank").summary
ph = ph[["test_statistic", "p"]].rename(index=LABELS)
display(Markdown("**Schoenfeld residual test (Model A)**: small p means the HR changes over time"))
display(ph.style.format({"test_statistic": "{:.2f}", "p": "{:.3f}"}))

def split_long(X, time, cut=26):
    """Episode-split each patient at `cut` weeks; returns covariates interacted with period."""
    rows = []
    for i, (idx, x) in enumerate(X.iterrows()):
        T = time.loc[idx]
        rows.append({"id": i, "start": 0, "stop": min(T, cut), "event": int(T <= cut),
                     **{f"{k}__early": v for k, v in x.items()}, **{f"{k}__late": 0.0 for k in x.index}})
        if T > cut:
            rows.append({"id": i, "start": cut, "stop": T, "event": 1,
                         **{f"{k}__early": 0.0 for k in x.index}, **{f"{k}__late": v for k, v in x.items()}})
    return pd.DataFrame(rows)

L = split_long(XA, dm.time)
ctv = CoxTimeVaryingFitter().fit(L, id_col="id", start_col="start", stop_col="stop", event_col="event")
s = ctv.summary
tv = pd.DataFrame({
    "HR 0–26 wk": s.loc[[f"{v}__early" for v in XA.columns], "exp(coef)"].values,
    "CI 0–26": [f"{a:.2f}–{b:.2f}" for a, b in s.loc[[f"{v}__early" for v in XA.columns], ["exp(coef) lower 95%", "exp(coef) upper 95%"]].values],
    "HR > 26 wk": s.loc[[f"{v}__late" for v in XA.columns], "exp(coef)"].values,
    "CI > 26": [f"{a:.2f}–{b:.2f}" for a, b in s.loc[[f"{v}__late" for v in XA.columns], ["exp(coef) lower 95%", "exp(coef) upper 95%"]].values],
}, index=[LABELS[v] for v in XA.columns])
n_late = int((dm.time > 26).sum())
display(Markdown(f"**Period-specific HRs** (all {len(dm)} patients contribute to 0–26 weeks; the {n_late} still alive at 26 weeks contribute afterwards)"))
display(tv.style.format({"HR 0–26 wk": "{:.2f}", "HR > 26 wk": "{:.2f}"}))
tv.to_csv(OUT / "table4_time_varying_hr.csv")

# %% [markdown]
# ## 4. Radiomic predictors
# ### 4.1 Feature-by-feature screen
# Each baseline feature is transformed with Yeo–Johnson to reduce skew and standardised, so HRs are **per 1 SD**.
# It is then tested in a Cox model (a) on its own and (b) adjusted for age, PS, WHO edition, RT regimen and tumour volume.
# p-values are corrected for testing ~100 features per sequence using the Benjamini–Hochberg false-discovery rate (q).
# The colour shows how strongly each feature tracks tumour volume, because many "texture" features are just size measures in disguise.

# %%
ADJ = ["age10", "ps2", "ps3", "who2021", "hypo", "unk", "lv"]

def screen(feat_df, prefix):
    ids = feat_df.index.intersection(dm.index)
    Z = pd.DataFrame(PowerTransformer().fit_transform(feat_df.loc[ids]), index=ids, columns=feat_df.columns)
    base = dm.loc[ids, ["time", "event"] + ADJ]
    out = []
    for f in Z.columns:
        u = CoxPHFitter().fit(base[["time", "event"]].assign(z=Z[f]), "time", "event").summary.loc["z"]
        a = CoxPHFitter().fit(base.assign(z=Z[f]), "time", "event").summary.loc["z"]
        out.append(dict(feature=f.replace("original_", ""), HR=u["exp(coef)"], p=u.p, HR_adj=a["exp(coef)"], p_adj=a.p,
                        r_volume=np.corrcoef(Z[f], dm.loc[ids, "lv"])[0, 1]))
    r = pd.DataFrame(out).set_index("feature")
    r["q"] = multipletests(r.p, method="fdr_bh")[1]
    r["q_adj"] = multipletests(r.p_adj, method="fdr_bh")[1]
    r["sequence"] = prefix
    return r

scr = {"T1-Gd": screen(t1gd, "T1-Gd"), "FLAIR": screen(flair, "FLAIR")}
summary_rows = []
for k, r in scr.items():
    vol_like = r.r_volume.abs() >= 0.7
    summary_rows.append({"Sequence": k, "Features tested": len(r), "Significant, unadjusted (q<0.05)": int((r.q < .05).sum()),
                         "…of which volume surrogates (|r|≥0.7)": int(((r.q < .05) & vol_like).sum()),
                         "Significant after adjustment (q<0.05)": int((r.q_adj < .05).sum())})
display(pd.DataFrame(summary_rows).set_index("Sequence"))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
for ax, (k, r) in zip(axes, scr.items()):
    vl = r.r_volume.abs() >= 0.7
    ax.scatter(np.log2(r.HR[~vl]), -np.log10(r.p[~vl]), s=22, color=C[1], edgecolor=SURF, linewidth=0.8, label="Not volume-related (|r| < 0.7)")
    ax.scatter(np.log2(r.HR[vl]), -np.log10(r.p[vl]), s=22, color=C[0], edgecolor=SURF, linewidth=0.8, label="Volume surrogate (|r| ≥ 0.7)")
    thr = r.p[r.q < 0.05].max() if (r.q < 0.05).any() else None
    if thr:
        ax.axhline(-np.log10(thr), color=MUTED, lw=1, ls="--"); ax.text(ax.get_xlim()[0] if False else -0.55, -np.log10(thr) + 0.1, "FDR 5%", color=MUTED, fontsize=8)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_xticks(np.log2([0.6, 0.8, 1, 1.25, 1.5])); ax.set_xticklabels(["0.6", "0.8", "1", "1.25", "1.5"])
    ax.set_xlabel("Unadjusted HR per 1 SD (log scale)"); ax.set_title(f"t0 {k}: {len(r)} features", loc="left")
axes[0].set_ylabel("−log10 p"); axes[0].legend(loc="upper left", fontsize=8)
fig.tight_layout(); savefig(fig, "fig3_radiomics_screen"); plt.show()

top_flair = scr["FLAIR"].sort_values("p_adj").head(12)[["HR", "q", "HR_adj", "q_adj", "r_volume"]]
pd.concat(scr.values()).to_csv(OUT / "table5_radiomics_screen.csv")
display(Markdown("**Top FLAIR features after adjustment** (HR per 1 SD)"))
display(top_flair.style.format("{:.3g}"))
display(Markdown("**Top T1-Gd features (unadjusted)**: note the high correlation with volume"))
display(scr["T1-Gd"].sort_values("p").head(8)[["HR", "q", "HR_adj", "q_adj", "r_volume"]].style.format("{:.3g}"))

# %% [markdown]
# ### 4.2 FLAIR texture: one main signal behind many correlated features
# The 93 FLAIR intensity and texture features are highly redundant. Principal-component analysis compresses them so we can see how many *independent* signals there are.

# %%
ids_f = flair.index.intersection(dm.index)
Zf = PowerTransformer().fit_transform(flair.loc[ids_f])
pca = PCA(6, random_state=SEED).fit(Zf)
PCs = pd.DataFrame(pca.transform(Zf)[:, :5], index=ids_f, columns=[f"PC{i}" for i in range(1, 6)])
PCs = PCs / PCs.std()
base = dm.loc[ids_f, ["time", "event"] + ADJ]
pc_tab = []
for pc in PCs.columns:
    a = CoxPHFitter().fit(base.assign(z=PCs[pc]), "time", "event").summary.loc["z"]
    load = pd.Series(pca.components_[int(pc[2:]) - 1], index=[c.replace("original_", "") for c in flair.columns])
    top = load.abs().sort_values(ascending=False).index[:3]
    pc_tab.append({"Component": pc, "Variance explained": pca.explained_variance_ratio_[int(pc[2:]) - 1],
                   "Adjusted HR per SD": a["exp(coef)"], "p": a.p,
                   "Top loadings": ", ".join(f"{t} ({'+' if load[t] > 0 else '−'})" for t in top)})
display(pd.DataFrame(pc_tab).set_index("Component").style.format({"Variance explained": "{:.1%}", "Adjusted HR per SD": "{:.2f}", "p": "{:.3g}"}))

# %% [markdown]
# ## 5. Building an individualized survival-prediction model
# ### 5.1 Candidate models and how they are compared
# All candidates are evaluated with **10 × repeated 5-fold cross-validation** on the 255 patients who have both T1-Gd and FLAIR at t0, so every model sees exactly the same folds.
# All tuning happens **inside** the training folds: spline knots, Yeo–Johnson transforms, PCA, and the elastic-net penalty (chosen by an inner 5-fold CV).
# That means the test-fold numbers are honest estimates of performance on new patients from the same population.
#
# | Metric | Meaning | Better |
# |---|---|---|
# | **C-index** | Probability that, of two random patients, the one predicted to live longer actually does | higher (0.5 = coin flip) |
# | **AUC(t)** | How well the predicted probability of death by week *t* separates who died by *t* from who did not | higher |
# | **Integrated Brier score** | Average squared error of the predicted survival curve over 4–104 weeks; it captures both discrimination **and** calibration | lower (the no-covariate Kaplan–Meier curve is the floor to beat) |
# | **Calibration at 1 year** | Mean predicted minus observed proportion alive at 52 weeks | close to 0 |

# %%
from joblib import Parallel, delayed

ALPHAS = np.exp(np.linspace(np.log(0.5), np.log(0.005), 25))
I26, I52, I104 = (np.searchsorted(GRID, t) for t in (26, 52, 104))

def surv_y(d):
    return Surv.from_arrays(d.event.astype(bool).values, d.time.astype(float).values)

def eval_stepfns(fns):
    return np.vstack([f(np.clip(GRID, f.x[0], f.x[-1])) for f in fns])

class KMModel:
    def fit(self, d):
        self.km = KaplanMeierFitter().fit(d.time, d.event); return self
    def survival(self, d):
        return np.tile(self.km.survival_function_at_times(GRID).values, (len(d), 1))

class CoxModel:
    """Cox PH on the clinical design, optionally plus the first k principal components of FLAIR texture.

    Only texture features that are not size surrogates (|r| with log-volume < max_vol_r in the training data)
    enter the PCA, so the components carry information *beyond* tumour volume, which is already in the model.
    """
    def __init__(self, design, flair_pcs=0, max_vol_r=0.5):
        self.design, self.k, self.max_vol_r = design, flair_pcs, max_vol_r
    def _X(self, d, fit=False):
        if fit: self.design.fit(d)
        X = self.design.transform(d)
        if self.k:   # FLAIR features come from `d` itself if present (new patients), else from the cohort table
            F = (d[flair.columns] if set(flair.columns) <= set(d.columns) else flair.loc[d.index]).values
            if fit:
                self.pt = PowerTransformer().fit(F)
                Z = self.pt.transform(F)
                r = np.array([np.corrcoef(Z[:, j], d.lv)[0, 1] for j in range(Z.shape[1])])
                self.keep = np.abs(r) < self.max_vol_r
                self.pca = PCA(self.k, random_state=SEED).fit(Z[:, self.keep])
            P = self.pca.transform(self.pt.transform(F)[:, self.keep])
            X = X.join(pd.DataFrame(P, index=d.index, columns=[f"PC{i + 1}" for i in range(self.k)]))
        return X
    def fit(self, d):
        self.cph = CoxPHFitter().fit(self._X(d, fit=True).assign(time=d.time, event=d.event), "time", "event")
        return self
    def survival(self, d):
        return self.cph.predict_survival_function(self._X(d), times=GRID).T.values

class TVCoxModel(CoxModel):
    """Cox model whose coefficients for `tv` variables change at `cut` weeks (step function)."""
    def __init__(self, design, tv=("ps2", "hypo", "unk", "lv"), cut=26):
        super().__init__(design); self.tv, self.cut = list(tv), cut
    def fit(self, d):
        X = self._X(d, fit=True); rows = []
        for i, idx in enumerate(X.index):
            x, T = X.loc[idx].to_dict(), d.time.loc[idx]
            rows.append({"id": i, "start": 0, "stop": min(T, self.cut), "event": int(T <= self.cut), **x, **{f"{v}_late": 0.0 for v in self.tv}})
            if T > self.cut:
                rows.append({"id": i, "start": self.cut, "stop": T, "event": 1, **x, **{f"{v}_late": x[v] for v in self.tv}})
        self.ctv = CoxTimeVaryingFitter().fit(pd.DataFrame(rows), id_col="id", start_col="start", stop_col="stop", event_col="event")
        H0 = self.ctv.baseline_cumulative_hazard_.iloc[:, 0]
        pos = np.searchsorted(H0.index.values, GRID, side="right") - 1
        self.H0 = np.where(pos >= 0, H0.values[np.clip(pos, 0, None)], 0.0)
        return self
    def survival(self, d):
        X = self._X(d); b = self.ctv.params_; mu = self.ctv._norm_mean[b.index]
        ev, lv = list(X.columns), [f"{v}_late" for v in self.tv]
        base = (X[ev] - mu[ev]).values @ b[ev].values
        early = np.exp(base - mu[lv].values @ b[lv].values)
        late = np.exp(base + (X[self.tv].values - mu[lv].values) @ b[lv].values)
        Hcut = np.interp(self.cut, GRID, self.H0)
        Hc, Hl = np.minimum(self.H0, Hcut), np.clip(self.H0 - Hcut, 0, None)
        return np.exp(-(early[:, None] * Hc[None, :] + late[:, None] * Hl[None, :]))

class CoxnetModel:
    """Elastic-net Cox on radiomic blocks; clinical design columns (if given) are left unpenalized."""
    def __init__(self, blocks, design=None, l1_ratio=0.5, seed=0):
        self.blocks, self.design, self.l1, self.seed = blocks, design, l1_ratio, seed
    def _X(self, d, fit=False):
        R = np.hstack([b.loc[d.index].values for b in self.blocks])
        if fit: self.pt = PowerTransformer().fit(R)
        parts = [self.pt.transform(R)]
        if self.design is not None:
            if fit: self.design.fit(d)
            parts.insert(0, self.design.transform(d).values)
        return np.hstack(parts)
    def fit(self, d):
        X, y = self._X(d, fit=True), surv_y(d)
        n_rad = sum(b.shape[1] for b in self.blocks)
        pf = np.r_[np.zeros(X.shape[1] - n_rad), np.ones(n_rad)]
        base = CoxnetSurvivalAnalysis(l1_ratio=self.l1, alphas=ALPHAS, penalty_factor=pf, fit_baseline_model=True, max_iter=100000)
        score = np.zeros(len(ALPHAS))
        for itr, iva in KFold(5, shuffle=True, random_state=self.seed).split(X):
            m = clone(base).fit(X[itr], y[itr])
            for j, a in enumerate(ALPHAS):
                score[j] += concordance_index_censored(y[iva]["event"], y[iva]["time"], m.predict(X[iva], alpha=a))[0]
        self.alpha = ALPHAS[int(np.argmax(score))]
        self.est = clone(base).fit(X, y)
        coef = self.est.coef_[:, list(ALPHAS).index(self.alpha)]
        self.n_selected = int((np.abs(coef[X.shape[1] - n_rad:]) > 1e-8).sum())
        return self
    def survival(self, d):
        return eval_stepfns(self.est.predict_survival_function(self._X(d), alpha=self.alpha))

class RSFModel:
    def __init__(self, seed=0): self.seed = seed
    def _X(self, d):
        return np.hstack([ClinicalDesign(volume=True, spline=False).fit(d).transform(d).values,
                          t1gd.loc[d.index].values, flair.loc[d.index].values])
    def fit(self, d):
        self.rsf = RandomSurvivalForest(n_estimators=500, min_samples_leaf=15, max_features="sqrt",
                                        n_jobs=1, random_state=self.seed).fit(self._X(d), surv_y(d))
        return self
    def survival(self, d):
        return eval_stepfns(self.rsf.predict_survival_function(self._X(d)))

MODELS = {
    "No covariates (Kaplan–Meier)":                 lambda s: KMModel(),
    "Clinical, no volume (Cox)":                    lambda s: CoxModel(ClinicalDesign(volume=False, spline=False)),
    "Clinical + volume, linear (Cox) [Model A]":    lambda s: CoxModel(ClinicalDesign(volume=True, spline=False)),
    "Clinical + volume, splines (Cox) [Model B]":   lambda s: CoxModel(ClinicalDesign(volume=True, spline=True)),
    "Model B + time-varying effects (Cox)":         lambda s: TVCoxModel(ClinicalDesign(volume=True, spline=True)),
    "Radiomics T1-Gd only (elastic-net Cox)":       lambda s: CoxnetModel([t1gd], seed=s),
    "Radiomics FLAIR only (elastic-net Cox)":       lambda s: CoxnetModel([flair], seed=s),
    "Radiomics T1-Gd + FLAIR (elastic-net Cox)":    lambda s: CoxnetModel([t1gd, flair], seed=s),
    "Model B + FLAIR texture (elastic-net Cox)":    lambda s: CoxnetModel([flair], design=ClinicalDesign(True, True), seed=s),
    "Model B + 5 FLAIR principal components (Cox)": lambda s: CoxModel(ClinicalDesign(True, True), flair_pcs=5),
    "Random survival forest (all features)":        lambda s: RSFModel(seed=s),
}

def metrics(T, S):
    rmst = np.trapezoid(S[:, GRID <= 156], GRID[GRID <= 156], axis=1)      # restricted mean survival to 3 years
    out = {"C-index": concordance_index_censored(np.ones(len(T), bool), T.astype(float), -rmst)[0]}
    for t, i in ((26, I26), (52, I52), (104, I104)):
        lab = (T <= t).astype(int)
        out[f"AUC {t} wk"] = roc_auc_score(lab, 1 - S[:, i]) if np.ptp(S[:, i]) > 0 else 0.5
    bt = (GRID >= 4) & (GRID <= 104)
    out["Integrated Brier"] = np.trapezoid((((T[:, None] > GRID[None, bt]) * 1.0 - S[:, bt]) ** 2).mean(0), GRID[bt]) / 100
    out["Calibration 1 y"] = S[:, I52].mean() - (T > 52).mean()
    return out

def pred_median(S):
    below = S <= 0.5
    return np.where(below.any(1), GRID[np.argmax(below, axis=1)], np.nan)

cvdat = dm.loc[dm.index.intersection(flair.index)].copy()

def run_fold(r, k, tr, te):
    TR, TE = cvdat.iloc[tr], cvdat.iloc[te]
    res, oof = [], []
    for name, make in MODELS.items():
        S = make(r * 10 + k).fit(TR).survival(TE)
        res.append({"rep": r, "fold": k, "model": name, **metrics(TE.time.values, S)})
        oof.append(pd.DataFrame({"rep": r, "model": name, "id": TE.index, "S26": S[:, I26],
                                 "S52": S[:, I52], "S104": S[:, I104], "median": pred_median(S)}))
    return res, pd.concat(oof)

cache = OUT / "cv_results.csv"
if cache.exists():
    cvres = pd.read_csv(cache); oof = pd.read_csv(OUT / "cv_oof_predictions.csv")
else:
    tasks = [(r, k, tr, te) for r in range(N_REPEATS)
             for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=SEED + r).split(cvdat))]
    out = Parallel(n_jobs=-1)(delayed(run_fold)(*t) for t in tasks)
    cvres = pd.DataFrame([row for res, _ in out for row in res]); oof = pd.concat([o for _, o in out])
    cvres.to_csv(cache, index=False); oof.to_csv(OUT / "cv_oof_predictions.csv", index=False)
print(f"n = {len(cvdat)} patients, {cvres.rep.nunique()} repeats × 5 folds = {cvres.groupby(['rep', 'fold']).ngroups} train/test splits per model")

# %% [markdown]
# ### 5.2 Cross-validated performance

# %%
cols = ["C-index", "AUC 26 wk", "AUC 52 wk", "AUC 104 wk", "Integrated Brier", "Calibration 1 y"]
perf = cvres.groupby("model")[cols].mean()
perf_sd = cvres.groupby("model")[cols].std()
ref = "Clinical + volume, splines (Cox) [Model B]"
paired = cvres.pivot_table(index=["rep", "fold"], columns="model", values="C-index")
perf["ΔC vs Model B"] = (paired.sub(paired[ref], axis=0)).mean()
perf["% splits better than B"] = (paired.gt(paired[ref], axis=0)).mean() * 100
perf = perf.loc[list(MODELS)]
perf.to_csv(OUT / "table6_cv_performance.csv")
display(perf.style.format({c: "{:.3f}" for c in cols} | {"ΔC vs Model B": "{:+.3f}", "% splits better than B": "{:.0f}%"})
        .format({"Calibration 1 y": "{:+.3f}"}))

order = perf["C-index"].sort_values().index
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
for ax, metric, xlabel in [(axes[0], "C-index", "C-index (higher is better)"),
                           (axes[1], "Integrated Brier", "Integrated Brier score, 4–104 wk (lower is better)")]:
    m, sd = perf.loc[order, metric], perf_sd.loc[order, metric]
    colors = [C[1] if "Radiomics" in n and "Model B" not in n else (MUTED if "Kaplan" in n else C[0]) for n in order]
    ax.hlines(range(len(order)), m - sd, m + sd, color=colors, lw=2, alpha=0.45)
    ax.scatter(m, range(len(order)), color=colors, s=60, zorder=3, edgecolor=SURF, linewidth=2)
    for i, v in enumerate(m):
        ax.text(v, i + 0.28, f"{v:.3f}", ha="center", fontsize=8, color=INK2)
    ax.set_xlabel(xlabel); ax.grid(axis="y", visible=False)
axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels(order)
axes[0].axvline(0.5, color=AXIS, lw=1)
axes[0].set_title("Discrimination (mean ± SD over 50 test folds)", loc="left")
axes[1].set_title("Overall accuracy of the predicted survival curve", loc="left")
fig.text(0.01, -0.02, "Blue = clinical-based models, orange = radiomics-only, grey = no-covariate reference", color=INK2, fontsize=9)
fig.tight_layout(); savefig(fig, "fig4_cv_performance"); plt.show()

# %% [markdown]
# **Reading the comparison**
# * Every model with covariates clearly beats the no-covariate reference, but discrimination is **modest** for all of them (C ≈ 0.61–0.64). That is typical for glioblastoma when the strongest known prognostic markers (MGMT promoter methylation, IDH status, extent of resection) are not available.
# * **Radiomics alone underperforms the clinical model.** Most of the T1-Gd signal is tumour size, which the clinical model already captures through volume.
# * **High-dimensional radiomics overfits.** Adding 93 FLAIR features through an elastic net (ΔC ≈ −0.004) does worse than compressing them first into 5 principal components (ΔC ≈ +0.008, better in ~2/3 of splits). The PC version also clearly improves 2-year discrimination (AUC 0.67 vs 0.63).
# * A random survival forest is no better than the Cox models, so there is no evidence of strong interactions or non-linearities that a Cox model misses.
#
# **Final model ("Model C") = Model B + 5 FLAIR texture components.** It has the best C-index and 2-year AUC, an integrated Brier score as good as any candidate, and is still a transparent Cox model.
# **Model B** (clinical + volume only) is kept as the **bedside version**. It needs only age, PS, planned RT regimen and tumour volume, and loses only about 0.01 in C-index.

# %% [markdown]
# ### 5.3 The final model

# %%
FINAL_NAME, BEDSIDE_NAME = "Model B + 5 FLAIR principal components (Cox)", "Clinical + volume, splines (Cox) [Model B]"
final = CoxModel(ClinicalDesign(volume=True, spline=True), flair_pcs=5).fit(cvdat)
bedside = CoxModel(ClinicalDesign(volume=True, spline=True)).fit(dm)

print(f"FLAIR features entering the PCA: {final.keep.sum()} of {len(final.keep)} "
      f"(excluded as volume surrogates: {', '.join(c.replace('original_', '') for c in flair.columns[~final.keep])})")
loads = pd.DataFrame(final.pca.components_.T, index=[c.replace("original_", "") for c in flair.columns[final.keep]],
                     columns=[f"PC{i + 1}" for i in range(5)])
pc_desc = {pc: ", ".join(f"{f} ({'+' if loads.loc[f, pc] > 0 else '−'})" for f in loads[pc].abs().sort_values(ascending=False).index[:3])
           for pc in loads.columns}
ftab = final.cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].copy()
ftab.columns = ["HR", "95% CI low", "95% CI high", "p"]
pc_sd = final._X(cvdat)[[f"PC{i + 1}" for i in range(5)]].std()
for pc in pc_sd.index:   # express PC effects per 1 SD of the component
    ftab.loc[pc, ["HR", "95% CI low", "95% CI high"]] = np.exp(np.log(ftab.loc[pc, ["HR", "95% CI low", "95% CI high"]].astype(float)) * pc_sd[pc])
ftab.index = [LABELS.get(i, f"FLAIR {i} (per SD): {pc_desc.get(i, '')}") for i in ftab.index]
ftab.to_csv(OUT / "table7_final_model.csv")
display(Markdown(f"**Model C coefficients** (n = {len(cvdat)}). The spline terms are jointly interpretable through the curves in §3.4; the FLAIR components are listed with their three largest loadings."))
display(ftab.style.format({"HR": "{:.2f}", "95% CI low": "{:.2f}", "95% CI high": "{:.2f}", "p": "{:.3g}"}))

# %% [markdown]
# ### 5.4 Calibration: do predicted survival probabilities match what actually happened?
# Each patient's prediction is **out-of-fold**: it comes from models that never saw that patient, averaged over the 10 CV repeats.
# Patients are grouped into fifths by predicted probability of being alive at 6, 12 and 24 months, and compared with the observed proportion alive in each group. Because nobody is censored, the observed proportion is exact.
# Points on the diagonal mean perfect calibration.

# %%
def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h

oof_avg = {m: oof[oof.model == m].groupby("id")[["S26", "S52", "S104", "median"]].mean().join(cvdat[["time"]])
           for m in [FINAL_NAME, BEDSIDE_NAME]}
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
cal_rows = []
for ax, (t, col) in zip(axes, [(26, "S26"), (52, "S52"), (104, "S104")]):
    for j, (m, lab) in enumerate([(FINAL_NAME, "Model C (clinical + FLAIR)"), (BEDSIDE_NAME, "Model B (bedside)")]):
        o = oof_avg[m]; q = pd.qcut(o[col].rank(method="first"), 5, labels=False)
        g = o.groupby(q).agg(pred=(col, "mean"), k=("time", lambda x: (x > t).sum()), n=("time", "size"))
        g["obs"] = g.k / g.n; lo, hi = wilson(g.k, g.n)
        off = (j - 0.5) * 0.008
        ax.errorbar(g.pred + off, g.obs, yerr=[g.obs - lo, hi - g.obs], fmt="o", color=C[j], ms=6, lw=1.2, capsize=0,
                    label=lab, mec=SURF, mew=1.5)
        cll = np.log(-np.log(np.clip(o[col], 1e-6, 1 - 1e-6)))
        slope = CoxPHFitter().fit(pd.DataFrame({"time": o.time, "event": 1, "cll": cll}), "time", "event").params_["cll"]
        cal_rows.append({"Model": lab, "Horizon": f"{t} wk", "Mean predicted": o[col].mean(), "Observed": (o.time > t).mean(),
                         "Calibration slope": slope})
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1, ls="--")
    lim = 0.5 if t == 104 else 1.0
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel(f"Predicted probability alive at {t} weeks"); ax.set_ylabel("Observed proportion alive")
    ax.set_title(f"{t} weeks ({t / 52:.1f} year{'s' if t > 52 else ''})" if t != 26 else "26 weeks (6 months)", loc="left")
axes[0].legend(loc="upper left")
fig.tight_layout(); savefig(fig, "fig5_calibration"); plt.show()
cal = pd.DataFrame(cal_rows).set_index(["Model", "Horizon"])
display(cal.style.format("{:.2f}"))

# %% [markdown]
# *Calibration slope*: 1 is ideal. Below 1 means predictions are too extreme, i.e. the model is somewhat over-confident, which is expected with ~250 patients.

# %% [markdown]
# ### 5.5 Risk groups: does the model separate patients in practice?
# Patients are split into thirds by their **out-of-fold** predicted median survival from Model C, so these curves are not optimistic re-fits.

# %%
o = oof_avg[FINAL_NAME].copy()
o["group"] = pd.qcut(o["median"].rank(method="first"), 3, labels=["High risk", "Intermediate risk", "Low risk"])
fig, ax = plt.subplots(figsize=(8.5, 5))
grp_rows = []
for i, gname in enumerate(["Low risk", "Intermediate risk", "High risk"]):
    g = o[o.group == gname]; k = KaplanMeierFitter().fit(g.time, np.ones(len(g)))
    k.plot_survival_function(ax=ax, ci_show=True, color=C[i], label=f"{gname} (n={len(g)})", ci_alpha=0.12)
    ax.text(k.median_survival_time_ + 3, 0.52, f"{k.median_survival_time_:.0f} wk", color=INK2, fontsize=8)
    grp_rows.append({"Group": gname, "n": len(g), "Median survival, weeks": k.median_survival_time_,
                     "Alive at 6 mo": k.predict(26), "Alive at 1 y": k.predict(52), "Alive at 2 y": k.predict(104),
                     "Mean predicted median": g["median"].mean()})
pv = multivariate_logrank_test(o.time, o.group, np.ones(len(o))).p_value
ax.axhline(0.5, color=AXIS, lw=1, ls=":")
ax.set_xlim(0, 200); ax.set_xticks([0, 26, 52, 78, 104, 130, 156, 182])
ax.set_xlabel("Weeks since t0 MRI"); ax.set_ylabel("Proportion alive")
ax.set_title(f"Survival by cross-validated risk group (Model C), log-rank p = {pv:.1g}", loc="left")
fig.tight_layout(); savefig(fig, "fig6_risk_groups"); plt.show()
grp = pd.DataFrame(grp_rows).set_index("Group")
display(grp.style.format({"Median survival, weeks": "{:.0f}", "Alive at 6 mo": "{:.0%}", "Alive at 1 y": "{:.0%}",
                          "Alive at 2 y": "{:.0%}", "Mean predicted median": "{:.0f}"}))

# %% [markdown]
# ### 5.6 Individualized predictions
# `predict_patient()` returns a full survival curve for a new patient, together with:
# * **Predicted median survival**: the time at which the patient's predicted probability of being alive falls to 50%.
# * **50% prediction interval**: the range between the 25th and 75th percentile of the predicted survival time. It shows how much individual outcomes vary even when the risk profile is known exactly.
# * **Probability of being alive** at 6, 12 and 24 months.
#
# Without FLAIR features it uses the bedside Model B. If a FLAIR feature vector is supplied (PyRadiomics, same settings, GTV mask at t0), it uses Model C.
# `who2021=1` is the default because new patients are diagnosed under the current WHO 2021 classification.
# Uncertainty in the model itself is shown with 95% bootstrap intervals (200 resamples, refitting the whole pipeline each time).

# %%
def make_patient(age, ps, regimen, volume_cm3, who2021=1, flair_features=None):
    row = {"age": age, "age10": (age - 65) / 10, "ps2": int(ps == 2), "ps3": int(ps == 3), "who2021": who2021,
           "hypo": int(regimen == "Hypofractionated"), "unk": int(regimen == "Unknown"),
           "lv": np.log2(volume_cm3) - np.log2(30), "time": np.nan, "event": np.nan}
    p = pd.DataFrame([row])
    if flair_features is not None:
        p = p.join(pd.DataFrame([pd.Series(flair_features)[flair.columns].values], columns=flair.columns))
    return p

def summarize_curve(S):
    t_at = lambda level: GRID[np.argmax(S <= level)] if (S <= level).any() else np.nan
    return {"Median, weeks": t_at(0.5), "50% PI low": t_at(0.75), "50% PI high": t_at(0.25),
            "Alive 6 mo": S[I26], "Alive 1 y": S[I52], "Alive 2 y": S[I104]}

def predict_patient(age, ps, regimen, volume_cm3, who2021=1, flair_features=None, model=None):
    """Individualized survival prediction. regimen: 'Conventional' | 'Hypofractionated' | 'Unknown'."""
    p = make_patient(age, ps, regimen, volume_cm3, who2021, flair_features)
    model = model or (final if flair_features is not None else bedside)
    S = model.survival(p)[0]
    return S, summarize_curve(S)

examples = pd.DataFrame([
    {"Patient": "A", "age": 52, "ps": 0, "regimen": "Conventional", "volume_cm3": 20},
    {"Patient": "B", "age": 64, "ps": 1, "regimen": "Conventional", "volume_cm3": 40},
    {"Patient": "C", "age": 73, "ps": 2, "regimen": "Hypofractionated", "volume_cm3": 60},
    {"Patient": "D", "age": 79, "ps": 3, "regimen": "Hypofractionated", "volume_cm3": 100},
]).set_index("Patient")

# FLAIR texture profiles: the cohort's typical (feature-wise median) texture, and the actual texture vectors of the
# patients at the 10th / 90th percentile of the Model C radiomic risk score
Xc = final._X(cvdat); pcs = [f"PC{i + 1}" for i in range(5)]
rad_score = Xc[pcs] @ final.cph.params_[pcs]
typical_flair = flair.loc[cvdat.index].median()
fav_flair = flair.loc[(rad_score - rad_score.quantile(0.10)).abs().idxmin()]
unfav_flair = flair.loc[(rad_score - rad_score.quantile(0.90)).abs().idxmin()]

rng = np.random.default_rng(SEED)
B = 200
boot_S = {p: [] for p in examples.index}
for b in range(B):
    idx = rng.choice(len(cvdat), len(cvdat), replace=True)
    bd = cvdat.iloc[idx].copy(); bd.index = np.arange(len(bd))
    bflair = flair.loc[cvdat.index[idx]].copy(); bflair.index = bd.index
    bd = bd.join(bflair)
    try:
        bm = CoxModel(ClinicalDesign(volume=True, spline=True), flair_pcs=5).fit(bd)
    except Exception:
        continue
    for pid, e in examples.iterrows():
        boot_S[pid].append(bm.survival(make_patient(e.age, e.ps, e.regimen, e.volume_cm3, 1, typical_flair))[0])

pred_rows, curves = [], {}
for pid, e in examples.iterrows():
    S, summ = predict_patient(e.age, e.ps, e.regimen, e.volume_cm3, flair_features=typical_flair)
    Sb = np.array(boot_S[pid]); curves[pid] = (S, np.percentile(Sb, 2.5, 0), np.percentile(Sb, 97.5, 0))
    meds = np.array([summarize_curve(s)["Median, weeks"] for s in Sb])
    _, summ_bed = predict_patient(e.age, e.ps, e.regimen, e.volume_cm3)
    pred_rows.append({"Patient": pid, "Profile": f"{e.age} y, PS {e.ps}, {e.regimen.lower()} RT, {e.volume_cm3} cm³",
                      "Median (wk)": summ["Median, weeks"],
                      "95% CI": f"{np.nanpercentile(meds, 2.5):.0f}–{np.nanpercentile(meds, 97.5):.0f}",
                      "50% of patients die between (wk)": f"{summ['50% PI low']:.0f}–{summ['50% PI high']:.0f}",
                      "Alive 6 mo": summ["Alive 6 mo"],
                      "Alive 1 y": summ["Alive 1 y"],
                      "1 y, 95% CI": f"{np.percentile(Sb[:, I52], 2.5):.0%}–{np.percentile(Sb[:, I52], 97.5):.0%}",
                      "Alive 2 y": summ["Alive 2 y"],
                      "Bedside Model B median (wk)": summ_bed["Median, weeks"]})
pred_tab = pd.DataFrame(pred_rows).set_index("Patient")
pred_tab.to_csv(OUT / "table8_example_predictions.csv")
display(Markdown(f"**Example predictions** (Model C with typical FLAIR texture; {len(boot_S['A'])} successful bootstrap refits)"))
display(pred_tab.style.format({"Median (wk)": "{:.0f}", "Alive 6 mo": "{:.0%}", "Alive 1 y": "{:.0%}", "Alive 2 y": "{:.0%}",
                               "Bedside Model B median (wk)": "{:.0f}"}))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
for i, pid in enumerate(examples.index):
    S, lo, hi = curves[pid]
    ax.fill_between(GRID, lo, hi, color=C[i], alpha=0.12, lw=0)
    ax.plot(GRID, S, color=C[i], label=f"{pid}: {pred_tab.loc[pid, 'Profile']}")
ax.axhline(0.5, color=AXIS, lw=1, ls=":")
ax.set_xlim(0, 156); ax.set_xticks([0, 26, 52, 78, 104, 130, 156]); ax.set_ylim(0, 1)
ax.set_xlabel("Weeks since t0 MRI"); ax.set_ylabel("Predicted probability alive")
ax.set_title("Predicted survival for four example patients (95% bootstrap bands)", loc="left"); ax.legend(loc="upper right", fontsize=8)

ax = axes[1]
e = examples.loc["B"]
for i, (lab, fv) in enumerate([("Favourable FLAIR texture (10th pct risk)", fav_flair), ("Typical FLAIR texture", typical_flair),
                               ("Unfavourable FLAIR texture (90th pct risk)", unfav_flair)]):
    S, summ = predict_patient(e.age, e.ps, e.regimen, e.volume_cm3, flair_features=fv)
    ax.plot(GRID, S, color=C[i], label=f"{lab}: median {summ['Median, weeks']:.0f} wk, 1 y {summ['Alive 1 y']:.0%}")
S, summ = predict_patient(e.age, e.ps, e.regimen, e.volume_cm3)
ax.plot(GRID, S, color=MUTED, ls="--", lw=1.5, label=f"Bedside Model B (no radiomics): median {summ['Median, weeks']:.0f} wk")
ax.axhline(0.5, color=AXIS, lw=1, ls=":")
ax.set_xlim(0, 156); ax.set_xticks([0, 26, 52, 78, 104, 130, 156]); ax.set_ylim(0, 1)
ax.set_xlabel("Weeks since t0 MRI"); ax.set_title(f"Patient B ({pred_tab.loc['B', 'Profile']}): effect of FLAIR texture", loc="left")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout(); savefig(fig, "fig7_individual_predictions"); plt.show()

# %% [markdown]
# ## 6. Additional analyses
# ### 6.1 Early response on the first follow-up MRI (exploratory, *not* a baseline predictor)
# RANO response between t0 and t1 is only known for patients who **survived long enough to have a t1 scan**. Patients with a t1 assessment had median survival of 57 weeks vs 28 weeks for those without.
# Using response as a baseline predictor would therefore create **immortal-time bias**. Instead we ask: *among patients who reached the first follow-up MRI, does early progression add information beyond the baseline factors?* This is the question a physician faces at that visit.
# Caveats: the exact date of t1 is not in the shared files, so a formal landmark analysis is not possible. And radiological "progression" soon after chemoradiation is partly **pseudoprogression**.

# %%
r = dm[dm.rano_t0_to_t1.notna()].copy()
r["resp"] = r.rano_t0_to_t1.str.extract(r"\((\w+)\)")[0].replace({"CR": "PR"}).map({"PD": "Progression", "SD": "Stable", "PR": "Partial/complete response"})
r["PD"] = (r.resp == "Progression").astype(int); r["PRCR"] = (r.resp == "Partial/complete response").astype(int)
r["vol_change_doublings"] = np.log2(r["size_t1 (cm3)"].clip(lower=0.1) / r["size_t0 (cm3)"])
print(f"Median survival with vs without a t1 RANO assessment: {r.time.median():.0f} vs "
      f"{dm.loc[~dm.index.isin(r.index), 'time'].median():.0f} weeks")
fig, ax = plt.subplots(figsize=(8.5, 4.6))
for i, lev in enumerate(["Partial/complete response", "Stable", "Progression"]):
    g = r[r.resp == lev]; k = KaplanMeierFitter().fit(g.time, g.event)
    k.plot_survival_function(ax=ax, ci_show=False, color=C[i], label=f"{lev} at t1 (n={len(g)}, median {k.median_survival_time_:.0f} wk)")
ax.set_xlim(0, 200); ax.set_xticks([0, 26, 52, 78, 104, 130, 156, 182])
ax.set_xlabel("Weeks since t0 MRI"); ax.set_ylabel("Proportion alive")
ax.set_title(f"Survival by RANO response t0→t1, log-rank p = {multivariate_logrank_test(r.time, r.resp, r.event).p_value:.2g}", loc="left")
fig.tight_layout(); savefig(fig, "fig8_rano_t1"); plt.show()
Xr = ClinicalDesign(volume=True, spline=False).fit(r).transform(r)
s1 = CoxPHFitter().fit(Xr.assign(PD=r.PD, PRCR=r.PRCR, time=r.time, event=r.event), "time", "event").summary
s2 = CoxPHFitter().fit(Xr.assign(chg=r.vol_change_doublings, time=r.time, event=r.event), "time", "event").summary
rt = pd.DataFrame({"HR": [s1.loc["PD", "exp(coef)"], s1.loc["PRCR", "exp(coef)"], s2.loc["chg", "exp(coef)"]],
                   "95% CI low": [s1.loc["PD", "exp(coef) lower 95%"], s1.loc["PRCR", "exp(coef) lower 95%"], s2.loc["chg", "exp(coef) lower 95%"]],
                   "95% CI high": [s1.loc["PD", "exp(coef) upper 95%"], s1.loc["PRCR", "exp(coef) upper 95%"], s2.loc["chg", "exp(coef) upper 95%"]],
                   "p": [s1.loc["PD", "p"], s1.loc["PRCR", "p"], s2.loc["chg", "p"]]},
                  index=["Progression at t1 (vs stable)", "Partial/complete response at t1 (vs stable)", "Tumour volume change t0→t1 (per doubling)"])
display(Markdown(f"**Adjusted for age, PS, WHO edition, RT regimen and baseline volume** (n = {len(r)})"))
display(rt.style.format({"HR": "{:.2f}", "95% CI low": "{:.2f}", "95% CI high": "{:.2f}", "p": "{:.3g}"}))

# %% [markdown]
# ### 6.2 A counter-intuitive finding: a longer wait before radiotherapy goes with *longer* survival
# Each extra week between diagnosis and the start of radiotherapy is associated with *lower* hazard. This is almost certainly **confounding by indication**, not a benefit of delay: the sickest patients (worse PS, more symptomatic) are started urgently. It is a good example of why the treatment variables in this observational dataset must not be read causally. The same applies to the hypofractionated regimen, which is chosen for older and frailer patients.

# %%
dl = df[df.rt_delay.notna()].copy()
dl["Delay to RT"] = pd.cut(dl.rt_delay, [-1, 0, 1, 2, 10], labels=["0 wk", "1 wk", "2 wk", "≥3 wk"])
display(dl.groupby("Delay to RT", observed=True).agg(n=("time", "size"), **{"Median survival (wk)": ("time", "median"),
        "Mean PS": ("ps", "mean"), "% PS ≥ 2": ("ps", lambda x: (x >= 2).mean() * 100), "Median age": ("age", "median")})
        .style.format({"Median survival (wk)": "{:.0f}", "Mean PS": "{:.2f}", "% PS ≥ 2": "{:.0f}%", "Median age": "{:.0f}"}))

# %% [markdown]
# ## 7. Limitations
# 1. **Single centre, n = 264, no external validation.** Cross-validation estimates performance in *similar* patients. The model needs validation in an independent cohort, ideally from another institution with different MRI scanners, before any clinical use.
# 2. **Key molecular and surgical prognostic factors are missing**: MGMT promoter methylation, IDH status, extent of resection, and concurrent/adjuvant temozolomide. These are among the strongest known predictors in glioblastoma, so they explain much of the remaining unexplained variation (C ≈ 0.64).
# 3. **Treatment variables are not randomized.** The HRs for RT regimen and delay are *prognostic* (they reflect clinicians' judgement of frailty), not causal estimates of treatment benefit.
# 4. **WHO edition is a proxy.** "WHO 2021" probably flags tumours classified under the IDH-wildtype-only definition, and possibly a later calendar period. New patients will all be classified under 2021, so predictions should use `who2021=1`.
# 5. **Proportional hazards is only approximately met.** Frailty markers (PS 2, hypofractionated RT) act mainly in the first 6 months. A time-varying model fits slightly better (lower Brier score) but does not improve discrimination, so the simpler PH model was kept.
# 6. **2-year predictions for the best-prognosis patients are optimistic** (§5.4), and calibration slopes < 1 indicate mild over-fitting. Shrinking the coefficients, or re-calibrating on external data, would be a sensible next step.
# 7. **Radiomics reproducibility**: features depend on the scanner, preprocessing and segmentation. The FLAIR components must be computed with exactly the same pipeline (PyRadiomics settings, intensity standardization, GTV mask) for Model C to apply.
#
# ## 8. Reproducibility
# Python environment: pandas, numpy, scipy, lifelines, scikit-survival, scikit-learn, statsmodels, matplotlib (see `requirements.txt`). Cross-validation results are cached in `outputs/cv_results.csv`; delete that file to recompute (~3 min on 8 cores). All figures and tables are written to `outputs/`.
