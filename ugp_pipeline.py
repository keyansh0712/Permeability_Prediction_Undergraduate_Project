#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-entry UGP runner. Input is a folder of images: --data_dir.

For each noise level:
  1) Run a patched copy of predictions_permeability.py to create predictions.csv
     under runs/noise_<level>/
  2) Patch ugp-code-new.ipynb to read ./predictions.csv and ./data, then execute it
     inside that run directory.

Usage:
  python ugp_pipeline.py --data_dir data --notebook ugp-code-new.ipynb --noise_levels 0,0.02,0.05,0.1

Noise levels:
  Set with --noise_levels. Comma list or a:b:step, e.g. 0.00:0.10:0.02.
"""

import os, sys, math, shutil, argparse, re
from pathlib import Path
from typing import List

def parse_noise_levels(s: str) -> List[float]:
    s = s.strip()
    if ":" in s:
        a,b,step = s.split(":")
        a=float(a); b=float(b); step=float(step)
        out=[]; v=a
        while v <= b + 1e-12:
            out.append(round(v,10)); v += step
        return out
    return [float(x) for x in s.split(",") if x.strip()!=""]

def safe_symlink(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src, target_is_directory=src.is_dir())
    except Exception:
        if src.is_dir(): shutil.copytree(src, dst, dirs_exist_ok=True)
        else: shutil.copy2(src, dst)

def render_predictions_script(template_text: str, data_dir: Path, out_dir: Path, noise_level: float, seed: int) -> str:
    t = template_text

    # 1) Force directories
    t = re.sub(r'^IMAGES_DIR\s*=.*$', 'IMAGES_DIR = Path(r"%s")' % (str(data_dir.resolve())), t, flags=re.M)
    t = re.sub(r'^OUTDIR\s*=.*$',     'OUTDIR     = Path(r"%s"); OUTDIR.mkdir(parents=True, exist_ok=True)' % (str(out_dir.resolve())), t, flags=re.M)

    # 2) Fix RNG and noise
    t = re.sub(r'^RANDOM_STATE\s*=.*$', 'RANDOM_STATE = %d' % seed, t, flags=re.M)
    t = re.sub(r'^NOISE_ENABLED\s*=.*$', 'NOISE_ENABLED = True', t, flags=re.M)
    t = re.sub(r'^NOISE_REL_STD\s*=.*$', 'NOISE_REL_STD = %s' % noise_level, t, flags=re.M)

    # 3) Force non-interactive matplotlib backend
    if "matplotlib.use(" not in t:
        t = t.replace("import matplotlib.pyplot as plt",
                      "import matplotlib; matplotlib.use('Agg')\nimport matplotlib.pyplot as plt")

    # 4) Remove any Colab zip block that writes to /content
    t = re.sub(r'# ================== Zip outputs ==================[\s\S]*?print\("Also:.*"\)\s*$', 
               '# Zip block removed by wrapper', t, flags=re.M)

    return t

def run_predictions_once(data_dir: Path, out_dir: Path, noise_level: float, seed: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_symlink(data_dir.resolve(), out_dir/"data")

    script_text = render_predictions_script(PREDICTIONS_TEMPLATE, data_dir, out_dir, noise_level, seed)
    tmp_script = out_dir / "_pred_run.py"
    tmp_script.write_text(script_text, encoding="utf-8")

    cmd = f"python \"{tmp_script}\""
    print(f"[pred] noise={noise_level:.4f} -> {out_dir}  cmd: {cmd}")
    rc = os.system(cmd)
    if rc != 0:
        raise SystemExit(f"predictions script failed with code {rc}")

    csv_path = out_dir / "predictions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"predictions.csv not found in {out_dir}")
    return csv_path

def patch_and_execute_notebook(notebook_path: Path, workdir: Path, timeout: int = 1200):
    try:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
        from nbconvert import HTMLExporter
    except Exception:
        print("[warn] nbconvert/nbformat missing. Skipping notebook execution.")
        return None, None

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Patch csv_path and images_root in the first cell that defines them
    for cell in nb.cells:
        if cell.cell_type == "code":
            src = "".join(cell.source)
            if "csv_path" in src and "images_root" in src:
                src = re.sub(r'csv_path\s*=.*', 'csv_path = "./predictions.csv"', src)
                src = re.sub(r'images_root\s*=.*', 'images_root = "./data"', src)
                cell.source = src
                break

    ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(workdir)}})

    executed_ipynb = workdir / (notebook_path.stem + ".executed.ipynb")
    with open(executed_ipynb, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    html_exporter = HTMLExporter()
    (body, resources) = html_exporter.from_notebook_node(nb)
    executed_html = workdir / (notebook_path.stem + ".executed.html")
    executed_html.write_text(body, encoding="utf-8")

    return executed_ipynb, executed_html

def main():
    ap = argparse.ArgumentParser(description="UGP multi-noise runner (single input: images folder)")
    ap.add_argument("--data_dir", type=Path, required=True, help="Folder with images")
    ap.add_argument("--notebook", type=Path, default=Path("ugp-code-new.ipynb"))
    ap.add_argument("--outdir", type=Path, default=Path("runs"))
    ap.add_argument("--noise_levels", type=str, default="0,0.02,0.05,0.1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    if not args.data_dir.exists():
        raise SystemExit(f"data_dir missing: {args.data_dir}")
    if args.notebook and not args.notebook.exists():
        print(f"[warn] notebook not found: {args.notebook}. Will skip notebook execution.")
        args.notebook = None

    levels = parse_noise_levels(args.noise_levels)
    print(f"[info] noise levels: {levels}")

    for lvl in levels:
        run_dir = args.outdir / f"noise_{lvl:.4f}"
        run_dir.mkdir(parents=True, exist_ok=True)

        csv_path = run_predictions_once(args.data_dir, run_dir, noise_level=float(lvl), seed=args.seed)
        print(f"[ok] predictions: {csv_path}")

        if args.notebook:
            try:
                ipynb, html = patch_and_execute_notebook(args.notebook, workdir=run_dir, timeout=args.timeout)
                if ipynb: print(f"[ok] executed notebook: {ipynb}")
                if html:  print(f"[ok] exported HTML:     {html}")
            except Exception as e:
                print(f"[warn] notebook failed for noise={lvl:.4f}: {e}")

    print("[done] All levels complete.")

# Embedded predictions_permeability.py template (minimal, headless-safe)
PREDICTIONS_TEMPLATE = r"""# ================== Imports & Config ==================
import os, re, warnings, math, random, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

IMAGES_DIR = Path("data")
LABELS_CSV = Path("labels.csv")
OUTDIR     = Path("outputs"); OUTDIR.mkdir(parents=True, exist_ok=True)

# Default A if no labels (A = d^2 / C)
DEFAULT_D_METERS = 150e-6
DEFAULT_C = 150.0
A_DEFAULT = (DEFAULT_D_METERS**2) / DEFAULT_C

RANDOM_STATE = 42
random.seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)

# =============== Noise config (new) ==================
NOISE_ENABLED = True
NOISE_REL_STD = 0.05
NOISE_ABS_MIN_STD = 1e-12
_rng = np.random.default_rng(RANDOM_STATE)

# ================== Utils ==================
def list_images(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    files = [p for p in folder.glob("*") if p.suffix.lower() in exts]
    if not files:
        raise FileNotFoundError(f"No images in {folder} with {exts}")
    return sorted(files)

def parse_phi_from_title(name: str):
    m = re.search(r"([01](?:\.\d+)?)[-_]?porosity", name.lower())
    if m: return float(m.group(1))
    m2 = re.search(r"([0-9]+\.[0-9]+)", name)
    if m2: return float(m2.group(1))
    raise ValueError(f"Cannot parse porosity from filename: {name}")

def kc_feature(phi: np.ndarray):
    eps = 1e-12
    return (phi**3) / np.maximum((1.0 - phi)**2, eps)

def parity(y_true, y_pred, title, outpng):
    plt.figure(figsize=(5.0,5.0))
    plt.scatter(y_true, y_pred, s=18, alpha=0.8)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([lo,hi],[lo,hi],"--")
    plt.xlabel("Actual permeability"); plt.ylabel("Predicted permeability")
    plt.title(title); plt.tight_layout(); plt.savefig(outpng, dpi=180); plt.close()

def add_gaussian_noise(arr: np.ndarray, rel_std: float = NOISE_REL_STD,
                       abs_min_std: float = NOISE_ABS_MIN_STD,
                       rng: np.random.Generator = _rng) -> np.ndarray:
    arr = np.array(arr, dtype=float)
    std = np.maximum(rel_std * np.abs(arr), abs_min_std)
    noise = rng.normal(loc=0.0, scale=std, size=arr.shape)
    noisy = arr + noise
    return np.clip(noisy, 1e-18, None)

# ================== Build table (φ only) ==================
files = list_images(IMAGES_DIR)
phi_title = [parse_phi_from_title(p.name) for p in files]
phi_arr = np.clip(np.array(phi_title, dtype=float), 1e-6, 1-1e-6)

df = pd.DataFrame({
    "filename": [p.name for p in files],
    "phi": phi_arr
})
df["KC_X"] = kc_feature(df["phi"].to_numpy())
df["k_KC_default"] = A_DEFAULT * df["KC_X"].to_numpy()

if NOISE_ENABLED:
    df["k_KC_default_noisy"] = add_gaussian_noise(df["k_KC_default"].to_numpy())

plt.figure(figsize=(5.2,3.4))
plt.hist(df["phi"], bins=20, alpha=0.9)
plt.xlabel("Porosity (from title)"); plt.ylabel("Count")
plt.title("Porosity distribution (title-derived)")
plt.tight_layout(); plt.savefig(OUTDIR/"hist_phi.png", dpi=180); plt.close()

# ================== Optional: labels -> Calibrate & ML ==================
have_labels = LABELS_CSV.exists()
if have_labels:
    lab = pd.read_csv(LABELS_CSV)
    if not {"filename","permeability"}.issubset(lab.columns):
        warnings.warn("labels.csv must have columns: filename, permeability. Skipping labels.")
        have_labels = False

if have_labels:
    md = df.merge(lab[["filename","permeability"]], on="filename", how="inner").copy()
    md = md.dropna(subset=["permeability"])
    y = md["permeability"].to_numpy().astype(float)
    X_kc = md[["KC_X"]].to_numpy()
    X_phi = md[["phi"]].to_numpy()
    X_hyb = md[["phi","KC_X"]].to_numpy()

    from sklearn.linear_model import LinearRegression
    A_fit_model = LinearRegression(fit_intercept=False)
    A_fit_model.fit(X_kc, y)
    A_fit = float(A_fit_model.coef_[0])
    md["k_KC_calibrated"] = A_fit * md["KC_X"].to_numpy()

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    ml_phi  = GradientBoostingRegressor(random_state=RANDOM_STATE)
    ml_hyb  = GradientBoostingRegressor(random_state=RANDOM_STATE)
    ml_phi.fit(X_phi, y)
    ml_hyb.fit(X_hyb, y)

    md["k_ML_phi"] = ml_phi.predict(X_phi)
    md["k_ML_hybrid"] = ml_hyb.predict(X_hyb)

    def metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred); rmse = math.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred);  r2  = r2_score(y_true, y_pred)
        return mse, rmse, mae, r2

    kc_mse, kc_rmse, kc_mae, kc_r2 = metrics(y, md["k_KC_calibrated"])
    p_mse,  p_rmse,  p_mae,  p_r2  = metrics(y, md["k_ML_phi"])
    h_mse,  h_rmse,  h_mae,  h_r2  = metrics(y, md["k_ML_hybrid"])

    parity(y, md["k_KC_calibrated"].to_numpy(), f"KC (A_fit={A_fit:.2e})", OUTDIR/"kc_parity.png")
    parity(y, md["k_ML_phi"].to_numpy(), "ML (φ only)", OUTDIR/"ml_phi_parity.png")
    parity(y, md["k_ML_hybrid"].to_numpy(), "ML (Hybrid: φ + KC_X)", OUTDIR/"ml_hybrid_parity.png")

    df = df.merge(md[["filename","k_KC_calibrated","k_ML_phi","k_ML_hybrid"]], on="filename", how="left")

    if NOISE_ENABLED:
        df["k_KC_calibrated_noisy"] = add_gaussian_noise(df["k_KC_calibrated"].to_numpy())
        df["k_ML_phi_noisy"]       = add_gaussian_noise(df["k_ML_phi"].to_numpy())
        df["k_ML_hybrid_noisy"]    = add_gaussian_noise(df["k_ML_hybrid"].to_numpy())

df = df.sort_values("filename")
df.to_csv(OUTDIR/"predictions.csv", index=False)

with open(OUTDIR/"report.md","w",encoding="utf-8") as f:
    f.write("# Permeability via KC using φ from filename\n\n")
    f.write(f"- Images: {len(df)}\n")
    f.write(f"- KC feature: X = φ³/(1-φ)²\n")
    f.write(f"- Default A = d²/C with d={DEFAULT_D_METERS:.2e} m, C={DEFAULT_C:.1f}\n")

print("Done.")
print(f"- Images dir: {IMAGES_DIR}")
print(f"- Outputs dir: {OUTDIR}")
print("Saved: predictions.csv, hist_phi.png, report.md")
"""

if __name__ == "__main__":
    main()
