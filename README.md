# Human Action Recognition (Streamlit + PyTorch)

End‑to‑end project to **train** a human action classifier and **serve** it with a **Streamlit** frontend,
using the Hugging Face dataset:

```
visual-layer/human-action-recognition-vl-enriched
```

> Your timezone: Asia/Kolkata. Tested on Windows 10/11 and Ubuntu.

---

## 🧰 1) Prerequisites

- **VS Code** + Python extension
- **Python 3.9–3.11**
- **Git** (optional but recommended)

Create and activate a virtual environment (PowerShell on Windows):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux (bash/zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 📦 2) Install dependencies

> ⚠️ Install **PyTorch** first (choose CPU or your CUDA version): https://pytorch.org/get-started/locally/

**CPU only example:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Then install the rest:

```bash
pip install -r requirements.txt
```

---

## 🗂️ 3) Project structure

```
har-streamlit/
├─ app.py                    # Streamlit frontend (predict + explore)
├─ requirements.txt
├─ README.md
├─ artifacts/                # Trained model & label mapping saved here
├─ data/                     # Local image cache
└─ src/
   ├─ __init__.py
   ├─ dataset_utils.py       # Download/cache images, datasets & transforms
   ├─ model.py               # ResNet model factory
   ├─ train.py               # Training script (fine-tune ResNet)
   ├─ infer.py               # Inference utility
   └─ utils.py               # Helpers (seed, confusion matrix plot)
```

---

## ⛏️ 4) Train a baseline model

This dataset contains **image URIs** (remote links). The training script will **download and cache**
images to `data/images/` automatically.

Basic run (downloads a subset of images to keep it lightweight on first run):

```bash
python -m src.train --max-samples 5000 --epochs 5 --batch-size 32 --image-size 224
```

Useful flags:

- `--max-samples` : limit dataset size for quick experiments (set `-1` for all)
- `--exclude-issues blurry dark` : skip images flagged as blurry/dark
- `--val-split 0.2` : validation ratio
- `--arch resnet18` : or `resnet50`
- `--lr 3e-4` : learning rate
- `--num-workers 4` : DataLoader workers

Outputs:

- `artifacts/model_best.pt` – best weights
- `artifacts/labels.json` – class → index mapping
- `artifacts/confusion_matrix.png` – saved after eval

---

## 🚀 5) Run the Streamlit app (frontend)

```bash
streamlit run app.py
```

Features:

- **Predict**: upload an image or paste an image URL → top‑5 action predictions
- **Explore**: browse dataset samples, filter by label, see image issues & object labels
- **Stats**: label distribution chart (computed on the fly)

> If `artifacts/model_best.pt` is missing, the app will still run (explorer works)
> but prediction tab will show a friendly notice.

---

## 🧪 6) Quick sanity check (no training)

If you want to try the app first, run it without training. Use the **Explore** tab to view dataset samples.
Then come back and train a small model (e.g., 5 epochs, 5k samples) and test **Predict**.

---

## 🧯 7) Troubleshooting

- **Torch install errors**: Reinstall using the correct index‑url for your CUDA/CPU.
- **SSL/Network timeouts** while downloading images: rerun training; the cache resumes.
- **Out of VRAM**: reduce `--batch-size` and/or image size, or train on CPU.
- **Windows PowerShell**: If `activate` is blocked, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once.

---

## 📜 License

MIT, do whatever you want. Attribution appreciated.
