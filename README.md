# 🫀 PPG Signal Classification & XAI Dashboard

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-MIMIC--IV%20v2.2-sky)](https://physionet.org/content/mimiciv/)
[![Model](https://img.shields.io/badge/Classifier-Random%20Forest-purple)](train_rf.py)
[![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.946-emerald)](train_rf.py)
[![Deployed on Vercel](https://img.shields.io/badge/Deploy-Vercel-black?logo=vercel)](vercel.json)

A serene, minimal, and seraphic frontend interface for medical AI photoplethysmography (PPG) signal classification using the **MIMIC-IV** dataset, enhanced with **Explainable AI (XAI)** interpretability (SHAP, feature attributions, and what-if sensitivity analysis).

---

## 🌟 Visual & Design Principles

- **Seraphic Medical Aesthetic**: Built with soft off-whites, sky blue (`#0284c7`), pastel lavender (`#9333ea`), mint emerald (`#059669`), and calm ambient glow effects.
- **Glassmorphism & Micro-Animations**: Backdrop blur (`backdrop-filter: blur(16px)`), ultra-smooth rounded cards (`rounded-2xl`), ambient pulse wave canvas, and subtle hover elevation.
- **Dual Serene Mode**: Toggle effortlessly between **Serene Light Mode** and **Deep Navy Dark Mode**.

---

## 🔥 Key Features

1. **Preset MIMIC-IV Clinical Records & CSV Upload**:
   - 4 pre-loaded MIMIC-IV patient cases (*Stage 2 Hypertension*, *Normotensive Baseline*, *Prehypertension Borderline*, *Stage 1 Hypertension*).
   - Drag-and-drop file uploader for custom optical PPG waveform CSV datasets.

2. **3-Stage Waveform Processing Pipeline**:
   - **Stage 1 (Raw Signal)**: Optical signal with respiratory baseline wander and motion artifacts.
   - **Stage 2 (Filtered Signal)**: 4th-order 0.5–8.0 Hz Butterworth bandpass filtered signal.
   - **Stage 3 (Delineated Beat)**: Extracted single-beat fiducial landmarks for Systolic Peak ($P_1$), Dicrotic Notch ($V$), and Diastolic Peak ($P_2$).
   - Live stream play/pause playback animation.

3. **Explainable AI (XAI) Panel**:
   - **Automated Human-Readable Summary**: Translates feature anomalies into clinical physiological explanations.
   - **Global Feature Importance**: Bar chart displaying Gini importance weights (Augmentation Index, Stiffness Index, Pulse Transit Time).
   - **Local SHAP Contribution Waterfall Plot**: Patient-specific waterfall plot displaying feature impact relative to baseline $E[f(x)] = 0.32$.
   - **Temporal Waveform Attribution Map**: Visualizes signal upstroke, dicrotic notch, and diastolic decay SHAP impacts.
   - **Interactive "What-If" Sensitivity Simulator**: Sliders for Augmentation Index, Stiffness Index, and Pulse Transit Time that recalculate predicted hypertension risk live!

4. **Model Performance & Nested CV**:
   - **ROC-AUC Curve** chart ($AUC = 0.946$).
   - **Pooled Confusion Matrix** heatmap and 5-Fold Stratified Group Cross-Validation metrics summary matching `train_rf.py`.

---

## 📁 Repository Structure

```
ppg_vt/
├── index.html        # Main HTML5 entry point (React 18 + Tailwind + Chart.js)
├── app.js            # Complete React application logic & XAI visualizers
├── styles.css        # Serene theme CSS variables, glassmorphism, & dark mode
├── server.py         # Python local server launcher (http://localhost:8000)
├── train_rf.py       # Python nested Stratified Group K-Fold Random Forest trainer
├── vercel.json       # Vercel static deployment configuration
├── package.json      # Node package manifest
├── requirements.txt  # Python ML dependencies
├── LICENSE           # MIT open source license
└── README.md         # Documentation & deployment guide
```

---

## 💻 Local Quickstart

### Option 1: Using Python HTTP Server (Zero Install)
Simply run the included server launcher:

```bash
python server.py
```
Open **`http://localhost:8000`** in your browser.

---

### Option 2: Train Random Forest Model (Python ML Pipeline)
To train the nested cross-validation Random Forest classifier locally:

```bash
pip install -r requirements.txt
python train_rf.py
```

---

## 🚀 How to Deploy on Vercel

Deploying this application to **Vercel** is instant and free!

### Method A: Via GitHub Integration (Recommended)

1. **Initialize Git & Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: PPG Signal Classification & XAI Dashboard"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/ppg-signal-classification-xai.git
   git push -u origin main
   ```

2. **Deploy on Vercel**:
   - Go to [Vercel Dashboard](https://vercel.com/dashboard) and click **"Add New" -> "Project"**.
   - Select your GitHub repository `ppg-signal-classification-xai`.
   - Vercel will automatically detect `vercel.json` and set the Framework Preset to **Other** / **Static Site**.
   - Click **Deploy**!
   - Your site will be live at `https://ppg-signal-classification-xai.vercel.app`.

---

### Method B: Via Vercel CLI

1. Install Vercel CLI:
   ```bash
   npm install -g vercel
   ```

2. Run `vercel` in your project folder:
   ```bash
   vercel
   ```
   Follow the prompts and choose `y` to deploy!

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
