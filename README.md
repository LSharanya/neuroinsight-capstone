EEGBASED — Explainable Stress Assistant (SAM 40)

Capstone project: a concept-bottleneck EEG network (CB-EEGNet) that classifies Relax vs Stroop (cognitive stress) from 32-channel EEG, explains its output through six interpretable EEG concepts (TAR, FAA, EI, FMθ, HAI, PAC), and wraps it in a Gradio app with a local LLM (Ollama) for plain-language explanations.

Research prototype. Not a medical device and not a diagnosis.

Pipeline
Load SAM 40 recordings, band-pass 0.5–45 Hz, cut into 4 s windows (50 % overlap)
Filter bank → 5 band experts → gating → 6 concepts → classifier
Subject-wise train/val/test split (tests on unseen people)
RAG knowledge base (Ollama nomic-embed-text + FAISS, TF-IDF fallback)
Gradio UI: stress score, timeline, XAI plots, data-quality check, chat
Setup
1. Clone
bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
2. Install (pick one)
Linux / macOS / Codespaces: bash setup.sh
Windows: install Ollama from https://ollama.com, then run setup.bat

Manual equivalent:

bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
ollama pull llama3.2
ollama pull nomic-embed-text
3. Get the dataset

Download SAM 40 (not included in this repo) and place it so the folder contains sub-folders with "relax" and "stroop" in their names:

data/SAM40/
├── Relax/    ... recordings (.csv, or extension-less files with "_sub_" in the name)
└── Stroop/   ... recordings

Or point to it elsewhere:

bash
export SAM40_ROOT_DIR=/path/to/SAM40        # Windows: set SAM40_ROOT_DIR=C:\path\to\SAM40
4. Run

Make sure Ollama is running (ollama serve, or the desktop app), then:

bash
jupyter notebook project_capstone.ipynb

Run all cells top to bottom (Shift+Enter). The last cell launches the Gradio app. Training writes sam40_cbeegnet_model.pt, concept_stats.npz and test_report.json. The first run caches preprocessed data to sam40_binary_processed_ep4.npz (git-ignored).

Notes
No GPU needed; training uses CUDA automatically if available.
If Ollama is not running, the app falls back to knowledge-base snippets.
The helpline text is India-specific (Tele-MANAS 14416); change it in the notebook for other regions and verify it.
Add a LICENSE file before making the repo public, and cite the SAM 40 dataset paper.
