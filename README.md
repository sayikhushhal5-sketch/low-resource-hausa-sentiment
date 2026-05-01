# Low-Resource Hausa Sentiment Analysis with Transfer Learning

End-to-end deep learning project for three-class sentiment classification
(negative / neutral / positive) on Hausa text. Built on
`xlm-roberta-base` fine-tuned on the AfriSenti Hausa benchmark and
evaluated out-of-domain on NollySenti Hausa movie reviews.

---

## Live links

| Resource | URL |
|---|---|
| Live Gradio demo (Hugging Face Space) | https://huggingface.co/spaces/sayikhushhal/low-resource-hausa-sentiment |
| Fine-tuned model weights (Hugging Face Hub) | https://huggingface.co/sayikhushhal/xlm-r-hausa-sentiment |
| Colab notebook (training + evaluation + deployment) | https://colab.research.google.com/drive/1_rY5UxPA3rui9YJv4Hnpk9uzh3wBFnQ9 |

The Hugging Face Space serves the live demo with five pre-loaded
NollySenti example chips — anyone can try the model in the browser
in under 10 seconds without cloning or installing anything.

---

## Repository layout

```
.
├── README.md                                       — this file
├── notebook/
│   └── SayiKhushhalGadde_A00074661_MS4.ipynb       — full training, evaluation, deployment pipeline
├── gradio-app/
│   ├── app.py                                      — Gradio interface (loads model from HF Hub)
│   ├── requirements.txt                            — runtime deps for the Space
│   └── README.md                                   — Space metadata and one-line summary
├── report/
│   └── SayiKhushhalGadde_A00074661_MS4.pdf         — IEEE-format final report
└── requirements.txt                                — deps for re-running the notebook
```

---

## Reproducing the results — Colab path (recommended)

The notebook was developed and tested on Google Colab with an NVIDIA T4
GPU. The full top-to-bottom run takes approximately **25 minutes** and
trains 9 model variants plus the deployment pipeline.

1. Open the Colab notebook URL above (or upload `notebook/SayiKhushhalGadde_A00074661_MS4.ipynb` to your own Colab).
2. **Runtime → Change runtime type → Hardware accelerator: T4 GPU** (or H100 if available).
3. **Runtime → Run all**.
4. Wait for the notebook to finish. All figures and metrics print inline.
5. The final cell launches a Gradio interface inside the notebook with a public `*.gradio.live` URL — usable for ~72 hours after launch.

Datasets are downloaded automatically from the Hugging Face Hub:

- AfriSenti Hausa: `masakhane/afrisenti`, config `hau`
- NollySenti Hausa: `Davlan/nollysenti`, config `ha`

A random seed of 42 is fixed throughout, but CUDA operations are not
fully deterministic across hardware, so a fresh run on a different GPU
will produce results within ±0.5 F1 of the reported numbers.

---

## Reproducing locally (optional)

The notebook can also be executed locally on a machine with a CUDA GPU
(8 GB+ VRAM recommended) and Python 3.10 or 3.11.

```bash
git clone <this-repo-url>
cd <this-repo>
python -m venv venv
source venv/bin/activate           # on Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebook/SayiKhushhalGadde_A00074661_MS4.ipynb
```

Then **Cell → Run All** in Jupyter. Without a CUDA GPU, training will
fall back to CPU and take several hours instead of 25 minutes.

---

## Running the Gradio app locally

The deployed Space loads the fine-tuned weights from
`sayikhushhal/xlm-r-hausa-sentiment` on the Hugging Face Hub, so you
do not need to retrain to run the demo locally:

```bash
cd gradio-app
pip install -r requirements.txt
python app.py
```

The app prints a local URL (e.g., `http://127.0.0.1:7860`) — open it in
a browser. The model is downloaded on first launch (~1 GB).

---

## Expected results — sanity check

After a successful top-to-bottom notebook run, the final results table
should show:

| Configuration | Test Accuracy | Test Macro F1 |
|---|---:|---:|
| TF-IDF + Logistic Regression (baseline) | 0.7430 | 0.7445 |
| XLM-R MAX (deployed model) | 0.7422 | **0.7430** |
| mBERT MAX | 0.7469 | 0.7469 |

Out-of-domain evaluation on NollySenti Hausa:

| Dataset | Accuracy | Macro F1 |
|---|---:|---:|
| NollySenti Hausa (binary) | 0.6604 | 0.6604 |

The 8.3-point Macro F1 drop reflects the Twitter-to-movie-reviews
domain shift, not a label mismatch — the three-class output is
collapsed to binary by comparing the positive and negative logits.

If your re-run lands within ±0.5 F1 of these numbers, the pipeline
worked end-to-end.

---

## Dependencies

Top-level `requirements.txt` (for the notebook):

```
transformers>=4.38
torch>=2.0
datasets>=2.16
huggingface_hub>=0.20
accelerate>=0.27
sentencepiece>=0.1.99
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
gradio>=4.20
```

`gradio-app/requirements.txt` (for the deployed Space) is intentionally
minimal — just the runtime dependencies for inference.

---

## Acknowledgements

Built on prior work in low-resource African NLP, in particular:

- AfriSenti — Muhammad et al., EMNLP 2023
- NollySenti — Shode et al., ACL 2023
- XLM-RoBERTa — Conneau et al., ACL 2020

Full citations are in the references section of the project report.