import re
import gradio as gr
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_REPO = "sayikhushhal/xlm-r-hausa-sentiment"
LABEL_NAMES = ["negative", "neutral", "positive"]
MAX_LENGTH = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO).to(device).eval()


def clean_tweet(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = text.replace("#", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_sentiment(text):
    text = clean_tweet(text or "")
    if not text:
        return {name: 0.0 for name in LABEL_NAMES}
    enc = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        probs = F.softmax(model(**enc).logits, dim=-1)[0].cpu().tolist()
    return {LABEL_NAMES[i]: float(probs[i]) for i in range(len(LABEL_NAMES))}


EXAMPLES = [
    ["Ambato shirin. Cikakken shiri. Naso dukkan daƙiƙa ma shirin. Nayi fatan bai ƙare ba."],
    ["Shekarun1980, masu sufurin ƙwaya, kyawawan baƙaƙen mata ... Ina cikin aljanna"],
    ["wannan shirin kyauta ce mai kyau a cikin nau'in."],
    ["Wannan shirin yayi tsayi da rashin ban sha'awa. Ko sautin ciki yayi ɗumi."],
    ["Hanya mai kyau wanda gaba ɗaya zai lalata maka rana"],
]

DESCRIPTION = (
    "Three-class sentiment classifier (negative / neutral / positive) for Hausa text. "
    "Built on `xlm-roberta-base` fine-tuned on the AfriSenti Hausa benchmark "
    "(Muhammad et al., EMNLP 2023) and evaluated out-of-domain on NollySenti "
    "Hausa movie reviews (Shode et al., ACL 2023). "
    "Click an example below or paste your own Hausa text."
)

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Shigar da jumla a Hausa...", label="Hausa text"),
    outputs=gr.Label(num_top_classes=3, label="Sentiment"),
    title="Low-Resource Hausa Sentiment Classifier",
    description=DESCRIPTION,
    examples=EXAMPLES,
    cache_examples=False,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
