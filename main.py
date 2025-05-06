import re
import pickle
import numpy as np
import gradio as gr
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import nltk

nltk.download("punkt")
nltk.download("stopwords")


# Load the model
def load_model(filename="clustering_model.pkl"):
    with open(filename, "rb") as file:
        model_dict = pickle.load(file)
    return model_dict


model = load_model("clustering_model.pkl")


# Preprocessing functions
def clean_text(text):
    text = re.sub(r'["\']', "", text)
    text = text.strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text.lower()


def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    return [word for word in tokens if word not in stop_words and len(word) > 1]


def get_sentence_vector(tokens, model, vector_size):
    vec = np.zeros(vector_size)
    count = 0
    for word in tokens:
        if word in model.wv:
            vec += model.wv[word]
            count += 1
    return vec / count if count != 0 else vec


# Prediction function with HTML output
def predict(text):
    cleaned_text = clean_text(text)
    tokens = preprocess_text(cleaned_text)
    vector = get_sentence_vector(tokens, model["word2vec_model"], model["vector_size"])
    vector = vector.reshape(1, -1)

    cluster = model["kmeans_model"].predict(vector)[0]
    tag = model["cluster_to_tag"][cluster]

    return f"""
    <div style="padding: 20px; border: 2px solid #4CAF50; border-radius: 10px; background-color: #f0fdf4;">
        <h3 style="color: #256029;">Prediction Result</h3>
        <p><strong>Cluster:</strong> {cluster}</p>
        <p><strong>Tag:</strong> {tag} </p>
    </div>
    """


# Gradio UI
with gr.Blocks(
    css=".gr-button { background-color: #4CAF50; color: white; font-weight: bold; }"
) as demo:
    gr.Markdown(
        """
        # Sentence Intent Classifier
        Type a sentence to classify it into a cluster and get the predicted tag.
        """
    )
    with gr.Row():
        input_text = gr.Textbox(
            label="Enter your sentence",
            placeholder="e.g., Can I track learning progress of my team?",
        )
    submit_btn = gr.Button("Predict")
    output_html = gr.HTML()

    submit_btn.click(fn=predict, inputs=input_text, outputs=output_html)


# Launch
if __name__ == "__main__":
    demo.launch()
