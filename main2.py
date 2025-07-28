import fitz  # PyMuPDF
import os
import json
import re
import joblib
import numpy as np
from sklearn.cluster import KMeans

MODEL_PATH = "/app/model/heading_model.pkl"
FONT_THRESHOLD = 3

def is_valid_line(spans):
    text = " ".join(span['text'] for span in spans).strip()
    if not text:
        return False
    is_bold = any('bold' in span.get('font', '').lower() for span in spans)
    is_large_enough = spans[0]['size'] >= FONT_THRESHOLD
    not_just_caps = not (text.isupper() and not any(char.isdigit() for char in text))
    is_short = len(text.split()) <= 10
    return is_bold and is_large_enough and not_just_caps and is_short and re.search(r"[a-zA-Z]", text)

def extract_features(pdf_path):
    doc = fitz.open(pdf_path)
    features = []
    metadata = []

    for page_num, page in enumerate(doc, start=1):
        page_height = page.rect.height
        blocks = page.get_text("dict")["blocks"]

        for b in blocks:
            if 'lines' not in b:
                continue

            for l in b['lines']:
                spans = l['spans']
                if not spans or not is_valid_line(spans):
                    continue

                text = " ".join(span['text'] for span in spans).strip()
                size = spans[0]['size']
                y0 = l['bbox'][1]
                y_rel = y0 / page_height
                is_bold = 1 if any('bold' in span.get('font', '').lower() for span in spans) else 0
                word_count = len(text.split())

                features.append([size, y_rel, is_bold, word_count])
                metadata.append({
                    "text": text,
                    "page": page_num,
                    "font_size": size,
                    "y": y_rel
                })

    return np.array(features), metadata

def train_model(all_features, n_clusters=4):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    model.fit(all_features)
    joblib.dump(model, MODEL_PATH)
    return model

def assign_labels(cluster_id, label_map):
    return label_map.get(cluster_id, "UNKNOWN")

def get_label_mapping(model):
    # Sort clusters by average font size (assumes cluster centers are in same order)
    centers = model.cluster_centers_
    sorted_indices = np.argsort(-centers[:, 0])  # Larger font = more important
    labels = ['Title', 'H1', 'H2', 'H3']
    return {idx: labels[i] for i, idx in enumerate(sorted_indices)}

def process_pdf(pdf_path, model, label_map):
    features, metadata = extract_features(pdf_path)
    if len(features) == 0:
        return {"title": "", "outline": []}

    preds = model.predict(features)
    result = []
    title_text = ""

    for i, pred in enumerate(preds):
        label = assign_labels(pred, label_map)
        entry = {
            "level": label,
            "text": metadata[i]["text"],
            "page": metadata[i]["page"]
        }
        if label == "Title" and not title_text:
            title_text = metadata[i]["text"]
        elif label != "Title":
            result.append(entry)

    return {
        "title": title_text,
        "outline": result
    }

def main2():
    input_dir = "/app/input"
    output_dir = "/app/output"
    model_dir = "/app/model"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    all_features = []
    file_list = []

    # Gather all features from PDFs
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            features, _ = extract_features(pdf_path)
            if len(features) > 0:
                all_features.extend(features)
                file_list.append(filename)

    if not all_features:
        print("No features found in any PDF.")
        return

    model = train_model(np.array(all_features))
    label_map = get_label_mapping(model)

    for filename in file_list:
        pdf_path = os.path.join(input_dir, filename)
        result = process_pdf(pdf_path, model, label_map)
        output_path = os.path.join(output_dir, filename.replace(".pdf", ".json"))

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Processing complete. Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main2()
