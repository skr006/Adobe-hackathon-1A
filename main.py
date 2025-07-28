import fitz  # PyMuPDF
import spacy
import re
import os
import json
from collections import Counter
from main2 import main2  
USE_NER = True

if USE_NER:
    nlp = spacy.load("en_core_web_md")

# Relaxed heading validation
def is_valid_heading(line_text, spans):
    is_bold = any('bold' in span.get('font', '').lower() for span in spans)
    return (
        is_bold and
        3 <= len(line_text) and
        1 <= len(line_text.split()) <= 10 and
        re.search(r'[A-Za-z]', line_text)
    )

# Main PDF processing
def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    heading_candidates = []

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")['blocks']

        for b in blocks:
            if 'lines' not in b:
                continue

            for l in b['lines']:
                spans = l['spans']
                if not spans:
                    continue

                line_text = " ".join(span['text'] for span in spans).strip()
                font_size = spans[0]['size']
                
                if is_valid_heading(line_text, spans):
                    heading_candidates.append((font_size, line_text, page_num, l['bbox'][1]))

    if not heading_candidates:
        return {"title": "", "outline": []}

    # Determine title from largest font
    sorted_fonts = sorted(set(f[0] for f in heading_candidates), reverse=True)
    title_font_size = sorted_fonts[0]
    
    title = ""
    other_headings = []

    for font_size, text, page_num, y0 in heading_candidates:
        if font_size == title_font_size and title == "":
            title = text
        else:
            other_headings.append((font_size, text, page_num, y0))

    # Map remaining fonts to H1, H2, H3
    remaining_fonts = sorted(set(f[0] for f in other_headings), reverse=True)
    font_to_level = {f: lvl for f, lvl in zip(remaining_fonts, ['H1', 'H2', 'H3'])}

    headings = []
    for font_size, text, page_num, y0 in other_headings:
        level = font_to_level.get(font_size)
        if level:
            if USE_NER:
                ner_doc = nlp(text)
                entities = [(ent.text, ent.label_) for ent in ner_doc.ents]
            else:
                entities = []

            headings.append({
                'level': level,
                'text': text.strip(),
                'page': page_num
            })

    # Sort by page then position (optional)
    headings = sorted(headings, key=lambda x: (x['page']))

    return {
        "title": title,
        "outline": headings
    }

# Main loop
def main():
    input_dir = "/app/input"
    output_dir = "/app/output"
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            result = process_pdf(pdf_path)

            output_path = os.path.join(output_dir, filename.replace(".pdf", ".json"))
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

# Run
if __name__ == "__main__":
    main2()
