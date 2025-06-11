import torch
import os
import re
import pandas as pd
import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import google.generativeai as genai
import time
from google.api_core.exceptions import ResourceExhausted, GoogleAPIError

#API key
genai.configure(api_key="")

# Paths to model, tokenizer, and dataset
model_path = "Model/finetuned"
tokenizer_path = "Model/tokenizer"
file_path = "Data/Ambiguous Dataset.tsv"

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

SPECIAL_START_TOKEN = "[TGT]"
SPECIAL_END_TOKEN = "[/TGT]"

print("Loading dataset...")
df_org = pd.read_csv(file_path, sep='\t')
df_org = df_org.sample(frac=1.0, random_state=42).drop_duplicates()
df_org['cleaned_text'] = df_org['Marathi'].str.replace(r'[^\u0900-\u097F\s]', '', regex=True).str.strip()

def find_target_index(row):
    sentence = row['cleaned_text']
    target = row['Ambiguous'].strip()
    words = sentence.split()
    try:
        return words.index(target)
    except ValueError:
        for i, word in enumerate(words):
            if target in word:
                return i
        matches = list(re.finditer(re.escape(target), sentence))
        if matches:
            char_pos = matches[0].start()
            words_before = len(sentence[:char_pos].split())
            return words_before
        print(f"Warning: Target word '{target}' not found in sentence: '{sentence}'")
        return -1

df_org['Target_Index'] = df_org.apply(find_target_index, axis=1)
df_org = df_org[df_org['Target_Index'] >= 0]

labels = df_org['Category'].unique().tolist()
labels = [s.strip() for s in labels]
label2id = {label: id for id, label in enumerate(labels)}
id2label = {id: label for id, label in enumerate(labels)}

def clean_text(text):
    return re.sub(r'[^\u0900-\u097F\s]', '', text).strip()

def has_english_script(text):
    """
    Check if text contains English/Latin script characters
    Returns True if English script is found, False otherwise
    """
    if not text:
        return False
    
    # Check for English alphabets (a-z, A-Z)
    english_chars = re.findall(r'[a-zA-Z]', text)
    return len(english_chars) > 0

def has_devanagari_script(text):
    """
    Check if text contains Devanagari script characters (Marathi)
    Returns True if Devanagari script is found, False otherwise
    """
    if not text:
        return False
    
    # Check for Devanagari characters
    devanagari_chars = re.findall(r'[\u0900-\u097F]', text)
    return len(devanagari_chars) > 0

def is_valid_marathi_input(text):
    """
    Validate input text for Marathi processing
    Returns True if text is valid Marathi input, False otherwise
    """
    if not text or not text.strip():
        return False
    
    # Reject if contains English script
    if has_english_script(text):
        return False
    
    # Accept if contains Devanagari script
    if has_devanagari_script(text):
        return True
    
    # If no script detected, reject
    return False

def mark_target_word(sentence, target_index):
    words = sentence.split()
    if 0 <= target_index < len(words):
        words[target_index] = f"{SPECIAL_START_TOKEN} {words[target_index]} {SPECIAL_END_TOKEN}"
        return " ".join(words)
    return sentence

def predict_sense(sentence, target_word):
    # Check script validation - English script not allowed
    if has_english_script(sentence):
        return "Error", "Error: English script detected. Please enter text in Marathi (Devanagari script) only."
    
    if has_english_script(target_word):
        return "Error", "Error: English script detected. Please enter the target word in Marathi (Devanagari script) only."
    
    # Validate that both inputs contain Devanagari script
    if not is_valid_marathi_input(sentence):
        return "Error", "Error: Input sentence should be in Marathi (Devanagari script). Please enter Marathi text only."
    
    if not is_valid_marathi_input(target_word):
        return "Error", "Error: Target word should be in Marathi (Devanagari script). Please enter a Marathi word only."
    
    cleaned_sentence = clean_text(sentence)
    cleaned_target = clean_text(target_word)
    
    # Additional check after cleaning
    if not cleaned_sentence:
        return "Error", "Error: Input sentence appears to be empty or contains no valid Marathi characters."
    
    if not cleaned_target:
        return "Error", "Error: Target word appears to be empty or contains no valid Marathi characters."
    
    row = {'cleaned_text': cleaned_sentence, 'Ambiguous': cleaned_target}
    target_index = find_target_index(row)

    if target_index < 0:
        return "Error", f"Error: Target word '{cleaned_target}' not found in sentence."

    marked_sentence = mark_target_word(cleaned_sentence, target_index)
    inputs = tokenizer(marked_sentence, truncation=True, padding="max_length", return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0]
    probabilities = torch.nn.functional.softmax(logits, dim=0)
    predicted_label = torch.argmax(logits).item()
    predicted_sense = id2label[predicted_label]

    top_indices = torch.argsort(probabilities, descending=True)[:3]
    top_predictions = [
        [id2label[idx.item()], f"{probabilities[idx].item() * 100:.2f}%"]
        for idx in top_indices
    ]

    return predicted_sense, top_predictions

def translate_with_gemini(text, sense, max_retries=3):
    """
    Translate text with Gemini API with error handling and retry logic
    """
    prompt = (
        f"Translate the following Marathi sentence to English, ensuring the translation strictly follows the given context.\n\n"
        f"Marathi Sentence: {text}\n"
        f"Context (Word Sense): {sense}\n\n"
        f"Provide only the English translation, without any explanations or extra details."
    )
    
    # Try different models in order of preference
    models_to_try = ["gemini-1.5-flash", "gemini-1.0-pro"]
    
    for model_name in models_to_try:
        print(f"Trying model: {model_name}")
        
        for attempt in range(max_retries):
            try:
                model = genai.GenerativeModel(model_name)
                
                # Configure generation with safety settings to avoid blocks
                generation_config = {
                    "temperature": 0.1,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 200,
                }
                
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                return response.text.strip()
                
            except ResourceExhausted as e:
                print(f"Quota exceeded for {model_name} on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    # Exponential backoff: 5, 10, 20 seconds
                    retry_delay = 5 * (2 ** attempt)
                    print(f"Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                else:
                    print(f"All retries exhausted for {model_name}")
                    break  # Try next model
                    
            except GoogleAPIError as e:
                print(f"Google API error for {model_name} on attempt {attempt + 1}: {e}")
                if "quota" in str(e).lower() or "limit" in str(e).lower():
                    print(f"Quota-related error for {model_name}, trying next model")
                    break  # Try next model
                elif attempt < max_retries - 1:
                    time.sleep(3)
                else:
                    break
                    
            except Exception as e:
                print(f"Unexpected error for {model_name} on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    break
    
    return "Translation service temporarily unavailable. All API quotas exceeded. Please try again in a few hours."

def wsd_interface(sentence, target_word):
    if not sentence or not target_word:
        return "Please enter both sentence and target word.", []
    
    # Check script validation - English script not allowed
    if has_english_script(sentence):
        return "Error: English script detected. Please enter text in Marathi (Devanagari script) only.", []
    
    if has_english_script(target_word):
        return "Error: English script detected. Please enter the target word in Marathi (Devanagari script) only.", []
    
    # Validate Marathi input before processing
    if not is_valid_marathi_input(sentence):
        return "Error: Input sentence should be in Marathi (Devanagari script) only.", []
    
    if not is_valid_marathi_input(target_word):
        return "Error: Target word should be in Marathi (Devanagari script) only.", []
    
    return predict_sense(sentence, target_word)

# Add a simple rate limiter to prevent rapid API calls
last_api_call_time = 0
min_api_interval = 10  # Minimum 10 seconds between API calls for new accounts

def rate_limited_translate(text, sense):
    global last_api_call_time
    current_time = time.time()
    time_since_last_call = current_time - last_api_call_time
    
    if time_since_last_call < min_api_interval:
        wait_time = min_api_interval - time_since_last_call
        print(f"Rate limiting: waiting {wait_time:.1f} seconds...")
        time.sleep(wait_time)
    
    last_api_call_time = time.time()
    return translate_with_gemini(text, sense)

with gr.Blocks(css=".gradio-container {background-color: #e6f7ff; padding: 20px; border-radius: 10px;}") as demo:
    with gr.Column():
        gr.Markdown("# <span style='color:#000000;'>Marathi Word Sense Disambiguation</span>", elem_id="title", visible=True)
        gr.Markdown("### <span style='color:#000000;'>Enter a Marathi sentence and an ambiguous word to predict its sense. (English text is not allowed)</span>", elem_id="sub_title")

        with gr.Row():
            sentence_input = gr.Textbox(label="Enter Marathi Sentence (मराठी वाक्य)", lines=3, placeholder="मराठी वाक्य येथे टाइप करा...", elem_id="input_box", elem_classes=["input_box_style"])
            target_word_input = gr.Textbox(label="Enter Ambiguous Word (अस्पष्ट शब्द)", placeholder="अस्पष्ट शब्द येथे टाइप करा...", elem_id="input_box")

        with gr.Row():
            prediction_output = gr.Textbox(label="Top Predicted Sense", lines=2, elem_id="predicted_sense_box", elem_classes=["output_box_style"])
            top_predictions_output = gr.Dataframe(label="Top Predictions", headers=["Sense", "Confidence"], elem_id="predictions_table")

        submit_button = gr.Button("Predict", elem_id="submit_button")

        # Feedback Section
        feedback_label = gr.Label("Is the predicted sense correct?")
        with gr.Row():
            thumbs_up_button = gr.Button("👍")
            thumbs_down_button = gr.Button("👎")

        correct_sense_input = gr.Textbox(label="Enter Correct Sense", visible=False)
        translate_button = gr.Button("Translate", visible=False)
        translation_output = gr.Textbox(label="Translated Sentence", lines=2, visible=False)

        # Define interactions
        def on_predict(sentence, target_word):
            # Validate inputs first
            if not sentence or not target_word:
                return "Please enter both sentence and target word.", []
            
            if not sentence.strip() or not target_word.strip():
                return "Please enter both sentence and target word.", []
            
            # Check for English script
            if has_english_script(sentence):
                return "Error: English script detected. Please enter text in Marathi (Devanagari script) only.", []
            
            if has_english_script(target_word):
                return "Error: English script detected. Please enter the target word in Marathi (Devanagari script) only.", []
            
            # Validate Marathi script
            if not is_valid_marathi_input(sentence):
                return "Error: Input sentence should be in Marathi (Devanagari script) only.", []
            
            if not is_valid_marathi_input(target_word):
                return "Error: Target word should be in Marathi (Devanagari script) only.", []
            
            predicted_sense, top_predictions = predict_sense(sentence, target_word)
            return predicted_sense, top_predictions

        def on_thumbs_up(sentence, predicted_sense):
            # Validate inputs before translation
            if not sentence or not predicted_sense:
                return gr.update(visible=False), gr.update(visible=False), gr.update(value="Error: Missing sentence or predicted sense", visible=True)
            
            if has_english_script(sentence):
                return gr.update(visible=False), gr.update(visible=False), gr.update(value="Error: English script not allowed", visible=True)
            
            if not is_valid_marathi_input(sentence):
                return gr.update(visible=False), gr.update(visible=False), gr.update(value="Error: Sentence should be in Marathi (Devanagari script)", visible=True)
            
            if predicted_sense.startswith("Error"):
                return gr.update(visible=False), gr.update(visible=False), gr.update(value="Cannot translate due to prediction error", visible=True)
            
            translation = rate_limited_translate(sentence, predicted_sense)
            return gr.update(visible=False), gr.update(visible=False), gr.update(value=translation, visible=True)

        def on_thumbs_down():
            return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)

        def on_translate(sentence, correct_sense):
            # Validate inputs before translation
            if not sentence or not correct_sense:
                return gr.update(value="Error: Missing sentence or correct sense", visible=True)
            
            if has_english_script(sentence):
                return gr.update(value="Error: English script not allowed", visible=True)
            
            if not is_valid_marathi_input(sentence):
                return gr.update(value="Error: Sentence should be in Marathi (Devanagari script)", visible=True)
            
            if not correct_sense.strip():
                return gr.update(value="Error: Please enter a correct sense", visible=True)
            
            translation = rate_limited_translate(sentence, correct_sense)
            return gr.update(value=translation, visible=True)

        submit_button.click(on_predict, inputs=[sentence_input, target_word_input], outputs=[prediction_output, top_predictions_output])
        thumbs_up_button.click(on_thumbs_up, inputs=[sentence_input, prediction_output], outputs=[correct_sense_input, translate_button, translation_output])
        thumbs_down_button.click(on_thumbs_down, outputs=[correct_sense_input, translate_button, translation_output])
        translate_button.click(on_translate, inputs=[sentence_input, correct_sense_input], outputs=[translation_output])


demo.css = """
#title {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #003366;
    font-size: 40px;
    font-weight: 900;
    text-align: center;
    text-shadow: 1px 1px 2px #b0c4de;
    margin-bottom: 10px;
    background-color: #f0f8ff; /* soft light blue */
    padding: 15px 20px;
    border-radius: 10px;
    display: inline-block;
    margin-left: auto;
    margin-right: auto;
}

#sub_title {
    color: #000000;
    font-size: 20px;
    text-align: center;
    margin-bottom: 30px;
}

#input_box {
    font-size: 18px;
    padding: 12px;
    border-radius: 10px;
    border: 1px solid #B0E0E6;
}
.input_box_style input {
    background-color: #d6ecf7;
}
#predicted_sense_box {
    font-size: 18px;
    color: #3c3c3c;
    background-color: #f0f8ff;
    border-radius: 8px;
    padding: 15px;
    border: 1px solid #B0E0E6;
}
.output_box_style input {
    background-color: #d6ecf7;
}
#submit_button {
    background-color: #1989b9;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    padding: 10px 20px;
    width: 200px;
    border: none;
}
#submit_button:hover {
    background-color: #4682B4;
}
#predictions_table {
    background-color: #ffffff;
    border-radius: 8px;
}
"""

demo.launch()
