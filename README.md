# Marathi-Word-Sense-Disambiguation

This project implements a transformer-based Word Sense Disambiguation (WSD) system for the Marathi language. It leverages IndicBERT for sense prediction and integrates Google Gemini API for context-aware English translations. A Gradio-based UI allows seamless interaction with the model, making the system accessible to users.

🚀 Features
-  Sense disambiguation of ambiguous Marathi words using fine-tuned IndicBERT
-  Custom sense-labeled Marathi-English dataset in TSV format
-  Modular pipeline: Preprocessing → Tokenization → Classification → Translation
-  Context-based English translation using Gemini API
-  Fine-tuned on Google Colab with GPU support
-  Interactive and user-friendly interface using Gradio
-  Confidence-based prediction with fallback logic
-  Feedback system for model improvement

📂 Project Structure
├── Data/
  └── Ambiguous Dataset.tsv
  
├── Model/
  ├── finetuned/ 
    └── config.json
    └── model.safetensors #Due to GitHub's limitation on file upload size, we cannot upload files completely. 
  ├── tokenizer/ 
    └── added_tokens.json
    └── special_tokens_map.json
    └── spiece.model
    └── tokenizer_config.json
    └── tokenizer.json
├── be.py


📊 Model Details
Model: IndicBERT (fine-tuned)
Architecture: Transformer-based BERT model for Indian languages
Training Data: 60 sense labels, 300 sentences per label (~18,000 samples)
Training Environment: Google Colab with GPU
Evaluation: Accuracy, Precision, Recall, F1-Score (macro averaged)

⚠️ Note: Due to size constraints, the full training tensors and model checkpoints are not uploaded. You can regenerate them locally by fine-tuning the model according to your requirements.

