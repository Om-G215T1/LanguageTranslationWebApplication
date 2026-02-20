from flask import Flask, render_template, request, jsonify
from transformers import MarianMTModel, MarianTokenizer
import torch

app = Flask(__name__)

# ── SUPPORTED LANGUAGES ───────────────────────────────────────────────────────
# Each entry: "Language Name": "Helsinki-NLP model suffix"
LANGUAGES = {
    "Hindi":        "hi",
    "French":       "fr",
    "German":       "de",
    "Spanish":      "es",
    "Italian":      "it",
    "Portuguese":   "pt",
    "Dutch":        "nl",
    "Russian":      "ru",
    "Ukrainian":    "uk",
    "Polish":       "pl",
    "Czech":        "cs",
    "Slovak":       "sk",
    "Romanian":     "ro",
    "Bulgarian":    "bg",
    "Croatian":     "hr",
    "Serbian":      "sr",
    "Slovenian":    "sl",
    "Lithuanian":   "lt",
    "Latvian":      "lv",
    "Estonian":     "et",
    "Finnish":      "fi",
    "Swedish":      "sv",
    "Norwegian":    "no",
    "Danish":       "da",
    "Icelandic":    "is",
    "Greek":        "el",
    "Turkish":      "tr",
    "Arabic":       "ar",
    "Hebrew":       "he",
    "Persian":      "fa",
    "Urdu":         "ur",
    "Bengali":      "bn",
    "Gujarati":     "gu",
    "Marathi":      "mr",
    "Punjabi":      "pa",
    "Tamil":        "ta",
    "Telugu":       "te",
    "Kannada":      "kn",
    "Malayalam":    "ml",
    "Sinhala":      "si",
    "Nepali":       "ne",
    "Thai":         "th",
    "Vietnamese":   "vi",
    "Indonesian":   "id",
    "Malay":        "ms",
    "Tagalog":      "tl",
    "Swahili":      "sw",
    "Yoruba":       "yo",
    "Zulu":         "zu",
    "Afrikaans":    "af",
    "Albanian":     "sq",
    "Hungarian":    "hu",
    "Catalan":      "ca",
    "Welsh":        "cy",
    "Irish":        "ga",
    "Macedonian":   "mk",
    "Azerbaijani":  "az",
    "Georgian":     "ka",
    "Armenian":     "hy",
    "Mongolian":    "mn",
    "Belarusian":   "be",
}

# ── MODEL CACHE ───────────────────────────────────────────────────────────────
# Models are loaded on first use and cached to avoid reloading
_model_cache = {}

def get_model(target_lang_code):
    """Load and cache tokenizer + model for a given target language."""
    if target_lang_code in _model_cache:
        return _model_cache[target_lang_code]

    model_name = f"Helsinki-NLP/opus-mt-en-{target_lang_code}"
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        model.eval()  # Set to inference mode
        _model_cache[target_lang_code] = (tokenizer, model)
        return tokenizer, model
    except Exception as e:
        raise ValueError(f"Model not available for language code '{target_lang_code}': {e}")


def translate(text, target_lang_code):
    """Translate English text to the target language."""
    tokenizer, model = get_model(target_lang_code)

    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    # Generate translation with beam search for better accuracy
    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            num_beams=4,           # Beam search for better quality
            length_penalty=0.6,    # Slightly prefer shorter outputs
            early_stopping=True,
            max_length=512
        )

    # Decode result
    result = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return result


# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET', 'POST'])
def index1():
    translated_msg = ""
    selected_lang = "Hindi"        # Default language
    error_msg = ""

    if request.method == 'POST':
        data = request.form.get('message', '').strip()
        selected_lang = request.form.get('target_language', 'Hindi')

        if data:
            lang_code = LANGUAGES.get(selected_lang)
            if lang_code:
                try:
                    translated_msg = translate(data, lang_code)
                except ValueError as e:
                    error_msg = f"Model unavailable for {selected_lang}. Try another language."
                except Exception as e:
                    error_msg = "Translation failed. Please try again."
            else:
                error_msg = "Invalid language selected."

    return render_template(
        'index1.html',
        translated_msg=translated_msg,
        selected_lang=selected_lang,
        languages=list(LANGUAGES.keys()),
        error_msg=error_msg
    )


if __name__ == '__main__':
    app.run(debug=True)