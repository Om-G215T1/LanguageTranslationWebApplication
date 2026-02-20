from pydoc import text

from flask import Flask, render_template, request
from transformers import MarianMTModel, MarianTokenizer  # Translation model \& tokenizer

# Model name from Hugging Face by Helsinki-NLP

model_name = "Helsinki-NLP/opus-mt-en-hi"

# Load tokenizer (converts text → tokens/numbers)

tokenizer = MarianTokenizer.from_pretrained(model_name)

# Load pre-trained translation model

model = MarianMTModel.from_pretrained(model_name)

def output(message):
 # Convert input text into model-readable tokens (PyTorch format)
  inputs = tokenizer(message, return_tensors="pt", padding=True)
# Generate translated tokens using the model
  translated = model.generate(**inputs)
# Convert translated tokens back to readable text
  result = tokenizer.decode(translated[0], skip_special_tokens=True)
  # Return final translated text
  return result

app=Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    translated_msg=""
    if request.method == 'POST':
        data= request.form['message']
        translated_msg= output(data)
    return render_template('index.html', translated_msg=translated_msg)

if __name__=='__main__':
    app.run(debug=True)
