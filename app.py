from flask import Flask, render_template, request
import torch
from transformers import MarianMTModel, MarianTokenizer
import sentencepiece

app = Flask(__name__)
# cv = pickle.load(open("models/cv.pkl"))
# clf = pickle.load(open("models/clf.pkl"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
        input_text = request.form.get('content')
    
        model_name = f'Helsinki-NLP/opus-mt-{"en"}-{"hi"}'
        model = MarianMTModel.from_pretrained(model_name)
        tokenizer = MarianTokenizer.from_pretrained(model_name)

        input_text = input_text.strip()
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        with torch.no_grad():
            output = model.generate(input_ids, max_length=150, num_return_sequences=1)

        translation = tokenizer.decode(output[0], skip_special_tokens=True)

        
    # tokenized_email = cv.transform([email]) # X 
    # prediction = clf.predict(tokenized_email)
    # prediction = 1 if prediction == 1 else -1
        return render_template("index.html", translation=translation, text=input_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)