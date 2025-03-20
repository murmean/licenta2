import os

import gensim
import pandas as pd
import spacy
from flask import Flask, request, jsonify, render_template
from gensim import corpora
from textblob import TextBlob

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__, template_folder="templates", static_folder="static")
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


class TextMiner:
    def __init__(self):
        self.documents = []
        self.processed_docs = []
        self.model_trained = False
        self.model = None

    def load_data(self, text):
        self.documents = [text] if isinstance(text, str) else text
        return {"message": f"Loaded {len(self.documents)} documents."}

    def preprocess(self, lowercase, remove_punct, remove_stopwords, lemmatize):
        if not self.documents:
            return {"error": "No documents available. Load data first!"}

        self.processed_docs = []
        for doc in self.documents:
            tokens = [token for token in nlp(doc)]

            if lowercase:
                tokens = [token.text.lower() for token in tokens]
            else:
                tokens = [token.text for token in tokens]

            if remove_punct:
                tokens = [token for token in tokens if token.isalnum()]

            if remove_stopwords:
                tokens = [token for token in tokens if not nlp.vocab[token].is_stop]

            if lemmatize:
                tokens = [nlp(token)[0].lemma_ for token in tokens]

            self.processed_docs.append(tokens)
        return {"message": "Text preprocessing completed."}

    def sentiment_analysis(self):
        if not self.processed_docs:
            return {"error": "No processed documents. Run preprocessing first!"}
        sentiments = [TextBlob(" ".join(doc)).sentiment.polarity for doc in self.processed_docs]
        return {"sentiments": sentiments}

    def generate_word_cloud(self):
        # words = [word for doc in self.processed_docs for word in doc]
        # if not words:
        #     return {"error": "No words available for word cloud."}

        # wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(words))
        # wordcloud_path = os.path.join("static", "wordcloud.png")
        # wordcloud.to_file(wordcloud_path)
        # return {"wordcloud": wordcloud_path}

        return {"wordcloud": None}  # Return None instead of generating the word cloud

    def train_model(self):
        if not self.processed_docs:
            return {"error": "No data available for training."}

        dictionary = corpora.Dictionary(self.processed_docs)
        corpus = [dictionary.doc2bow(doc) for doc in self.processed_docs]
        self.model = gensim.models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)
        self.model_trained = True
        return {"message": "Model training completed."}

    def topic_modeling(self):
        if not self.model_trained:
            return {"error": "Train the model first!"}

        # Extract top topics
        topics = self.model.print_topics(num_words=5)

        # Extract Named Entities and noun chunks for better topic labels
        entity_topics = set()
        for doc in self.documents:
            nlp_doc = nlp(doc)
            for ent in nlp_doc.ents:  # Named Entities
                entity_topics.add(ent.text)
            for chunk in nlp_doc.noun_chunks:  # Noun phrases
                entity_topics.add(chunk.text)

        # Keep only meaningful topics
        filtered_topics = [topic for topic in entity_topics if len(topic.split()) > 1]

        return {"topics": filtered_topics[:5]}  # Return top 5 topics


miner = TextMiner()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.form.get("text", "").strip()
    file = request.files.get("file")

    # Handle file upload
    if file:
        filename = file.filename
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Read text from file
        if filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
        elif filename.endswith(".csv"):
            df = pd.read_csv(filepath)
            if 'text' in df.columns:
                text = " ".join(df['text'].dropna().tolist())
            else:
                return jsonify({"error": "CSV must contain a 'text' column"})

    if not text:
        return jsonify({"error": "No text provided"})

    lowercase = request.form.get("lowercase", "false") == "true"
    remove_punct = request.form.get("remove_punct", "false") == "true"
    remove_stopwords = request.form.get("remove_stopwords", "false") == "true"
    lemmatize = request.form.get("lemmatize", "false") == "true"

    miner.load_data(text)
    miner.preprocess(lowercase, remove_punct, remove_stopwords, lemmatize)
    sentiment = miner.sentiment_analysis()
    miner.train_model()
    topics = miner.topic_modeling()

    return jsonify({
        "sentiment": sentiment.get("sentiments", [0])[0],
        "topics": topics.get("topics", []),
        "processed_text": " ".join(miner.processed_docs[0]),  # Processed text
        "keywords": list(topics.get("topics", []))  # Keywords are extracted topics
    })


if __name__ == "__main__":
    app.run(debug=True)
