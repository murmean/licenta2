import os

import gensim
import pandas as pd
import spacy
import yake
from flask import Flask, request, jsonify, render_template
from gensim import corpora
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

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

    def train_model(self):
        """
        Trains the LDA model for topic modeling.
        """
        if not self.processed_docs:
            return {"error": "No data available for training."}

        dictionary = corpora.Dictionary(self.processed_docs)
        corpus = [dictionary.doc2bow(doc) for doc in self.processed_docs]
        self.model = gensim.models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)
        self.model_trained = True
        return {"message": "Model training completed."}

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

    def extract_keywords_tfidf(self, top_n=10):
        """
        Extracts keywords using TF-IDF for multiple documents.
        """
        if not self.processed_docs:
            return {"error": "No processed documents available!"}

        vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        tfidf_matrix = vectorizer.fit_transform([" ".join(doc) for doc in self.processed_docs])
        feature_names = vectorizer.get_feature_names_out()

        keywords = []
        for doc_vector in tfidf_matrix:
            scores = zip(feature_names, doc_vector.toarray().flatten())
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            keywords.append([word for word, score in sorted_scores[:top_n]])

        return {"keywords": keywords}

    def extract_keywords_yake(self, text, top_n=10):
        """
        Extracts keywords using YAKE for a single document.
        """
        kw_extractor = yake.KeywordExtractor(n=1, dedupLim=0.9, top=top_n)
        keywords = kw_extractor.extract_keywords(text)
        return {"keywords": [word for word, score in keywords]}

    import string

    def topic_modeling(self, num_topics=5, num_words=5):
        """
        Extracts distinct topics from LDA without redundant or meaningless words.
        """
        if not self.model_trained:
            return {"error": "Train the model first!"}

        # Extract top words from LDA topics
        raw_topics = self.model.print_topics(num_topics=num_topics, num_words=num_words)

        unique_topics = set()  # Store unique topics
        lda_topics = []

        stopwords = set(nlp.Defaults.stop_words)  # Get spaCy stopwords

        for topic in raw_topics:
            topic_words = topic[1]  # Extract topic words
            words = [word.split('*')[1].replace('"', '').strip() for word in topic_words.split('+')]

            # ✅ Remove stopwords & short words (length < 3)
            filtered_words = [word for word in words if word not in stopwords and len(word) > 2 and word.isalpha()]

            if not filtered_words:
                continue  # Skip empty topics

            # Convert words into a sorted, comma-separated string
            topic_str = ", ".join(sorted(set(filtered_words)))

            if topic_str not in unique_topics:  # Avoid repeating topics
                unique_topics.add(topic_str)
                lda_topics.append(topic_str)

        # Backup: Extract Named Entities if LDA fails
        entity_topics = set()
        for doc in self.documents:
            nlp_doc = nlp(doc)
            for ent in nlp_doc.ents:
                entity_topics.add(ent.text)
            for chunk in nlp_doc.noun_chunks:
                entity_topics.add(chunk.text)

        filtered_entities = [topic for topic in entity_topics if len(topic.split()) > 1]

        # If LDA has no topics, use Named Entities
        return {"topics": lda_topics[:num_topics] if lda_topics else filtered_entities[:num_topics]}



    def summarize_text(self, text, num_sentences=3):
        """
        Generates a summary of the text using TextRank.
        """
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, num_sentences)

        return {"summary": " ".join(str(sentence) for sentence in summary)}


miner = TextMiner()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.form.get("text", "").strip()
    file = request.files.get("file")

    if file:
        filename = file.filename
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

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

    # ✅ Train LDA Model Before Extracting Topics
    miner.train_model()

    sentiment = miner.sentiment_analysis()
    topics = miner.topic_modeling()

    if isinstance(text, str) and len(text.split()) < 100:
        keywords = miner.extract_keywords_yake(text, top_n=5)
    else:
        keywords = miner.extract_keywords_tfidf(top_n=5)

    return jsonify({
        "sentiment": sentiment.get("sentiments", [0])[0],
        "topics": topics.get("topics", []),  # ✅ Now using LDA-based topics
        "processed_text": " ".join(miner.processed_docs[0]),
        "keywords": keywords.get("keywords", [])  # Still using TF-IDF/YAKE for keywords
    })


@app.route("/summarize", methods=["POST"])
def summarize():
    text = request.form.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided for summarization."})

    summary = miner.summarize_text(text, num_sentences=3)

    return jsonify({
        "summary": summary.get("summary", "")
    })


if __name__ == "__main__":
    app.run(debug=True)
