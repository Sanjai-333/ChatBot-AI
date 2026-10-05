from flask import Flask, render_template, request, jsonify

import pickle
import random
import json
import re
import os

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from openai import OpenAI


app = Flask(__name__)


# =========================================================
# OPENAI
# =========================================================

OPENAI_MODEL = "gpt-6-luna"

openai_client = None

if os.getenv("OPENAI_API_KEY"):
    openai_client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )


# =========================================================
# NLP PREPROCESSING
# =========================================================

def preprocess_text(text):

    text = text.lower()

    replacements = {
        "what's": "what is",
        "whats": "what is",
        "who's": "who is",
        "who're": "who are",
        "how's": "how is",
        "how're": "how are",
        "can't": "cannot",
        "don't": "do not",
        "doesn't": "does not",
        "isn't": "is not",
        "aren't": "are not",
        "i'm": "i am",
        "im": "i am",
        "you're": "you are",
        "youre": "you are",
        "it's": "it is",
        "its": "it is"
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    informal_words = {
        "abt": "about",
        "pls": "please",
        "plz": "please",
        "u": "you",
        "ur": "your",
        "r": "are",
        "wat": "what",
        "wht": "what",
        "hw": "how"
    }

    words = text.split()

    words = [
        informal_words.get(word, word)
        for word in words
    ]

    text = " ".join(words)

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# ENTITY EXTRACTION
# =========================================================

def extract_entities(message):

    entities = {
        "name": None,
        "technology": [],
        "topic": [],
        "day": []
    }

    lower_message = message.lower()

    # NAME

    name_patterns = [
        r"my name is ([a-zA-Z]+)",
        r"i am ([a-zA-Z]+)",
        r"call me ([a-zA-Z]+)"
    ]

    for pattern in name_patterns:

        match = re.search(
            pattern,
            message,
            re.IGNORECASE
        )

        if match:

            entities["name"] = match.group(1)

            break

    # TECHNOLOGIES

    technologies = [
        "Python",
        "Java",
        "C",
        "C++",
        "JavaScript",
        "HTML",
        "CSS",
        "Flask",
        "React",
        "SQL",
        "MongoDB",
        "TensorFlow",
        "PyTorch",
        "Scikit-learn",
        "R",
        "Git",
        "GitHub"
    ]

    for technology in technologies:

        if technology.lower() in lower_message:

            entities["technology"].append(
                technology
            )

    # AI / COMPUTER TOPICS

    topics = {

        "AI": [
            "ai",
            "artificial intelligence"
        ],

        "Machine Learning": [
            "machine learning",
            "ml"
        ],

        "Data Science": [
            "data science"
        ],

        "Deep Learning": [
            "deep learning"
        ],

        "Programming": [
            "programming",
            "programming language"
        ],

        "NLP": [
            "nlp",
            "natural language processing"
        ],

        "Computer Vision": [
            "computer vision"
        ]
    }

    for topic, keywords in topics.items():

        for keyword in keywords:

            if keyword in lower_message:

                if topic not in entities["topic"]:

                    entities["topic"].append(
                        topic
                    )

                break

    # DAYS

    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    for day in days:

        if day.lower() in lower_message:

            entities["day"].append(day)

    return entities


# =========================================================
# LOAD INTENTS
# =========================================================

with open(
    "intents.json",
    "r",
    encoding="utf-8"
) as file:

    intents_data = json.load(file)


# =========================================================
# OPTIONAL LOCAL ML MODEL
# =========================================================

model = None
vectorizer = None
ML_AVAILABLE = False

try:

    with open(
        "chatbot_model.pkl",
        "rb"
    ) as file:

        model_data = pickle.load(file)

    model = model_data["model"]
    vectorizer = model_data["vectorizer"]

    ML_AVAILABLE = True

    print("-----------------------------------")
    print("Local ML model loaded successfully.")
    print("-----------------------------------")

except Exception as e:

    ML_AVAILABLE = False

    print("-----------------------------------")
    print("Local ML model could not be loaded.")
    print("Reason:", str(e))
    print("OpenAI API will be used instead.")
    print("-----------------------------------")


# =========================================================
# SENTIMENT ANALYZER
# =========================================================

sentiment_analyzer = SentimentIntensityAnalyzer()


# =========================================================
# MEMORY
# =========================================================

memory = {

    "name": None,

    "messages": [],

    "last_intent": None
}


# =========================================================
# CONFIDENCE LEVELS
# =========================================================
# These are used internally only.
# They are NOT sent to the webpage.

HIGH_CONFIDENCE = 0.70

MEDIUM_CONFIDENCE = 0.40


# =========================================================
# SENTIMENT
# =========================================================

def detect_sentiment(message):

    scores = sentiment_analyzer.polarity_scores(
        message
    )

    compound = scores["compound"]

    if compound >= 0.05:

        return "Positive"

    elif compound <= -0.05:

        return "Negative"

    else:

        return "Neutral"


# =========================================================
# INTENT RESPONSE
# =========================================================

def get_intent_response(intent_name):

    for intent in intents_data["intents"]:

        if intent["tag"] == intent_name:

            return random.choice(
                intent["responses"]
            )

    return None


# =========================================================
# CONTEXT RESPONSE
# =========================================================

def get_context_response(intent_name):

    context_responses = {

        "python": [
            "Python is used for web development, artificial intelligence, machine learning, data science, automation, and software development.",
            "Python is commonly used in AI, machine learning, data science, web development, automation, and scripting."
        ],

        "machine_learning": [
            "Machine learning is used to make predictions, recognize patterns, recommend products, detect fraud, and automate decisions.",
            "Machine learning is widely used in recommendation systems, image recognition, fraud detection, forecasting, and many AI applications."
        ],

        "ai": [
            "Artificial Intelligence is used in virtual assistants, recommendation systems, robotics, healthcare, finance, autonomous systems, and many other applications.",
            "AI is used to make computers perform tasks that normally require human intelligence, such as learning, reasoning, and decision-making."
        ],

        "data_science": [
            "Data Science is used to analyze data, discover patterns, create predictions, build dashboards, and support better decision-making.",
            "Data Science is commonly used for data analysis, prediction, visualization, business intelligence, and machine learning."
        ],

        "programming": [
            "Programming is used to create websites, mobile apps, desktop software, games, AI systems, and automation tools.",
            "Programming allows us to give instructions to computers and build software applications."
        ],

        "study": [
            "Studying helps you understand concepts, prepare for exams, develop skills, and improve your knowledge.",
            "Good study habits include understanding concepts, practicing regularly, reviewing mistakes, and taking short breaks."
        ]
    }

    if intent_name in context_responses:

        return random.choice(
            context_responses[intent_name]
        )

    return None


# =========================================================
# OPENAI RESPONSE
# =========================================================

def get_openai_response(message):

    if openai_client is None:

        print("OpenAI API key is not loaded.")

        return None

    try:

        response = openai_client.responses.create(

            model=OPENAI_MODEL,

            instructions=(
                "You are SmartChat AI, a helpful and "
                "educational chatbot. Give clear, accurate "
                "and easy-to-understand answers. "
                "Use simple English unless the user asks "
                "for another language. "
                "Keep answers appropriate for a student."
            ),

            input=message
        )

        return response.output_text

    except Exception as e:

        print("-----------------------------------")
        print("OpenAI API Error:", str(e))
        print("-----------------------------------")

        return None


# =========================================================
# CHATBOT RESPONSE
# =========================================================

def get_response(message):

    message = message.strip()

    lower_message = message.lower()

    # ENTITY EXTRACTION

    entities = extract_entities(message)

    print("-----------------------------------")

    print("Message:", message)

    print("Entities:", entities)

    # NAME MEMORY

    if entities["name"]:

        memory["name"] = entities["name"]

    # SAVE MESSAGE

    memory["messages"].append({
        "user": message
    })

    # ASK NAME

    if lower_message in [
        "what is my name",
        "what's my name",
        "do you know my name",
        "remember my name"
    ]:

        if memory["name"]:

            response = (
                f"Your name is "
                f"{memory['name']}."
            )

        else:

            response = (
                "You haven't told me "
                "your name yet."
            )

        memory["messages"][-1]["bot"] = response

        return (
            response,
            entities
        )

    # NAME INTRODUCTION

    if entities["name"]:

        response = (
            f"Nice to meet you, "
            f"{entities['name']}!"
        )

        memory["messages"][-1]["bot"] = response

        return (
            response,
            entities
        )

    # FOLLOW-UP

    follow_up_words = [

        "what is it used for",
        "what is it used to do",
        "what are its uses",
        "what is its use",
        "where is it used",
        "why is it important",
        "tell me more",
        "explain more",
        "more about it",
        "how does it work"
    ]

    is_follow_up = (

        lower_message in follow_up_words

        or lower_message.endswith("used for?")

        or lower_message.endswith("used for")
    )

    # =====================================================
    # LOCAL ML MODEL
    # =====================================================

    if ML_AVAILABLE:

        try:

            cleaned_message = preprocess_text(
                message
            )

            print(
                "Processed:",
                cleaned_message
            )

            message_vector = vectorizer.transform(
                [cleaned_message]
            )

            prediction = model.predict(
                message_vector
            )[0]

            probabilities = model.predict_proba(
                message_vector
            )[0]

            confidence = max(probabilities)

            print(
                "Predicted intent:",
                prediction
            )

            print(
                "Internal confidence:",
                round(
                    confidence * 100,
                    2
                ),
                "%"
            )

            # FOLLOW-UP

            if (
                is_follow_up
                and memory["last_intent"]
            ):

                previous_intent = (
                    memory["last_intent"]
                )

                context_response = (
                    get_context_response(
                        previous_intent
                    )
                )

                if context_response:

                    response = context_response

                    memory["messages"][-1][
                        "bot"
                    ] = response

                    return (
                        response,
                        entities
                    )

            # OPENAI FALLBACK

            if confidence < HIGH_CONFIDENCE:

                print(
                    "Local confidence below 70%."
                )

                print(
                    "Sending question to OpenAI..."
                )

                openai_response = (
                    get_openai_response(
                        message
                    )
                )

                if openai_response:

                    response = openai_response

                    memory["messages"][-1][
                        "bot"
                    ] = response

                    print(
                        "Response source: OpenAI API"
                    )

                    print("-----------------------------------")

                    return (
                        response,
                        entities
                    )

            # SAVE INTENT

            memory["last_intent"] = prediction

            # LOCAL RESPONSE

            response = get_intent_response(
                prediction
            )

            if response:

                memory["messages"][-1][
                    "bot"
                ] = response

                print(
                    "Response source: Local ML"
                )

                print(
                    "Bot:",
                    response
                )

                print("-----------------------------------")

                return (
                    response,
                    entities
                )

        except Exception as e:

            print("-----------------------------------")
            print("Local ML prediction failed.")
            print("Reason:", str(e))
            print("Switching to OpenAI API.")
            print("-----------------------------------")

    # =====================================================
    # OPENAI DIRECT MODE
    # =====================================================

    print(
        "Using OpenAI API..."
    )

    openai_response = get_openai_response(
        message
    )

    if openai_response:

        response = openai_response

        memory["messages"][-1][
            "bot"
        ] = response

        print(
            "Response source: OpenAI API"
        )

        print("-----------------------------------")

        return (
            response,
            entities
        )

    # =====================================================
    # FINAL FALLBACK
    # =====================================================

    response = (
        "Sorry, I couldn't process "
        "your question right now."
    )

    memory["messages"][-1]["bot"] = response

    return (
        response,
        entities
    )


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# CHAT API
# =========================================================

@app.route(
    "/chat",
    methods=["POST"]
)
def chat():

    data = request.get_json()

    if not data:

        return jsonify({

            "response":
            "No message received.",

            "sentiment":
            "Neutral",

            "entities": {}

        })

    user_message = data.get(
        "message",
        ""
    ).strip()

    if user_message == "":

        return jsonify({

            "response":
            "Please type something.",

            "sentiment":
            "Neutral",

            "entities": {}

        })

    response, entities = get_response(
        user_message
    )

    sentiment = detect_sentiment(
        user_message
    )

    # IMPORTANT:
    # Confidence is NOT returned to the webpage.

    return jsonify({

        "response":
        response,

        "sentiment":
        sentiment,

        "entities":
        entities

    })


# =========================================================
# CLEAR MEMORY
# =========================================================

@app.route(
    "/clear",
    methods=["POST"]
)
def clear_memory():

    memory["name"] = None

    memory["messages"] = []

    memory["last_intent"] = None

    return jsonify({
        "status": "cleared"
    })


# =========================================================
# HISTORY
# =========================================================

@app.route(
    "/history",
    methods=["GET"]
)
def get_history():

    return jsonify({
        "history": memory["messages"]
    })


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    print("-----------------------------------")
    print("          SMARTCHAT AI")
    print("-----------------------------------")

    print(
        "AI Chatbot: Starting..."
    )

    print(
        "Local ML Model:",
        "Available"
        if ML_AVAILABLE
        else "Unavailable"
    )

    print(
        "NLP Preprocessing: Enabled"
    )

    print(
        "Sentiment Analysis: VADER"
    )

    print(
        "Conversation Context: Enabled"
    )

    print(
        "Memory: Enabled"
    )

    print(
        "Entity Extraction: Enabled"
    )

    print(
        "OpenAI API:",
        "Enabled"
        if openai_client
        else "Not configured"
    )

    print(
        "OpenAI Model:",
        OPENAI_MODEL
    )

    print("-----------------------------------")

    print(
        "Open in your browser:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print("-----------------------------------")

    app.run(
        debug=True
    )