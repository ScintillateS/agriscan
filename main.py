"""Main entrypoint of the application."""
import os
from dotenv import load_dotenv

from flask import Flask, render_template
from pymongo import MongoClient

from text import send_twilio_message

app = Flask(__name__)

load_dotenv()

MONGODB_USER = os.environ.get('MONGODB_USER')
MONGODB_PASS = os.environ.get('MONGODB_PASS')

client = MongoClient(f"mongodb+srv://{MONGODB_USER}:{MONGODB_PASS}"
                     "@main.hup8pvq.mongodb.net/?retryWrites=true&w=majority")

db = client.development
users = db.users

@app.route("/")
def home():
    """Home page of the application."""
    return render_template("home.html")

if __name__ == "__main__":
    app.run()
