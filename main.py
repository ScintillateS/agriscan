"""Main entrypoint of the application."""
import os
from dotenv import load_dotenv

from flask import Flask, render_template, request
from pymongo import MongoClient

from text import send_twilio_message

app = Flask(__name__)

load_dotenv()

MONGODB_USER = os.environ.get('MONGODB_USER')
MONGODB_PASS = os.environ.get('MONGODB_PASS')

client = MongoClient(f"mongodb+srv://{MONGODB_USER}:{MONGODB_PASS}"
                     "@main.hup8pvq.mongodb.net/?retryWrites=true&w=majority")

db = client.development
reports = db.reports

@app.route("/")
def home():
    """Home page of the application."""
    return render_template("home.html")

@app.route("/scan", methods=["GET", "POST"])
def scan():
    """Scan a plant."""
    if request.method == "GET":
        return render_template("scan.html")
 
    elif request.method == "POST":
        phoneno = request.form.get("phoneno")
        latitude = request.form.get("latitude")
        longitude = request.form.get("longitude")
        # disease = request.form.get("disease")

        print(latitude, longitude)

        report = {
            "phoneno": phoneno,
            "latitude": latitude,
            "longitude": longitude,
            # "disease": disease
        }

        reports.insert_one(report)

        send_twilio_message("Your plant has been scanned! Thank you for your input.", phoneno.replace(" ", ""))

        return render_template("scan.html")

@app.route("/map")
def display_map():
    """Map of plants."""
    return render_template("map.html")

if __name__ == "__main__":
    app.run()
