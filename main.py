"""Main entrypoint of the application."""
from flask import Flask

app = Flask(__name__)

@app.route("/")
def main():
    """Home page of the application."""
    return "Hello World!"

if __name__ == "__main__":
    app.run()
