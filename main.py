"""Main entrypoint of the application."""
import os
import time
from dotenv import load_dotenv

from flask import Flask, render_template, request
from pymongo import MongoClient
from text import send_twilio_message

import tensorflow as tf
from PIL import Image
import torch
import torch.nn as nn           # for creating  neural networks
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader


#tensormodel = tf.keras.models.load_model('./Model_Files/disease-classification_1')

# Check its architecture

class ImageClassificationBase(nn.Module):

    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                   # Generate prediction
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)          # Calculate accuracy
        return {"val_loss": loss.detach(), "val_accuracy": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()       # Combine loss
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        # Combine accuracies
        return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_accuracy']))


class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)  # out_dim : 128 x 64 x 64
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))

        self.conv3 = ConvBlock(128, 256, pool=True)  # out_dim : 256 x 16 x 16
        self.conv4 = ConvBlock(256, 512, pool=True)  # out_dim : 512 x 4 x 44
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Linear(512, num_diseases))

    def forward(self, xb):  # xb is the loaded batch
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


torchmodel = torch.load(
    './Model_Files/disease-classification_2.pth', map_location=torch.device('cpu'))


def diseasemodel1():
    return 1

# def diseasemodel2(model, data):
#     convert_tensor = transforms.ToTensor()
#     model.eval()

#     output = model(convert_tensor(data))
#     prediction = torch.argmax(output)
#     return prediction


def diseasemodel2(model, img):
    """Converts image to array and return the predicted class
        with highest probability"""
    # Convert to a batch of 1
    # Get predictions from model
    img = img.resize((256, 256))
    convert_tensor = transforms.ToTensor()
    img = convert_tensor(img)
    yb = model(torch.unsqueeze(img, 0))
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    # Retrieve the class label

    return diseaselist2[preds[0].item()]


diseaselist2 = ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy", "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy", "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
                "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"]

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'


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

        image = request.files["image"]
        filename = str(int(time.time())) + ".jpg"
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(path)

        img = Image.open(path)

        width, height = img.size

        if width != height:
            # Determine the size of the square
            size = min(width, height)

            # Determine the coordinates of the top-left corner of the square
            left = (width - size) / 2
            top = (height - size) / 2
            right = (width + size) / 2
            bottom = (height + size) / 2

            # Crop the image to the square
            img = img.crop((left, top, right, bottom))

        # Resize the image to 256 by 256 pixels
        img = img.resize((256, 256))

        result = diseasemodel2(torchmodel, img)

        report = {
            "phoneno": phoneno,
            "latitude": latitude,
            "longitude": longitude,
            "disease": result
        }

        reports.insert_one(report)

        send_twilio_message(
            f"Your plant has been scanned! Thank you for your input. Your image was identified with: {result}", phoneno.replace(" ", ""))

        return render_template("scan.html")


@app.route("/map")
def display_map():
    """Map of plants."""
    all_reports = {"data": list(reports.find({}, {"_id": 0}))}

    print(all_reports)

    return render_template("map.html", reports=all_reports)


if __name__ == "__main__":
    app.run()
