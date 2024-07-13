import os
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np

import numpy as np2
import torch
import pandas as pd

import os

# Get the directory of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the paths to the CSV files
disease_info_path = os.path.join(base_dir, 'disease_info.csv')
supplement_info_path = os.path.join(base_dir, 'supplement_info.csv')

# Read the CSV files
disease_info = pd.read_csv(disease_info_path, encoding='cp1252')
supplement_info = pd.read_csv(supplement_info_path, encoding='cp1252')



# disease_info = pd.read_csv('C:\\Users\\Suyash\\OneDrive\\Desktop\\Plant leaf detection\\Flask Deployed App\\disease_info.csv', encoding='cp1252')
# supplement_info = pd.read_csv('C:\\Users\\Suyash\\OneDrive\\Desktop\\Plant leaf detection\\Flask Deployed App\\supplement_info.csv', encoding='cp1252')

# disease_info = pd.read_csv('C:\Users\Suyash\OneDrive\Desktop\Plant leaf detection\Flask Deployed App\disease_info.csv' , encoding='cp1252')
# supplement_info = pd.read_csv('C:\Users\Suyash\OneDrive\Desktop\Plant leaf detection\Flask Deployed App\supplement_info.csv',encoding='cp1252')



# Get the directory of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the model file
model_path = os.path.join(base_dir, 'plant_disease_model_1_latest.pt')
model = CNN.CNN(39) 
# Load the model weights
model.load_state_dict(torch.load(model_path))



# model = CNN.CNN(39)    
# model.load_state_dict(torch.load('C:\\Users\\Suyash\\OneDrive\\Desktop\\Plant leaf detection\\Flask Deployed App\\plant_disease_model_1_latest.pt'))

# model.load_state_dict(torch.load("C:\Users\Suyash\OneDrive\Desktop\Plant leaf detection\Flask Deployed App\plant_disease_model_1_latest.pt"))
model.eval()
 
def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index


from flask import Flask, render_template

app = Flask(__name__,template_folder='templates')

@app.route('/')
def home(): 
    return render_template('home.html')

@app.route('/index')
def aiengine(): 
    return render_template('index.html') 


@app.route('/contact') 
def contact(): 
    return render_template('ContactUs.html')


@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']

        # Get the directory of the current script
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Assuming 'static/uploads' is the directory where you want to save uploaded files
        uploads_dir = os.path.join(base_dir, 'static', 'uploads')

        # Assuming 'image' is the uploaded file object
        filename = image.filename
        file_path = os.path.join(uploads_dir, filename)



        # filename = image.filename
        # file_path = os.path.join('C:\\Users\\Suyash\\OneDrive\\Desktop\\Plant leaf detection\\Flask Deployed App\\static\\uploads', filename)
 

 
        # file_path = os.path.join('"C:\Users\Suyash\OneDrive\Desktop\Plant leaf detection\Flask Deployed App\static\uploads"', filename)
        image.save(file_path)
        print(file_path)
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description =disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html' , title = title , desc = description , prevent = prevent , 
                               image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)
