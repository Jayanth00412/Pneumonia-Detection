import os
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19

app = Flask(__name__)

# Load model with weights (updated approach)
try:
    # Option 1: Try loading the entire model if it was saved that way
    model_03 = load_model('vgg_unfrozen.h5')
    print("Loaded full model")
except:
    try:
        # Option 2: If that fails, build model architecture and load weights
        base_model = VGG19(include_top=False, input_shape=(128, 128, 3))
        
        # Freeze the base VGG19 layers if you want transfer learning
        for layer in base_model.layers:
            layer.trainable = False
        
        # Build custom top layers
        x = base_model.output
        x = Flatten()(x)
        x = Dense(4608, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(1152, activation='relu')(x)
        predictions = Dense(2, activation='softmax')(x)
        
        model_03 = Model(inputs=base_model.input, outputs=predictions)
        
        # Load weights with skip_mismatch in case of partial matching
        model_03.load_weights('vgg_unfrozen.h5', skip_mismatch=True)
        print("Loaded weights with possible skip")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Handle error or exit

print('Model loaded. Check http://127.0.0.1:5000/')

def get_className(classNo):
    return "Normal" if classNo == 0 else "Pneumonia"

def getResult(img_path):
    try:
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError("Could not read image")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = Image.fromarray(image)
        image = image.resize((128, 128))
        image = np.array(image) / 255.0  # Normalize to [0,1]
        
        input_img = np.expand_dims(image, axis=0)
        result = model_03.predict(input_img)
        return np.argmax(result, axis=1)[0]  # Return single class number
    except Exception as e:
        print(f"Error in prediction: {e}")
        return -1  # Error case

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded", 400
            
        f = request.files['file']
        if f.filename == '':
            return "No selected file", 400
            
        try:
            # Create uploads directory if it doesn't exist
            os.makedirs('uploads', exist_ok=True)
            
            file_path = os.path.join('uploads', secure_filename(f.filename))
            f.save(file_path)
            
            value = getResult(file_path)
            if value == -1:
                return "Error processing image", 500
                
            return get_className(value)
        except Exception as e:
            return f"Error: {str(e)}", 500
            
    return None

if __name__ == '__main__':
    app.run(debug=True)