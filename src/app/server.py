import os
import cv2
import json
from roboflow import Roboflow
from dotenv import load_dotenv
from PIL import Image, ExifTags
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename

load_dotenv()

class_map = {
    0: 'ash_dieback'
}

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploaded_images')
PREDICTION_FOLDER = os.path.join(os.getcwd(), 'static', 'image_predictions')
PREDICTION_DATA = os.path.join(os.getcwd(), 'static', 'prediction_data')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def resize_coords(image, points, new_size=(640,640)):
    y_ = image.shape[0]
    x_ = image.shape[1]

    image = cv2.resize(image, new_size)

    x_scale = new_size[0] / x_
    y_scale = new_size[1] / y_

    xmin = int(points[0] * x_scale)
    ymin = int(points[1] * y_scale)
    xmax = int(points[2] * x_scale)
    ymax = int(points[3] * y_scale)

    return xmin, ymin, xmax, ymax

def yolo_format_to_xyxy(x,y,w,h):
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return int(x1), int(y1), int(x2), int(y2)

def extract_gps_data(image_location):
    img = Image.open(image_location)
    exif = { ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS }
    return exif['GPSInfo']

def convert_DMS_to_DD(gps_data):
    if gps_data[1] == 'S':
        deg, mins, seconds = gps_data[2]
        latitude = float(deg * -1) - float(mins / 60) - float(seconds / 3600)
    else:
        deg, mins, seconds = gps_data[2]
        latitude = float(deg) + float(mins / 60) + float(seconds / 3600)

    if gps_data[3] == 'W':
        deg, mins, seconds = gps_data[4]
        longitude = float(deg * -1) - float(mins / 60) - float(seconds / 3600)
    else:
        deg, mins, seconds = gps_data[4]
        latitude = float(deg) + float(mins / 60) + float(seconds / 3600)

    return latitude, longitude


rf = Roboflow(api_key=os.getenv('ROBOFLOW_API_KEY'))
project = rf.workspace().project("tree-diseases")
model = project.version(4).model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICTION_FOLDER'] = PREDICTION_FOLDER
app.config['PREDICTION_DATA'] = PREDICTION_DATA

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/error', methods=['GET'])
def error():
    return render_template('error.html')

@app.route('/', methods=['GET'])
def home():
    return render_template("home.html")

@app.route('/upload-images', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        files = request.files.getlist('file')
        if len(files) == 0:
            return redirect(url_for('error'))

        for file in files:
            if file.filename == '':
                return redirect(url_for('error'))
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                
        return render_template('upload.html')
    return render_template("upload.html")

@app.route('/current-dataset', methods=['GET'])
def show_dataset():
    images = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('current_dataset.html', images=images)

@app.route('/predictions', methods=['GET'])
def predictions():
    images = os.listdir(app.config['PREDICTION_FOLDER'])
    return render_template('predictions.html', images=images)

@app.route('/predict', methods=['POST'])
def generate_predictions():
    image_area_total = 640 * 640
    image_paths = [os.path.join(app.config['UPLOAD_FOLDER'], image_name) for image_name in os.listdir(app.config['UPLOAD_FOLDER'])]

    for image_path in image_paths:
        prediction_details = {
            'image_name': None,
            'latitude': 0,
            'longitude': 0,
            'found_classifications': [],
            'number_of_bboxes': 0,
            'screen_density': 0,
            'box_details': []
        }
        
        bbox_area_total = 0
        prediction_details['image_name'] = os.path.basename(os.path.abspath(image_path))
        lat, long = convert_DMS_to_DD(extract_gps_data(image_path))
        prediction_details['latitude'] = lat
        prediction_details['longitude'] = long

        image = cv2.imread(image_path)
        resized = cv2.resize(image, (640, 640))

        results = model.predict(image_path, confidence=60, overlap=20).json()

        for idx, details in enumerate(results['predictions']):
            x, y, w, h = details['x'], details['y'], details['width'], details['height']
            x1, y1, x2, y2 = resize_coords(image, yolo_format_to_xyxy(x, y, w, h))
            bbox_area_total += (x2 - x1) * (y2 - y1)
            prediction_details['box_details'].append(
                {
                    'id': idx + 1, 'box': (x1, y1, x2, y2),
                    'class': class_map[details['class_id']],
                    'confidence': f"{details['confidence']:.2%}"
                }
            )

            cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 255, 255), 2)

        prediction_details['found_classifications'] = list(set(classification['class'] for classification in prediction_details['box_details']))
        prediction_details['screen_density'] = bbox_area_total / image_area_total
        prediction_details['number_of_bboxes'] = len(prediction_details['box_details'])

        cv2.imwrite((os.path.join(app.config['PREDICTION_FOLDER'], prediction_details['image_name'])), resized)
        with open(os.path.join(app.config['PREDICTION_DATA'], (prediction_details['image_name'].split('.')[0] + '.json')), 'x') as f:
            json.dump(prediction_details, f, indent=4)

    return redirect(url_for('predictions'))

@app.route('/prediction-detail/<image_id>')
def prediction_image_details(image_id):
    with open(os.path.join(app.config['PREDICTION_DATA'], image_id + '.json'), 'r') as f:
        data = json.load(f)

    data['screen_density'] = f'{data["screen_density"]:.2%}'
    data['found_classifications'] = ','.join(data['found_classifications'])
    
    return render_template('prediction_details.html', name=image_id, image=image_id + '.JPG', data=data)

@app.route('/clear-dataset', methods=['POST'])
def clear_existing_dataset():
    items = [os.path.join(app.config['UPLOAD_FOLDER'], item) for item in os.listdir(app.config['UPLOAD_FOLDER'])]
    for item in items:
        os.remove(item)
    return redirect('/current-dataset')

@app.route('/clear-predictions', methods=['POST'])
def clear_existing_predicted():
    items = [os.path.join(app.config['PREDICTION_FOLDER'], item) for item in os.listdir(app.config['PREDICTION_FOLDER'])]
    for item in items:
        os.remove(item)

    items = [os.path.join(app.config['PREDICTION_DATA'], item) for item in os.listdir(app.config['PREDICTION_DATA'])]
    for item in items:
        os.remove(item)
    return redirect(url_for('predictions'))

if __name__ == '__main__':
    app.run(debug=True)