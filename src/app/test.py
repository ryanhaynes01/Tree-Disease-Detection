import os
import cv2
from roboflow import Roboflow
rf = Roboflow(api_key="p7fKwIgRvWqrantEwNQ2")
project = rf.workspace().project("tree-diseases")
model = project.version(4).model

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

test_image_path = os.path.join(os.getcwd(), 'static', 'uploaded_images')
images = [os.path.join(test_image_path, image) for image in os.listdir(test_image_path)]

image = cv2.imread(str(images[0]))
resized = cv2.resize(image, (640, 640))


# infer on a local image
results = model.predict(images[0], confidence=50, overlap=20).json()
print(results['predictions'])

for bbox in results['predictions']:
    x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
    x1, y1, x2, y2 = resize_coords(image, yolo_format_to_xyxy(x, y, w, h))

    cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 255, 255), 2)



# visualize your prediction
# model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())