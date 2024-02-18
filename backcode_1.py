from ultralytics import YOLO
from flask import request, Response, Flask
from PIL import Image
import json

app = Flask(__name__)

@app.route("/")
def root():
    
    # Site main page handler function.
    # :return: Content of index.html file
    
    with open("index.html") as file:
        return file.read()


@app.route("/detect", methods=["POST"])
def detect():

    buf = request.files["image_file"]
    boxes = detect_objects_on_image(Image.open(buf.stream))
    return Response(
      json.dumps(boxes),  
      mimetype='application/json'
    )

def detect_objects_on_image(buf):

    model = YOLO("best.pt")
    results = model.predict(buf)
    result = results[0]
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [
          round(x) for x in box.xyxy[0].tolist()
        ]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([
          x1, y1, x2, y2, result.names[class_id], prob
        ])
    return output

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
