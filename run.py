import cv2
import numpy
from PIL import Image
import base64
import io
from flask import Flask, render_template, request, jsonify
import glob
from models.train_classifiers import dog_breed, face_detector

app = Flask(__name__)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')

def index():
 
    return render_template('master.html')


# web page that handles user query and displays model results
@app.route('/go', methods=['POST'])
def go():
     # save user input in query
    query = cv2.imdecode(numpy.fromstring(request.files['query'].read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
    #request.files.get('query')#request.args.get('query', '') 

    filename = request.args.get('query', '')

    breed = dog_breed(query).replace('es/train/', '')

    result_dict = {}
    result_dict['dog_breed'] = "Predicted breed: " + breed.split('.')[1].replace('_', ' ')

    #get input image
    im = Image.open("static/img/dog.png")
    data = io.BytesIO()
    im.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())
    #classification_result = result_dict[classification_label]#dict(zip(dog_breed(query), classification_labels))

    im2 = Image.open(glob.glob(f"models/data/dog_images/train/{breed}/*.jpg")[0])
    data2 = io.BytesIO()
    im2.save(data2, "JPEG")
    encoded_img_data2 = base64.b64encode(data2.getvalue())

     # This will render the go.html Please see that file. 
    return render_template(
         'go.html',
         query=query,
         result_dict=result_dict,
         filename = filename,
         img_data=encoded_img_data.decode('utf-8'),
         img_data2=encoded_img_data2.decode('utf-8')
         #img_data=img_data
     )


def main():
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()

