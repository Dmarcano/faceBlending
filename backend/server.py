from lib.face_coordinates import detect_face_coordinates
from flask import Flask, flash, request, redirect, url_for, jsonify
import cv2 
import numpy as np 
import numpy 
from werkzeug.utils import secure_filename

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/v1/helloWorld')
def hello_world():
    img = cv2.imread("images/obama.jpg")
    faces = detect_face_coordinates(img)
    content = { 'x' : int(faces[0][0]), 'y' : int(faces[0][1]), 'w' : int(faces[0][2]), 'h'  : int(faces[0][3]) }
    
    return jsonify(content)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        request_file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if request_file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if request_file and allowed_file(request_file.filename):
            filename = secure_filename(request_file.filename)
            # request_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img_from_buffer =  cv2.imdecode(np.fromstring(b"".join(request_file.stream.readlines()), np.uint8),1)
            faces = detect_face_coordinates(img_from_buffer)
            face_box = list(map(lambda val : int(val),faces[0]))
            resp_content = {'x' : face_box[0], 'y' :face_box[1], "w" : face_box[2], 'h' : face_box[3] }
            return jsonify(resp_content)

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''
    
# TODO DO BIT MAPPING


