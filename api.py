import base64
import warnings
import os

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from flask import Flask, jsonify, request, send_from_directory, send_file

import argparse
import uuid
import glob
from deepface import DeepFace
from mtcnn import MTCNN
import cv2

warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)


@app.route('/verify', methods=['POST'])
def verify():
    req = request.get_json()
    if 'img' not in list(req.keys()):
        raise VerificationException("you must pass img")
    if 'id' not in list(req.keys()):
        raise VerificationException("you must pass id")
    try:
        verification_image_base64 = req['img']
        verification_image = load_base64_img(verification_image_base64)
        employee_id = req['id']

        employee_image_filename, employee_image = find_employee_image(employee_id)
        employee_filename_parts = employee_image_filename.split("_")

        face_detector = MTCNN()
        verification_faces = face_detector.detect_faces(
            cv2.cvtColor(verification_image, cv2.COLOR_BGR2RGB))
        employee_faces = face_detector.detect_faces(employee_image)
        if len(verification_faces) < 1:
            raise VerificationException("No faces on image")
        if len(verification_faces) > 1:
            raise VerificationException("More than one face on image")
        if len(employee_faces) < 1:
            raise VerificationException("Db error - no faces on employee image")
        if len(employee_faces) > 1:
            raise VerificationException("Db error - more than one face on image")

        verify_result = DeepFace.verify(verification_image,
                                        employee_image_filename,
                                        model_name="VGG-Face",
                                        distance_metric="cosine",
                                        detector_backend="mtcnn")
        identification_id = str(uuid.uuid4())
        save_verification_result_image(identification_id, verification_faces, employee_faces, verification_image,
                                       employee_image, verify_result)
        plt.savefig('./results/' + identification_id + '.jpg')
        resp_obj = {'verified': verify_result['verified'], 'confidence': 1 - verify_result['distance'],
                    'threshold': 1 - verify_result['max_threshold_to_verify'], 'name': employee_filename_parts[1],
                    'position': employee_filename_parts[2], 'position_id': employee_filename_parts[3],
                    'identification_id': identification_id, 'faces_number': 1}
        return resp_obj, 200
    except VerificationException as ex:
        return jsonify({'success': False, 'error': str(ex)}), 400


@app.route('/result/<filename>')
def get_verification_result(filename):
    return send_file("./results/" + filename + '.jpg')


def find_employee_image(employee_id):
    files = glob.glob('./database/' + employee_id + '_?*_?*_?*.jpg')
    if not files:
        raise VerificationException("employee not found")
    if len(files) > 1:
        raise VerificationException("db error - more than one employee found")
    filename = files[0]
    return filename, cv2.imread(filename)


def save_verification_result_image(identification_id, verification_faces, employee_faces, verification_image,
                                   employee_image, verify_result):
    f, subplots = plt.subplots(nrows=1, ncols=2)
    subplots[0].axis('off')
    subplots[1].axis('off')
    subplots[0].set_title("Verified: " + str(verify_result['verified']) + "\nConfidence: " + str(
        round(1 - verify_result['distance'], 2)))
    subplots[0].imshow(verification_image)
    subplots[0].add_patch(
        matplotlib.patches.Rectangle((verification_faces[0]['box'][0], verification_faces[0]['box'][1]),
                                     verification_faces[0]['box'][2],
                                     verification_faces[0]['box'][3], color="green", fill=None))
    subplots[1].imshow(employee_image)
    subplots[1].add_patch(
        matplotlib.patches.Rectangle((employee_faces[0]['box'][0], employee_faces[0]['box'][1]),
                                     employee_faces[0]['box'][2],
                                     employee_faces[0]['box'][3], color="green", fill=None))
    plt.savefig('./results/' + identification_id + '.jpg')


def load_base64_img(uri):
    nparr = np.fromstring(base64.b64decode(uri), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


class VerificationException(Exception):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--port',
        type=int,
        default=5000,
        help='Port of serving api')
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=args.port)
