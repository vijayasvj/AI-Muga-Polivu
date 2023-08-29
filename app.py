from flask import Flask, request, jsonify
import base64
import numpy as np
import cv2
from gfpgan import GFPGANer
import cv2
import os
import torch
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from torchvision.transforms.functional import normalize

from gfpgan.archs.gfpgan_bilinear_arch import GFPGANBilinear
from gfpgan.archs.gfpganv1_arch import GFPGANv1
from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

# Initialize GFPGANer (You can use your initialization code here)
model_path = r'experiments\pretrained_models\GFPGANv1.3.pth'  # Path to your GFPGAN model
gfpganer = GFPGANer(model_path=model_path, upscale=3)

def enhance_image(img):
    # Enhance the input image using the model
    cropped_faces, restored_faces, restored_img = gfpganer.enhance(img)
    return cropped_faces, restored_faces, restored_img

@app.route('/enhance', methods=['POST'])
def enhance_api():
    try:
        data = request.json  # Assuming JSON input containing 'base64_image'
        base64_image = data.get('base64_image', None)

        if base64_image is None:
            return jsonify({'error': 'Missing base64_image'}), 400

        # Convert base64 image to NumPy array
        image_data = base64.b64decode(base64_image)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Enhance the image
        cropped_faces, restored_faces, restored_img = enhance_image(img)

        # Convert images to base64
        restored_faces_base64 = [base64.b64encode(cv2.imencode('.png', face)[1]).decode() for face in restored_faces]
        restored_img_base64 = base64.b64encode(cv2.imencode('.png', restored_img)[1]).decode()

        response_data = {
            'restored_faces': restored_faces_base64,
            'restored_img': restored_img_base64
        }

        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
