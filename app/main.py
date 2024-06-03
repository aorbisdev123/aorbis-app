from flask import Flask, request, jsonify
from PIL import Image
import os
import torch
import base64
from io import BytesIO
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.datasets import letterbox
import numpy as np
import xml.etree.ElementTree as ET
import xml.dom.minidom
from utils.plots import plot_one_box
from pdf2img import pdf_page_to_image, resize_and_pad_image
import time

app = Flask(__name__)

# Select device (GPU if available, otherwise CPU)
device = select_device('0' if torch.cuda.is_available() else 'cpu')
# Load the model
model = attempt_load('aorbis.pt', map_location=device)
model.to(device).eval()

# Set Path in Docker Enviroment 
FIXED_PATH = os.getenv('FIXED_PATH', '/app/static/')
# FIXED_PATH = os.getenv('FIXED_PATH', 'static/')

# Function to prettify XML output
def prettify(elem):
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = xml.dom.minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

# Function to write detection results into an XML format
def write_detection_xml(detections,filename, img_path, img_size):
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = os.path.basename(os.path.dirname(img_path))
    ET.SubElement(root, "filename").text = os.path.basename(img_path)
    ET.SubElement(root, "path").text = img_path
    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Aorbis"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(img_size[0])
    ET.SubElement(size, "height").text = str(img_size[1])
    ET.SubElement(size, "depth").text = '3'
    ET.SubElement(root, "segmented").text = "0"
    for det in detections:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = det['class_name']
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(int(det['coordinates'][0]))
        ET.SubElement(bbox, "ymin").text = str(int(det['coordinates'][1]))
        ET.SubElement(bbox, "xmax").text = str(int(det['coordinates'][2]))
        ET.SubElement(bbox, "ymax").text = str(int(det['coordinates'][3]))
    
    formatted_xml = prettify(root)
    with open(filename, 'w') as f:
        f.write(formatted_xml)
   

# Testing endpoint to verify if the API is working
@app.route('/', methods=['GET'])
def get_req():
    return 'OK API is working'

# Endpoint for processing images and performing detection
# prameter For Get Binary Image output process
  # input-image --> save that image in flask api -> convert binary and return binary Image
  # important parameter: flag=>binary , images=>array([0]=>img1,[1]=>img2)

# prameter For Get Image Path output process
  # input image_dir_path -> read image dir path -> and save xml nd detected folder image
  # important parameter: dir=>(dir path where to read files )

@app.route('/detect', methods=['POST'])
def process_images():

    start = time.time()
    
    if 'dir' not in request.json:
        return jsonify({'status': 400, "message": "Missing dir parameter ({'dir': (send dir where images saved)})"})
    
    dynamic_path = request.json['dir']
    image_dir = os.path.join(FIXED_PATH, dynamic_path)
   
    if not os.path.exists(image_dir):
        return jsonify({'status': 400, "message": "Directory does not exist"})

    detected_folder = os.path.join(image_dir, 'detected_images')
    if not os.path.exists(detected_folder):
        os.makedirs(detected_folder)

    xml_folder = image_dir  # Save XML files in the same directory as the images

    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    response = []
    results = []
    for img_path in image_files:
        try:
            img = Image.open(img_path).convert('RGB')
            img_result, detections = perform_detection(img)

            base_filename = os.path.splitext(os.path.basename(img_path))[0]
            detected_filename = f"{base_filename}_detected.jpg"
            detected_path = os.path.join(detected_folder, detected_filename)
            img_result.save(detected_path)

            xml_filename = f"{base_filename}.xml"
            xml_path = os.path.join(xml_folder, xml_filename)
            write_detection_xml(detections, xml_path, img_path, img.size)

            results.append({
                'original': img_path,
                'detected': detected_path,
                'xml': xml_path
            })
        except Exception as e:
            print(f"Failed to process {img_path}: {str(e)}")
            continue
    
    end = time.time()
    execution_time = end - start
    response = {"status":200,'data':results,'execution_time':execution_time}
    return jsonify(response)

# Function to perform detection on an image
def perform_detection(image):
    img0 = np.array(image)
    img_size = 1024  # Example size, adjust based on your model's input requirements
    img = letterbox(img0, img_size, stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)
    img_tensor = torch.from_numpy(img).float()
    img_tensor /= 255.0  # Normalize to [0, 1]
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

    pred = model(img_tensor, augment=False)[0]
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

    detection_data = []
    if len(pred):
        det = pred[0]
        det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img0.shape).round()  # Rescale boxes to original image size
        for *xyxy, conf, cls in reversed(det):
            label = f'{model.names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, img0, label=label, color=(0, 0, 0), line_thickness=1)
            detection_data.append({
                'coordinates': [int(x) for x in xyxy],
                'class_name': model.names[int(cls)],
                'confidence': conf.item()
            })

    return Image.fromarray(img0), detection_data

# Endpoint for resizing and padding an image
# parameter:-
    # Input:-  Binary Image Part , relative croped image path , filename (what is name of file)
    # Output:- Cropped Image Output Path

@app.route('/resize_image', methods=['POST'])
def resize_image():
    start = time.time()
    # Retrieve image path and filename from POST data
    if 'relative_cropped_image_path' not in request.form:
        return jsonify(
            {
                'status': 400, 
                'message': "Missing parameter. Parameters should be: {'relative_cropped_image_path': (where file is saved)}"
            }
        )
    cropped_img_path = request.form.get('relative_cropped_image_path')
    cropped_img_path = os.path.join(FIXED_PATH, cropped_img_path)

    # Ensure the file exists before proceeding
    if not os.path.exists(cropped_img_path):
        return jsonify({'status': 404,'message': f"The file {cropped_img_path} does not exist."})
    
    # Call resize_and_pad_image function
    multiple_status = True if request.form.get('multiple') or 'multiple' not in request.form else False

    if multiple_status and os.path.isdir(cropped_img_path):
        for filename in os.listdir(cropped_img_path):
            image_file_path = os.path.join(cropped_img_path,filename)
            resized_and_padded_image = resize_and_pad_image(image_file_path)
            resized_and_padded_image.save(image_file_path)
    else:
        if os.path.isdir(cropped_img_path):
            return jsonify({'status': 404,'message': f"You Send Diretory Instant of File Path, If you want to use Directory then Use THis Multiple:true"})
        else:
            resized_and_padded_image = resize_and_pad_image(cropped_img_path)
            resized_and_padded_image.save(cropped_img_path)

    end = time.time()
    execution_time = end - start

    # Send response
    response = {
        'status': 200,
        'message': 'Image processed successfully',
        'server_path': cropped_img_path,
        'execution_time': execution_time
    }

    return jsonify(response)


# Endpoint for converting PDF to images

@app.route('/pdf2img', methods=['POST'])
def pdf_to_image():

    start_time = time.time()

    if 'relative_pdf_file_path' not in request.form or 'relative_pdf_file_image_path' not in request.form:
        return jsonify({'status': 400, 'message': 'Missing required parameters (This is Format "relative_pdf_file_image_path", relative_pdf_file_path)'})

    
    pdf_path_with_name = request.form['relative_pdf_file_path']

    pdf_path_with_name = os.path.join(FIXED_PATH, pdf_path_with_name)
    

    # Mapping Container Folder 
    save_pdf_img_folder_name = request.form['relative_pdf_file_image_path']

    save_pdf_img_folder_name = os.path.join(FIXED_PATH,save_pdf_img_folder_name)

    pdf_process = False

    if not os.path.exists(save_pdf_img_folder_name):
        os.makedirs(save_pdf_img_folder_name)


    if os.listdir(save_pdf_img_folder_name):
        pdf_process = False
    else:
        pdf_process = True

    try:
        if pdf_process:
            pdf_page_to_image(pdf_path_with_name, save_pdf_img_folder_name)

        end_time = time.time()
        execution_time = end_time - start_time

        response = {
            'status': 200,
            'message': "Conversion successful",
            'execution_time': execution_time,
            'images_path': save_pdf_img_folder_name
        }
        print(response)
        return jsonify(response)

    except Exception as e:
        print(f"Error during PDF processing: {str(e)}")
        return jsonify({'status': 500, 'message': f"Error during PDF processing: {str(e)}"}), 500

if __name__ == '__main__':  
    from waitress import serve
    # Serve the app using Waitress
    print('Server is Running 8089 ....')
    serve(app, host='0.0.0.0', port=8089, threads=12)
    
