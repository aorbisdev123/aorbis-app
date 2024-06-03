# # import torch

# # from models.yolov7 import YOLOv7

# # # Load the YOLOv7 model




# import torch
# import cv2
# from models.experimental import attempt_load  # Assuming you have the necessary imports

# # Define the path to your model weights file(s)
# weights_path = "\\orion\ASDU\Developers\Manish\Python Code\detection_api\yolov7\aorbis.pt"

# # Load the model or ensemble of models
# model = attempt_load(weights_path)

# # Now you can use the model for inference
# # For example, you can pass an image through the model to get predictions
# # image = ...  # Your image data
# # predictions = model(image)



# model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
# model.eval()

# # Load and preprocess the image
# image = cv2.imread('image.jpg')  # Replace 'image.jpg' with the path to your image file
# # Preprocess image (resize, normalize, etc.)

# # Convert image to PyTorch tensor
# image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0).unsqueeze(0)

# # Perform inference
# with torch.no_grad():
#     detections = model(image_tensor)

# # Postprocess detections (e.g., filter by confidence threshold)

# # Visualize or save results
# # (e.g., draw bounding boxes on the original image)






