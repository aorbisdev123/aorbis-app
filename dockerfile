# # Start from the PyTorch base image
# FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# # Set the working directory in the container
# WORKDIR /app

# # Copy the contents of the yolov7 directory into the /app/yolov7 directory in the container
# COPY app /app

# # Copy the requirements.txt file into the /app directory in the container
# COPY requirements.txt /app

# # List the contents of the /app directory for debugging purposes
# RUN ls -l /app

# # Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir  -r requirements.txt

# # Expose the port the app runs on
# EXPOSE 8089

# # Run main.py when the container launches
# CMD ["python", "main.py"]

# Start from the PyTorch base image
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Install necessary system dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Copy the contents of the app directory into the /app directory in the container
COPY app /app

# Copy the requirements.txt file into the /app directory in the container
COPY requirements.txt /app

# List the contents of the /app directory for debugging purposes
RUN ls -l /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8089

# Run main.py when the container launches
CMD ["python", "main.py"]
