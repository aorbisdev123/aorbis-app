import fitz
from PIL import Image, ImageOps
import os
import tempfile
from io import BytesIO


def resize_and_pad_image(image_path, target_width=1024, target_height=1024):
    # Open the image file
    img = Image.open(image_path)
    
    # Convert image to RGB mode if it's in RGBA mode
    if img.mode == 'RGBA':
        img = img.convert('RGB')
        
    # Determine target dimensions based on original size
    width, height = img.size
    print("Original size:", width, height)
    
    # Calculate scaling factor to maintain aspect ratio
    width_ratio = target_width / width
    height_ratio = target_height / height
    scale_factor = min(width_ratio, height_ratio)
    
    # Calculate resized dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Resize the image while maintaining aspect ratio
    img = img.resize((new_width, new_height), Image.BICUBIC)
    print("Resized size:", new_width, new_height)
    
    # Calculate padding
    pad_width_left = (target_width - new_width) // 2
    pad_width_right = target_width - new_width - pad_width_left
    pad_height_top = (target_height - new_height) // 2
    pad_height_bottom = target_height - new_height - pad_height_top
    
    # Add padding
    padded_img = ImageOps.expand(img, (pad_width_left, pad_height_top, pad_width_right, pad_height_bottom), fill='white')
    
    print("Padded size:", padded_img.size)
    return padded_img

def pdf_page_to_image(pdf_content, save_img_folder_name='images', dpi=300):
    try:
        #Save PDF content to a temporary file
        # with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        #     temp_pdf.write(pdf_content)
        #     temp_pdf_path = temp_pdf.name

        temp_pdf_path = pdf_content
        # Get File Name 
        file_name = os.path.splitext(os.path.basename(temp_pdf_path))[0]

        # last Directory 
        last_directiory_name = os.path.basename(os.path.normpath(save_img_folder_name))

        os.makedirs(save_img_folder_name, exist_ok=True)

        # Open the PDF file
        pdf_file = fitz.open(temp_pdf_path)

        for page_number in range(pdf_file.page_count):
            # Get the specified page
            page = pdf_file[page_number]  # Page numbers are 0-based in fitz

            # Render the page as an image with higher resolution
            image = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))

            # Convert to a PIL Image with anti-aliasing
            pil_image = Image.frombytes("RGB", [image.width, image.height], image.samples, "raw", "RGB", 0, 1)

            # Save the image with original dimensions
            # replace string with underscore where white space, hiphen,and special character
            # last_directiory_name = last_directiory_name.replace(' ', '_').replace('-', '_')
    
            # last_directiory_name = last_directiory_name.replace(' ', '_')

            output_image_path = f'{save_img_folder_name}/{last_directiory_name}_{page_number}.jpg'

            pil_image.save(output_image_path, dpi=(dpi, dpi))

            return False
            # Remove return if you want to convert all pages 
            
            # # Convert Images To Fix Dimetion
            # resized_and_padded_image = resize_and_pad_image(output_image_path)

            #  # Save the resized and padded image to the output folder
            # resized_and_padded_image.save(output_image_path)
            print(f"Saved image: {output_image_path}")
            
    except OSError as error:
        print(error)