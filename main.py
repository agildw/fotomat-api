from typing import Union
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import  StreamingResponse
import cv2
import numpy as np
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/")
async def read_image_matrix(file: UploadFile = File(...)):
    # if image mimetype is not image
    if file.content_type.split('/')[0] != 'image':
        return {"Error": "Invalid file type"}
    
    # read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return {"image": image.shape}
    

@app.post("/red")
async def red_image(file: UploadFile = File(...)):
    """
    Process an uploaded image and change its colors to red.

    Parameters:
    - file: UploadFile object representing the uploaded image file.

    Returns:
    - StreamingResponse: Response containing the processed image in PNG format.

    Raises:
    - HTTPException: If the uploaded file is not an image or an error occurs during processing.
    """
    
    # if image mimetype is not image
    if file.content_type.split('/')[0] != 'image':
        return {"Error": "Invalid file type"}
    
    # Check if the uploaded file is an image
    try:
        # Read image file into memory
        image_stream = await file.read()
        image_array = np.fromstring(image_stream, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Process the image to change colors to red
        red_image = image.copy()
        red_image[:, :, 0] = 0  # Zero out the blue channel
        red_image[:, :, 1] = 0  # Zero out the green channel

        # Encode the processed image to send it back
        _, encoded_image = cv2.imencode('.png', red_image)
        encoded_image_bytes = encoded_image.tobytes()
        return StreamingResponse(BytesIO(encoded_image_bytes), media_type="image/png")
    except Exception as e:
        return {"Error": str(e)}
    
@app.post('/blue')
async def blue_image(file: UploadFile = File(...)):
    """
    Process an uploaded image and change its colors to blue.

    Parameters:
    - file: UploadFile object representing the uploaded image file.

    Returns:
    - StreamingResponse: Response containing the processed image in PNG format.

    Raises:
    - HTTPException: If the uploaded file is not an image or an error occurs during processing.
    """
    
    # if image mimetype is not image
    if file.content_type.split('/')[0] != 'image':
        return {"Error": "Invalid file type"}
    
    # Check if the uploaded file is an image
    try:
        # Read image file into memory
        image_stream = await file.read()
        image_array = np.fromstring(image_stream, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Process the image to change colors to blue
        blue_image = image.copy()
        blue_image[:, :, 1] = 0  # Zero out the green channel
        blue_image[:, :, 2] = 0  # Zero out the red channel

        # Encode the processed image to send it back
        _, encoded_image = cv2.imencode('.png', blue_image)
        encoded_image_bytes = encoded_image.tobytes()
        return StreamingResponse(BytesIO(encoded_image_bytes), media_type="image/png")
    except Exception as e:
        return {"Error": str(e)}
    
@app.post('/green')
async def green_image(file: UploadFile = File(...)):
    """
    Process an uploaded image and change its colors to green.

    Parameters:
    - file: UploadFile object representing the uploaded image file.

    Returns:
    - StreamingResponse: Response containing the processed image in PNG format.

    Raises:
    - HTTPException: If the uploaded file is not an image or an error occurs during processing.
    """
    
    # if image mimetype is not image
    if file.content_type.split('/')[0] != 'image':
        return {"Error": "Invalid file type"}
    
    # Check if the uploaded file is an image
    try:
        # Read image file into memory
        image_stream = await file.read()
        image_array = np.fromstring(image_stream, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Process the image to change colors to green
        green_image = image.copy()
        green_image[:, :, 0] = 0  # Zero out the blue channel
        green_image[:, :, 2] = 0  # Zero out the red channel

        # Encode the processed image to send it back
        _, encoded_image = cv2.imencode('.png', green_image)
        encoded_image_bytes = encoded_image.tobytes()
        return StreamingResponse(BytesIO(encoded_image_bytes), media_type="image/png")
    except Exception as e:
        return {"Error": str(e)}
    
@app.post('/gray')
async def gray_image(file: UploadFile = File(...)):
    """
    Process an uploaded image and convert it to grayscale.

    Parameters:
    - file: UploadFile object representing the uploaded image file.

    Returns:
    - StreamingResponse: Response containing the processed image in PNG format.

    Raises:
    - HTTPException: If the uploaded file is not an image or an error occurs during processing.
    """
    
    # if image mimetype is not image
    if file.content_type.split('/')[0] != 'image':
        return {"Error": "Invalid file type"}
    
    # Check if the uploaded file is an image
    try:
        # Read image file into memory
        image_stream = await file.read()
        image_array = np.fromstring(image_stream, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Encode the processed image to send it back
        _, encoded_image = cv2.imencode('.png', gray_image)
        encoded_image_bytes = encoded_image.tobytes()
        return StreamingResponse(BytesIO(encoded_image_bytes), media_type="image/png")
    except Exception as e:
        return {"Error": str(e)}
    
@app.post('/compress')
async def compress_image(file: UploadFile = File(...), quality: int = 90):
    """
    Compress an uploaded image with a specified quality level.

    Parameters:
    - file: UploadFile object representing the uploaded image file.
    - quality: Integer representing the compression quality level (0-100).

    Returns:
    - StreamingResponse: Response containing the compressed image in PNG format.

    Raises:
    - HTTPException: If the uploaded file is not an image or an error occurs during processing.
    """
    
    # if image mimetype is not image
    if file.content_type.split('/')[0] != 'image':
        return {"Error": "Invalid file type"}
    
    # Check if the uploaded file is an image
    try:
        # Read image file into memory
        image_stream = await file.read()
        image_array = np.fromstring(image_stream, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Compress the image with the specified quality level
        _, encoded_image = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, quality])
        encoded_image_bytes = encoded_image.tobytes()
        return StreamingResponse(BytesIO(encoded_image_bytes), media_type="image/png")
    except Exception as e:
        return {"Error": str(e)}