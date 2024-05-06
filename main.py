from typing import Union
from fastapi import FastAPI, File, UploadFile,HTTPException
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

async def image_matrix(file: UploadFile = File(...)):
    """
    Read the matrix representation of an uploaded image.

    Parameters:
    - file: UploadFile object representing the uploaded image file.

    Returns:
    - dict: Dictionary containing the shape and matrix representation of the image.

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

        # Convert the image matrix to a string
        image_matrix_str = str(image)

        return {"shape": image.shape, "matrix": image_matrix_str}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    

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
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    
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
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    
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
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    
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
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    
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

        
        print(quality)

        # Compress the image with the specified quality level
        _, encoded_image = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, int(quality)])

        # print image size
        print("Original image size: ", image.size)
        print("Compressed image size: ", encoded_image.size)

        encoded_image_bytes = encoded_image.tobytes()
        return StreamingResponse(BytesIO(encoded_image_bytes), media_type="image/png")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post('/sharpen')
async def sharpen_image(file: UploadFile = File(...)):
    """
    Process an uploaded image and apply a sharpening filter.

    Parameters:
    - file: UploadFile object representing the uploaded image file.

    Returns:
    - StreamingResponse: Response containing the sharpened image in PNG format.

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

        # Apply a sharpening filter to the image
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened_image = cv2.filter2D(image, -1, kernel)

        # Encode the processed image to send it back
        _, encoded_image = cv2.imencode('.png', sharpened_image)
        encoded_image_bytes = encoded_image.tobytes()
        return StreamingResponse(BytesIO(encoded_image_bytes), media_type="image/png")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
        # return {"Error": str(e)}
    
@app.post('/fourier')
async def fourier_transform(file: UploadFile = File(...)):
    """
    Process an uploaded image and apply a Fourier transform.

    Parameters:
    - file: UploadFile object representing the uploaded image file.
    
    Returns:
    - StreamingResponse: Response containing the Fourier transformed image in PNG format.

    Raises:
    - HTTPException: If the uploaded file is not an image or an error occurs during processing.
    """
    
    
    # if image mimetype is not image
    if file.content_type.split('/')[0] != 'image':
        return {"Error": "Invalid file type"}
    
    
    try:
       
        image_stream = await file.read()
        image_array = np.fromstring(image_stream, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
        # image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        # calculating the discrete Fourier transform
        DFT = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
        
        # reposition the zero-frequency component to the spectrum's middle
        shift = np.fft.fftshift(DFT)
        row, col = image.shape
        center_row, center_col = row // 2, col // 2
        
        # create a mask with a centered square of 1s
        mask = np.zeros((row, col, 2), np.uint8)
        mask[center_row - 30:center_row + 30, center_col - 30:center_col + 30] = 1
        
        # put the mask and inverse DFT in place.
        fft_shift = shift * mask
        fft_ifft_shift = np.fft.ifftshift(fft_shift)
        imageThen = cv2.idft(fft_ifft_shift)
        
        # calculate the magnitude of the inverse DFT
        imageThen = cv2.magnitude(imageThen[:,:,0], imageThen[:,:,1])

        # normalize the magnitude for display
        
        imageThen = cv2.normalize(imageThen, None, 0, 255, cv2.NORM_MINMAX)
        imageThen = imageThen.astype(np.uint8)

        _, encoded_image = cv2.imencode('.png', imageThen)
        encoded_image_bytes = encoded_image.tobytes()
        return StreamingResponse(BytesIO(encoded_image_bytes), media_type="image/png")

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
        
    