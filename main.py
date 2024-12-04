from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from azure.storage.blob import BlobServiceClient
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from models import ImageAnalysisResponse, DenseCaption
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional, List
import logging
from caption_enhance import CaptionEnhance


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

caption_enhance = CaptionEnhance()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Azure credentials with validation
STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
VISION_ENDPOINT = os.getenv("VISION_ENDPOINT")
VISION_KEY = os.getenv("VISION_KEY")

# Validate environment variables
if not VISION_ENDPOINT or not VISION_KEY or not STORAGE_CONNECTION_STRING:
    error_message = "Missing environment variables:\n"
    if not VISION_ENDPOINT:
        error_message += "- VISION_ENDPOINT\n"
    if not VISION_KEY:
        error_message += "- VISION_KEY\n"
    if not STORAGE_CONNECTION_STRING:
        error_message += "- AZURE_STORAGE_CONNECTION_STRING\n"
    raise ValueError(error_message)


@app.get("/")
async def root():
    return {
        "message": "Vision API is running",
        "vision_endpoint_configured": bool(VISION_ENDPOINT),
        "vision_key_configured": bool(VISION_KEY),
        "storage_configured": bool(STORAGE_CONNECTION_STRING)
    }


def upload_to_blob(file_data, filename):
    try:
        # Validate connection string
        if not STORAGE_CONNECTION_STRING or "AccountName" not in STORAGE_CONNECTION_STRING:
            raise ValueError("Invalid storage connection string")

        # Create blob service client
        logger.info("Creating blob service client")
        blob_service_client = BlobServiceClient.from_connection_string(
            STORAGE_CONNECTION_STRING)

        # Get container client
        container_name = "vision-images"
        logger.info(f"Getting container client for {container_name}")
        container_client = blob_service_client.get_container_client(
            container_name)

        # Create container if it doesn't exist
        try:
            container_properties = container_client.get_container_properties()
            logger.info("Container exists")
        except Exception as e:
            logger.info("Creating container")
            container_client.create_container()

        # Create blob name
        blob_name = f"{filename}"
        logger.info(f"Uploading blob: {blob_name}")

        # Get blob client
        blob_client = container_client.get_blob_client(blob_name)

        # Upload file
        blob_client.upload_blob(file_data, overwrite=True)

        # Return the blob URL
        return blob_client.url

    except ValueError as ve:
        logger.error(f"Configuration error: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Blob storage error: {str(e)}")
        raise


@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Read image data
        image_data = await file.read()

        # Upload to blob storage
        try:
            blob_url = upload_to_blob(image_data, file.filename)
            logger.info(f"File uploaded successfully to: {blob_url}")
        except Exception as storage_error:
            logger.error(f"Storage error: {str(storage_error)}")

        try:
            # Initialize Vision client
            logger.debug("Initializing Vision client")
            vision_client = ImageAnalysisClient(
                endpoint=VISION_ENDPOINT,
                credential=AzureKeyCredential(VISION_KEY)
            )

            # Analyze image
            logger.debug("Analyzing image")
            result = vision_client.analyze(
                image_data=image_data,
                visual_features=[
                    VisualFeatures.CAPTION,
                    VisualFeatures.DENSE_CAPTIONS,
                    VisualFeatures.READ
                ]
            )

            # Process dense captions and sort by confidence
            dense_captions = []
            if result.dense_captions:
                dense_captions = [
                    DenseCaption(
                        text=caption.text,
                        confidence=caption.confidence
                    ) for caption in result.dense_captions.list
                ]
                # Sort by confidence in descending order
                dense_captions.sort(
                    key=lambda x: x.confidence or 0, reverse=True)

            # Process results
                response = ImageAnalysisResponse(
                    caption=result.caption.text if result.caption else None,
                    dense_captions=[
                        DenseCaption(
                            text=caption.text,
                            confidence=caption.confidence
                        ) for caption in result.dense_captions.list
                    ] if result.dense_captions else [],
                    text_content=" ".join(
                        [line.text for block in result.read.blocks for line in block.lines]
                    ) if result.read else None
                )

            # Convert to dict for enhancement
            response_dict = response.dict()

            # Enhance the response
            enhanced_dict = caption_enhance.enhance_response(response_dict)

            # Convert back to ImageAnalysisResponse
            enhanced_response = ImageAnalysisResponse(**enhanced_dict)

            return enhanced_response

        except Exception as vision_error:
            logger.error(f"Vision API error: {str(vision_error)}")
            raise HTTPException(status_code=500, detail=str(vision_error))

    except Exception as e:
        logger.error(f"General error: {str(e)}")
        return ImageAnalysisResponse(error=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
