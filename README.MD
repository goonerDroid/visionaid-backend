# Vision Analysis API

A FastAPI service that leverages Azure's Computer Vision services to analyze images.

## Features

- Image upload and analysis
- Azure Blob Storage integration
- Computer Vision analysis including:
  - Image captioning
  - Text extraction (OCR)
  - Object detection
  - Image tagging

## Prerequisites

- Python 3.8+
- Azure account with:
  - Computer Vision API subscription
  - Storage account

## Environment Variables

Create a `.env` file with the following variables:
```
AZURE_STORAGE_CONNECTION_STRING=your_storage_connection_string
VISION_ENDPOINT=your_vision_endpoint
VISION_KEY=your_vision_key
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

- `GET /`: Health check endpoint
- `POST /analyze-image`: Upload and analyze an image

## Development

Make sure to run tests before committing changes:
```bash
pytest
```