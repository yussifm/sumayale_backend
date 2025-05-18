from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for Flutter app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Flutter app's domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication setup
security = HTTPBearer()
SECRET_TOKEN = os.getenv("SECRET_TOKEN", "default-token")  # Set via environment variable

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the authentication token."""
    if credentials.credentials != SECRET_TOKEN:
        logger.warning("Authentication failed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Load model and processor at startup
logger.info("Loading model and processor...")
try:
    model = WhisperForConditionalGeneration.from_pretrained("./whisper_model", use_safetensors=True)
    processor = WhisperProcessor.from_pretrained("./whisper_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("Model and processor loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise RuntimeError("Model loading failed") from e

# Middleware to log requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests and response status."""
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

@app.post("/transcribe", dependencies=[Depends(verify_token)])
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe an uploaded audio file using the custom speech model.
    
    Args:
        file: Audio file uploaded from the client (e.g., Flutter app)
    
    Returns:
        JSON response with the transcription or an error message
    """
    try:
        # Load audio from the uploaded file
        waveform, sample_rate = torchaudio.load(file.file)
        
        # Resample to 16kHz if necessary (Whisper standard)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Process audio with feature extractor
        input_features = processor.feature_extractor(
            waveform.squeeze().numpy(), 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(device)
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        
        # Decode the transcription
        transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        logger.info(f"Transcription successful for file: {file.filename}")
        return {"transcription": transcription}
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint to verify API status."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)