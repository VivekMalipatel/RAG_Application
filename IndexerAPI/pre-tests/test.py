import os
import torch
import logging
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get HF token from environment variable (if available)
hf_token = os.environ.get("HF_TOKEN")

# Set up custom directory for model storage (absolute path)
current_file = os.path.abspath(__file__)
models_dir = os.path.join(os.path.dirname(current_file), ".models")
os.makedirs(models_dir, exist_ok=True)

# Set environment variables to force HF to use our custom cache directory
os.environ["HF_HOME"] = models_dir
os.environ["TRANSFORMERS_CACHE"] = os.path.join(models_dir, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(models_dir, "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(models_dir, "hub")

logger.info(f"Models will be stored at: {models_dir}")
logger.info(f"HF_HOME set to: {os.environ['HF_HOME']}")
logger.info(f"TRANSFORMERS_CACHE set to: {os.environ['TRANSFORMERS_CACHE']}")

model_name = "nomic-ai/colnomic-embed-multimodal-3b"

# Load the model with proper error handling
try:
    logger.info(f"Loading model: {model_name} on CPU")
    
    # Model loading options - using CPU instead of CUDA
    model_kwargs = {
        "torch_dtype": torch.float32,  # Use float32 for CPU instead of bfloat16
        "device_map": "cpu",  # Changed from cuda:0 to cpu
        "cache_dir": os.environ["TRANSFORMERS_CACHE"],
        "local_files_only": False
    }
    
    # Flash attention is not applicable for CPU
    # No need to check for is_flash_attn_2_available()
    
    # Load model
    logger.info("Starting model loading process...")
    model = ColQwen2_5.from_pretrained(
        model_name,
        token=hf_token,
        **model_kwargs
    ).eval()
    
    # Load processor
    logger.info("Loading processor...")
    processor = ColQwen2_5_Processor.from_pretrained(
        model_name,
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
        token=hf_token
    )
    
    logger.info(f"Successfully loaded model: {model_name}")
    
    mixed_inputs = [
        {
            "image": Image.new("RGB", (128, 128), color="white"),
            "text": "R&D department structure diagram"  
        },
    ]

    # Process interleaved inputs - make sure device is CPU
    logger.info("Processing inputs...")
    processed_batch = processor.process_images(
        images=[item["image"] for item in mixed_inputs],
        context_prompts=[item["text"] for item in mixed_inputs]  # <-- This is key
    )
    # No need to manually move to device as we're using CPU
    
    # Generate unified embeddings
    logger.info("Generating embeddings...")
    with torch.no_grad():
        combined_embeddings = model(**processed_batch)

    # Print embedding shape to verify success
    logger.info(f"Generated embeddings with shape: {combined_embeddings.shape}")
    
except Exception as e:
    logger.error(f"Error loading model {model_name}: {str(e)}")
    import traceback
    logger.error(f"Error details: {traceback.format_exc()}")
    raise

logger.info("Test completed successfully")
