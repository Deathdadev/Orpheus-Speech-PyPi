from snac import SNAC
import numpy as np
import torch
import asyncio
import threading
import queue
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for token processing
TOKEN_PREFIX_LENGTH = 14  # Length of "<custom_token_"
TOKEN_OFFSET = 10  # Offset value used in token ID calculation
CODES_MAX_VALUE = 4096  # Maximum valid code value
FRAME_SIZE = 7  # Number of tokens per frame
MIN_BUFFER_SIZE = 28  # Minimum buffer size for processing

# Initialize SNAC model with proper device detection
class SNACModel:
    _instance = None

    @classmethod
    def get_instance(cls):
        """Singleton pattern to ensure only one model instance exists"""
        if cls._instance is None:
            cls._instance = SNACModel()
        return cls._instance

    def __init__(self):
        """Initialize the SNAC model with appropriate device"""
        # Determine the best available device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            logger.warning("CUDA is not available. Using CPU for SNAC model, which may be slow.")
        else:
            logger.info(f"Using {self.device} for SNAC model")

        # Load the model
        try:
            self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
            self.model = self.model.to(self.device)
            logger.info("SNAC model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SNAC model: {str(e)}")
            raise

    def decode(self, codes):
        """Decode audio from codes using the SNAC model"""
        with torch.inference_mode():
            return self.model.decode(codes)

    def cleanup(self):
        """Clean up resources when no longer needed"""
        if hasattr(self, 'model'):
            # Move model to CPU to free up CUDA memory
            if self.device == "cuda":
                self.model = self.model.cpu()
                torch.cuda.empty_cache()
                logger.info("SNAC model resources cleaned up")

# Get the model instance
snac_model = SNACModel.get_instance()
snac_device = snac_model.device


def convert_to_audio(multiframe, count):
    """Convert token frames to audio bytes.

    Args:
        multiframe (list): List of token IDs to convert to audio
        count (int): Current token count for tracking progress

    Returns:
        bytes or None: Audio data as bytes if successful, None if invalid input

    Raises:
        RuntimeError: If there's an error during audio decoding
    """
    try:
        # Validate input
        if not multiframe or len(multiframe) < FRAME_SIZE:
            logger.warning(f"Input frame too small: {len(multiframe) if multiframe else 0} tokens (minimum {FRAME_SIZE} required)")
            return None

        # Initialize code tensors
        codes_0 = torch.tensor([], device=snac_device, dtype=torch.int32)
        codes_1 = torch.tensor([], device=snac_device, dtype=torch.int32)
        codes_2 = torch.tensor([], device=snac_device, dtype=torch.int32)

        # Calculate number of complete frames
        num_frames = len(multiframe) // FRAME_SIZE
        frame = multiframe[:num_frames*FRAME_SIZE]

        # Process each frame
        for j in range(num_frames):
            i = FRAME_SIZE * j

            # Process codes for the first codebook
            if codes_0.shape[0] == 0:
                codes_0 = torch.tensor([frame[i]], device=snac_device, dtype=torch.int32)
            else:
                codes_0 = torch.cat([codes_0, torch.tensor([frame[i]], device=snac_device, dtype=torch.int32)])

            # Process codes for the second codebook
            if codes_1.shape[0] == 0:
                codes_1 = torch.tensor([frame[i+1]], device=snac_device, dtype=torch.int32)
                codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=snac_device, dtype=torch.int32)])
            else:
                codes_1 = torch.cat([codes_1, torch.tensor([frame[i+1]], device=snac_device, dtype=torch.int32)])
                codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=snac_device, dtype=torch.int32)])

            # Process codes for the third codebook
            if codes_2.shape[0] == 0:
                codes_2 = torch.tensor([frame[i+2]], device=snac_device, dtype=torch.int32)
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=snac_device, dtype=torch.int32)])
            else:
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+2]], device=snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=snac_device, dtype=torch.int32)])

        # Prepare codes for model input
        codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]

        # Validate code values
        for i, code in enumerate(codes):
            if torch.any(code < 0) or torch.any(code > CODES_MAX_VALUE):
                logger.warning(f"Invalid code values detected in codebook {i}: values must be between 0 and {CODES_MAX_VALUE}")
                return None

        # Decode audio using the SNAC model
        audio_hat = snac_model.decode(codes)

        # Process the audio output
        audio_slice = audio_hat[:, :, 2048:4096]  # Extract the relevant portion of audio
        detached_audio = audio_slice.detach().cpu()
        audio_np = detached_audio.numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)  # Convert to 16-bit PCM
        audio_bytes = audio_int16.tobytes()

        return audio_bytes

    except Exception as e:
        logger.error(f"Error converting tokens to audio: {str(e)}")
        return None

def turn_token_into_id(token_string, index):
    """Convert a token string to its numerical ID.

    Args:
        token_string (str): The token string to convert
        index (int): Current token index for calculating the offset

    Returns:
        int or None: The token ID if successful, None if invalid token
    """
    # Strip whitespace
    token_string = token_string.strip()

    # Find the last token in the string
    token_prefix = "<custom_token_"
    last_token_start = token_string.rfind(token_prefix)

    if last_token_start == -1:
        logger.debug("No token found in the string")
        return None

    # Extract the last token
    last_token = token_string[last_token_start:]

    # Process the last token
    if last_token.startswith(token_prefix) and last_token.endswith(">"):
        try:
            # Extract the number from the token
            number_str = last_token[TOKEN_PREFIX_LENGTH:-1]

            # Calculate the token ID with the appropriate offset
            # Formula: token_id = raw_number - TOKEN_OFFSET - ((index % FRAME_SIZE) * CODES_MAX_VALUE)
            token_id = int(number_str) - TOKEN_OFFSET - ((index % FRAME_SIZE) * CODES_MAX_VALUE)

            return token_id
        except ValueError as e:
            logger.warning(f"Failed to parse token number: {str(e)}")
            return None
    else:
        logger.debug(f"Invalid token format: {last_token}")
        return None


async def tokens_decoder(token_gen, max_buffer_size=1000):
    """Decode tokens into audio samples asynchronously.

    Args:
        token_gen: Async generator that yields token strings
        max_buffer_size (int): Maximum buffer size to prevent memory issues

    Yields:
        bytes: Audio samples as they become available
    """
    buffer = []
    count = 0

    try:
        async for token_sim in token_gen:
            # Convert token string to ID
            token = turn_token_into_id(token_sim, count)

            if token is not None and token > 0:
                # Add valid token to buffer
                buffer.append(token)
                count += 1

                # Limit buffer size to prevent memory issues
                if len(buffer) > max_buffer_size:
                    logger.warning(f"Buffer exceeded max size ({max_buffer_size}). Trimming to prevent memory issues.")
                    buffer = buffer[-max_buffer_size:]

                # Process buffer when we have enough tokens and at frame boundaries
                if count % FRAME_SIZE == 0 and count > MIN_BUFFER_SIZE - 1:
                    # Use only the most recent tokens needed for processing
                    buffer_to_proc = buffer[-MIN_BUFFER_SIZE:]

                    # Convert tokens to audio
                    audio_samples = convert_to_audio(buffer_to_proc, count)

                    if audio_samples is not None:
                        yield audio_samples
    except Exception as e:
        logger.error(f"Error in tokens_decoder: {str(e)}")
        # Re-raise to propagate the error
        raise


# ------------------ Synchronous Tokens Decoder Wrapper ------------------ #
def tokens_decoder_sync(syn_token_gen, max_buffer_size=1000, timeout=60):
    """Synchronous wrapper for the asynchronous tokens_decoder function.

    This function converts a synchronous token generator to an asynchronous one,
    processes the tokens through tokens_decoder, and yields audio chunks synchronously.

    Args:
        syn_token_gen: Synchronous generator that yields token strings
        max_buffer_size (int): Maximum buffer size to pass to tokens_decoder
        timeout (int): Timeout in seconds for queue operations

    Yields:
        bytes: Audio samples as they become available

    Raises:
        RuntimeError: If there's an error in the async processing thread
    """
    # Create a thread-safe queue for passing audio chunks between threads
    audio_queue = queue.Queue()
    error_queue = queue.Queue()  # For passing exceptions from the thread

    # Convert the synchronous token generator into an async generator
    async def async_token_gen():
        try:
            for token in syn_token_gen:
                yield token
        except Exception as e:
            logger.error(f"Error in token generator: {str(e)}")
            raise

    async def async_producer():
        try:
            # Process tokens through the async decoder
            async for audio_chunk in tokens_decoder(async_token_gen(), max_buffer_size=max_buffer_size):
                audio_queue.put(audio_chunk)
        except Exception as e:
            # Put the exception in the error queue to propagate it to the main thread
            error_queue.put(e)
        finally:
            # Always signal completion
            audio_queue.put(None)  # Sentinel

    def run_async():
        try:
            asyncio.run(async_producer())
        except Exception as e:
            # Catch any exceptions not caught in async_producer
            error_queue.put(e)
            audio_queue.put(None)  # Ensure the main thread doesn't hang

    # Start the async processing in a separate thread
    thread = threading.Thread(target=run_async)
    thread.daemon = True  # Make thread daemon so it doesn't block program exit
    thread.start()

    try:
        # Yield audio chunks as they become available
        while True:
            # Check for errors from the async thread
            if not error_queue.empty():
                exception = error_queue.get()
                raise RuntimeError(f"Error in async processing thread: {str(exception)}") from exception

            # Get the next audio chunk with timeout
            try:
                audio = audio_queue.get(timeout=timeout)
                if audio is None:
                    break
                yield audio
            except queue.Empty:
                logger.warning(f"Timeout ({timeout}s) waiting for audio chunks")
                break
    finally:
        # Clean up resources
        if thread.is_alive():
            logger.info("Waiting for async processing thread to complete")
            thread.join(timeout=5)  # Wait for thread to finish with timeout