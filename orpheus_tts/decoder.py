from snac import SNAC
import numpy as np
import torch
import asyncio
import threading
import queue
import logging
import os
import time

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
        if num_frames == 0:
            logger.warning("No complete frames available for processing")
            return None
            
        frame = multiframe[:num_frames*FRAME_SIZE]

        # Process each frame
        for j in range(num_frames):
            i = FRAME_SIZE * j
            
            # Ensure we don't go out of bounds
            if i + FRAME_SIZE > len(frame):
                logger.warning(f"Frame index {i} would exceed frame length {len(frame)}")
                break

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

        # Validate code values and clamp them to valid range if needed
        for i, code in enumerate(codes):
            if torch.any(code < 0) or torch.any(code > CODES_MAX_VALUE):
                logger.warning(f"Invalid code values detected in codebook {i}: values must be between 0 and {CODES_MAX_VALUE}")
                # Clamp values to valid range instead of returning None
                codes[i] = torch.clamp(code, 0, CODES_MAX_VALUE)
                
        # Validate code shapes for SNAC model
        expected_shapes = [
            (1, codes_0.shape[0]),
            (1, codes_1.shape[0]),
            (1, codes_2.shape[0])
        ]
        
        for i, (code, expected_shape) in enumerate(zip(codes, expected_shapes)):
            if code.shape != expected_shape:
                logger.warning(f"Invalid code shape for codebook {i}: expected {expected_shape}, got {code.shape}")
                # Try to reshape if possible, otherwise return None
                if code.numel() == expected_shape[0] * expected_shape[1]:
                    codes[i] = code.reshape(expected_shape)
                else:
                    logger.error(f"Cannot reshape code {i} to expected shape {expected_shape}")
                    return None

        # Decode audio using the SNAC model with error handling
        try:
            audio_hat = snac_model.decode(codes)
        except RuntimeError as e:
            if "CUDA" in str(e) or "index" in str(e).lower() or "out of bounds" in str(e).lower():
                logger.error(f"CUDA error during audio decoding: {str(e)}")
                # Return empty audio instead of failing
                return b''
            else:
                # Re-raise other errors
                raise

        # Process the audio output safely
        if audio_hat is None or audio_hat.shape[2] < 4096:
            logger.warning("Audio output is None or too small")
            return b''
            
        # Extract the relevant portion of audio safely
        try:
            audio_slice = audio_hat[:, :, 2048:4096]  # Extract the relevant portion of audio
            detached_audio = audio_slice.detach().cpu()
            audio_np = detached_audio.numpy()
            audio_int16 = (audio_np * 32767).astype(np.int16)  # Convert to 16-bit PCM
            audio_bytes = audio_int16.tobytes()
            return audio_bytes
        except Exception as e:
            logger.error(f"Error processing audio output: {str(e)}")
            return b''

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
    consecutive_errors = 0
    max_consecutive_errors = 3  # Maximum number of consecutive errors before yielding silence

    try:
        async for token_sim in token_gen:
            # Convert token string to ID
            token = turn_token_into_id(token_sim, count)

            if token is not None and token >= 0:  # Changed from > 0 to >= 0 to include 0 as valid
                # Add valid token to buffer
                buffer.append(token)
                count += 1

                # Limit buffer size to prevent memory issues
                if len(buffer) > max_buffer_size:
                    logger.warning(f"Buffer exceeded max size ({max_buffer_size}). Trimming to prevent memory issues.")
                    # Keep a multiple of FRAME_SIZE tokens to maintain frame alignment
                    keep_tokens = (max_buffer_size // FRAME_SIZE) * FRAME_SIZE
                    buffer = buffer[-keep_tokens:]

                # Process buffer when we have enough tokens and at frame boundaries
                if count % FRAME_SIZE == 0 and count > MIN_BUFFER_SIZE - 1:
                    # Calculate how many complete frames we can process
                    num_frames = len(buffer) // FRAME_SIZE
                    # Use at least MIN_BUFFER_SIZE tokens, but always in complete frames
                    tokens_to_use = max(MIN_BUFFER_SIZE, num_frames * FRAME_SIZE)
                    # Ensure tokens_to_use is a multiple of FRAME_SIZE
                    tokens_to_use = (tokens_to_use // FRAME_SIZE) * FRAME_SIZE
                    # Use the most recent complete frames
                    buffer_to_proc = buffer[-tokens_to_use:]

                    # Convert tokens to audio
                    audio_samples = convert_to_audio(buffer_to_proc, count)

                    if audio_samples is not None and len(audio_samples) > 0:
                        consecutive_errors = 0  # Reset error counter on success
                        yield audio_samples
                    else:
                        consecutive_errors += 1
                        logger.warning(f"Failed to generate audio (attempt {consecutive_errors}/{max_consecutive_errors})")
                        
                        if consecutive_errors >= max_consecutive_errors:
                            # Generate silence as a fallback after multiple failures
                            logger.warning("Generating silence as fallback after multiple failures")
                            # Generate 0.1 seconds of silence at 24kHz (16-bit)
                            silence_samples = b'\x00\x00' * 2400
                            yield silence_samples
                            consecutive_errors = 0  # Reset after yielding silence
            elif token is not None:
                logger.warning(f"Invalid token value: {token} (must be >= 0)")
    except Exception as e:
        logger.error(f"Error in tokens_decoder: {str(e)}")
        # Yield silence as a fallback when an exception occurs
        silence_samples = b'\x00\x00' * 2400
        yield silence_samples
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
    stall_timeout = timeout // 2  # Timeout for detecting stalled generation

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
                if audio_chunk is not None and len(audio_chunk) > 0:
                    audio_queue.put(audio_chunk)
                else:
                    # If we get empty audio, put a small silence chunk
                    audio_queue.put(b'\x00\x00' * 1200)  # 0.05 seconds of silence
        except Exception as e:
            # Put the exception in the error queue to propagate it to the main thread
            error_queue.put(e)
            # Put a silence chunk to prevent blocking
            audio_queue.put(b'\x00\x00' * 2400)  # 0.1 seconds of silence
        finally:
            # Always signal completion
            audio_queue.put(None)  # Sentinel

    def run_async():
        try:
            asyncio.run(async_producer())
        except Exception as e:
            # Catch any exceptions not caught in async_producer
            error_queue.put(e)
            # Ensure the main thread doesn't hang
            audio_queue.put(b'\x00\x00' * 2400)  # 0.1 seconds of silence
            audio_queue.put(None)  # Sentinel

    # Start the async processing in a separate thread
    thread = threading.Thread(target=run_async)
    thread.daemon = True  # Make thread daemon so it doesn't block program exit
    thread.start()

    try:
        # Yield audio chunks as they become available
        last_audio_time = time.time()
        consecutive_timeouts = 0
        max_consecutive_timeouts = 3
        
        while True:
            # Check for errors from the async thread
            if not error_queue.empty():
                exception = error_queue.get()
                logger.error(f"Error in async processing thread: {str(exception)}")
                # Continue with silence instead of raising the exception
                yield b'\x00\x00' * 4800  # 0.2 seconds of silence
                consecutive_timeouts = 0
                last_audio_time = time.time()
                continue

            # Get the next audio chunk with timeout
            try:
                audio = audio_queue.get(timeout=timeout)
                if audio is None:
                    # End of generation
                    break
                    
                # Reset counters on successful audio retrieval
                consecutive_timeouts = 0
                last_audio_time = time.time()
                
                # Yield the audio chunk
                yield audio
                
            except queue.Empty:
                consecutive_timeouts += 1
                current_time = time.time()
                stall_duration = current_time - last_audio_time
                
                logger.warning(f"Timeout ({timeout}s) waiting for audio chunks (attempt {consecutive_timeouts}/{max_consecutive_timeouts})")
                
                if stall_duration > stall_timeout:
                    logger.warning(f"Audio generation stalled for {stall_duration:.1f}s, yielding silence")
                    # Yield silence to maintain audio flow
                    yield b'\x00\x00' * 4800  # 0.2 seconds of silence
                    last_audio_time = current_time
                
                if consecutive_timeouts >= max_consecutive_timeouts:
                    logger.error("Too many consecutive timeouts, ending generation")
                    break
    finally:
        # Clean up resources
        if thread.is_alive():
            logger.info("Waiting for async processing thread to complete")
            thread.join(timeout=5)  # Wait for thread to finish with timeout