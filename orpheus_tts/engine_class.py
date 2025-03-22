import asyncio
import torch
import platform
import threading
import queue
import logging
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from .decoder import tokens_decoder_sync

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Only import vllm if not on Windows
if platform.system() != "Windows":
    try:
        from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
        logger.info("Successfully imported vllm for optimized inference")
    except ImportError as e:
        logger.warning(f"Failed to import vllm: {str(e)}. Will fall back to transformers if needed.")

class OrpheusModel:
    def __init__(self, model_name, dtype=torch.bfloat16):
        self.model_name = self._map_model_params(model_name)
        self.dtype = dtype
        self.platform = platform.system()  # Set the platform attribute
        self.engine = self._setup_engine()
        self.available_voices = ["tara", "jess", "leo", "leah", "dan", "mia", "zac", "zoe"]
        self.tokeniser = AutoTokenizer.from_pretrained(model_name)


    def _map_model_params(self, model_name):
        """Map model name to repository ID and validate model compatibility.

        Args:
            model_name (str): Name of the model to use

        Returns:
            str: Repository ID for the model

        Raises:
            ValueError: If the model is not supported or invalid
        """
        # Define supported models and their repository IDs
        model_map = {
            # Currently only medium-3b is supported
            "medium-3b": {
                "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
                "min_ram_gb": 8,
                "min_vram_gb": 6,
            },
            # Future models (commented out until available)
            # "nano-150m": {
            #     "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
            #     "min_ram_gb": 4,
            #     "min_vram_gb": 2,
            # },
            # "micro-400m": {
            #     "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
            #     "min_ram_gb": 4,
            #     "min_vram_gb": 2,
            # },
            # "small-1b": {
            #     "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
            #     "min_ram_gb": 6,
            #     "min_vram_gb": 4,
            # },
        }

        # List of models that are recognized but not yet supported
        unsupported_models = ["nano-150m", "micro-400m", "small-1b"]

        # Check if model is in the unsupported list
        if model_name in unsupported_models:
            raise ValueError(
                f"Model '{model_name}' is not yet supported. Currently only 'medium-3b' is supported. "
                f"The smaller models (nano, micro, small) will be released soon."
            )

        # Check if model is in the supported list
        elif model_name in model_map:
            logger.info(f"Using predefined model: {model_name}")
            return model_map[model_name]["repo_id"]

        # If it's a custom model path or Hugging Face model ID
        else:
            # Check if it's a local path that exists
            if os.path.exists(model_name):
                logger.info(f"Using local model at: {model_name}")
                return model_name

            # Assume it's a Hugging Face model ID
            logger.info(f"Using custom Hugging Face model: {model_name}")
            return model_name

    def validate_voice(self, voice):
        """Validate that the requested voice is available.

        Args:
            voice (str or None): Voice to validate, or None for default voice

        Raises:
            ValueError: If the voice is not available for this model
        """
        if voice is None:
            # No voice specified, using default
            return

        if not isinstance(voice, str):
            raise ValueError(f"Voice must be a string, got {type(voice).__name__}")

        if voice not in self.available_voices:
            available_voices_str = ", ".join(f"'{v}'" for v in self.available_voices)
            raise ValueError(
                f"Voice '{voice}' is not available for model '{self.model_name}'. "
                f"Available voices are: {available_voices_str}"
            )

    def _format_prompt(self, prompt, voice="tara", model_type="larger"):
        if model_type == "smaller":
            if voice:
                return f"<custom_token_3>{prompt}[{voice}]<custom_token_4><custom_token_5>"
            else:
                return f"<custom_token_3>{prompt}<custom_token_4><custom_token_5>"
        else:
            if voice:
                adapted_prompt = f"{voice}: {prompt}"
                prompt_tokens = self.tokeniser(adapted_prompt, return_tensors="pt")
                start_token = torch.tensor([[ 128259]], dtype=torch.int64)
                end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
                all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
                prompt_string = self.tokeniser.decode(all_input_ids[0])
                return prompt_string
            else:
                prompt_tokens = self.tokeniser(prompt, return_tensors="pt")
                start_token = torch.tensor([[ 128259]], dtype=torch.int64)
                end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
                all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
                prompt_string = self.tokeniser.decode(all_input_ids[0])
                return prompt_string

    def _setup_engine(self):
        """Set up the appropriate engine based on the platform.

        Returns:
            object or None: Engine instance for non-Windows platforms, None for Windows

        Raises:
            RuntimeError: If engine setup fails
        """
        try:
            if self.platform == "Windows":
                # For Windows, use transformers directly
                logger.info(f"Setting up transformers engine for {self.model_name}")

                # Check if CUDA is available
                if torch.cuda.is_available():
                    logger.info(f"Using CUDA for model inference")
                else:
                    logger.warning("CUDA not available. Model inference will be slow on CPU.")

                # Load the model
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.dtype,
                    device_map="auto",  # Let transformers decide the best device mapping
                    low_cpu_mem_usage=True  # Reduce memory usage during loading
                )

                return None  # No separate engine needed for transformers
            else:
                # For non-Windows, use vllm if available
                if 'AsyncLLMEngine' not in globals():
                    logger.warning("vllm not available. Falling back to transformers.")
                    # Fall back to transformers
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=self.dtype,
                        device_map="auto",
                        low_cpu_mem_usage=True
                    )
                    return None

                # Use vllm for optimized inference
                logger.info(f"Setting up vllm engine for {self.model_name}")

                # Configure vllm engine
                engine_args = AsyncEngineArgs(
                    model=self.model_name,
                    dtype=str(self.dtype).split('.')[-1],
                    gpu_memory_utilization=0.8,
                    tensor_parallel_size=1  # Use a single GPU by default
                )

                return AsyncLLMEngine.from_engine_args(engine_args)

        except Exception as e:
            error_msg = f"Failed to set up engine: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e


    def generate_tokens_sync(self, prompt, voice=None, request_id="req-001", temperature=0.6, top_p=0.8, max_tokens=1200, stop_token_ids=[49158], repetition_penalty=1.3):
        """Generate tokens synchronously using the appropriate backend.

        Args:
            prompt (str): Text to convert to speech
            voice (str, optional): Voice to use for speech generation
            request_id (str, optional): Identifier for this generation request
            temperature (float, optional): Sampling temperature (higher = more random)
            top_p (float, optional): Nucleus sampling parameter
            max_tokens (int, optional): Maximum number of tokens to generate
            stop_token_ids (list, optional): Token IDs that signal the end of generation
            repetition_penalty (float, optional): Penalty for repeating tokens

        Returns:
            generator: Generator yielding tokens as they're produced

        Raises:
            ValueError: If voice validation fails
            RuntimeError: If token generation fails
        """
        try:
            # Validate the voice parameter
            self.validate_voice(voice)

            # Format the prompt with the selected voice
            prompt_string = self._format_prompt(prompt, voice)
            logger.info(f"Generating speech for prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")

            # Use different generation methods based on platform and available engines
            if hasattr(self, 'model') and self.model is not None:
                # Use transformers-based generation
                logger.info("Using transformers for token generation")
                return self._generate_tokens_transformers(
                    prompt_string,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stop_token_ids=stop_token_ids,
                    repetition_penalty=repetition_penalty
                )
            elif hasattr(self, 'engine') and self.engine is not None:
                # Use vllm-based generation
                logger.info("Using vllm for token generation")
                return self._generate_tokens_vllm(
                    prompt_string,
                    request_id=request_id,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stop_token_ids=stop_token_ids,
                    repetition_penalty=repetition_penalty
                )
            else:
                # No valid engine available
                raise RuntimeError("No valid generation engine available. Model initialization may have failed.")

        except Exception as e:
            logger.error(f"Error in token generation: {str(e)}")
            raise

    def _generate_tokens_transformers(self, prompt_string, temperature=0.6, top_p=0.8, max_tokens=1200, stop_token_ids=[49158], repetition_penalty=1.3):
        """Generate tokens using transformers for Windows compatibility"""
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError("Transformers model is not available. Make sure you're running on Windows or initialize the model manually.")

        token_queue = queue.Queue()

        def generate_tokens():
            # Encode the prompt
            inputs = self.tokeniser(prompt_string, return_tensors="pt").to(self.model.device)
            input_length = inputs.input_ids.shape[1]

            # Setup generation parameters
            gen_config = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "do_sample": True,
                "eos_token_id": stop_token_ids,
            }

            # Generate tokens one by one
            generated = inputs.input_ids
            past_key_values = None

            for _ in range(max_tokens):
                with torch.no_grad():
                    if past_key_values is None:
                        outputs = self.model(**inputs, return_dict=True, use_cache=True)
                    else:
                        outputs = self.model(
                            input_ids=generated[:, -1:],
                            past_key_values=past_key_values,
                            return_dict=True,
                            use_cache=True
                        )

                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :]

                # Apply temperature
                next_token_logits = next_token_logits / temperature

                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for seq_idx in range(generated.shape[0]):
                        for token_idx in set(generated[seq_idx].tolist()):
                            next_token_logits[seq_idx, token_idx] /= repetition_penalty

                # Apply top-p sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    for batch_idx in range(next_token_logits.shape[0]):
                        indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                        next_token_logits[batch_idx, indices_to_remove] = -float('Inf')

                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Check if we've reached a stop token
                if next_token.item() in stop_token_ids:
                    break

                # Add the token to the generated sequence
                generated = torch.cat((generated, next_token), dim=1)

                # Decode the new token and put it in the queue
                new_token = self.tokeniser.decode(next_token[0], skip_special_tokens=False)
                token_queue.put(new_token)

            token_queue.put(None)  # Signal completion

        # Start generation in a separate thread
        thread = threading.Thread(target=generate_tokens)
        thread.start()

        # Yield tokens as they become available
        while True:
            token = token_queue.get()
            if token is None:
                break
            yield token

        thread.join()

    def _generate_tokens_vllm(self, prompt_string, request_id="req-001", temperature=0.6, top_p=0.8, max_tokens=1200, stop_token_ids=[49158], repetition_penalty=1.3):
        """Generate tokens using vllm for non-Windows platforms"""
        if self.engine is None:
            raise RuntimeError("vllm engine is not available on this platform. Use _generate_tokens_transformers instead.")

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids,
            repetition_penalty=repetition_penalty,
        )

        token_queue = queue.Queue()

        async def async_producer():
            async for result in self.engine.generate(prompt=prompt_string, sampling_params=sampling_params, request_id=request_id):
                # Place each token text into the queue.
                token_queue.put(result.outputs[0].text)
            token_queue.put(None)  # Sentinel to indicate completion.

        def run_async():
            asyncio.run(async_producer())

        thread = threading.Thread(target=run_async)
        thread.start()

        while True:
            token = token_queue.get()
            if token is None:
                break
            yield token

        thread.join()

    def generate_speech(self, **kwargs):
        """Generate speech audio from text.

        This is the main method to use for text-to-speech generation.
        It handles the full pipeline from text to audio chunks.

        Args:
            **kwargs: All arguments are passed to generate_tokens_sync
                prompt (str): Text to convert to speech
                voice (str, optional): Voice to use for speech generation
                temperature (float, optional): Sampling temperature
                top_p (float, optional): Nucleus sampling parameter
                max_tokens (int, optional): Maximum number of tokens to generate
                repetition_penalty (float, optional): Penalty for repeating tokens

        Returns:
            generator: Generator yielding audio chunks as they're produced

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If speech generation fails
        """
        try:
            # Get token generation parameters with defaults
            max_buffer_size = kwargs.pop('max_buffer_size', 1000)
            timeout = kwargs.pop('timeout', 60)

            # Generate tokens and convert to audio
            token_generator = self.generate_tokens_sync(**kwargs)
            return tokens_decoder_sync(
                token_generator,
                max_buffer_size=max_buffer_size,
                timeout=timeout
            )
        except Exception as e:
            logger.error(f"Error in speech generation: {str(e)}")
            raise

    def cleanup(self):
        """Clean up resources used by the model.

        This method should be called when the model is no longer needed to free up
        memory and GPU resources. It's especially important for CUDA resources.
        """
        try:
            # Clean up transformers model if it exists
            if hasattr(self, 'model') and self.model is not None:
                logger.info("Cleaning up transformers model resources")
                if hasattr(self.model, 'to') and torch.cuda.is_available():
                    # Move model to CPU to free up CUDA memory
                    self.model = self.model.cpu()
                    torch.cuda.empty_cache()
                self.model = None

            # Clean up vllm engine if it exists
            if hasattr(self, 'engine') and self.engine is not None:
                logger.info("Cleaning up vllm engine resources")
                # vllm doesn't have an explicit cleanup method, so we just remove the reference
                self.engine = None

            # Clean up tokenizer
            if hasattr(self, 'tokeniser'):
                self.tokeniser = None

            # Force garbage collection
            import gc
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Resource cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def __del__(self):
        """Destructor to ensure resources are cleaned up."""
        try:
            self.cleanup()
        except Exception as e:
            # Can't use logger here as it might be gone already
            print(f"Error during automatic cleanup: {str(e)}")
