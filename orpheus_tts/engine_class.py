import asyncio
import torch
import platform
import threading
import queue
from transformers import AutoTokenizer, AutoModelForCausalLM
from .decoder import tokens_decoder_sync

# Only import vllm if not on Windows
if platform.system() != "Windows":
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

class OrpheusModel:
    def __init__(self, model_name, dtype=torch.bfloat16):
        self.model_name = self._map_model_params(model_name)
        self.dtype = dtype
        self.platform = platform.system()  # Set the platform attribute
        self.engine = self._setup_engine()
        self.available_voices = ["zoe", "zac","jess", "leo", "mia", "julia", "leah"]
        self.tokeniser = AutoTokenizer.from_pretrained(model_name)

    
    def _map_model_params(self, model_name):
        model_map = {
            # "nano-150m":{
            #     "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
            # }, 
            # "micro-400m":{
            #     "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
            # }, 
            # "small-1b":{
            #     "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
            # },
            "medium-3b":{
                "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
            },
        }
        unsupported_models = ["nano-150m", "micro-400m", "small-1b"]
        if (model_name  in unsupported_models):
            raise ValueError(f"Model {model_name} is not supported. Only medium-3b is supported, small, micro and nano models will be released very soon")
        elif model_name in model_map:
            return model_map[model_name]["repo_id"]
        else:
            return model_name
        
    def validate_voice(self, voice):
        if voice:
            if voice not in self.available_voices:
                raise ValueError(f"Voice {voice} is not available for model {self.model_name}")
    
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
        """Set up the appropriate engine based on the platform."""
        if self.platform == "Windows":
            # For Windows, use transformers directly
            print(f"Setting up transformers engine for {self.model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map="auto"
            )
            return None  # No separate engine needed for transformers
        else:
            # For non-Windows, use vllm
            print(f"Setting up vllm engine for {self.model_name}")
            engine_args = AsyncEngineArgs(
                model=self.model_name,
                dtype=str(self.dtype).split('.')[-1],
                gpu_memory_utilization=0.8,
            )
            return AsyncLLMEngine.from_engine_args(engine_args)


    def generate_tokens_sync(self, prompt, voice=None, request_id="req-001", temperature=0.6, top_p=0.8, max_tokens=1200, stop_token_ids=[49158], repetition_penalty=1.3):
        prompt_string = self._format_prompt(prompt, voice)
        print(prompt)
        
        # Use different generation methods based on platform
        if self.platform == "Windows":
            return self._generate_tokens_transformers(
                prompt_string, 
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop_token_ids=stop_token_ids,
                repetition_penalty=repetition_penalty
            )
        else:
            return self._generate_tokens_vllm(
                prompt_string, 
                request_id=request_id,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop_token_ids=stop_token_ids,
                repetition_penalty=repetition_penalty
            )
    
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
        return tokens_decoder_sync(self.generate_tokens_sync(**kwargs))