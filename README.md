# Orpheus Speech Models

This package provides a Python interface to the Orpheus Text-to-Speech models, enabling high-quality speech synthesis from text.

## Features

- High-quality text-to-speech conversion
- Multiple voice options
- Cross-platform support (Windows, Linux, macOS)
- Streaming audio generation
- Adjustable generation parameters

## System Requirements

- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **GPU**: CUDA-compatible GPU with 6GB+ VRAM recommended for optimal performance
- **Disk Space**: ~5GB for model files

## Installation

### Basic Installation

```bash
pip install git+https://github.com/Deathdadev/Orpheus-Speech-PyPi
pip install accelerate
```

### With Specific CUDA Version

If you need a specific CUDA version, use the extra-index-url parameter:

```bash
pip install git+https://github.com/Deathdadev/Orpheus-Speech-PyPi --extra-index-url https://download.pytorch.org/whl/cu124
pip install accelerate
```

## Platform Support

This package supports multiple platforms with optimized backends:

- **Windows**: Uses the Hugging Face Transformers library for model inference
- **Linux/macOS**: Uses vllm for optimized inference when available, falls back to Transformers if not

The appropriate backend is automatically selected based on your operating system and available hardware.

## Usage

### Basic Usage

```python
from orpheus_tts import OrpheusModel

# Initialize the model
model = OrpheusModel("medium-3b")

# Generate speech from text
audio_chunks = model.generate_speech(
    prompt="Hello, this is a test of the Orpheus text-to-speech system.",
    voice="zoe"
)

# Process the audio chunks (e.g., save to file, play, etc.)
for chunk in audio_chunks:
    # Process each audio chunk
    pass
```

### Saving Audio to a File

```python
import wave
import io
from orpheus_tts import OrpheusModel

def save_audio_chunks_to_wav(audio_chunks, output_file, sample_rate=24000):
    with wave.open(output_file, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono audio
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(sample_rate)

        # Concatenate all audio chunks
        for chunk in audio_chunks:
            wav_file.writeframes(chunk)

# Initialize the model
model = OrpheusModel("medium-3b")

# Generate speech
audio_chunks = list(model.generate_speech(
    prompt="Hello, this is a test of the Orpheus text-to-speech system.",
    voice="zoe"
))

# Save to file
save_audio_chunks_to_wav(audio_chunks, "output.wav")
```

### Available Voices

The following voices are currently available:

- `zoe` - Female voice
- `zac` - Male voice
- `jess` - Female voice
- `leo` - Male voice
- `mia` - Female voice
- `julia` - Female voice
- `leah` - Female voice

### Advanced Parameters

```python
audio_chunks = model.generate_speech(
    prompt="Hello, this is a test of the Orpheus text-to-speech system.",
    voice="zoe",
    temperature=0.7,  # Controls randomness (higher = more random)
    top_p=0.9,        # Controls diversity
    max_tokens=1500,  # Maximum number of tokens to generate
    repetition_penalty=1.2,  # Penalizes repetition
    max_buffer_size=1000,  # Maximum buffer size for token processing
    timeout=60  # Timeout in seconds for audio generation
)
```

### Resource Management

To properly free up GPU memory and other resources when you're done with the model:

```python
from orpheus_tts import OrpheusModel

# Initialize the model
model = OrpheusModel("medium-3b")

try:
    # Use the model
    audio_chunks = list(model.generate_speech(
        prompt="Hello, this is a test of the Orpheus text-to-speech system.",
        voice="zoe"
    ))

    # Process audio chunks...

finally:
    # Clean up resources when done
    model.cleanup()
```

## Roadmap

- [x] Implement for Windows using Transformers
- [x] Implement for Linux/macOS using vllm
- [ ] Add support for smaller models (nano-150m, micro-400m, small-1b)
- [ ] Create a VLLM wheel for Windows to improve speed: *via* https://github.com/vllm-project/vllm/pull/14891
- [ ] Add more voice options
- [ ] Improve documentation and examples

## Troubleshooting

### CUDA Issues

If you encounter CUDA-related errors:

1. Ensure you have compatible NVIDIA drivers installed
2. Try installing with a specific CUDA version as shown in the installation section
3. If CUDA is unavailable, the model will fall back to CPU (which will be significantly slower)

### Memory Issues

If you encounter out-of-memory errors:

1. Try using a machine with more RAM/VRAM
2. Reduce the `max_tokens` parameter
3. Consider using a smaller model when they become available

## License

MIT License

## Acknowledgements

For more information, see the [Orpheus TTS GitHub repository](https://github.com/canopyai/Orpheus-TTS-0.1).
