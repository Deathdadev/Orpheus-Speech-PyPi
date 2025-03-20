# Orpheus Speech Models

This package provides a Python interface to the Orpheus Text-to-Speech models.

## Installation

```bash
pip install orpheus-speech
```

## Platform Support

This package supports both Windows and non-Windows platforms:

- **Windows**: Uses the Hugging Face Transformers library for model inference
- **Linux/macOS**: Uses vllm for optimized inference

The appropriate backend is automatically selected based on your operating system.

## Usage

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

For more information, see the [Orpheus TTS GitHub repository](https://github.com/canopyai/Orpheus-TTS-0.1).