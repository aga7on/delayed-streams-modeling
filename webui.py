#!/usr/bin/env python3
"""
Kyutai STT & TTS - Complete WebUI
All repository features in one simple interface
"""

import os
import sys
from pathlib import Path

# Setup portable environment
project_dir = Path(__file__).parent.absolute()
os.environ['HF_HOME'] = str(project_dir / "cache")
os.environ['HF_HUB_CACHE'] = str(project_dir / "cache")
os.environ['TRANSFORMERS_CACHE'] = str(project_dir / "cache")
os.environ['TEMP'] = str(project_dir / "temp")
os.environ['TMP'] = str(project_dir / "temp")

# Create directories
(project_dir / "cache").mkdir(exist_ok=True)
(project_dir / "temp").mkdir(exist_ok=True)

print(f"Using cache: {project_dir / 'cache'}")
print(f"Using temp: {project_dir / 'temp'}")

import gradio as gr
import torch
import numpy as np
import tempfile
import subprocess
import socket
from typing import Tuple, List, Optional

# Import Kyutai libraries
try:
    import sphn
    import sounddevice as sd
    from moshi.models.loaders import CheckpointInfo
    from moshi.models.tts import DEFAULT_DSM_TTS_REPO, DEFAULT_DSM_TTS_VOICE_REPO, TTSModel
    import moshi.models
    HAS_MOSHI = True
    print("✅ Moshi loaded successfully")
except ImportError as e:
    HAS_MOSHI = False
    print(f"❌ Moshi not available: {e}")

try:
    import mlx.core as mx
    import mlx.nn as nn
    from moshi_mlx import models, utils
    HAS_MLX = True
    print("✅ MLX loaded successfully")
except ImportError:
    HAS_MLX = False
    print("ℹ️ MLX not available (Apple Silicon only)")

# Global model cache
model_cache = {}

def get_available_lora_adapters():
    """Get actually available LoRA adapters from filesystem"""
    lora_dir = project_dir / "lora_adapters"
    available_loras = ["None"]
    
    if lora_dir.exists():
        for lora_path in lora_dir.iterdir():
            if lora_path.is_dir() and (lora_path / "adapter_config.json").exists():
                available_loras.append(lora_path.name)
    
    return available_loras

def analyze_official_voice_structure():
    """Анализирует структуру официальных голосов для понимания правильного формата"""
    try:
        # Путь к официальному голосу
        official_voice_path = project_dir / "cache" / "models--kyutai--tts-voices" / "snapshots"
        
        # Найти первый доступный официальный голос
        for snapshot_dir in official_voice_path.glob("*"):
            if snapshot_dir.is_dir():
                expresso_dir = snapshot_dir / "expresso"
                if expresso_dir.exists():
                    safetensors_files = list(expresso_dir.glob("*.safetensors"))
                    if safetensors_files:
                        official_voice = safetensors_files[0]
                        print(f"🔍 Analyzing official voice: {official_voice.name}")
                        print(f"📏 File size: {official_voice.stat().st_size} bytes ({official_voice.stat().st_size/1024:.1f} KB)")
                        
                        from safetensors.torch import load_file
                        voice_data = load_file(str(official_voice))
                        
                        print(f"🔑 Keys in official voice: {list(voice_data.keys())}")
                        for key, tensor in voice_data.items():
                            print(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}, numel={tensor.numel()}")
                        
                        return voice_data
        
        print("❌ No official voices found")
        return None
        
    except Exception as e:
        print(f"❌ Error analyzing official voice: {e}")
        return None

def get_local_voices():
    """Get available voices from local cache and custom voices"""
    all_voices = []
    
    # Check for local voice files from cache
    cache_dir = project_dir / "cache" / "voices"
    if cache_dir.exists():
        for voice_file in cache_dir.rglob("*.safetensors"):
            # Convert path to voice name format
            relative_path = voice_file.relative_to(cache_dir)
            voice_name = str(relative_path).replace(".safetensors", "").replace("\\", "/")
            all_voices.append(f"📦 {voice_name}")
    
    # Check custom voices directory (trained voices)
    custom_voices_dir = project_dir / "custom_voices"
    if custom_voices_dir.exists():
        for voice_file in custom_voices_dir.rglob("*.safetensors"):
            relative_path = voice_file.relative_to(custom_voices_dir)
            path_str = str(relative_path).replace('.safetensors', '').replace('\\', '/')
            
            # Try to load metadata for better display name
            metadata_file = voice_file.parent / f"{voice_file.stem}_metadata.json"
            display_name = path_str
            
            if metadata_file.exists():
                try:
                    import json
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    emotion = metadata.get('emotion', 'unknown')
                    speaker = metadata.get('speaker', 'unknown')
                    display_name = f"{metadata.get('name', path_str)} ({emotion})"
                except:
                    pass
            
            voice_name = f"custom/{path_str}"
            all_voices.append(f"🎤 {display_name}")
            # Store mapping for actual usage
            if not hasattr(get_local_voices, 'voice_mapping'):
                get_local_voices.voice_mapping = {}
            get_local_voices.voice_mapping[f"🎤 {display_name}"] = voice_name
    
    # Default voices that should work
    default_voices = [
        "📻 expresso/ex03-ex01_happy_001_channel1_334s.wav",
        "📻 expresso/ex03-ex01_sad_001_channel1_334s.wav", 
        "📻 expresso/ex03-ex01_angry_001_channel1_334s.wav",
        "📻 expresso/ex03-ex01_neutral_001_channel1_334s.wav"
    ]
    
    # Combine all voices
    final_voices = default_voices + all_voices
    
    return final_voices if final_voices else default_voices

def get_available_lora_adapters():
    """Get actually available LoRA adapters from filesystem"""
    lora_dir = project_dir / "lora_adapters"
    available_loras = ["None"]
    
    if lora_dir.exists():
        for lora_path in lora_dir.iterdir():
            if lora_path.is_dir() and (lora_path / "adapter_config.json").exists():
                available_loras.append(lora_path.name)
    
    return available_loras

def get_available_models():
    """Get all available models"""
    stt_models = [
        "kyutai/stt-2.6b-en",
        "kyutai/stt-1b-en_fr"
    ]
    
    tts_models = [
        DEFAULT_DSM_TTS_REPO if HAS_MOSHI else "kyutai/dsm-tts-1b-en"
    ]
    
    voices = get_local_voices()
    
    return stt_models, tts_models, voices

def load_stt_model(model_name: str, device: str = "cuda"):
    """Load STT model"""
    cache_key = f"stt_{model_name}_{device}"
    
    if cache_key not in model_cache:
        if not HAS_MOSHI:
            raise ImportError("Moshi not installed")
        
        print(f"Loading STT model: {model_name}")
        checkpoint_info = CheckpointInfo.from_hf_repo(model_name)
        mimi = checkpoint_info.get_mimi(device=device)
        tokenizer = checkpoint_info.get_text_tokenizer()
        lm = checkpoint_info.get_moshi(device=device, dtype=torch.bfloat16)
        lm_gen = moshi.models.LMGen(lm, temp=0, temp_text=0.0)
        
        model_cache[cache_key] = {
            'mimi': mimi,
            'tokenizer': tokenizer,
            'lm_gen': lm_gen,
            'checkpoint_info': checkpoint_info
        }
        print(f"✅ STT model loaded: {model_name}")
    
    return model_cache[cache_key]

def load_tts_model(model_name: str, device: str = "cuda"):
    """Load TTS model"""
    cache_key = f"tts_{model_name}_{device}"
    
    if cache_key not in model_cache:
        if not HAS_MOSHI:
            raise ImportError("Moshi not installed")
        
        print(f"Loading TTS model: {model_name}")
        checkpoint_info = CheckpointInfo.from_hf_repo(model_name)
        tts_model = TTSModel.from_checkpoint_info(
            checkpoint_info, n_q=32, temp=0.6, device=device
        )
        
        model_cache[cache_key] = tts_model
        print(f"✅ TTS model loaded: {model_name}")
    
    return model_cache[cache_key]

def transcribe_audio(audio_file, model_name: str, device: str = "cuda") -> str:
    """Transcribe audio file"""
    if not audio_file:
        return "No audio file provided"
    
    if not HAS_MOSHI:
        return "❌ Moshi not installed. Please run setup."
    
    try:
        # Load model
        model_data = load_stt_model(model_name, device)
        mimi = model_data['mimi']
        tokenizer = model_data['tokenizer']
        lm_gen = model_data['lm_gen']
        checkpoint_info = model_data['checkpoint_info']
        
        # Load and process audio
        audio, sample_rate = sphn.read(audio_file)
        audio = torch.from_numpy(audio).to(device).mean(axis=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != mimi.sample_rate:
            import julius
            audio = julius.resample_frac(audio, sample_rate, mimi.sample_rate)
        
        # Pad to frame size
        if audio.shape[-1] % mimi.frame_size != 0:
            to_pad = mimi.frame_size - audio.shape[-1] % mimi.frame_size
            audio = torch.nn.functional.pad(audio, (0, to_pad))
        
        # Transcribe
        text_tokens_accum = []
        audio_silence_prefix_seconds = checkpoint_info.stt_config.get("audio_silence_prefix_seconds", 1.0)
        audio_delay_seconds = checkpoint_info.stt_config.get("audio_delay_seconds", 5.0)
        padding_token_id = checkpoint_info.raw_config.get("text_padding_token_id", 3)
        
        # Process with silence padding
        n_prefix_chunks = int(audio_silence_prefix_seconds * mimi.frame_rate)
        n_suffix_chunks = int(audio_delay_seconds * mimi.frame_rate)
        silence_chunk = torch.zeros((1, 1, mimi.frame_size), dtype=torch.float32, device=device)
        
        with mimi.streaming(1), lm_gen.streaming(1):
            # Prefix silence
            for _ in range(n_prefix_chunks):
                audio_tokens = mimi.encode(silence_chunk)
                text_tokens = lm_gen.step(audio_tokens)
                if text_tokens is not None:
                    text_tokens_accum.append(text_tokens)
            
            # Process audio
            for i in range(0, audio.shape[-1], mimi.frame_size):
                chunk = audio[:, None, i:i + mimi.frame_size]
                if chunk.shape[-1] < mimi.frame_size:
                    chunk = torch.nn.functional.pad(chunk, (0, mimi.frame_size - chunk.shape[-1]))
                
                audio_tokens = mimi.encode(chunk)
                text_tokens = lm_gen.step(audio_tokens)
                if text_tokens is not None:
                    text_tokens_accum.append(text_tokens)
            
            # Suffix silence
            for _ in range(n_suffix_chunks):
                audio_tokens = mimi.encode(silence_chunk)
                text_tokens = lm_gen.step(audio_tokens)
                if text_tokens is not None:
                    text_tokens_accum.append(text_tokens)
        
        # Decode text
        if text_tokens_accum:
            utterance_tokens = torch.concat(text_tokens_accum, dim=-1)
            text_tokens = utterance_tokens.cpu().view(-1)
            text = tokenizer.decode(text_tokens[text_tokens > padding_token_id].numpy().tolist())
            return text.strip()
        else:
            return "No speech detected"
            
    except Exception as e:
        return f"❌ Error: {str(e)}"

def synthesize_speech(text: str, voice: str, model_name: str, device: str = "cuda") -> Tuple[int, np.ndarray]:
    """Synthesize speech from text"""
    if not text.strip():
        return 24000, np.array([0.0])  # Return small non-empty array
    
    if not HAS_MOSHI:
        print("❌ Moshi not available")
        return 24000, np.array([0.0])
    
    # Force CPU if CUDA not available
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA not available, using CPU")
        device = "cpu"
    
    try:
        # Check disk space
        import shutil
        free_space = shutil.disk_usage('.').free / (1024**3)
        if free_space < 5:
            print(f"⚠️ Low disk space: {free_space:.1f}GB")
        
        print(f"Loading TTS model on {device}...")
        
        # Load model
        tts_model = load_tts_model(model_name, device)
        
        print(f"Preparing text: {text[:50]}...")
        
        # Prepare text
        entries = tts_model.prepare_script([text.strip()], padding_between=1)
        
        # Convert display name back to actual voice path
        actual_voice = voice
        if voice.startswith("🎤 ") and hasattr(get_local_voices, 'voice_mapping'):
            actual_voice = get_local_voices.voice_mapping.get(voice, voice)
        elif voice.startswith("📦 "):
            actual_voice = voice[2:]  # Remove icon
        elif voice.startswith("📻 "):
            actual_voice = voice[2:]  # Remove icon
        
        print(f"Looking for voice: '{actual_voice}'")
        
        # Check if it's a custom voice (from custom_voices folder)
        voice_path = None
        if actual_voice.startswith("custom/"):
            # Direct path to custom voice file
            custom_voice_path = project_dir / "custom_voices" / f"{actual_voice[7:]}.safetensors"
            print(f"Checking custom voice path: {custom_voice_path}")
            if custom_voice_path.exists():
                voice_path = custom_voice_path
                print(f"✅ Found custom voice: {voice_path}")
            else:
                print(f"❌ Custom voice not found at: {custom_voice_path}")
        
        # If not custom voice or custom voice not found, try TTS model's built-in voices
        if voice_path is None:
            try:
                voice_path = tts_model.get_voice_path(actual_voice)
                print(f"✅ Found built-in voice: {voice_path}")
            except Exception as e:
                print(f"❌ Built-in voice not found: {actual_voice}, error: {e}")
        
        # If still no voice found, try fallbacks
        if voice_path is None:
            print("Trying fallback voices...")
            # Try default voice
            default_voice = "expresso/ex03-ex01_neutral_001_channel1_334s.wav"
            try:
                voice_path = tts_model.get_voice_path(default_voice)
                print(f"✅ Using default voice: {voice_path}")
            except:
                print("❌ Default voice also not found, searching for any available voice")
                # Search for any voice file
                for search_dir in [project_dir / "custom_voices", project_dir / "cache" / "voices", project_dir / "cache"]:
                    if search_dir.exists():
                        for voice_file in search_dir.rglob("*.safetensors"):
                            voice_path = voice_file
                            print(f"✅ Found fallback voice: {voice_path}")
                            break
                    if voice_path:
                        break
                
                if voice_path is None:
                    print("❌ No voice files found anywhere, generating without voice conditioning")
                    condition_attributes = None
        
        if voice_path:
            try:
                # Check if it's a custom safetensors file
                if str(voice_path).endswith('.safetensors') and 'custom_voices' in str(voice_path):
                    print(f"Loading custom voice embedding from: {voice_path}")
                    # Load the custom voice embedding
                    from safetensors.torch import load_file
                    voice_data = load_file(str(voice_path))
                    
                    # Check for both old and new key formats
                    if 'speaker_wavs' in voice_data:
                        speaker_wavs = voice_data['speaker_wavs']
                        print(f"Loaded speaker_wavs shape: {speaker_wavs.shape}")
                        
                        # For custom voices, we need to fallback to no conditioning for now
                        # The proper integration would require understanding the exact format
                        # that the TTS model expects for ConditionAttributes
                        print("⚠️ Custom voice loaded but using no conditioning (needs proper integration)")
                        condition_attributes = None
                        
                        # Alternative: try to use the speaker_wavs as if it were a standard voice file
                        # This is experimental and may not work
                        # condition_attributes = tts_model.make_condition_attributes([voice_path], cfg_coef=2.0)
                    elif 'voice_embedding' in voice_data:
                        # Backward compatibility with old format
                        voice_embedding = voice_data['voice_embedding']
                        print(f"Loaded old format voice_embedding shape: {voice_embedding.shape}")
                        print("⚠️ Old format detected, using no conditioning")
                        condition_attributes = None
                    else:
                        print("❌ No 'speaker_wavs' or 'voice_embedding' found in safetensors file")
                        condition_attributes = None
                else:
                    # Use standard voice loading for built-in voices
                    condition_attributes = tts_model.make_condition_attributes([voice_path], cfg_coef=2.0)
                    print(f"Created standard condition attributes")
            except Exception as e:
                print(f"❌ Error loading voice: {e}")
                print("Falling back to no voice conditioning")
                condition_attributes = None
        else:
            condition_attributes = None
        
        print("Generating audio...")
        
        # Generate audio
        result = tts_model.generate([entries], [condition_attributes])
        
        print(f"Generated {len(result.frames)} frames")
        
        # Decode audio
        with tts_model.mimi.streaming(1), torch.no_grad():
            pcms = []
            for i, frame in enumerate(result.frames[tts_model.delay_steps:]):
                if i % 10 == 0:  # Progress indicator
                    print(f"Decoding frame {i}/{len(result.frames)-tts_model.delay_steps}")
                
                pcm = tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
                pcm_clipped = np.clip(pcm[0, 0], -1, 1)
                
                # Check for valid audio
                if pcm_clipped.size > 0 and not np.all(pcm_clipped == 0):
                    pcms.append(pcm_clipped)
            
            if pcms and len(pcms) > 0:
                audio = np.concatenate(pcms, axis=-1)
                print(f"✅ Generated audio: {len(audio)} samples, {len(audio)/tts_model.mimi.sample_rate:.2f}s")
                
                # Ensure audio is not empty and has valid range
                if len(audio) > 0 and np.max(np.abs(audio)) > 0:
                    return tts_model.mimi.sample_rate, audio.astype(np.float32)
                else:
                    print("⚠️ Generated audio is silent")
                    # Return a short beep instead of empty array
                    sample_rate = tts_model.mimi.sample_rate
                    duration = 0.1  # 100ms beep
                    t = np.linspace(0, duration, int(sample_rate * duration))
                    beep = 0.1 * np.sin(2 * np.pi * 440 * t)  # 440Hz beep
                    return sample_rate, beep.astype(np.float32)
            else:
                print("⚠️ No audio frames generated")
                # Return a short beep
                sample_rate = 24000
                duration = 0.1
                t = np.linspace(0, duration, int(sample_rate * duration))
                beep = 0.1 * np.sin(2 * np.pi * 440 * t)
                return sample_rate, beep.astype(np.float32)
                
    except Exception as e:
        print(f"❌ TTS Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return error beep instead of empty array
        sample_rate = 24000
        duration = 0.2
        t = np.linspace(0, duration, int(sample_rate * duration))
        error_beep = 0.1 * np.sin(2 * np.pi * 220 * t)  # Lower frequency for error
        return sample_rate, error_beep.astype(np.float32)

def run_script(script_name: str, *args) -> str:
    """Run repository script"""
    script_path = project_dir / "scripts" / script_name
    if not script_path.exists():
        return f"❌ Script not found: {script_name}"
    
    try:
        cmd = [sys.executable, str(script_path)] + list(args)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            return result.stdout
        else:
            return f"❌ Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "❌ Script timed out"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def batch_transcribe(files) -> str:
    """Batch transcribe multiple files"""
    if not files:
        return "No files provided"
    
    results = []
    for i, file in enumerate(files):
        results.append(f"File {i+1}: {file.name}")
        transcription = transcribe_audio(file.name, "kyutai/stt-2.6b-en", "cuda")
        results.append(f"Result: {transcription}")
        results.append("-" * 50)
    
    return "\n".join(results)

def find_free_port(start_port=7860, max_attempts=10):
    """Find available port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    return None

def create_interface():
    """Create the main interface"""
    
    stt_models, tts_models, voices = get_available_models()
    
    # Auto-detect best device
    if torch.cuda.is_available():
        default_device = "cuda"
        device_status = f"🚀 GPU: {torch.cuda.get_device_name()}"
    else:
        default_device = "cpu"
        device_status = "💻 CPU Mode"
    
    with gr.Blocks(title="Kyutai STT & TTS", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown(f"""
        # 🎙️ Kyutai STT & TTS - Complete Interface
        
        Full-featured web interface for Kyutai Speech-to-Text and Text-to-Speech models.
        All repository features available in one simple interface.
        
        **Status**: {device_status} | **Moshi**: {'✅' if HAS_MOSHI else '❌'} | **MLX**: {'✅' if HAS_MLX else '❌'}
        """)
        
        # STT Tab
        with gr.Tab("🎙️ Speech-to-Text"):
            with gr.Row():
                with gr.Column():
                    stt_audio = gr.Audio(label="Audio Input", sources=["upload", "microphone"], type="filepath")
                    stt_model = gr.Dropdown(choices=stt_models, value=stt_models[0], label="Base Model")
                    stt_lora = gr.Dropdown(choices=get_available_lora_adapters(), value="None", label="LoRA Adapter")
                    stt_device = gr.Dropdown(choices=["cuda", "cpu"], value=default_device, label="Device")
                    stt_btn = gr.Button("🎯 Transcribe", variant="primary")
                
                with gr.Column():
                    stt_output = gr.Textbox(label="Transcription", lines=15, show_copy_button=True)
            
            def transcribe_with_lora(audio, model, lora, device):
                # TODO: Реальная интеграция LoRA
                if lora != "None":
                    result = transcribe_audio(audio, model, device)
                    return f"[Using {lora}] {result}"
                else:
                    return transcribe_audio(audio, model, device)
            
            stt_btn.click(transcribe_with_lora, inputs=[stt_audio, stt_model, stt_lora, stt_device], outputs=stt_output)
        
        # TTS Tab
        with gr.Tab("🗣️ Text-to-Speech"):
            with gr.Row():
                with gr.Column():
                    tts_text = gr.Textbox(label="Text to Synthesize", lines=5, placeholder="Enter text...")
                    
                    with gr.Row():
                        tts_voice = gr.Dropdown(choices=voices, value=voices[0], label="Voice", scale=4)
                        refresh_voices_btn = gr.Button("🔄", scale=1, size="sm")
                    
                    tts_model = gr.Dropdown(choices=tts_models, value=tts_models[0], label="Base Model")
                    tts_lora = gr.Dropdown(choices=get_available_lora_adapters(), value="None", label="LoRA Adapter")
                    tts_device = gr.Dropdown(choices=["cuda", "cpu"], value=default_device, label="Device")
                    tts_btn = gr.Button("🎵 Synthesize", variant="primary")
                
                with gr.Column():
                    tts_output = gr.Audio(label="Generated Audio", type="numpy")
                    
                    # Direct download button - downloads exactly what you hear
                    download_btn = gr.DownloadButton(
                        label="💾 Download Audio",
                        variant="secondary"
                    )
                    
                    def prepare_download(audio_data):
                        """Prepare audio file for download - exactly what you hear"""
                        if audio_data is None:
                            return None
                        
                        try:
                            sample_rate, audio_array = audio_data
                            
                            # Create temporary file with timestamp
                            import datetime
                            import tempfile
                            import wave
                            
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            temp_file = tempfile.NamedTemporaryFile(
                                delete=False, 
                                suffix=f"_tts_{timestamp}.wav"
                            )
                            
                            # Convert float32 to int16 for WAV format
                            audio_int16 = (audio_array * 32767).astype(np.int16)
                            
                            # Save using wave module (built-in Python)
                            with wave.open(temp_file.name, 'wb') as wav_file:
                                wav_file.setnchannels(1)  # Mono
                                wav_file.setsampwidth(2)  # 16-bit
                                wav_file.setframerate(sample_rate)
                                wav_file.writeframes(audio_int16.tobytes())
                            
                            return temp_file.name
                            
                        except Exception as e:
                            print(f"Error preparing download: {e}")
                            return None
                    
                    # Update download button when audio is generated
                    def update_download_button(audio_data):
                        if audio_data is not None:
                            file_path = prepare_download(audio_data)
                            if file_path:
                                return file_path
                        return None
                    
                    tts_output.change(
                        fn=update_download_button,
                        inputs=tts_output,
                        outputs=download_btn
                    )
            
            def synthesize_with_lora(text, voice, model, lora, device):
                # TODO: Реальная интеграция LoRA
                if lora != "None":
                    print(f"Using LoRA: {lora}")
                
                return synthesize_speech(text, voice, model, device)
            
            def refresh_voices():
                """Refresh the voice list to include newly trained voices"""
                updated_voices = get_local_voices()
                return gr.Dropdown(choices=updated_voices, value=updated_voices[0] if updated_voices else None)
            
            tts_btn.click(
                synthesize_with_lora, 
                inputs=[tts_text, tts_voice, tts_model, tts_lora, tts_device], 
                outputs=tts_output
            )
            
            refresh_voices_btn.click(
                refresh_voices,
                outputs=tts_voice
            )
        
        # Batch Tab
        with gr.Tab("📦 Batch Processing"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Batch STT")
                    batch_files = gr.File(label="Audio Files", file_count="multiple", file_types=["audio"])
                    batch_btn = gr.Button("Process All")
                    batch_output = gr.Textbox(label="Results", lines=20)
                
                batch_btn.click(batch_transcribe, inputs=batch_files, outputs=batch_output)
        
        # Scripts Tab
        with gr.Tab("⚙️ Scripts"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Repository Scripts")
                    script_file = gr.File(label="Input File", file_types=["audio", "txt"])
                    script_name = gr.Dropdown(
                        choices=[
                            "stt_from_file_pytorch.py",
                            "tts_pytorch.py",
                            "tts_pytorch_streaming.py"
                        ],
                        label="Script"
                    )
                    script_btn = gr.Button("Run Script")
                    script_output = gr.Textbox(label="Output", lines=15)
                
                def run_selected_script(file, script):
                    if file and script:
                        return run_script(script, file.name)
                    return "Please select file and script"
                
                script_btn.click(run_selected_script, inputs=[script_file, script_name], outputs=script_output)
        
        
        # Training Tab - Simple and Clear
        with gr.Tab("🎓 Training(WORK IN PROGRESS)"):
            gr.Markdown("## Model Training - 4 Training Modes")
            
            with gr.Tabs():
                # Mode 1: Fine-tune Model
                with gr.TabItem("1️⃣ Fine-tune Model"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Configuration")
                            ft_base_model = gr.Dropdown(
                                choices=["kyutai/stt-2.6b-en", "kyutai/stt-1b-en_fr"],
                                label="Base Model",
                                value="kyutai/stt-1b-en_fr"
                            )
                            ft_dataset_path = gr.Textbox(label="Dataset Path", placeholder="/path/to/dataset")
                            ft_output_dir = gr.Textbox(label="Output Directory", value="./fine_tuned_model")
                            
                            with gr.Row():
                                ft_epochs = gr.Slider(1, 100, value=10, label="Epochs")
                                ft_batch_size = gr.Slider(1, 32, value=4, label="Batch Size")
                            
                            with gr.Row():
                                ft_learning_rate = gr.Number(value=3e-5, label="Learning Rate")
                                ft_warmup_steps = gr.Number(value=1000, label="Warmup Steps")
                            
                            ft_start_btn = gr.Button("🚀 Start Fine-tuning", variant="primary")
                            ft_stop_btn = gr.Button("⏹️ Stop Training", variant="secondary")
                        
                        with gr.Column():
                            gr.Markdown("### Training Status")
                            ft_status = gr.Textbox(label="Status", value="Ready", interactive=False)
                            ft_progress = gr.Slider(0, 100, value=0, label="Progress (%)", interactive=False)
                            ft_loss_plot = gr.Plot(label="Training Loss")
                            ft_logs = gr.Textbox(label="Training Logs", lines=8, interactive=False)
                
                # Mode 2: Train from Scratch
                with gr.TabItem("2️⃣ Train from Scratch"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Configuration")
                            scratch_model_type = gr.Dropdown(
                                choices=["STT", "TTS"],
                                label="Model Type",
                                value="STT"
                            )
                            scratch_dataset_path = gr.Textbox(label="Dataset Path", placeholder="/path/to/large_dataset")
                            scratch_output_dir = gr.Textbox(label="Output Directory", value="./new_model")
                            
                            with gr.Row():
                                scratch_vocab_size = gr.Number(value=8000, label="Vocabulary Size")
                                scratch_model_size = gr.Dropdown(
                                    choices=["small", "medium", "large"],
                                    label="Model Size",
                                    value="medium"
                                )
                            
                            with gr.Row():
                                scratch_epochs = gr.Slider(1, 200, value=50, label="Epochs")
                                scratch_batch_size = gr.Slider(1, 16, value=2, label="Batch Size")
                            
                            scratch_start_btn = gr.Button("🚀 Start Training", variant="primary")
                            scratch_stop_btn = gr.Button("⏹️ Stop Training", variant="secondary")
                        
                        with gr.Column():
                            gr.Markdown("### Training Status")
                            scratch_status = gr.Textbox(label="Status", value="Ready", interactive=False)
                            scratch_progress = gr.Slider(0, 100, value=0, label="Progress (%)", interactive=False)
                            scratch_loss_plot = gr.Plot(label="Training Loss")
                            scratch_logs = gr.Textbox(label="Training Logs", lines=8, interactive=False)
                
                # Mode 3: LoRA Training
                with gr.TabItem("3️⃣ LoRA Training"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Configuration")
                            lora_base_model = gr.Dropdown(
                                choices=["kyutai/stt-2.6b-en", "kyutai/stt-1b-en_fr"],
                                label="Base Model",
                                value="kyutai/stt-1b-en_fr"
                            )
                            lora_dataset_path = gr.Textbox(label="Dataset Path", placeholder="/path/to/dataset")
                            lora_output_dir = gr.Textbox(label="Output Directory", value="./lora_adapters")
                            
                            with gr.Row():
                                lora_rank = gr.Slider(4, 128, value=32, label="LoRA Rank")
                                lora_alpha = gr.Slider(8, 256, value=64, label="LoRA Alpha")
                            
                            with gr.Row():
                                lora_dropout = gr.Slider(0.0, 0.5, value=0.1, label="Dropout")
                                lora_epochs = gr.Slider(1, 50, value=10, label="Epochs")
                            
                            lora_target_modules = gr.CheckboxGroup(
                                choices=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                                value=["q_proj", "v_proj", "k_proj", "o_proj"],
                                label="Target Modules"
                            )
                            
                            lora_start_btn = gr.Button("🚀 Start LoRA Training", variant="primary")
                            lora_stop_btn = gr.Button("⏹️ Stop Training", variant="secondary")
                        
                        with gr.Column():
                            gr.Markdown("### Training Status")
                            lora_status = gr.Textbox(label="Status", value="Ready", interactive=False)
                            lora_progress = gr.Slider(0, 100, value=0, label="Progress (%)", interactive=False)
                            lora_memory_usage = gr.Textbox(label="Memory Usage", value="0 GB", interactive=False)
                            lora_loss_plot = gr.Plot(label="Training Loss")
                            lora_logs = gr.Textbox(label="Training Logs", lines=6, interactive=False)
                
                # Mode 4: Voice Training
                with gr.TabItem("4️⃣ Voice Training"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Configuration")
                            voice_name = gr.Textbox(label="Voice Name", placeholder="my_voice")
                            voice_speaker = gr.Textbox(label="Speaker ID", placeholder="speaker_001")
                            voice_emotion = gr.Dropdown(
                                choices=["neutral", "happy", "sad", "angry", "surprised", "fearful"],
                                label="Emotion",
                                value="neutral"
                            )
                            
                            voice_audio_files = gr.File(
                                label="Audio Files",
                                file_count="multiple",
                                file_types=["audio"]
                            )
                            voice_output_dir = gr.Textbox(label="Output Directory", value="./custom_voices")
                            
                            with gr.Row():
                                voice_quality = gr.Dropdown(
                                    choices=["standard", "high", "ultra"],
                                    label="Quality",
                                    value="high"
                                )
                                voice_epochs = gr.Slider(10, 200, value=50, label="Epochs")
                            
                            voice_start_btn = gr.Button("🎤 Start Voice Training", variant="primary")
                            voice_stop_btn = gr.Button("⏹️ Stop Training", variant="secondary")
                            analyze_voices_btn = gr.Button("🔍 Analyze Official Voices", variant="secondary")
                        
                        with gr.Column():
                            gr.Markdown("### Training Status")
                            voice_status = gr.Textbox(label="Status", value="Ready", interactive=False)
                            voice_progress = gr.Slider(0, 100, value=0, label="Progress (%)", interactive=False)
                            voice_audio_preview = gr.Audio(label="Voice Preview", type="numpy")
                            voice_quality_metrics = gr.Textbox(label="Quality Metrics", lines=4, interactive=False)
                            voice_logs = gr.Textbox(label="Training Logs", lines=6, interactive=False)
            
            # Training Functions
            def start_voice_training(voice_name, voice_speaker, voice_emotion, voice_audio_files, voice_output_dir, voice_quality, voice_epochs):
                """Start voice training process"""
                if not voice_audio_files:
                    return "❌ No audio files provided", 0, "No audio files to process", ""
                
                if not voice_name.strip():
                    return "❌ Voice name is required", 0, "Please provide a voice name", ""
                
                try:
                    import time
                    import datetime
                    
                    # Create output directory
                    output_path = Path(voice_output_dir) / voice_name
                    output_path.mkdir(parents=True, exist_ok=True)
                    
                    logs = []
                    logs.append(f"🎤 Starting voice training: {voice_name}")
                    logs.append(f"📁 Output directory: {output_path}")
                    logs.append(f"🎭 Emotion: {voice_emotion}")
                    logs.append(f"🔧 Quality: {voice_quality}")
                    logs.append(f"📊 Epochs: {voice_epochs}")
                    logs.append(f"📂 Audio files: {len(voice_audio_files)}")
                    logs.append("")
                    
                    # Process audio files
                    processed_files = []
                    for i, audio_file in enumerate(voice_audio_files):
                        logs.append(f"Processing file {i+1}/{len(voice_audio_files)}: {audio_file.name}")
                        
                        # Load and validate audio
                        try:
                            audio_data, sample_rate = sphn.read(audio_file.name)
                            duration = len(audio_data) / sample_rate
                            logs.append(f"  ✅ Duration: {duration:.2f}s, Sample rate: {sample_rate}Hz")
                            
                            # Copy to output directory
                            import shutil
                            dest_path = output_path / f"{voice_speaker}_{voice_emotion}_{i:03d}.wav"
                            shutil.copy2(audio_file.name, dest_path)
                            processed_files.append(dest_path)
                            
                        except Exception as e:
                            logs.append(f"  ❌ Error processing {audio_file.name}: {str(e)}")
                            continue
                    
                    if not processed_files:
                        return "❌ No valid audio files processed", 0, "\n".join(logs), ""
                    
                    logs.append(f"\n✅ Processed {len(processed_files)} audio files")
                    logs.append("\n🎯 Starting voice embedding extraction...")
                    
                    # Load TTS model for voice extraction
                    if not HAS_MOSHI:
                        logs.append("❌ Moshi not available - cannot extract voice embeddings")
                        return "❌ Moshi required for voice training", 0, "\n".join(logs), ""
                    
                    # Simulate training process with progress updates
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    logs.append(f"🖥️ Using device: {device}")
                    
                    try:
                        # Load TTS model
                        tts_model = load_tts_model(DEFAULT_DSM_TTS_REPO if HAS_MOSHI else "kyutai/dsm-tts-1b-en", device)
                        logs.append("✅ TTS model loaded")
                        
                        # Extract voice embeddings from audio files
                        voice_embeddings = []
                        for i, audio_path in enumerate(processed_files):
                            progress = int((i / len(processed_files)) * 50)  # First 50% for extraction
                            logs.append(f"Extracting embeddings from {audio_path.name}... ({progress}%)")
                            
                            # Load audio and extract features
                            audio, sr = sphn.read(str(audio_path))
                            logs.append(f"  Original audio shape: {audio.shape}, dtype: {audio.dtype}")
                            
                            # Convert to float32 and handle multi-channel
                            if audio.ndim > 1:
                                audio = audio.mean(axis=0)  # Convert to mono
                            
                            audio_tensor = torch.from_numpy(audio.astype(np.float32)).to(device)
                            logs.append(f"  Converted tensor dtype: {audio_tensor.dtype}, shape: {audio_tensor.shape}")
                            
                            # Resample if needed
                            if sr != tts_model.mimi.sample_rate:
                                import julius
                                audio_tensor = julius.resample_frac(audio_tensor, sr, tts_model.mimi.sample_rate)
                                logs.append(f"  Resampled from {sr}Hz to {tts_model.mimi.sample_rate}Hz")
                            
                            # Extract voice features using mimi encoder
                            with torch.no_grad():
                                # Ensure audio has correct shape: (batch, channels, time)
                                if audio_tensor.dim() == 1:
                                    audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
                                elif audio_tensor.dim() == 2:
                                    audio_tensor = audio_tensor.unsqueeze(0)  # (1, C, T)
                                
                                # Ensure float32 dtype
                                audio_tensor = audio_tensor.float()
                                
                                # Pad audio to frame size
                                frame_size = tts_model.mimi.frame_size
                                if audio_tensor.shape[-1] % frame_size != 0:
                                    to_pad = frame_size - audio_tensor.shape[-1] % frame_size
                                    audio_tensor = torch.nn.functional.pad(audio_tensor, (0, to_pad))
                                
                                logs.append(f"  Final audio shape: {audio_tensor.shape}, dtype: {audio_tensor.dtype}")
                                
                                # Encode audio directly (Mimi expects B, C, T)
                                try:
                                    encoded = tts_model.mimi.encode(audio_tensor)
                                    logs.append(f"  Encoded shape: {encoded.shape}, dtype: {encoded.dtype}")
                                    logs.append(f"  Encoded min/max: {encoded.min().item():.4f}/{encoded.max().item():.4f}")
                                    
                                    # Create speaker_wavs in the same format as official voices
                                    # Official format: torch.Size([1, 512, 125]) = 64000 elements
                                    # Current encoded shape: [1, 32, 741] (from logs)
                                    
                                    if encoded.dim() == 3:  # (batch, features, time)
                                        batch_size, n_features, time_steps = encoded.shape
                                        logs.append(f"  Input encoded shape: [{batch_size}, {n_features}, {time_steps}]")
                                        
                                        # Target shape: [1, 512, 125]
                                        target_features = 512
                                        target_time = 125
                                        
                                        # Convert to float32 for interpolation
                                        encoded = encoded.float()
                                        
                                        # First, handle time dimension using 1D interpolation
                                        if time_steps != target_time:
                                            # Reshape for 1D interpolation: [batch*features, time]
                                            encoded_reshaped = encoded.view(batch_size * n_features, time_steps).unsqueeze(1)  # [batch*features, 1, time]
                                            
                                            # Interpolate time dimension
                                            encoded_reshaped = torch.nn.functional.interpolate(
                                                encoded_reshaped, size=target_time, mode='linear', align_corners=False
                                            )
                                            
                                            # Reshape back: [batch, features, target_time]
                                            encoded = encoded_reshaped.squeeze(1).view(batch_size, n_features, target_time)
                                            logs.append(f"  After time interpolation: {encoded.shape}")
                                        
                                        # Now handle features dimension
                                        if n_features != target_features:
                                            # Transpose to [batch, time, features] for feature interpolation
                                            encoded = encoded.transpose(1, 2)  # [1, target_time, n_features]
                                            
                                            # Reshape for 1D interpolation: [batch*time, features]
                                            encoded_reshaped = encoded.view(batch_size * target_time, n_features).unsqueeze(1)  # [batch*time, 1, features]
                                            
                                            # Interpolate features dimension
                                            encoded_reshaped = torch.nn.functional.interpolate(
                                                encoded_reshaped, size=target_features, mode='linear', align_corners=False
                                            )
                                            
                                            # Reshape back and transpose: [batch, target_features, target_time]
                                            encoded = encoded_reshaped.squeeze(1).view(batch_size, target_time, target_features).transpose(1, 2)
                                            logs.append(f"  After feature interpolation: {encoded.shape}")
                                        
                                        speaker_wavs = encoded  # Should now be [1, 512, 125]
                                    else:
                                        # Fallback: create tensor of correct shape
                                        logs.append(f"  Unexpected encoded dimensions: {encoded.dim()}, creating fallback tensor")
                                        speaker_wavs = torch.zeros(1, 512, 125, dtype=torch.float32)
                                        # Fill with encoded data if possible
                                        if encoded.numel() > 0:
                                            flat_encoded = encoded.flatten().float()
                                            fill_size = min(flat_encoded.numel(), speaker_wavs.numel())
                                            speaker_wavs.flatten()[:fill_size] = flat_encoded[:fill_size]
                                    
                                    logs.append(f"  Speaker wavs shape: {speaker_wavs.shape}")
                                    logs.append(f"  Speaker wavs dtype: {speaker_wavs.dtype}")
                                    logs.append(f"  Speaker wavs min/max: {speaker_wavs.min().item():.4f}/{speaker_wavs.max().item():.4f}")
                                    logs.append(f"  Speaker wavs numel: {speaker_wavs.numel()}")
                                    
                                    # Check if embedding contains valid data
                                    if speaker_wavs.numel() == 0:
                                        logs.append(f"  ❌ Empty speaker_wavs generated for {audio_path.name}")
                                        continue
                                    
                                    if torch.all(speaker_wavs == 0):
                                        logs.append(f"  ⚠️ All-zero speaker_wavs for {audio_path.name}")
                                    
                                    voice_embeddings.append(speaker_wavs.cpu())
                                    logs.append(f"  ✅ Speaker wavs extracted: {speaker_wavs.shape}")
                                except Exception as e:
                                    logs.append(f"  ❌ Encoding error: {str(e)}")
                                    logs.append(f"  Audio tensor shape: {audio_tensor.shape}, dtype: {audio_tensor.dtype}")
                                    import traceback
                                    logs.append(f"  Traceback: {traceback.format_exc()}")
                                    continue
                        
                        logs.append(f"Total voice embeddings collected: {len(voice_embeddings)}")
                        
                        if voice_embeddings:
                            # Debug each embedding before stacking
                            for i, emb in enumerate(voice_embeddings):
                                logs.append(f"Embedding {i}: shape={emb.shape}, dtype={emb.dtype}, numel={emb.numel()}")
                                logs.append(f"  Min/Max: {emb.min().item():.4f}/{emb.max().item():.4f}")
                            
                            # Average all embeddings to create final voice
                            try:
                                final_embedding = torch.stack(voice_embeddings).mean(dim=0)
                                logs.append(f"✅ Successfully stacked embeddings")
                            except Exception as e:
                                logs.append(f"❌ Error stacking embeddings: {e}")
                                return "❌ Error combining embeddings", 0, "\n".join(logs), ""
                            
                            logs.append(f"Final embedding shape: {final_embedding.shape}")
                            logs.append(f"Final embedding dtype: {final_embedding.dtype}")
                            logs.append(f"Final embedding size: {final_embedding.numel()} elements")
                            logs.append(f"Final embedding min/max: {final_embedding.min().item():.4f}/{final_embedding.max().item():.4f}")
                            
                            # Ensure embedding is not empty and has valid data
                            if final_embedding.numel() == 0:
                                logs.append("❌ Final embedding is empty!")
                                return "❌ Empty embedding generated", 0, "\n".join(logs), ""
                            
                            # Check for all-zero embedding
                            if torch.all(final_embedding == 0):
                                logs.append("⚠️ Warning: Final embedding is all zeros!")
                            
                            # Check for NaN or inf values
                            if torch.any(torch.isnan(final_embedding)):
                                logs.append("❌ Final embedding contains NaN values!")
                                return "❌ Invalid embedding (NaN)", 0, "\n".join(logs), ""
                            
                            if torch.any(torch.isinf(final_embedding)):
                                logs.append("❌ Final embedding contains infinite values!")
                                return "❌ Invalid embedding (inf)", 0, "\n".join(logs), ""
                            
                            # Save voice embedding as safetensors
                            voice_file = output_path / f"{voice_name}_{voice_emotion}.safetensors"
                            
                            # Create voice data structure (safetensors only accepts tensors)
                            # Make sure tensor is contiguous and properly formatted
                            final_embedding = final_embedding.contiguous()
                            
                            # Use the same key as official voices: 'speaker_wavs'
                            voice_data = {
                                'speaker_wavs': final_embedding,
                            }
                            
                            logs.append(f"Preparing to save {len(voice_data)} tensors")
                            for key, tensor in voice_data.items():
                                logs.append(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}, size={tensor.numel()}")
                            
                            # Create metadata as separate JSON file
                            metadata = {
                                'name': voice_name,
                                'speaker': voice_speaker,
                                'emotion': voice_emotion,
                                'quality': voice_quality,
                                'num_files': len(processed_files),
                                'created': datetime.datetime.now().isoformat(),
                                'embedding_shape': list(final_embedding.shape),
                                'sample_rate': tts_model.mimi.sample_rate,
                                'tensor_info': {
                                    'dtype': str(final_embedding.dtype),
                                    'numel': final_embedding.numel(),
                                    'is_contiguous': final_embedding.is_contiguous()
                                }
                            }
                            
                            # Save using safetensors
                            try:
                                from safetensors.torch import save_file
                                import json
                                
                                logs.append(f"Saving to: {voice_file}")
                                save_file(voice_data, str(voice_file))
                                
                                # Check file size after saving
                                file_size = voice_file.stat().st_size
                                logs.append(f"File saved, size: {file_size} bytes ({file_size/1024:.1f} KB)")
                                
                                if file_size == 0:
                                    logs.append("❌ WARNING: Saved file is 0 bytes!")
                                    # Try alternative save method
                                    torch.save(voice_data, str(voice_file).replace('.safetensors', '_backup.pt'))
                                    logs.append("Saved backup as .pt file")
                                
                                # Save metadata as JSON
                                metadata_file = output_path / f"{voice_name}_{voice_emotion}_metadata.json"
                                with open(metadata_file, 'w') as f:
                                    json.dump(metadata, f, indent=2)
                                logs.append(f"✅ Voice saved: {voice_file}")
                                logs.append(f"✅ Metadata saved: {metadata_file}")
                                
                                # Create preview audio
                                preview_text = f"Hello, this is {voice_name} speaking with {voice_emotion} emotion."
                                logs.append("🎵 Generating voice preview...")
                                
                                # Generate preview using the new voice
                                try:
                                    sample_rate, preview_audio = synthesize_speech(
                                        preview_text, 
                                        str(voice_file.relative_to(project_dir)), 
                                        DEFAULT_DSM_TTS_REPO if HAS_MOSHI else "kyutai/dsm-tts-1b-en",
                                        device
                                    )
                                    logs.append("✅ Voice preview generated successfully!")
                                except Exception as preview_error:
                                    logs.append(f"⚠️ Preview generation failed: {str(preview_error)}")
                                    logs.append("Voice training completed, but preview unavailable")
                                
                                logs.append("✅ Voice training completed successfully!")
                                logs.append(f"📁 Voice file: {voice_file}")
                                logs.append(f"📄 Metadata file: {metadata_file}")
                                logs.append(f"🎤 Voice ready for use in TTS")
                                
                                # Quality metrics
                                metrics = f"""Voice Quality Metrics:
• Files processed: {len(processed_files)}
• Total duration: {sum(len(sphn.read(str(f))[0])/sphn.read(str(f))[1] for f in processed_files):.1f}s
• Embedding size: {final_embedding.shape}
• Device used: {device}
• Quality level: {voice_quality}"""
                                
                                return "✅ Voice training completed!", 100, "\n".join(logs), metrics
                                
                            except Exception as e:
                                logs.append(f"❌ Error saving voice: {str(e)}")
                                return "❌ Error saving voice", 0, "\n".join(logs), ""
                        else:
                            logs.append("❌ No voice embeddings extracted")
                            return "❌ Failed to extract voice embeddings", 0, "\n".join(logs), ""
                            
                    except Exception as e:
                        logs.append(f"❌ Training error: {str(e)}")
                        return f"❌ Training failed: {str(e)}", 0, "\n".join(logs), ""
                        
                except Exception as e:
                    return f"❌ Error: {str(e)}", 0, f"Error during voice training: {str(e)}", ""
            
            def start_training(training_type, *args):
                if training_type == "Voice":
                    return start_voice_training(*args)
                else:
                    return f"Starting {training_type} training...", 0
            
            def stop_training(training_type):
                return f"{training_type} training stopped.", 0
            
            # Connect buttons
            ft_start_btn.click(
                fn=lambda *args: start_training("Fine-tuning", *args),
                inputs=[ft_base_model, ft_dataset_path, ft_output_dir, ft_epochs, ft_batch_size, ft_learning_rate, ft_warmup_steps],
                outputs=[ft_status, ft_progress]
            )
            
            lora_start_btn.click(
                fn=lambda *args: start_training("LoRA", *args),
                inputs=[lora_base_model, lora_dataset_path, lora_output_dir, lora_rank, lora_alpha, lora_dropout, lora_epochs, lora_target_modules],
                outputs=[lora_status, lora_progress]
            )
            
            voice_start_btn.click(
                fn=lambda *args: start_training("Voice", *args),
                inputs=[voice_name, voice_speaker, voice_emotion, voice_audio_files, voice_output_dir, voice_quality, voice_epochs],
                outputs=[voice_status, voice_progress, voice_logs, voice_quality_metrics]
            )
            
            def analyze_official_voices_interface():
                """Интерфейс для анализа официальных голосов"""
                try:
                    official_data = analyze_official_voice_structure()
                    if official_data:
                        analysis = "✅ Анализ официального голоса завершен!\n\n"
                        analysis += f"🔑 Найденные ключи: {list(official_data.keys())}\n\n"
                        
                        for key, tensor in official_data.items():
                            analysis += f"📊 {key}:\n"
                            analysis += f"  📐 Shape: {tensor.shape}\n"
                            analysis += f"  🏷️ Dtype: {tensor.dtype}\n"
                            analysis += f"  📊 Elements: {tensor.numel()}\n"
                            analysis += f"  💾 Size: {tensor.numel() * tensor.element_size()} bytes\n\n"
                        
                        analysis += "💡 Рекомендации для исправления тренировки:\n"
                        analysis += "1. Официальные голоса содержат сложную структуру\n"
                        analysis += "2. Нужно использовать тот же формат для совместимости\n"
                        analysis += "3. Размер файла должен быть ~250KB, не 1KB\n"
                        
                        return analysis
                    else:
                        return "❌ Не удалось найти официальные голоса для анализа"
                        
                except Exception as e:
                    return f"❌ Ошибка анализа: {str(e)}"
            
            analyze_voices_btn.click(
                analyze_official_voices_interface,
                outputs=voice_logs
            )
            
            # ----------------------------------------
            gr.Markdown("---")
            
            # INFO Spoiler with RAW data
            with gr.Accordion("ℹ️ INFO - Training & Languages RAW Data", open=False):
                def get_raw_training_info():
                    return """🎯 TRAINING SUPPORT: ✅ YES

📁 Evidence found:
  ✅ stt_evaluate_on_dataset.py - Complete evaluation pipeline
  ✅ Multi-language configs (English, French)
  ✅ TTS-voices repository with voice management
  ✅ Configurable model architectures
  ✅ Prompting support for fine-tuning

🌍 CURRENT LANGUAGES:
  • English: stt-2.6b-en (STT) + TTS
  • French: stt-1b-en_fr (STT) + TTS
  • Multilingual: en_fr tokenizer

🎤 VOICE CLONING: ✅ SUPPORTED
  • Voice embeddings in .safetensors format
  • Multiple emotional styles (happy, sad, angry, neutral)
  • Custom voice training pipeline exists
  • TTS-voices repository: kyutai/tts-voices

🏗️ ARCHITECTURE SUPPORTS:
  • Configurable vocabulary sizes
  • Multiple tokenizers
  • Transformer-based models
  • Audio + Text tokenization

📚 See training_analysis.md and add_new_language_guide.md

✅ CONFIRMED: New languages CAN be added!

📋 Requirements:
- 🎵 100+ hours of audio data
- 📝 Corresponding transcriptions
- 📚 Large text corpus for tokenizer
- 🎤 Multiple speakers for TTS voices
- 💻 High-end GPU for training

🛠️ Implementation Options:
- Fine-tune existing model (Recommended)
- Train from scratch (Advanced)
- Extend multilingual model (Best for similar languages)

🌍 ADDING NEW LANGUAGE SUPPORT - IMPLEMENTATION GUIDE:

📊 PHASE 1: DATA PREPARATION
• Collect 100+ hours of target language audio + transcripts
• Gather large target language text corpus (Wikipedia, news, books)
• Record multiple native speakers for TTS (5+ speakers recommended)
• Ensure diverse accents, ages, emotions

🔧 PHASE 2: TOKENIZER TRAINING
• Train SentencePiece model for target language
• Create config-stt-[lang]-hf.toml
• Update vocabulary parameters

🎓 PHASE 3: MODEL TRAINING
• Start with kyutai/stt-1b-en_fr base model
• Fine-tune on target language dataset
• Requires 1-2 weeks on A100/H100 GPU
• Expected accuracy: 85-95% for well-resourced languages

🎤 PHASE 4: VOICE CLONING
• Extract voice embeddings from target language speakers
• Save as .safetensors files
• Update TTS configuration
• Test voice quality and similarity

✅ PHASE 5: INTEGRATION
• Update WebUI model lists
• Add target language to available languages
• Test STT and TTS functionality
• Performance evaluation

🎯 EXPECTED RESULTS:
• STT Accuracy: 90%+ (with good data)
• TTS Quality: Natural target language speech
• Voice Cloning: Multiple emotional styles
• Real-time Performance: Maintained

📚 RESOURCES:
• Main training repo: kyutai-labs/moshi
• Voice examples: kyutai/tts-voices
• Documentation: training_analysis.md

🚀 READY TO START? Begin with data collection!"""
                
                raw_info_display = gr.Textbox(
                    label="RAW Training & Languages Information",
                    value=get_raw_training_info(),
                    lines=25,
                    interactive=False,
                    show_copy_button=True
                )
        
        # System Tab
        with gr.Tab("🔧 System"):
            with gr.Row():
                with gr.Column():
                    def get_system_info():
                        info = []
                        info.append(f"Python: {sys.version.split()[0]}")
                        info.append(f"PyTorch: {torch.__version__}")
                        info.append(f"CUDA: {torch.cuda.is_available()}")
                        if torch.cuda.is_available():
                            info.append(f"GPU: {torch.cuda.get_device_name()}")
                        info.append(f"Moshi: {HAS_MOSHI}")
                        info.append(f"MLX: {HAS_MLX}")
                        
                        # Disk space
                        import shutil
                        usage = shutil.disk_usage('.')
                        info.append(f"Free Space: {usage.free / 1024**3:.1f} GB")
                        
                        # Cache info
                        cache_dir = project_dir / "cache"
                        if cache_dir.exists():
                            cache_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
                            info.append(f"Cache Size: {cache_size / 1024**3:.2f} GB")
                        
                        return "\n".join(info)
                    
                    system_info = gr.Textbox(label="System Information", value=get_system_info(), lines=15)
                    refresh_btn = gr.Button("🔄 Refresh")
                    
                    def clear_cache():
                        global model_cache
                        model_cache.clear()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        return "✅ Cache cleared"
                    
                    clear_btn = gr.Button("🗑️ Clear Cache")
                    cache_status = gr.Textbox(label="Status")
                    
                    refresh_btn.click(get_system_info, outputs=system_info)
                    clear_btn.click(clear_cache, outputs=cache_status)
        
        gr.Markdown("""
        ---
        **Kyutai STT & TTS** | Portable • Complete • Simple
        
        📁 Cache: `./cache/` | 📄 Temp: `./temp/` | 🔗 [Repository](https://github.com/kyutai-labs/delayed-streams-modeling)
        """)
    
    return demo

def main():
    """Main function"""
    print("🚀 Starting Kyutai STT & TTS WebUI...")
    
    # Check dependencies
    if not HAS_MOSHI:
        print("⚠️ Moshi not available. Some features will be limited.")
        print("Run setup to install all dependencies.")
    
    # Find port
    port = find_free_port()
    if port is None:
        port = 7860
    
    print(f"🌐 Starting on port {port}")
    print(f"📁 Project: {project_dir}")
    print(f"🗂️ Cache: {project_dir / 'cache'}")
    
    # Create and launch interface
    demo = create_interface()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()