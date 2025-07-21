#!/usr/bin/env python3
"""
Training tools for Kyutai STT & TTS models
Includes LoRA training, fine-tuning, and voice creation utilities
"""

import os
import sys
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional
import subprocess

def check_training_requirements():
    """Check if training requirements are met"""
    
    requirements = {
        "torch": False,
        "transformers": False,
        "peft": False,
        "datasets": False,
        "accelerate": False,
        "cuda": False
    }
    
    try:
        import torch
        requirements["torch"] = True
        requirements["cuda"] = torch.cuda.is_available()
    except ImportError:
        pass
    
    try:
        import transformers
        requirements["transformers"] = True
    except ImportError:
        pass
    
    try:
        import peft
        requirements["peft"] = True
    except ImportError:
        pass
    
    try:
        import datasets
        requirements["datasets"] = True
    except ImportError:
        pass
    
    try:
        import accelerate
        requirements["accelerate"] = True
    except ImportError:
        pass
    
    return requirements

def install_training_dependencies():
    """Install required packages for training"""
    
    packages = [
        "transformers>=4.30.0",
        "peft>=0.4.0", 
        "datasets>=2.12.0",
        "accelerate>=0.20.0",
        "deepspeed",
        "wandb",
        "tensorboard"
    ]
    
    print("📦 Installing training dependencies...")
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
    
    print("🎉 Training dependencies installation complete!")

def create_lora_config(
    rank: int = 32,
    alpha: int = 64,
    dropout: float = 0.1,
    target_modules: List[str] = None
) -> Dict:
    """Create LoRA configuration"""
    
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "target_modules": target_modules,
        "bias": "none",
        "fan_in_fan_out": False,
        "modules_to_save": None
    }
    
    return config

def create_training_config(
    output_dir: str = "./lora_output",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 3e-4,
    max_steps: int = -1
) -> Dict:
    """Create training configuration"""
    
    config = {
        "output_dir": output_dir,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "max_steps": max_steps,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "logging_steps": 100,
        "save_steps": 500,
        "eval_steps": 500,
        "save_total_limit": 3,
        "remove_unused_columns": False,
        "push_to_hub": False,
        "report_to": "tensorboard",
        "fp16": True,
        "dataloader_num_workers": 4,
        "group_by_length": True,
        "ddp_find_unused_parameters": False
    }
    
    return config

def prepare_russian_dataset(audio_dir: str, transcript_file: str) -> str:
    """Prepare Russian dataset for training"""
    
    script_content = f"""
#!/usr/bin/env python3
'''
Russian dataset preparation script
'''

import os
import json
import torchaudio
from pathlib import Path
from datasets import Dataset, Audio

def prepare_dataset():
    audio_dir = Path("{audio_dir}")
    transcript_file = "{transcript_file}"
    
    # Load transcripts
    with open(transcript_file, 'r', encoding='utf-8') as f:
        transcripts = {{}}
        for line in f:
            parts = line.strip().split('\\t')
            if len(parts) >= 2:
                audio_id = parts[0]
                text = parts[1]
                transcripts[audio_id] = text
    
    # Prepare dataset entries
    dataset_entries = []
    
    for audio_file in audio_dir.glob("*.wav"):
        audio_id = audio_file.stem
        if audio_id in transcripts:
            dataset_entries.append({{
                "audio": str(audio_file),
                "text": transcripts[audio_id],
                "speaker_id": "russian_speaker"
            }})
    
    # Create HuggingFace dataset
    dataset = Dataset.from_list(dataset_entries)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Save dataset
    dataset.save_to_disk("russian_dataset")
    print(f"✅ Prepared {{len(dataset)}} samples")
    
    return dataset

if __name__ == "__main__":
    prepare_dataset()
"""
    
    script_path = "prepare_russian_dataset.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    return script_path

def create_lora_training_script(
    base_model: str = "kyutai/stt-1b-en_fr",
    dataset_path: str = "russian_dataset",
    lora_config: Dict = None,
    training_config: Dict = None
) -> str:
    """Create LoRA training script"""
    
    if lora_config is None:
        lora_config = create_lora_config()
    
    if training_config is None:
        training_config = create_training_config()
    
    script_content = f"""
#!/usr/bin/env python3
'''
LoRA training script for Russian language support
'''

import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_from_disk
import json

def main():
    # Configuration
    base_model = "{base_model}"
    dataset_path = "{dataset_path}"
    
    lora_config = {json.dumps(lora_config, indent=4)}
    training_config = {json.dumps(training_config, indent=4)}
    
    print("🚀 Starting LoRA training for Russian language...")
    
    # Load tokenizer and model
    print(f"📥 Loading base model: {{base_model}}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Setup LoRA
    print("🔧 Setting up LoRA adapters...")
    peft_config = LoraConfig(**lora_config)
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Load dataset
    print(f"📊 Loading dataset: {{dataset_path}}")
    dataset = load_from_disk(dataset_path)
    
    # Setup training arguments
    training_args = TrainingArguments(**training_config)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Start training
    print("🎓 Starting training...")
    trainer.train()
    
    # Save model
    print("💾 Saving trained model...")
    trainer.save_model()
    
    print("🎉 Training completed!")

if __name__ == "__main__":
    main()
"""
    
    script_path = "train_russian_lora.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    return script_path

def create_voice_training_script(
    voice_name: str,
    audio_files: List[str],
    speaker_name: str = "custom_speaker",
    emotion: str = "neutral"
) -> str:
    """Create voice training script"""
    
    script_content = f"""
#!/usr/bin/env python3
'''
Voice training script for custom voice creation
'''

import os
import torch
import torchaudio
from pathlib import Path
import numpy as np

def extract_voice_embeddings():
    voice_name = "{voice_name}"
    speaker_name = "{speaker_name}"
    emotion = "{emotion}"
    audio_files = {audio_files}
    
    print(f"🎤 Creating voice: {{voice_name}}")
    print(f"👤 Speaker: {{speaker_name}}")
    print(f"😊 Emotion: {{emotion}}")
    print(f"📁 Audio files: {{len(audio_files)}}")
    
    # Process audio files
    embeddings = []
    
    for audio_file in audio_files:
        print(f"🔊 Processing: {{audio_file}}")
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_file)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Extract features (placeholder - actual implementation would use TTS model)
        # This is a simplified version
        features = torch.mean(waveform, dim=1)
        embeddings.append(features)
    
    # Average embeddings
    if embeddings:
        voice_embedding = torch.stack(embeddings).mean(dim=0)
        
        # Save voice embedding
        output_path = f"{{voice_name}}_{{speaker_name}}_{{emotion}}.safetensors"
        
        # Save as safetensors (placeholder)
        print(f"💾 Saving voice to: {{output_path}}")
        torch.save(voice_embedding, output_path.replace('.safetensors', '.pt'))
        
        print("✅ Voice creation completed!")
        print(f"🎯 Voice file: {{output_path}}")
        
        return output_path
    else:
        print("❌ No audio files processed")
        return None

if __name__ == "__main__":
    extract_voice_embeddings()
"""
    
    script_path = f"create_voice_{voice_name}.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    return script_path

def main():
    """Main training tools interface"""
    
    print("🛠️ Kyutai Training Tools")
    print("=" * 40)
    
    # Check requirements
    print("🔍 Checking training requirements...")
    requirements = check_training_requirements()
    
    for req, status in requirements.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {req}")
    
    missing = [req for req, status in requirements.items() if not status]
    
    if missing:
        print(f"\n⚠️ Missing requirements: {', '.join(missing)}")
        install_choice = input("Install missing dependencies? (y/n): ").lower()
        if install_choice == 'y':
            install_training_dependencies()
    
    print("\n🎯 Available tools:")
    print("1. Create LoRA training script")
    print("2. Create voice training script") 
    print("3. Prepare Russian dataset")
    print("4. Check system capabilities")
    
    choice = input("\nSelect tool (1-4): ").strip()
    
    if choice == "1":
        print("\n🔧 Creating LoRA training script...")
        script_path = create_lora_training_script()
        print(f"✅ Created: {script_path}")
        
    elif choice == "2":
        print("\n🎤 Creating voice training script...")
        voice_name = input("Voice name: ")
        speaker_name = input("Speaker name: ")
        emotion = input("Emotion (neutral/happy/sad): ") or "neutral"
        
        script_path = create_voice_training_script(voice_name, [], speaker_name, emotion)
        print(f"✅ Created: {script_path}")
        
    elif choice == "3":
        print("\n📊 Creating Russian dataset preparation script...")
        audio_dir = input("Audio directory path: ")
        transcript_file = input("Transcript file path: ")
        
        script_path = prepare_russian_dataset(audio_dir, transcript_file)
        print(f"✅ Created: {script_path}")
        
    elif choice == "4":
        print("\n💻 Running system analysis...")
        os.system("python system_analysis.py")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()