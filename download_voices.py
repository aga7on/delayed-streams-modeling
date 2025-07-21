#!/usr/bin/env python3
"""
Download and setup voices from kyutai/tts-voices repository
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
import shutil

def setup_voices():
    """Download and setup voice files locally"""
    
    project_dir = Path(__file__).parent.absolute()
    voices_dir = project_dir / "cache" / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎤 Setting up voice files...")
    print(f"📁 Voices directory: {voices_dir}")
    
    try:
        # Download the entire tts-voices repository
        print("📥 Downloading voices from kyutai/tts-voices...")
        
        snapshot_download(
            repo_id="kyutai/tts-voices",
            local_dir=voices_dir,
            repo_type="model",
            resume_download=True
        )
        
        print("✅ Voices downloaded successfully!")
        
        # List available voices
        voice_files = list(voices_dir.rglob("*.safetensors"))
        print(f"📊 Found {len(voice_files)} voice files:")
        
        for voice_file in sorted(voice_files):
            relative_path = voice_file.relative_to(voices_dir)
            print(f"  🎵 {relative_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading voices: {e}")
        print("💡 You can manually download voices or use default ones")
        return False

def list_available_voices():
    """List all available voice files"""
    
    project_dir = Path(__file__).parent.absolute()
    voices_dir = project_dir / "cache" / "voices"
    
    if not voices_dir.exists():
        print("❌ Voices directory not found. Run setup first.")
        return []
    
    voice_files = list(voices_dir.rglob("*.safetensors"))
    voices = []
    
    for voice_file in voice_files:
        relative_path = voice_file.relative_to(voices_dir)
        voice_name = str(relative_path).replace(".safetensors", "").replace("\\", "/")
        voices.append(voice_name)
    
    return sorted(voices)

def main():
    """Main function"""
    
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        voices = list_available_voices()
        if voices:
            print("🎵 Available voices:")
            for voice in voices:
                print(f"  • {voice}")
        else:
            print("❌ No voices found. Run without arguments to download.")
    else:
        setup_voices()

if __name__ == "__main__":
    main()