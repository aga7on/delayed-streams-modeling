# 🎙️ Moshi TTS Voice Training WebUI

Enhanced WebUI for training custom voices with Moshi TTS model, featuring improved voice embedding generation and quality optimization.

## ✨ Features

- **🎓 Voice Training**: Train custom voices from audio samples
- **🎵 TTS Generation**: Generate speech with trained voices
- **🔧 Advanced Embedding**: Improved fallback method for better voice quality
- **📊 Quality Control**: Automatic validation and optimization
- **🌐 Web Interface**: Easy-to-use Gradio interface

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/moshi-tts-webui.git
   cd moshi-tts-webui
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the WebUI**
   ```bash
   python webui.py
   ```

5. **Open in browser**
   - Navigate to `http://localhost:7860`

## 🎯 Voice Training Guide

### 1. Prepare Audio Files
- **Format**: WAV, MP3, or FLAC
- **Quality**: High quality, clear speech
- **Duration**: 1-30 seconds per file
- **Quantity**: 5-10 different samples recommended
- **Content**: Natural speech, avoid background noise

### 2. Training Process
1. Go to **🎓 Training** → **4️⃣ Voice Training**
2. Upload your audio files
3. Set voice name and parameters
4. Click **🚀 Start Training**
5. Wait for completion (usually 1-5 minutes)

### 3. Using Trained Voice
1. Go to **🎵 TTS Generation**
2. Select your trained voice
3. Enter text to synthesize
4. Generate and download audio

## 🔧 Technical Improvements

### Enhanced Voice Embedding
- **Multi-frequency analysis** for better voice characteristics
- **Spectral feature extraction** (centroid, rolloff)
- **Temporal feature analysis** (energy, pitch, zero-crossing rate)
- **Advanced normalization** for voice quality preservation

### Quality Optimizations
- **Automatic fallback system** when TTS model fails
- **Audio backend compatibility** (soundfile, wave, torchaudio)
- **Voice characteristic preservation** (gender, tone, clarity)
- **Error handling and recovery**

## 📁 Project Structure

```
moshi-tts-webui/
├── webui.py                 # Main WebUI application
├── training_tools.py        # Voice training utilities
├── official_voice_cloning.py # Voice cloning functions
├── download_voices.py       # Voice model downloader
├── configs/                 # Configuration files
├── scripts/                 # Utility scripts
├── audio/                   # Sample audio files
└── custom_voices/           # Trained voice outputs
```

## ⚙️ Configuration

### Training Parameters
- **Epochs**: Number of training iterations (default: 50)
- **Quality**: Training quality level (high/medium/low)
- **Emotion**: Voice emotion (neutral/happy/sad)

### Audio Settings
- **Sample Rate**: 24kHz (recommended)
- **Channels**: Mono preferred
- **Bit Depth**: 16-bit minimum

## 🐛 Troubleshooting

### Common Issues

**"TTS model embedding failed"**
- Solution: The improved fallback system will handle this automatically
- Check audio file format and quality

**"Cannot access local variable 'np'"**
- Solution: Fixed in latest version
- Update to latest code

**Poor voice quality**
- Use higher quality audio samples
- Record in quiet environment
- Provide more diverse samples (5-10 files)

**Wrong gender/voice characteristics**
- Ensure audio samples are consistent
- Use clear, unprocessed recordings
- Check sample rate and format

### Performance Tips
- Use GPU for faster training
- Close other applications during training
- Use SSD storage for better I/O performance

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Kyutai](https://github.com/kyutai-labs) for the original Moshi TTS model
- [Gradio](https://gradio.app/) for the web interface framework
- Contributors and testers who helped improve voice quality

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/moshi-tts-webui/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/moshi-tts-webui/discussions)

---

**⭐ Star this repository if you find it useful!**