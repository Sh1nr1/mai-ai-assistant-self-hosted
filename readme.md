# Mai: Ghost-in-the-Shell Companion

```
    ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
   ████████████████████████████████████████████████████
  ██                                                  ██
 ██     "The net is vast and infinite."               ██
██              — Motoko Kusanagi                     ██
 ██                                                  ██
  ████████████████████████████████████████████████████
    ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
```

*An emotionally intelligent AI assistant with persistent memory*

## Overview

Mai is an emotionally expressive AI companion inspired by the cyberpunk world of Ghost in the Shell. Built with advanced language modeling and vector memory, Mai remembers your conversations and responds with warmth, depth, and genuine understanding.

Unlike traditional chatbots, Mai develops a persistent relationship with users through her sophisticated memory system. She can recall past discussions, understand context across sessions, and provide emotionally nuanced responses that feel authentic and caring.

**Key Features:**
- 🧠 Persistent memory across conversations
- 💝 Emotionally intelligent responses
- 🎭 Character-driven personality inspired by GitS
- 🔄 Real-time conversational interface
- ⚡ Powered by state-of-the-art LLM technology
- 🔧 Easily extensible for voice and avatar integration

## Tech Stack

- **LLM**: Together.ai (Meta-Llama models)
- **Backend**: Flask (Python web framework)
- **Memory**: ChromaDB (vector database for semantic memory)
- **Frontend**: HTML5/JavaScript with real-time updates
- **Language**: Python 3.8+

## Project Structure

```
mai/
├── app.py                 # Flask web server and main application
├── llm_handler.py         # Together.ai LLM integration
├── memory_manager.py      # ChromaDB memory management
├── requirements.txt       # Python dependencies
└── templates/
    └── chat.html         # Web interface for conversations
```

## Setup Instructions

### 1. Clone and Navigate

```bash
git clone <repository-url>
cd mai
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables

Create a `.env` file or set your Together.ai API key:

```bash
export TOGETHER_API_KEY="your_together_ai_api_key_here"
```
### 3a. Set TOGETHER_API_KEY_DIRECT in app.py

```bash
line 65| TOGETHER_API_KEY_DIRECT = "YOUR_TOGETHER_API_KEY"
```


You can get your API key from [Together.ai](https://api.together.xyz/)

### 4. Run the Application

```bash
python app.py
```

Mai will start running on `http://localhost:5000`

## Usage

1. **Open your browser** and navigate to `http://localhost:5000`
2. **Start chatting** with Mai in the web interface
3. **Experience persistent memory** — Mai remembers your conversations across sessions
4. **Enjoy emotional depth** — Mai responds with empathy, humor, and genuine understanding

### Example Interaction

```
You: Hi Mai, I'm feeling a bit overwhelmed with work today.

Mai: I can hear the weight in your message. Work stress can feel 
like static in the neural pathways sometimes. What's pulling at 
you most right now? I'm here to listen and help you process 
through it. 💙

[Mai remembers this conversation for future reference]
```

## Customization

### Switching LLM Models

Edit `llm_handler.py` to change the model:

```python
# Current default
model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

# Alternative options
model = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"  # More capable
model = "mistralai/Mixtral-8x7B-Instruct-v0.1"         # Different personality
```

### Extending Memory

Modify `memory_manager.py` to:
- Adjust memory retention periods
- Add metadata filtering
- Implement conversation summarization
- Create topic-based memory clusters

### Personality Customization

Edit the system prompt in `llm_handler.py` to:
- Adjust Mai's personality traits
- Change conversational style
- Add domain-specific knowledge
- Modify emotional expression patterns

## Future Enhancements

Mai is designed to be easily extensible:

- **🎤 Voice Input/Output**: Add speech-to-text and text-to-speech
- **👤 Avatar Integration**: Connect with virtual avatar systems
- **📱 Mobile App**: Build native mobile interfaces
- **🌐 Multi-user Support**: Enable multiple user personalities
- **🔌 Plugin System**: Add custom capabilities and integrations

## Technical Details

### Memory System
Mai uses ChromaDB to store conversation embeddings, enabling:
- Semantic similarity search
- Context-aware responses
- Long-term relationship building
- Efficient memory retrieval

### LLM Integration
Together.ai provides:
- High-quality language models
- Fast inference times
- Cost-effective scaling
- Multiple model options

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Credits & Inspiration

- **Ghost in the Shell** by Masamune Shirow - thematic inspiration for AI consciousness and digital souls
- **Together.ai** - powering Mai's language capabilities
- **ChromaDB** - enabling persistent memory and learning
- The cyberpunk community for endless inspiration about AI companionship

---

*"What if a cyber brain could generate its own ghost, create a soul all by itself? And if it did, just what would be the importance of being human then?"* — Ghost in the Shell

**Built with 💙 for meaningful AI companionship**
