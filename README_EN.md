# Hierarchical Text Summarization with LLM

A Python project for hierarchical summarization of long texts (such as books) using local LLM models through llama.cpp. The algorithm splits text into chunks, creates summaries, and recursively combines them into a coherent narrative.

## 📋 Features

- **Hierarchical Summarization**: Recursive merging of summaries to preserve context
- **Russian Language Support**: All prompts are optimized for Russian-language texts
- **Output Cleaning**: Removal of model internal reasoning and service tags
- **Language Detection**: Automatic detection and correction of English summaries
- **Logging**: Detailed logs of the summarization process
- **Flexible Configuration**: Customizable chunk parameters, overlap, and detail levels

## 🛠️ Technologies

- **Python 3.8+**
- **llama-cpp-python** - for working with GGUF models
- **Regular Expressions** - for text cleaning
- **Grok/Gemma models** - preferred models for summarization

## 📁 Project Structure

```
summarizer/
├── main.py                    # Main summarization script
├── README.md                  # Documentation
├── requirements.txt           # Python dependencies
├── example_config.py          # Configuration example
├── Master_i_Margarita.txt     # Example text for summarization
├── Summary_Detailed.txt       # Detailed summarization log (generated)
├── Output_summary.txt         # Final summary (generated)
└── hierarchical_log.txt       # Hierarchical process log (generated)
```

## ⚙️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/text-summarizer.git
cd text-summarizer
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install llama-cpp-python:**
```bash
# For CPU only
pip install llama-cpp-python

# For GPU (CUDA) support
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
```

## 📊 Model Requirements

Recommended models (GGUF format):
- **Grok-3-reasoning-gemma3-12B-distilled-HF.Q8_0.gguf** (used in example)
- **Mixtral-8x7B-Instruct-v0.1-GGUF**
- **Llama-3-8B-Instruct-GGUF**
- Any model with chat format support and Russian language capability

Download a model and specify its path in the code.

## 🚀 Usage

### Basic Usage

1. **Prepare a text file** in UTF-8 or cp1251 encoding

2. **Configure the model path** in the script:
```python
model_path = r"path/to/your/model.gguf"
```

3. **Specify the text file path:**
```python
book_file = r"path/to/your/book.txt"
```

4. **Run the summarization:**
```bash
python main.py
```

### Parameter Configuration

Main parameters to adjust in the `main()` function:

```python
# Model parameters
llm = Llama(
    model_path=model_path,
    chat_format="gemma",        # Or "chatml"
    n_ctx=32768,                # Context window size
    n_threads=8,                # Number of CPU threads
    n_gpu_layers=47,            # Layers on GPU (0 for CPU only)
    temperature=0.1,            # Generation temperature
    max_tokens=8192,            # Maximum tokens to generate
)

# Summarization parameters
chunk_size=3000                 # Chunk size in characters
overlap_sentences=3            # Overlap between chunks
max_group_size=5               # Group size for hierarchical summarization
detail_level=1                 # Detail level (1 = most detailed)
```

## 📝 Output Format

The program creates three files:

1. **Output_summary.txt** - Final summary:
   - Overall summary (10-20 sentences)
   - Hierarchical summary
   - Individual chunk summaries

2. **Summary_Detailed.txt** - Detailed process:
   - Detailed summaries of each chunk
   - Hierarchical merges
   - Intermediate results

3. **hierarchical_log.txt** - Technical log:
   - Structure of recursive calls
   - Number of processed items at each level
   - Execution time

## 🔧 Functions

### Main Functions

1. **`clean_model_output(text)`** - Cleans model output from `<think>`, `<reasoning>` tags and internal reasoning

2. **`combine_into_narrative(llm, text_list)`** - Combines a list of texts into a coherent narrative

3. **`chunk_text(text, chunk_size, overlap_sentences)`** - Splits text into overlapping chunks

4. **`summarize_chunk(chunk, level, summary_file)`** - Summarizes a single chunk with language detection

5. **`hierarchical_summarize(summaries, max_group_size, level, ...)`** - Recursive hierarchical summarization

### Algorithm Overview

1. **Load and clean text**
2. **Split into chunks** with overlap
3. **Summarize each chunk** with Russian language verification
4. **Hierarchically merge** summaries (recursively)
5. **Create final overall summary**
6. **Save results** in three different files

## 🎯 Example Usage

```python
# Quick start with your text
book_file = "your_book.txt"
model_path = "models/grok-3.gguf"

# Run summarization
# Results will be in Output_summary.txt and Summary_Detailed.txt
```

## ⚠️ Limitations and Recommendations

### Performance
- **Execution Time**: ~4-5 hours for a book on Grok-3 model (depends on text length)
- **Memory**: Sufficient VRAM/RAM required for model (12GB+ for 12B models)
- **Optimization**: Use quantized models (Q4_K_M, Q8_0) for better performance

### Summarization Quality
- **Best results** with models trained on Russian language
- **Chunk size** 2000-4000 characters optimal for context-detailing balance
- **Chunk overlap** (3-5 sentences) helps preserve context

### Troubleshooting
1. **English summaries**: Script automatically regenerates them in Russian
2. **Context loss**: Increase chunk overlap and decrease `max_group_size`
3. **Too general summaries**: Decrease `temperature` and increase `chunk_size`

## 📈 Future Improvements

- [ ] Support for various input formats (PDF, EPUB, DOCX)
- [ ] Web interface for easier usage
- [ ] Batch processing of multiple files
- [ ] Parameter configuration via config file
- [ ] Support for more languages
- [ ] Export to various formats (JSON, HTML, Markdown)

## 🤝 Contributing

Pull requests and issues are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. 

## ✨ Author

Vad_neld
Telegram  @Vad_neld
E-mail: v.bel@list.ru

---
⭐ If you found this project useful, please give it a star on GitHub!

## 🔗 Related Projects

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - LLM inference in C/C++
- [LangChain](https://github.com/langchain-ai/langchain) - Framework for LLM applications
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - State-of-the-art Machine Learning

## 🙏 Acknowledgments

- Thanks to the llama.cpp team for the efficient inference library
- Model creators and researchers for making LLMs accessible
- Open source community for continuous improvement and support

---

**Note**: This project is optimized for Russian language texts but can be adapted for other languages by modifying the system prompts and language detection logic.