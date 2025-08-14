# DocStringEval: Automated Docstring Generation and Evaluation Framework

A comprehensive framework for evaluating Large Language Models (LLMs) in automated docstring generation tasks for Python classes.

## Overview

This project evaluates the performance of various LLMs in generating docstrings for Python classes. The framework includes:

- **Data Extraction**: Automated extraction of Python classes from repositories
- **Code Preprocessing**: Cleaning and standardizing code for consistent evaluation
- **Docstring Generation**: Using multiple LLMs to generate docstrings
- **Evaluation Metrics**: Comprehensive evaluation using ROUGE, BLEU, and custom metrics
- **Analysis Tools**: Detailed analysis and comparison of model performance

## Project Structure

```
DocStringEval/
├── src/                    # Core source code
│   ├── data/              # Data extraction and preprocessing
│   ├── generation/        # Docstring generation modules
│   └── evaluation/        # Evaluation metrics and analysis
├── data/                  # Data storage
│   ├── raw/              # Original class files
│   ├── processed/        # Preprocessed data
│   └── outputs/          # Generated outputs
├── config/               # Configuration files
├── scripts/              # Execution scripts
├── tests/                # Unit tests
├── notebooks/            # Jupyter notebooks for analysis
└── results/              # Evaluation results and reports
```

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/DocStringEval.git
cd DocStringEval
pip install -r requirements.txt
```

### Basic Usage

1. **Extract Classes**:
```bash
python scripts/extract_classes.py --input data/raw/class_collection.csv
```

2. **Generate Docstrings**:
```bash
python scripts/generate_docstrings.py --model codellama-7b --input data/processed/
```

3. **Evaluate Results**:
```bash
python scripts/evaluate_results.py --output results/evaluation_report.json
```

## Supported Models

- CodeLlama (7B, 13B, 34B)
- Qwen2.5 Coder (7B, 32B)
- DeepSeek Coder (7B)
- CodeGemma (7B)
- Mistral (7B)
- Meta Llama 3 (8B)

## Evaluation Metrics

- **ROUGE-1**: Measures word overlap between generated and reference docstrings
- **BLEU**: Evaluates n-gram precision and brevity penalty
- **Conciseness**: Measures the length efficiency of generated docstrings
- **Custom Metrics**: Domain-specific evaluation criteria

## Results

Detailed evaluation results and model comparisons are available in the `results/` directory.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{docstringeval2024,
  title={DocStringEval: A Framework for Evaluating Automated Docstring Generation},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## Contact

For questions and support, please open an issue on GitHub or contact [your-email@domain.com].
