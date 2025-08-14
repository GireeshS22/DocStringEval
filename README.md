# DocStringEval: Evaluating the Effectiveness of Language Models for Code Explanation Through DocString Generation

A comprehensive framework for evaluating Large Language Models (LLMs) in automated docstring generation tasks for Python classes.

**Authors:** Gireesh Sundaram, Balaji Venktesh V, Sundharakumar K B

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
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules
├── LICENSE                     # MIT License
├── classes/                    # Python class files for evaluation
│   ├── _BaseEncoder.py
│   ├── Adamax.py
│   ├── AgglomerationTransform.py
│   ├── AveragePooling1D.py
│   ├── AveragePooling2D.py
│   ├── AveragePooling3D.py
│   ├── BayesianGaussianMixture.py
│   ├── Conv.py
│   ├── Conv1D.py
│   ├── Conv1DTranspose.py
│   ├── Conv2D.py
│   ├── Conv2DTranspose.py
│   ├── Conv3D.py
│   ├── Conv3DTranspose.py
│   ├── Cropping1D.py
│   ├── Cropping2D.py
│   ├── Cropping3D.py
│   ├── DBSCAN.py
│   ├── DepthwiseConv2D.py
│   ├── Embedding.py
│   ├── Flask.py
│   ├── FunctionTransformer.py
│   ├── GaussianMixture.py
│   ├── GlobalAveragePooling1D.py
│   ├── GlobalAveragePooling2D.py
│   ├── GlobalAveragePooling3D.py
│   ├── GlobalMaxPooling1D.py
│   ├── GlobalMaxPooling2D.py
│   ├── GlobalMaxPooling3D.py
│   ├── GlobalPooling1D.py
│   ├── GlobalPooling2D.py
│   ├── GlobalPooling3D.py
│   ├── GroupTimeSeriesSplit.py
│   ├── Kmeans.py
│   ├── LabelBinarizer.py
│   ├── LabelEncoder.py
│   ├── LinearRegression.py
│   ├── LogisticRegression.py
│   ├── Loss.py
│   ├── MaxPooling1D.py
│   ├── MaxPooling2D.py
│   ├── MaxPooling3D.py
│   ├── Metric.py
│   ├── MultiLabelBinarizer.py
│   ├── OneHotEncoder.py
│   ├── OPTICS.py
│   ├── OrdinalEncoder.py
│   ├── Pooling1D.py
│   ├── Pooling2D.py
│   ├── Pooling3D.py
│   ├── PrincipalComponentAnalysis.py
│   ├── RMSprop.py
│   ├── SelfTrainingClassifier.py
│   ├── SeparableConv.py
│   ├── SeparableConv1D.py
│   ├── SeparableConv2D.py
│   ├── SequentialFeatureSelector.py
│   ├── SGD.py
│   ├── SoftmaxRegression.py
│   ├── TargetEncoder.py
│   ├── TransactionEncoder.py
│   ├── UpSampling1D.py
│   ├── UpSampling2D.py
│   ├── UpSampling3D.py
│   ├── ZeroPadding1D.py
│   ├── ZeroPadding2D.py
│   └── ZeroPadding3D.py
├── Output/                     # Generated docstring outputs
│   ├── code-to-docstring-clean_codegemma-7b-it-dfe.json
│   ├── code-to-docstring-clean_codellama-7b-instruct-hf-yig.json
│   ├── code-to-docstring-clean_deepseek-coder-7b-instruct-v-riq.json
│   ├── code-to-docstring-clean_qwen2-5-coder-7b-instruct-ljc.json
│   ├── code-to-docstring-codegemma-7b-it-dfe.json
│   ├── code-to-docstring-codellama-34b-instruct-hf-kzi_NEW.json
│   ├── code-to-docstring-codellama-7b-instruct-hf-yig.json
│   ├── code-to-docstring-COT_codegemma-7b-it-dfe.json
│   ├── code-to-docstring-COT_codellama-7b-instruct-hf-yig.json
│   ├── code-to-docstring-COT_deepseek-coder-7b-instruct-v-fkg.json
│   ├── code-to-docstring-COT_qwen2-5-coder-7b-instruct-ljc.json
│   ├── code-to-docstring-deepseek-coder-7b-instruct-v-riq.json
│   ├── code-to-docstring-meta-llama-3-8b-instruct-gtq_NEW.json
│   ├── code-to-docstring-qwen2-5-7b-instruct-qne_NEW.json
│   ├── code-to-docstring-qwen2-5-coder-32b-instruct-qyb_NEW.json
│   └── code-to-docstring-qwen2-5-coder-7b-instruct-ljc.json
├── backup/                     # Backup files and previous outputs
│   ├── clean_code.json
│   ├── code-to-comments-clean-code-codellama-7b-instruct-hf-xfe.json
│   ├── code-to-comments-clean-code-qwen2-5-coder-7b-instruct-iul.json
│   ├── code-to-comments-codegemma-1-1-7b-it-uof.json
│   ├── code-to-comments-codellama-7b-instruct-hf-xfe.json
│   ├── code-to-comments-deepseek-coder-v2-lite-instr-zxn.json
│   ├── code-to-comments-llama-3-2-3b-instruct-msv.json
│   ├── code-to-comments-mistral-7b-instruct-v0-3-uly.json
│   ├── code-to-comments-qwen2-5-coder-7b-instruct-iul.json
│   ├── code-to-docstring-clean-code-codellama-7b-instruct-hf-xfe.json
│   ├── code-to-docstring-clean-codellama-13b-instruct-hf-abo.json
│   ├── code-to-docstring-clean-deepseek-coder-v2-lite-instr-atp.json
│   ├── code-to-docstring-clean-mistral-7b-instruct-v0-3-uly.json
│   └── code-to-docstring-clean-qwen2-5-coder-7b-instruct-iul-xfe.json
├── Core Scripts/               # Main evaluation and processing scripts
│   ├── analysis.py            # Analysis of evaluation results
│   ├── cleanup.py             # Code cleaning and preprocessing
│   ├── code to comments evaluation.py
│   ├── code to comments.py
│   ├── code to docstring evaluation.py
│   ├── code to docstring.py
│   ├── docstring generation.py
│   ├── extract_classes.py
│   ├── scoring.py
│   └── sample_class.py
├── Data Files/                 # Input data and processed files
│   ├── class collection.csv
│   ├── class collection.xlsx
│   ├── class_files_df.pkl
│   ├── clean_code
│   ├── LLM Hard questions.csv
│   ├── new scoring.xlsx
│   └── sdfsdf.csv
└── Results/                    # Evaluation results and reports
    ├── all llm scoring.xlsx
    └── all_scoiring.pkl
```

## Quick Start

### Installation

```bash
git clone https://github.com/GireeshS22/DocStringEval.git
cd DocStringEval
pip install -r requirements.txt
```

### Basic Usage

1. **Extract Classes**:
```bash
python extract_classes.py
```

2. **Generate Docstrings**:
```bash
python "code to docstring.py"
```

3. **Evaluate Results**:
```bash
python "code to docstring evaluation.py"
```

4. **Analyze Results**:
```bash
python analysis.py
```

## Supported Models

- **CodeLlama** (7B, 13B, 34B)
- **Qwen2.5 Coder** (7B, 32B)
- **DeepSeek Coder** (7B)
- **CodeGemma** (7B)
- **Mistral** (7B)
- **Meta Llama 3** (8B)

## Evaluation Metrics

- **ROUGE-1**: Measures word overlap between generated and reference docstrings
- **BLEU**: Evaluates n-gram precision and brevity penalty
- **Conciseness**: Measures the length efficiency of generated docstrings
- **Custom Metrics**: Domain-specific evaluation criteria

## Project Components

### Data Processing
- `extract_classes.py`: Extracts Python classes from repositories
- `cleanup.py`: Cleans and standardizes code for evaluation
- `class_files_df.pkl`: Processed dataset of Python classes

### Docstring Generation
- `code to docstring.py`: Main script for generating docstrings using LLMs
- `docstring generation.py`: Alternative generation approach
- `Output/`: Directory containing generated docstrings for each model

### Evaluation
- `code to docstring evaluation.py`: Evaluates generated docstrings using ROUGE and BLEU
- `scoring.py`: Custom scoring mechanisms
- `analysis.py`: Comprehensive analysis of evaluation results

### Results
- `all llm scoring.xlsx`: Comprehensive scoring results
- `all_scoiring.pkl`: Pickled scoring data for further analysis

## Results

Detailed evaluation results and model comparisons are available in the `Results/` directory and `Output/` directory.

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
@INPROCEEDINGS{11108633,
  author={Sundaram, Gireesh and Venktesh V, Balaji and K B, Sundharakumar},
  booktitle={2025 International Conference on Emerging Technologies in Computing and Communication (ETCC)}, 
  title={DocStringEval: Evaluating the Effectiveness of Language Models for Code Explanation Through DocString Generation}, 
  year={2025},
  volume={},
  number={},
  pages={1-7},
  keywords={Measurement;Codes;Accuracy;Large language models;Computational modeling;Computer architecture;Benchmark testing;Python;Large Language Models (LLMs);Code Explanation;Docstring Generation;Chain of Thought (CoT) Prompting;Code Summarization},
  doi={10.1109/ETCC65847.2025.11108633}
}
```

## Latest Updates

🚀 **Paper Published!** Our work "DocStringEval: Evaluating the Effectiveness of Language Models for Code Explanation through DocString Generation" has been published at IEEE ETCC 2025! 

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">🧵 🚀 NEW PAPER: &quot;DocStringEval: Evaluating the Effectiveness of Language Models for Code Explanation through DocString Generation&quot;<br>Published at IEEE ETCC 2025! 📄 <br>With <a href="https://twitter.com/Balajivenky4288?ref_src=twsrc%5Etfw">@Balajivenky4288</a> <br>We benchmarked how well LLMs can explain Python code by generating DocStrings.</p>&mdash; Gireesh (@GireeshS22) <a href="https://twitter.com/GireeshS22/status/1956054735523303689?ref_src=twsrc%5Etfw">August 14, 2025</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

## Contact

For questions and support, please:
- Open an issue on [GitHub](https://github.com/GireeshS22/DocStringEval/issues)
- Follow us on [X (Twitter)](https://x.com/GireeshS22) for updates and discussions
