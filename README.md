# Semantic-text-similarity

## Overview

Semantic Text Similarity is a Natural Language Processing (NLP) project that aims to assess the degree of semantic equivalence between two given sentences. The project leverages the power of BERT, a transformer-based language model, to generate contextual embeddings and calculate cosine similarity to measure the semantic similarity between text pairs.

## Features

- Utilizes the pre-trained BERT model for semantic text embeddings.
- Calculates cosine similarity to quantify the degree of similarity.
- Supports multiple domains and languages for a broad range of applications.
- Provides a scalable and robust solution for text-based similarity assessments.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/coding-bot007/semantic-text-similarity.git
cd semantic-text-similarity
```

2. Create and activate a virtual environment (optional, but recommended):

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Edit the `text1` and `text2` variables in `main_stream.py` with the sentences you want to compare.

2. Load the Semantics_model.sav file to the ML model

3. Run the script to calculate the semantic text similarity:

```bash
python main_stream.py
```

3. The script will output the cosine similarity score between the two sentences.

## License

No such Licenses yet.

## Contributing

Contributions are welcome! If you find any issues or have improvements to suggest, please open an issue or submit a pull request.

## Acknowledgments

- The project utilizes the Hugging Face `transformers` library for BERT integration.
- Special thanks to the Streamlit community for hosting the application.

## Contact

For questions or inquiries, please contact [ameernadaf0@gmail.com].
