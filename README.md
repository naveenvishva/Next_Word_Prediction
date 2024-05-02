
# Text Generation using LSTM Neural Network

This repository contains code for generating text using a Long Short-Term Memory (LSTM) neural network. The model is trained on the adventures of Sherlock Holmes dataset and is capable of generating text based on a seed text input.

## Overview

The code provided here demonstrates how to train an LSTM model on a text dataset and then use the trained model to generate new text based on an initial seed text. The LSTM model is built using TensorFlow and Keras.

## Prerequisites

Make sure you have the following dependencies installed:

- Python (>=3.6)
- TensorFlow (2.7.0)
- NumPy (1.21.5)
- Requests (2.26.0)

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Dataset

The model is trained on the "Sherlock Holmes" adventures dataset, which is provided in the file `sherlock-holm.es_stories_plain-text_advs.txt`. Ensure that the dataset file is placed in the same directory as the code file.

## Usage

1. Clone this repository:

```bash
git clone https://github.com/your-username/your-repository.git
```

2. Navigate to the cloned directory:

```bash
cd your-repository
```

3. Run the script:

```bash
python text_generation.py
```

This will train the LSTM model on the dataset and generate text based on the provided seed text.

## Parameters

You can adjust the following parameters in the script according to your requirements:

- `max_sequence_len`: Maximum length of input sequences.
- `total_words`: Total number of unique words in the dataset.
- LSTM model parameters such as the number of LSTM units and embedding dimensions can also be modified in the model architecture section.

## Credits

This project is inspired by various tutorials and examples available online for text generation using LSTM neural networks.

