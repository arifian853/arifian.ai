<h1 align="center">  Arifian.AI </h1>

<p align="center"> 
    Building a personal AI chatbot using a pre-trained and fine-tuned LLM, or building an NLP model from scratch that can interact like a virtual AI version/persona of myself.
</p>

<div align="center">
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
    <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white">
    <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">
    <img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white">
    <img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white">
    <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white"> 
</div>

## Dataset

- Dataset format is in JSON with pair of "prompt" and "response" of my public personal data for method 1: 

```
[
    {
        "prompt": "Arifian, apa hobi kamu?",
        "response": "Hobi saya adalah bermain game dan mendengarkan musik."
    },
    {
        "prompt": "Arifian, berapa umurmu?",
        "response": "Saya berumur 22 tahun."
    },
]
```

- Dataset format is in JSON with pair of "question" and "answer" of my public personal data for method 2: 

```
[
    {
        "question": "Siapakah kamu, Arifian?",
        "answer": "Namaku Arifian Saputra, Salam kenal!"
    },
    {
        "question": "Arifian, berapa umurmu?",
        "answer": "Saat ini aku berumur 22 tahun."
    },
]
```
## Methods

### Method 1 - Fine-tune small LLMs
<details>
<summary> Click to see details </summary>

LLM used : GPT-2, GPT Neo 1.3B, Google T5 Base, Google T5 Large

Worth to try : Meta Llama, Mistral (Canceled)

Fine-tuning history : 

- GPT-2 (Result : Bad)
```
https://colab.research.google.com/drive/1TojHMkPg8UXHeC7rekZWy9f6RY03WolM?usp=sharing
```

- GPT Neo 1.3B (Result : Bad)
``` 
https://colab.research.google.com/drive/1zowBZTTdbs1-X57uFRxfNrK1v2JAsJpo?usp=sharing
```

- GPT Neo 1.3B with LoRA (Result : Bad)
```
https://colab.research.google.com/drive/1y8MWtBxcKYgAc8TmQmEpeYOONZvjpuhe?usp=sharing
```

- Google T5 Base (Result : Bad)
```
https://colab.research.google.com/drive/1P5hID7iAIZqHQ8cS5WaYoQ4QrcM08WAK?usp=sharing
```

- Google T5 Large (Result : Worse)
```
https://colab.research.google.com/drive/1t7Z6ZSjnIa8UNFmhnQ9G71UreMFBoEG7?usp=sharing
```

- Conclusions
This method is canceled.
</details>


### Method 2 - Build manual NLP
<details>
<summary> Click to see details </summary>
- Bidirectional LSTM (Result: Bad, need more dataset)
```
https://colab.research.google.com/drive/16rCFvfDOn1Dtc7b8_SQasTdYTVy8C3i4?usp=sharing
```

- SVM (Support Vector Machine) (Result: Better, need to extend the context)
```
https://colab.research.google.com/drive/1HA50eQx8brVg043e4SgaH47DfvDhJ_49?usp=sharing
```

- Conclusions
This method is canceled.
</details>

### Method 3 - Fine-tuning with pre-trained model T5 and USE feature extraction

- Result: Moderate

- T5 (Text-to-Text Transfer Transformer) and USE (Universal Sentence Encoder) 
```
https://colab.research.google.com/drive/17fm1X9B88tkIkvxSboUn6wsoMNifbRnr?usp=sharing
```

## Result

The method used in this project is Method 3, fine-tuning with pre-trained model T5 and USE feature extraction.

- Result:
    - v0.1 (persona_dataset2.json)
    ```
    ROUGE-1: 0.4902
    ROUGE-2: 0.2448
    ROUGE-L: 0.4419
    BLEU Score: 24.6660 (24.66%)

    Epoch : 12
    Average Training Loss: 0.1760
    Learning Rate: 5e-4
    ```
    - Result visualization
    ![result1](/result/v0.1/loss-visualization.png)

    - v0.2 (persona_dataset3.json)
    ```
    ROUGE-1: 0.5797
    ROUGE-2: 0.3899
    ROUGE-L: 0.5502
    BLEU Score: 26.8416 (26.84%)

    Epoch : 12
    Average Training Loss: 0.1517
    Learning Rate: 5e-4
    ```
    - Result visualization
    ![result1](/result/v0.2/loss-visualization.png)

- Conclusion
This model need more data to achieve >= 0.9 ROUGE score and >= 30% BLEU score for greater result.

- How to use

    -  Requirements
    ```
    flask
    flask_cors
    transformers
    torch
    sentencepiece
    ```
    - Run with:
    ```
    py app.py
    ```
    - API endpoints
    ```
    http://127.0.0.1/5000/ask
    ```