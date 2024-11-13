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



### Method 1 - Fine-tune small LLMs
<details>
<summary> Here </summary>

LLM used : GPT-2, GPT Neo 1.3B, Google T5 Base, Google T5 Large

Worth to try : Meta Llama, Mistral

- Dataset format is in JSON with pair of "prompt" and "response" of my puclic personal data: 

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

- Bidirectional LSTM (Result: Bad, need more dataset)
```
https://colab.research.google.com/drive/16rCFvfDOn1Dtc7b8_SQasTdYTVy8C3i4?usp=sharing
```

- SVM (Support Vector Machine) (Result: Better, need to extend the context)
```
https://colab.research.google.com/drive/1HA50eQx8brVg043e4SgaH47DfvDhJ_49?usp=sharing
```

- T5 (Text-to-Text Transfer Transformer) and USE (Universal Sentence Encoder)
```
https://colab.research.google.com/drive/17fm1X9B88tkIkvxSboUn6wsoMNifbRnr?usp=sharing
```