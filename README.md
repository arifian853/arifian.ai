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

- Dataset format is in JSON with pair of "question" and "answer" of my public personal data for method 2 and 3: 

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
    - This method is canceled.
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
    - This method is canceled.
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
    Result Summary
    ROUGE-1: 0.4902
    ROUGE-2: 0.2448
    ROUGE-L: 0.4419
    BLEU Score: 24.6660

    Training Losses Summary
    Loss per epoch:
    Epoch 1: 0.3075
    Epoch 2: 0.2838
    Epoch 3: 0.2173
    Epoch 4: 0.2309
    Epoch 5: 0.2179
    Epoch 6: 0.1489
    Epoch 7: 0.1363
    Epoch 8: 0.1398
    Epoch 9: 0.1107
    Epoch 10: 0.1170
    Epoch 11: 0.1083
    Epoch 12: 0.0939

    Average Training Loss: 0.1760
    Learning Rate: 0.0005
    Count of question-answer pair: 104
    ```
    - Result visualization

    ![result1](/result/v0.1/loss-visualization.png)

    - v0.2 (persona_dataset3.json)
    ```
    Result Summary
    ROUGE-1: 0.5797
    ROUGE-2: 0.3899
    ROUGE-L: 0.5502
    BLEU Score: 26.8416

    Training Losses Summary
    Loss per epoch:
    Epoch 1: 0.5053
    Epoch 2: 0.3060
    Epoch 3: 0.2035
    Epoch 4: 0.1460
    Epoch 5: 0.1338
    Epoch 6: 0.0889
    Epoch 7: 0.0724
    Epoch 8: 0.0938
    Epoch 9: 0.0800
    Epoch 10: 0.0661
    Epoch 11: 0.0704
    Epoch 12: 0.0541

    Average Training Loss: 0.1517
    Learning Rate: 0.0005
    Count of question-answer pair: 81
    ```
    - Result visualization

    ![result1](/result/v0.2/loss-visualization.png)

    - v0.2 (persona_dataset4.json)
    ```
    Result Summary
    ROUGE-1: 0.5548
    ROUGE-2: 0.4657
    ROUGE-L: 0.5456
    BLEU Score: 39.2361

    Training Losses Summary
    Loss per epoch:
    Epoch 1: 2.8626
    Epoch 2: 1.3558
    Epoch 3: 0.9349
    Epoch 4: 0.7211
    Epoch 5: 0.5845
    Epoch 6: 0.4805
    Epoch 7: 0.4145
    Epoch 8: 0.3672
    Epoch 9: 0.3231
    Epoch 10: 0.2663
    Epoch 11: 0.2168
    Epoch 12: 0.2115

    Average Training Loss: 0.7282
    Learning Rate: 0.0005
    Count of question-answer pair: 268
    ```
    - Result visualization

    ![result1](/result/v0.3/loss-visualization.png)

- Conclusion
    - This model need more data to achieve >= 0.9 ROUGE score and >= 30% BLEU score for greater result.
    - With a significant increase in the number of question-answer pairs, the model can achieve a BLEU score of 39.23%. However, the ROUGE score remains stagnant, leading to a larger average loss during training, but has greater result when tested.
    - Current plan: increase the number of question-answer pairs in the dataset with lots of augmentations.

- How to use, (download the model at HuggingFace ![here](https://huggingface.co/arifian853/arifian.ai))

    - Clone the repository, or just the ```app.py``` file.
    - Make new virtual environment and activate it
    -  Install the requirements
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
    - Accessable API endpoint to test
    ```
    http://127.0.0.1/5000/ask
    ```