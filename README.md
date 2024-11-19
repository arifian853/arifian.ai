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
    
    v0.1 (persona_dataset2.json)
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
    Result visualization

    ![result1](/result/v0.1/loss-visualization.png)

    v0.2 (persona_dataset3.json)
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
    Result visualization

    ![result1](/result/v0.2/loss-visualization.png)

    v0.3 (persona_dataset4.json)
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
    Result visualization

    ![result1](/result/v0.3/loss_visualization.png)

    v0.3.1 (persona_dataset4.1.json)
    ```
    Result Summary
    ROUGE-1: 0.5903
    ROUGE-2: 0.4748
    ROUGE-L: 0.5678
    BLEU Score: 42.4546

    Training Losses Summary
    Loss per epoch:
    Epoch 1: 2.8721
    Epoch 2: 1.3137
    Epoch 3: 0.9076
    Epoch 4: 0.6958
    Epoch 5: 0.5559
    Epoch 6: 0.4559
    Epoch 7: 0.3847
    Epoch 8: 0.3312
    Epoch 9: 0.2707
    Epoch 10: 0.2375
    Epoch 11: 0.1969
    Epoch 12: 0.1854
    Epoch 13: 0.1579
    Epoch 14: 0.1361
    Epoch 15: 0.1339

    Average Training Loss: 0.5890
    Learning Rate: 0.0005
    Count of question-answer pair: 270

    ```
    Result visualization

    ![result1](/result/v0.3.1/loss_visualization.png)

    v0.4 & - v0.4.1 (persona_dataset5.json)
    ```
    v0.4
    Result Summary
    ROUGE-1: 0.6647
    ROUGE-2: 0.5067
    ROUGE-L: 0.6386
    BLEU Score: 50.0938

    16 Epochs with average training loss: 0.6282
    Learning Rate: 0.0005
    Count of question-answer pair: 297

    v0.4.1
    Result Summary
    ROUGE-1: 0.6640
    ROUGE-2: 0.5327
    ROUGE-L: 0.6441
    BLEU Score: 52.0508

    Training Losses Summary
    Loss per epoch:
    Epoch 1: 0.1451
    Epoch 2: 0.1388
    Epoch 3: 0.1398
    Epoch 4: 0.1283
    Epoch 5: 0.0957
    Epoch 6: 0.0946
    Epoch 7: 0.0858
    Epoch 8: 0.0819
    Epoch 9: 0.0683
    Epoch 10: 0.0650
    Epoch 11: 0.0708
    Epoch 12: 0.0693
    Epoch 13: 0.0688
    Epoch 14: 0.0599
    Epoch 15: 0.0617
    Epoch 16: 0.0601
    Epoch 17: 0.0458
    Epoch 18: 0.0575
    Epoch 19: 0.0582
    Epoch 20: 0.0429

    Average Training Loss: 0.0819
    Learning Rate: 0.0005
    Count of question-answer pair: 297
    ```
    Result visualization

    ![result1](/result/v0.4.1/loss_visualization.png)

- Conclusion (updated periodically)
    - This model need more data to achieve >= 0.9 ROUGE score and >= 30% BLEU score for greater result.
    - With a significant increase in the number of question-answer pairs, the model can achieve a BLEU score of 39.23%. However, the ROUGE score remains stagnant, leading to a larger average loss during training, and an almost stagnant result when tested.
    - Version 3.1 was the update with a slight improvement of BLEU and ROUGE score after increasing to 15 epochs after using 12 epoch before.
    - In version 4.0, after increasing the number of question-answer pairs, the model resulted in poor results due to training with many question-answer pairs that could only answer "I don't know about that" (out-of-context question-answer pairs, will be deleted in new v4.0).
    - Version 4.0.1 is the most significant improvement after creating new question-answer pairs and deleting all non-context question-answer pairs that could only answer "I don't know about that". It has an addition of 20+ question-answer pairs, and training was conducted with 16 and 20 epochs. The 20-epoch training performed the best.
    - The performance metrics of version 0.4 and 0.4.1 are as follows:
        - 16 epochs:
            - ROUGE-1: 0.6647
            - ROUGE-2: 0.5067
            - ROUGE-L: 0.6386
            - BLEU Score: 50.0938%
            - Average Training Loss: 0.6282
        - 20 epochs:
            - ROUGE-1: 0.6640
            - ROUGE-2: 0.5327
            - ROUGE-L: 0.6441
            - BLEU Score: 52.0508%
            - Average Training Loss: 0.0819
    
    - The model's performance is still not satisfactory, and further improvements are needed to achieve the desired performance.
    - Next update will use 18-20 epochs as defaults with a stagnant learning rate at 5e-4 (0.0005).
    - The model will available only for offline usage, download at [HuggingFace](https://huggingface.co/arifian853/arifian.ai).

- How to use

    - Download the model at HuggingFace [here](https://huggingface.co/arifian853/arifian.ai)
    - Adjust it to the path in the ```app.py``` code so the model path is : ```/trained_model/v0.3.1```
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
    http://127.0.0.1:5000/ask
    ```
    Or
    ```
    http://localhost:5000/ask
    ```