### Arifian.AI

Building a personal AI chatbot using LLM that can interact like a virtual version of myself.

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


Goals : The LLM can answers question related to my data.