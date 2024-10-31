import json
import os
from groq import Groq

# Inisialisasi Groq client
client = Groq(
    api_key="",
)

# Membaca dataset JSON
with open('data.json', 'r') as f:
    dataset = json.load(f)

# Mengunggah dataset ke Groq API
# Buat koleksi data untuk pencarian prompt-response
# Perluas sesuai instruksi dokumentasi Groq untuk mengatur koleksi
for item in dataset:
    client.documents.create(
        collection="arifian_collection",
        document=item
    )

# Fungsi untuk menerima input pengguna dan mencari respons terdekat
def get_response_from_groq(user_input):
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": user_input}],
        model="llama3-8b-8192",
        collection="arifian_collection"  # Tentukan nama koleksi
    )
    
    # Mengembalikan respons terdekat
    return chat_completion.choices[0].message.content

# Mengambil input dari pengguna
user_input = input("Tanyakan sesuatu: ")
response = get_response_from_groq(user_input)
print("Chatbot:", response)