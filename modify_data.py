import json

def modify_data(data):
    """
    Fungsi untuk memodifikasi data JSON sesuai permintaan.

    Args:
        data: Data JSON dalam bentuk list of dictionaries.

    Returns:
        List of dictionaries dengan format yang telah dimodifikasi.
    """

    modified_data = []
    for item in data:
        modified_item = {
            "prompt": f"Pertanyaan: {item['prompt']}",
            "response": f"Jawaban: {item['response']}"
        }
        modified_data.append(modified_item)

    return modified_data

# Load data dari file (sesuaikan dengan nama file Anda)
with open('data.json', 'r') as f:
    data = json.load(f)

# Modifikasi data
modified_data = modify_data(data)

# Simpan data yang telah dimodifikasi ke file baru
with open('modified_data.json', 'w') as f:
    json.dump(modified_data, f, indent=4)

print("Data telah berhasil dimodifikasi.")