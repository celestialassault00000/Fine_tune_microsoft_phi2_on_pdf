import PyPDF2
import json
from datasets import load_dataset
from constants.paths import output_file , pdf_file_path
def split_text_into_chunks(text, chunk_size):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

def save_chunks_to_jsonl(chunks, output_file):
    with open(output_file, 'w') as f:
        for chunk in chunks:
            data = {'text': chunk}
            f.write(json.dumps(data) + '\n')

def read_pdf_and_save_chunks(pdf_file, output_file=output_file, start_page=0, chunk_size=1000):
    with open(pdf_file, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(start_page - 1, len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        chunks = split_text_into_chunks(text, chunk_size)
        save_chunks_to_jsonl(chunks, output_file)


def load_dataset():
    pdf_file_path = input()
    read_pdf_and_save_chunks(pdf_file_path)
    dataset = load_dataset('json', data_files=output_file)
    return dataset
