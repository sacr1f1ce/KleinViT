import PyPDF2
import sys

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text

# Extract from each PDF
pdfs = ['carlsson2007.pdf', 'perea2013.pdf', 'Topological_Conv_Layers.pdf']

for pdf in pdfs:
    print(f"\n\n{'='*50}")
    print(f"Extracting from {pdf}")
    print('='*50)
    try:
        text = extract_text_from_pdf(pdf)
        print(text[:3000])  # Print first 3000 characters
    except Exception as e:
        print(f"Error extracting {pdf}: {e}")
