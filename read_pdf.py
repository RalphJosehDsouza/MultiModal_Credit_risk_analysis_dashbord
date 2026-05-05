import PyPDF2

reader = PyPDF2.PdfReader(r'c:\Mentor_MiniProject\credit-risk-prediction-model\log\Mini Project logbook 2025-26 (1).pdf')
print(f'Total pages: {len(reader.pages)}')

for i in range(len(reader.pages)):
    text = reader.pages[i].extract_text()
    print(f'=== PAGE {i+1} ===')
    print(text[:2500] if text else 'No text')
    print()