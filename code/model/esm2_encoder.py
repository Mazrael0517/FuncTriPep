import pandas as pd
import torch
from transformers import EsmForMaskedLM, EsmTokenizer

df = pd.read_excel('merged_data_updated1.xlsx')

sequences = df['seq'].tolist()
labels = df['label'].tolist()

model_name = "/home/Michael/Michael/model/esm2_t33_650M_UR50D"
tokenizer = EsmTokenizer.from_pretrained(model_name)
model = EsmForMaskedLM.from_pretrained(model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def encode_sequence(sequence):
    inputs = tokenizer(sequence, return_tensors='pt', padding=True, truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states[-1]
    
    sequence_embedding = torch.mean(hidden_states, dim=1).squeeze().cpu().numpy()
    
    return sequence_embedding

encoded_sequences = []
for i, seq in enumerate(sequences):
    if i % 100 == 0:
        print(f'编码进度: {i}/{len(sequences)}')
    embedding = encode_sequence(seq)
    encoded_sequences.append(embedding)

encoded_df = pd.DataFrame(encoded_sequences)
encoded_df['label'] = labels

encoded_df.to_csv('esm2_encoded_data.csv', index=False)
print('编码完成，结果已保存到 esm2_encoded_data.csv')