import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def extract_sequence_features(sequence):
    features = {}
    features['length'] = len(sequence)
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    for aa in amino_acids:
        features[f'freq_{aa}'] = sequence.count(aa) / len(sequence) if len(sequence) > 0 else 0
    hydrophobic = 'AILFMWV'
    features['hydrophobic_freq'] = sum(1 for aa in sequence if aa in hydrophobic) / len(sequence) if len(sequence) > 0 else 0
    polar = 'NQSTY'
    features['polar_freq'] = sum(1 for aa in sequence if aa in polar) / len(sequence) if len(sequence) > 0 else 0
    charged = 'DEKRH'
    features['charged_freq'] = sum(1 for aa in sequence if aa in charged) / len(sequence) if len(sequence) > 0 else 0
    acidic = 'DE'
    features['acidic_freq'] = sum(1 for aa in sequence if aa in acidic) / len(sequence) if len(sequence) > 0 else 0
    basic = 'KRH'
    features['basic_freq'] = sum(1 for aa in sequence if aa in basic) / len(sequence) if len(sequence) > 0 else 0
    features['has_proline'] = 1 if 'P' in sequence else 0
    features['has_cysteine'] = 1 if 'C' in sequence else 0
    features['has_tryptophan'] = 1 if 'W' in sequence else 0
    aromatic = 'FWYH'
    features['aromatic_freq'] = sum(1 for aa in sequence if aa in aromatic) / len(sequence) if len(sequence) > 0 else 0
    return features

def load_sequences(file_path):
    df = pd.read_excel(file_path)
    sequences = []
    seq_ids = []
    for idx, seq in enumerate(df.iloc[:, 0].tolist()):
        if isinstance(seq, str):
            sequences.append(seq.strip())
            seq_ids.append(idx + 1)
    return sequences, seq_ids

def main():
    class_labels = ['ACE抑制肽', '非ACE抑制肽']
    input_file = 'original_data.xlsx'
    print(f'\n加载序列文件: {input_file}')
    sequences, seq_ids = load_sequences(input_file)
    print(f'成功加载 {len(sequences)} 条序列')
    
    print('\n提取序列手工特征...')
    sequence_features = [extract_sequence_features(seq) for seq in sequences]
    df_features = pd.DataFrame(sequence_features)
    
    print('标准化手工特征...')
    scaler = StandardScaler()
    df_features_scaled = pd.DataFrame(scaler.fit_transform(df_features), columns=df_features.columns)
    
    print('加载ESM-2编码数据...')
    esm2_df = pd.read_csv('../embedding/esm2_encoded_data.csv')
    X_esm2 = esm2_df.drop('label', axis=1)
    
    print('合并特征 (ESM-2 + 手工特征 = 1310维)...')
    X_combined = pd.concat([X_esm2, df_features_scaled], axis=1).fillna(0)
    
    print('加载分类模型...')
    model = joblib.load('../best_final_classifier_model2.pkl')
    
    print('进行预测...')
    y_pred = model.predict(X_combined)
    y_prob = model.predict_proba(X_combined)
    
    print('处理预测结果...')
    results = []
    
    for i, (seq, pred, prob) in enumerate(zip(sequences, y_pred, y_prob)):
        ace_prob = prob[0]
        non_ace_prob = prob[2] + prob[4] + prob[5]
        
        total_prob = ace_prob + non_ace_prob
        if total_prob > 0:
            ace_prob_norm = ace_prob / total_prob
            non_ace_prob_norm = non_ace_prob / total_prob
        else:
            ace_prob_norm = ace_prob
            non_ace_prob_norm = non_ace_prob
        
        if pred == 0:
            pred_label = 'ACE抑制肽'
        else:
            pred_label = '非ACE抑制肽'
        
        max_prob_value = max(ace_prob_norm, non_ace_prob_norm)
        max_prob_func = 'ACE抑制肽' if ace_prob_norm >= non_ace_prob_norm else '非ACE抑制肽'
        
        results.append({
            '序列编号': seq_ids[i],
            '序列': seq,
            '预测功能': pred_label,
            '最可能功能': max_prob_func,
            '预测概率': max_prob_value,
            'ACE抑制肽概率': ace_prob_norm,
            '非ACE抑制肽概率': non_ace_prob_norm
        })
    
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('ACE抑制肽概率', ascending=False).reset_index(drop=True)
    result_df['排名'] = range(1, len(result_df) + 1)
    
    result_df = result_df[[
        '排名', '序列编号', '序列', '预测功能', '最可能功能', '预测概率',
        'ACE抑制肽概率', '非ACE抑制肽概率'
    ]]
    
    print('\n' + '=' * 70)
    print('预测统计结果')
    print('=' * 70)
    pred_counts = result_df['预测功能'].value_counts()
    for func in class_labels:
        count = pred_counts.get(func, 0)
        percentage = count / len(result_df) * 100
        print(f'{func}: {count} 条 ({percentage:.1f}%)')
    
    ace_peptides = result_df[result_df['预测功能'] == 'ACE抑制肽']
    print(f'\nACE抑制肽预测详情:')
    print(f'  - 预测为ACE抑制肽: {len(ace_peptides)} 条')
    print(f'  - ACE抑制肽概率最高的序列: {ace_peptides.iloc[0]["序列"]} (概率: {ace_peptides.iloc[0]["ACE抑制肽概率"]:.4f})')
    
    output_file = 'peptide_ace_binary_predictions.xlsx'
    with pd.ExcelWriter(output_file) as writer:
        result_df.to_excel(writer, sheet_name='二分类预测结果', index=False)
        
        ace_df = result_df[result_df['预测功能'] == 'ACE抑制肽'].copy()
        ace_df.sort_values('ACE抑制肽概率', ascending=False, inplace=True)
        ace_df.reset_index(drop=True, inplace=True)
        ace_df['功能内排名'] = range(1, len(ace_df) + 1)
        ace_df.to_excel(writer, sheet_name='ACE抑制肽', index=False)
        
        non_ace_df = result_df[result_df['预测功能'] == '非ACE抑制肽'].copy()
        non_ace_df.sort_values('非ACE抑制肽概率', ascending=False, inplace=True)
        non_ace_df.reset_index(drop=True, inplace=True)
        non_ace_df['功能内排名'] = range(1, len(non_ace_df) + 1)
        non_ace_df.to_excel(writer, sheet_name='非ACE抑制肽', index=False)
    
    print(f'\n预测结果已保存到: {output_file}')
    print('文件包含3个Sheet: 二分类预测结果、ACE抑制肽、非ACE抑制肽')

if __name__ == '__main__':
    main()