import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, matthews_corrcoef, classification_report
import joblib

# 提取序列特征
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

def main():
    print('=' * 70)
    print('     评估 best_final_classifier_model.pkl 的真实性能')
    print('     (模型在6类数据上训练，评估时只看4类)')
    print('=' * 70)
    
    df_encoded = pd.read_csv('./esm2_encoded_data.csv')
    
    filtered_indices = df_encoded['label'].isin([0, 2, 4, 5])
    df_encoded = df_encoded[filtered_indices].reset_index(drop=True)
    
    df_original = pd.read_excel('../data/final_data/merged_data_updated1.xlsx')
    df_original = df_original[filtered_indices].reset_index(drop=True)
    

    label_names_original = {0: '降压', 1: '抗氧化（负样本）', 2: '抗菌', 3: '抗糖尿病（负样本）', 4: '抗肿瘤', 5: '负样本'}
    
    print(f'数据量: {len(df_encoded)} 个样本')
    
    sequence_features = []
    for seq in df_original['seq']:
        features = extract_sequence_features(seq)
        sequence_features.append(features)
    df_features = pd.DataFrame(sequence_features)
    
    X_esm2 = df_encoded.drop('label', axis=1)
    y = df_encoded['label']  # 保持原始标签 0, 2, 4, 5
    
    scaler = StandardScaler()
    df_features_scaled = pd.DataFrame(scaler.fit_transform(df_features), columns=df_features.columns)
    X = pd.concat([X_esm2, df_features_scaled], axis=1).fillna(0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = joblib.load('../best_final_classifier_model2.pkl')
    print(f'\n模型类型: {type(model).__name__}')
    print(f'模型训练时的类别数: 6类 (0-降压, 1-抗氧化, 2-抗菌, 3-抗糖尿病, 4-抗肿瘤, 5-负样本)')
    print(f'模型特征数: 1310 (ESM-2: 1280 + 序列特征: 30)')
    
    y_pred = model.predict(X_test)
    
    mapping = {0: 0, 1: -1, 2: 1, 3: -1, 4: 2, 5: 3}
    y_pred_mapped = np.array([mapping[p] for p in y_pred])
    
    y_test_mapped = np.array([mapping[t] for t in y_test])
    
    cm = confusion_matrix(y_test_mapped, y_pred_mapped, labels=[0, 1, 2, 3])
    
    print('\n混淆矩阵 (4x4):')
    print('行=真实, 列=预测')
    print('标签: 0-降压, 1-抗菌, 2-抗肿瘤, 3-负样本, -1=抗氧化/抗糖尿病')
    print(cm)
    
    print('\n各类别详细性能:')
    print('-' * 70)
    
    label_names = {0: '降压', 1: '抗菌', 2: '抗肿瘤', 3: '负样本'}
    sn_list = []
    sp_list = []
    
    for i in range(4):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        
        sn = tp / (tp + fn) if (tp + fn) > 0 else 0
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * sn / (precision + sn) if (precision + sn) > 0 else 0
        
        sn_list.append(sn)
        sp_list.append(sp)
        
        print(f'{label_names[i]}:')
        print(f'  灵敏度(SN): {sn:.4f}')
        print(f'  特异度(SP): {sp:.4f}')
        print(f'  精确率: {precision:.4f}')
        print(f'  F1分数: {f1:.4f}')
        print(f'  正确: {tp}, 漏报: {fn}, 误报: {fp}')
        print('-' * 70)
    
    avg_sn = np.mean(sn_list)
    avg_sp = np.mean(sp_list)
    
    filtered_idx = y_test_mapped >= 0
    bacc = balanced_accuracy_score(y_test_mapped[filtered_idx], y_pred_mapped[filtered_idx])
    mcc = matthews_corrcoef(y_test_mapped[filtered_idx], y_pred_mapped[filtered_idx])
    
    print('\n整体性能:')
    print('-' * 70)
    print(f'平均灵敏度(SN): {avg_sn:.4f}')
    print(f'平均特异度(SP): {avg_sp:.4f}')
    print(f'平衡准确率(BACC): {bacc:.4f}')
    print(f'马修斯相关系数(MCC): {mcc:.4f}')
    
    print('\n分类报告:')
    print('-' * 70)
    print(classification_report(y_test_mapped[filtered_idx], y_pred_mapped[filtered_idx], 
                                labels=[0, 1, 2, 3],
                                target_names=[label_names[i] for i in range(4)]))
    
    print('\n' + '=' * 70)

if __name__ == '__main__':
    main()