import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler


df_encoded = pd.read_csv('esm2_encoded_data.csv')


df_original = pd.read_excel('../../data/merged_data_updated1.xlsx')

# 提取序列特征
def extract_sequence_features(sequence):
    features = {}
    # 序列长度
    features['length'] = len(sequence)
    
    # 氨基酸组成
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    for aa in amino_acids:
        features[f'freq_{aa}'] = sequence.count(aa) / len(sequence) if len(sequence) > 0 else 0
    
    # 物理化学性质分组
    # 疏水性氨基酸
    hydrophobic = 'AILFMWV'
    features['hydrophobic_freq'] = sum(1 for aa in sequence if aa in hydrophobic) / len(sequence) if len(sequence) > 0 else 0
    
    # 极性氨基酸
    polar = 'NQSTY'
    features['polar_freq'] = sum(1 for aa in sequence if aa in polar) / len(sequence) if len(sequence) > 0 else 0
    
    # 带电氨基酸
    charged = 'DEKRH'
    features['charged_freq'] = sum(1 for aa in sequence if aa in charged) / len(sequence) if len(sequence) > 0 else 0
    
    # 酸性氨基酸
    acidic = 'DE'
    features['acidic_freq'] = sum(1 for aa in sequence if aa in acidic) / len(sequence) if len(sequence) > 0 else 0
    
    # 碱性氨基酸
    basic = 'KRH'
    features['basic_freq'] = sum(1 for aa in sequence if aa in basic) / len(sequence) if len(sequence) > 0 else 0
    
    # 序列模式特征
    features['has_proline'] = 1 if 'P' in sequence else 0
    features['has_cysteine'] = 1 if 'C' in sequence else 0
    features['has_tryptophan'] = 1 if 'W' in sequence else 0
    
    # 芳香性氨基酸
    aromatic = 'FWYH'
    features['aromatic_freq'] = sum(1 for aa in sequence if aa in aromatic) / len(sequence) if len(sequence) > 0 else 0
    
    return features

sequence_features = []
for seq in df_original['seq']:
    features = extract_sequence_features(seq)
    sequence_features.append(features)

df_features = pd.DataFrame(sequence_features)

# 合并特征
X_esm2 = df_encoded.drop('label', axis=1)
y = df_encoded['label']

scaler = StandardScaler()
df_features_scaled = pd.DataFrame(scaler.fit_transform(df_features), columns=df_features.columns)

X = pd.concat([X_esm2, df_features_scaled], axis=1)

print(f'处理前特征数据形状: {X.shape}')
print(f'NaN值数量: {X.isna().sum().sum()}')
X = X.fillna(0)  # 用0填充NaN值
print(f'处理后特征数据形状: {X.shape}')
print(f'NaN值数量: {X.isna().sum().sum()}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

label_map = {0: '降压', 1: '抗氧化（负样本）', 2: '抗菌', 3: '抗糖尿病（负样本）', 4: '抗肿瘤', 5: '负样本'}
labels_to_show = {0: '降压', 2: '抗菌', 4: '抗肿瘤', 5: '负样本'}

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    sn_list = []
    sp_list = []
    print('各功能类别性能分析:')
    print('-' * 60)
    for i in range(len(cm)):
        if i in labels_to_show:
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            
            sn = tp / (tp + fn) if (tp + fn) > 0 else 0
            sp = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            sn_list.append(sn)
            sp_list.append(sp)
            
            print(f'{labels_to_show[i]} (标签{i}):')
            print(f'  灵敏度(SN): {sn:.4f}')
            print(f'  特异度(SP): {sp:.4f}')
            print(f'  正确预测: {tp}, 漏报: {fn}, 误报: {fp}')
            print(f'  测试集中该类别的样本数: {tp + fn}')
            print('-' * 60)
    
    avg_sn = np.mean(sn_list) if sn_list else 0
    avg_sp = np.mean(sp_list) if sp_list else 0
    
    bacc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    print('\n整体性能:')
    print('-' * 60)
    print(f'平均灵敏度(SN): {avg_sn:.4f}')
    print(f'平均特异度(SP): {avg_sp:.4f}')
    print(f'平衡准确率(BACC): {bacc:.4f}')
    print(f'马修斯相关系数(MCC): {mcc:.4f}')
    print('-' * 60)
    
    return avg_sn, avg_sp, bacc, mcc, cm

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42, probability=True),
    'MLP': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'LightGBM': LGBMClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    print(f'训练 {name}...')
    model.fit(X_train, y_train)
    sn, sp, bacc, mcc, cm = evaluate_model(model, X_test, y_test)
    results[name] = {'SN': sn, 'SP': sp, 'BACC': bacc, 'MCC': mcc, 'Confusion Matrix': cm}
    print(f'{name} 评估结果:')
    print(f'  SN: {sn:.4f}')
    print(f'  SP: {sp:.4f}')
    print(f'  BACC: {bacc:.4f}')
    print(f'  MCC: {mcc:.4f}')
    print()

print('训练集成模型...')
top_models = sorted(results.items(), key=lambda x: x[1]['BACC'], reverse=True)[:3]
en_estimators = [(name, model) for name, _ in top_models]
ensemble_model = VotingClassifier(estimators=en_estimators, voting='soft')
ensemble_model.fit(X_train, y_train)
ensemble_sn, ensemble_sp, ensemble_bacc, ensemble_mcc, ensemble_cm = evaluate_model(ensemble_model, X_test, y_test)
results['Ensemble'] = {'SN': ensemble_sn, 'SP': ensemble_sp, 'BACC': ensemble_bacc, 'MCC': ensemble_mcc, 'Confusion Matrix': ensemble_cm}
print('集成模型评估结果:')
print(f'  SN: {ensemble_sn:.4f}')
print(f'  SP: {ensemble_sp:.4f}')
print(f'  BACC: {ensemble_bacc:.4f}')
print(f'  MCC: {ensemble_mcc:.4f}')
print()

best_model_name = max(results, key=lambda x: results[x]['BACC'])
best_model = models[best_model_name] if best_model_name != 'Ensemble' else ensemble_model

print(f'最佳模型: {best_model_name}')
print(f'最佳模型性能:')
print(f'  SN: {results[best_model_name]["SN"]:.4f}')
print(f'  SP: {results[best_model_name]["SP"]:.4f}')
print(f'  BACC: {results[best_model_name]["BACC"]:.4f}')
print(f'  MCC: {results[best_model_name]["MCC"]:.4f}')

import joblib
joblib.dump(best_model, 'best_final_classifier_model2.pkl')
print('\n最佳模型已保存到 best_final_classifier_model2.pkl')