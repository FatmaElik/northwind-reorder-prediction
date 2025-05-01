#!/usr/bin/env python
"""


Bu script:
1. sqlcode.py içindeki load_snapshot_data() fonksiyonunu kullanarak Northwind snapshot verisini çeker.
2. Özellikleri ölçeklendirir ve eğitim/test setlerine ayırır.
3. Orijinal, Noise Injection ve SMOTE yöntemleriyle veriyi augment eder.
4. Her senaryo için scikit-learn'ün MLPClassifier modeliyle eğitip
   Accuracy ve ROC-AUC değerlerini karşılaştırır.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE

# sqlcode.py'den load_snapshot_data fonksiyonunu import edin
from sqlcode import load_snapshot_data

def augment_with_noise(X, y, factor=2, noise_level=0.01, random_state=42):
    np.random.seed(random_state)
    n_samples = X.shape[0]
    n_new = int(n_samples * (factor - 1))
    indices = np.random.randint(0, n_samples, size=n_new)
    X_selected = X[indices]
    noise = np.random.normal(scale=noise_level, size=X_selected.shape)
    X_noised = X_selected + noise
    y_noised = y[indices]
    X_aug = np.vstack([X, X_noised])
    y_aug = np.concatenate([y, y_noised])
    return X_aug, y_aug

def train_and_evaluate(X_train, y_train, X_test, y_test):
    clf = MLPClassifier(hidden_layer_sizes=(64,32),
                        max_iter=200,
                        random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return acc, auc

def main():
    # 1) Veriyi çek
    df = load_snapshot_data()

    # 2) Özellik ve etiket
    FEATURES = [
        'total_spent',
        'total_orders',
        'avg_order_value',
        'days_since_prev_order',
        'order_month'
    ]
    X = df[FEATURES].values
    y = df['reorder_6m'].values

    # 3) Ölçeklendirme ve eğitim/test ayrımı
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4) Orijinal veri ile değerlendirme
    print(">> Orijinal Veri")
    acc0, auc0 = train_and_evaluate(X_train, y_train, X_test, y_test)

    # 5) Noise injection ile augmentation
    print("\n>> Noise Injection Augmentation")
    X_noise, y_noise = augment_with_noise(X_train, y_train, factor=3, noise_level=0.02)
    acc1, auc1 = train_and_evaluate(X_noise, y_noise, X_test, y_test)

    # 6) SMOTE ile augmentation
    print("\n>> SMOTE Augmentation")
    sm = SMOTE(random_state=42)
    X_smote, y_smote = sm.fit_resample(X_train, y_train)
    acc2, auc2 = train_and_evaluate(X_smote, y_smote, X_test, y_test)

    # 7) Sonuçların özetlenmesi
    print("\n=== Performans Karşılaştırması ===")
    print(f"Orijinal   : Accuracy = {acc0:.3f}, ROC-AUC = {auc0:.3f}")
    print(f"NoiseAug   : Accuracy = {acc1:.3f}, ROC-AUC = {auc1:.3f}")
    print(f"SMOTEAug   : Accuracy = {acc2:.3f}, ROC-AUC = {auc2:.3f}")

if __name__ == "__main__":
    main()

