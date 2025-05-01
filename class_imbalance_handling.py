
#!/usr/bin/env python
"""


Bu script:
1. sqlcode.py içindeki load_snapshot_data() fonksiyonuyla snapshot verisini çeker.
2. Özellikleri ölçeklendirir ve eğitim/test setlerine ayırır.
3. İki yöntemle class imbalance sorununu çözer:
   a) Sınıf ağırlıkları (class_weight='balanced')  
   b) SMOTE ile azınlık sınıfı artırma  
4. sklearn MLPClassifier kullanarak her senaryoda modeli eğitir ve
   Accuracy ile ROC-AUC sonuçlarını karşılaştırır.
   
Gerekli paketler:
pip install imbalanced-learn
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

from sqlcode import load_snapshot_data

def train_and_evaluate(clf, X_train, y_train, X_test, y_test, label):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"\n>> {label}")
    print(f"Accuracy: {acc:.3f}, ROC-AUC: {auc:.3f}")
    print(classification_report(y_test, y_pred, zero_division=0))

def main():
    # 1) Veriyi çek
    df = load_snapshot_data()
    FEATURES = [
        'total_spent', 'total_orders', 'avg_order_value',
        'days_since_prev_order', 'order_month'
    ]
    X = df[FEATURES].values
    y = df['reorder_6m'].values

    # 2) Ölçeklendir ve ayır
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3) Baseline modele
    clf_base = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=200, random_state=42)
    train_and_evaluate(clf_base, X_train, y_train, X_test, y_test, "Baseline")

    # 4) class_weight yöntemi
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {cls: w for cls, w in zip(classes, weights)}
    clf_cw = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=200,
                           class_weight=class_weight_dict, random_state=42)
    train_and_evaluate(clf_cw, X_train, y_train, X_test, y_test, "Class Weight Balanced")

    # 5) SMOTE yöntemi
    sm = SMOTE(random_state=42)
    X_smote, y_smote = sm.fit_resample(X_train, y_train)
    clf_smote = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=200, random_state=42)
    train_and_evaluate(clf_smote, X_smote, y_smote, X_test, y_test, "SMOTE Augmented")

if __name__ == "__main__":
    main()
