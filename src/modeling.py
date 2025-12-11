from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.data_preparation import load_split_data, get_datasets
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import torch
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib

train_df, val_df, test_df = load_split_data()
X_train, X_val, X_test, y_train, y_val, y_test = get_datasets()

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

y_val_pred = lr_model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

y_test_pred = lr_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

c_m = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(c_m, annot=True, fmt="d", cmap="Reds")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.savefig('/Users/lailaalmohaymid/PycharmProjects/data_mining/reports/cm_lr.png')
plt.show()

svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)

y_val_pred_svm = svm_model.predict(X_val)
print("SVM Validation Accuracy:", accuracy_score(y_val, y_val_pred_svm))
print(classification_report(y_val, y_val_pred_svm))

y_test_pred_svm = svm_model.predict(X_test)
print("SVM Test Accuracy:", accuracy_score(y_test, y_test_pred_svm))
print(classification_report(y_test, y_test_pred_svm))

cm_svm = confusion_matrix(y_test, y_test_pred_svm)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - SVM')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('/Users/lailaalmohaymid/PycharmProjects/data_mining/reports/cm_svm.png')
plt.show()

rf_model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

y_val_pred_rf = rf_model.predict(X_val)
print("Random Forest Validation Accuracy:", accuracy_score(y_val, y_val_pred_rf))
print(classification_report(y_val, y_val_pred_rf))

y_test_pred_rf = rf_model.predict(X_test)
print("RandomForest Test Accuracy:", accuracy_score(y_test, y_test_pred_rf))
print(classification_report(y_test, y_test_pred_rf))

cm_rf = confusion_matrix(y_test, y_test_pred_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Purples')
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('/Users/lailaalmohaymid/PycharmProjects/data_mining/reports/cm_rf.png')
plt.show()

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

bert_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device=device)

X_train_emb = bert_model.encode(train_df["cleaned_text"].tolist(), convert_to_numpy=True, show_progress_bar=True)
X_val_emb = bert_model.encode(val_df["cleaned_text"].tolist(), convert_to_numpy=True, show_progress_bar=True)
X_test_emb = bert_model.encode(test_df["cleaned_text"].tolist(), convert_to_numpy=True, show_progress_bar=True)

y_train_nn = (train_df["label"] == "ai").astype(int).values
y_val_nn = (val_df["label"] == "ai").astype(int).values
y_test_nn = (test_df["label"] == "ai").astype(int).values

ffnn_model = models.Sequential([
    layers.Input(shape=(X_train_emb.shape[1],)),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

ffnn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
ffnn_model.fit(X_train_emb, y_train_nn, validation_data=(X_val_emb, y_val_nn), epochs=10, batch_size=32)

y_test_pred_nn = (ffnn_model.predict(X_test_emb) > 0.5).astype(int).flatten()
print("Neural Network Test Accuracy:", accuracy_score(y_test_nn, y_test_pred_nn))
print(classification_report(y_test_nn, y_test_pred_nn))

cm_nn = confusion_matrix(y_test_nn, y_test_pred_nn)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix - Neural Network')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('/Users/lailaalmohaymid/PycharmProjects/data_mining/reports/cm_nn.png')
plt.show()



joblib.dump(lr_model, '/Users/lailaalmohaymid/PycharmProjects/data_mining/models/logistic_regression.joblib')
joblib.dump(svm_model, '/Users/lailaalmohaymid/PycharmProjects/data_mining/models/svm.joblib')
joblib.dump(rf_model, '/Users/lailaalmohaymid/PycharmProjects/data_mining/models/random_forest.joblib')
ffnn_model.save('/Users/lailaalmohaymid/PycharmProjects/data_mining/models/neural_network.h5')

print("Models saved")