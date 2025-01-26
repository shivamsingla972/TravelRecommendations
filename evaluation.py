from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')

    # For precision@k and recall@k
    k = 10  
    precision_at_k = precision_score(y_test[:k], y_pred[:k], average='micro')
    recall_at_k = recall_score(y_test[:k], y_pred[:k], average='micro')

    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Precision@k: {precision_at_k}')
    print(f'Recall@k: {recall_at_k}')
