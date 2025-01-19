# Importet e nevojshme
import pandas as pd
import numpy as np
import re
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix, precision_recall_fscore_support
from imblearn.over_sampling import RandomOverSampler
from sklearn.naive_bayes import GaussianNB
import optuna
import seaborn as sns
import matplotlib.pyplot as plt

# Leximi i dataset-it dhe pastrimi i të dhënave
def load_and_clean_data(filepath):
    """Lexon dhe pastron dataset-in.""" 
    job_postings = pd.read_csv(filepath)
    selected_columns = ["description", "formatted_experience_level"]
    df = job_postings[selected_columns].dropna(subset=['formatted_experience_level']).reset_index(drop=True)
    df['description'] = df['description'].astype(str)
    df["formatted_experience_level"] = np.where(df["formatted_experience_level"] == 'Entry level', 1, 0)
    return df

# Gjenerimi i veçorive
def generate_features(df, column_name):
    """Gjeneron veçori të bazuara në analizën e tekstit.""" 
    feature_columns = ['word_cnt', 'sent_cnt', 'vocab_cnt', 'Avg_sent_word_cnt', 'lexical_richness', 'Readability_index']
    feature_data = []

    for index, row in df.iterrows():
        text = row[column_name]
        words = re.findall(r'\b\w+\b', text)
        word_cnt = len(words)
        sentences = re.split(r'[.!?]+', text)
        sentences = [sent for sent in sentences if sent.strip()]
        sent_cnt = len(sentences)
        avg_word_length = sum(len(word) for word in words) / word_cnt if word_cnt > 0 else 0
        avg_sent_length = word_cnt / sent_cnt if sent_cnt > 0 else 0
        vocab = set(words)
        vocab_cnt = len(vocab)
        lex_richness = vocab_cnt / word_cnt if word_cnt > 0 else 0
        ARI = 4.71 * avg_word_length + 0.5 * avg_sent_length - 21.43
        feature_data.append([word_cnt, sent_cnt, vocab_cnt, avg_sent_length, lex_richness, ARI])

    feature_df = pd.DataFrame(feature_data, columns=feature_columns)
    result_df = pd.concat([df, feature_df], axis=1)
    return result_df

# Krijimi i kolonave të personalizuara
def add_custom_columns(df):
    """Shton kolona të personalizuara të bazuara në fjalë kyçe.""" 
    keywords = {
        'Cust_Service': 'customer service',
        'diploma_ged': 'diploma ged',
        'per_hour': 'per hour',
        'diploma_equiv': 'diploma equivalent',
        'project_management': 'project management',
        'cross_functional': 'cross functional',
        'minimum_years': 'minimum years',
        'experience_working': 'experience working',
        'management': 'management',
        'track_record': 'track record'
    }
    for col, keyword in keywords.items():
        df[col] = df['description'].apply(lambda x: 1 if keyword in x.lower() else 0)
    return df

# Normalizimi dhe ndarja e të dhënave
def preprocess_data(df):
    """Normalizon dhe ndan të dhënat për trajnimin dhe testimin.""" 
    x = df.drop(['description', 'formatted_experience_level'], axis=1)
    y = df['formatted_experience_level']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    scaler = MinMaxScaler()
    X_train_normalized = scaler.fit_transform(X_train_resampled)
    X_test_normalized = scaler.transform(X_test)
    return pd.DataFrame(X_train_normalized, columns=X_train.columns), pd.DataFrame(X_test_normalized, columns=X_test.columns), y_train_resampled, y_test

# Funksioni i optimizimit të hyperparameter
def objective(trial, X_train, y_train):
    """Funksion për të optimizuar hiperparametrat.""" 
    C_value = trial.suggest_float('C', 1e-4, 1e3, log=True)
    logistic_model = LogisticRegression(C=C_value, penalty='l1', solver='liblinear', max_iter=1000)
    cv_scores = cross_val_score(logistic_model, X_train, y_train, cv=10, scoring='accuracy')
    return cv_scores.mean()

# Trajnimi i modelit dhe optimizimi
def train_model(X_train, y_train):
    """Trajnon modelin dhe optimizon hiperparametrat.""" 
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=10)
    return study.best_params, study.best_value

# Ekzekutimi i procesit
if __name__ == "__main__":
    filepath = "job_postings.csv"
    df = load_and_clean_data(filepath)
    df = generate_features(df, 'description')
    df = add_custom_columns(df)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    start_time = time.time()
    best_params, best_accuracy = train_model(X_train, y_train)
    elapsed_time = time.time() - start_time
    print(f"Train Time: {elapsed_time:.2f} seconds")
    print(f"Best Hyperparameters: {best_params}")
    print(f"Best 10-Fold CV Accuracy: {best_accuracy:.2%}")

    # Përdorimi i hiperparametrave më të mirë nga optimizimi
    logistic_fe = LogisticRegression(C=best_params['C'], penalty='l1', solver='liblinear', max_iter=1000)
    logistic_fe.fit(X_train, y_train)

    # Parashikimet për datasetin e testimit
    y_test_pred_lfe = logistic_fe.predict(X_test)

    # Saktësia e testimit
    test_acc_lfe = accuracy_score(y_test, y_test_pred_lfe)
    print(f"Test Accuracy: {test_acc_lfe:.2%}")

    print("Coefficient Weights on Test Data:")
    coef_weights_test = logistic_fe.coef_
    for feature, coef in zip(X_train.columns, coef_weights_test.flatten()):
        print(f"{feature}: {coef:.4f}")

    # Parashikimi i probabiliteteve, vlerësimi i false positive rate, true positive rate dhe AUC
    y_test_prob_lfe = logistic_fe.predict_proba(X_test)[:, 1]  # Probability of class 1 (positive)
    fpr_lfe, tpr_lfe, threshold_lfe = roc_curve(y_test, y_test_prob_lfe)
    auc_score_lfe = roc_auc_score(y_test, y_test_prob_lfe)
    precision_lfe, recall_lfe, f1_lfe, _ = precision_recall_fscore_support(y_test, y_test_pred_lfe)
    print(f"AUC Score: {auc_score_lfe:.2%}")
    print(f"Precision: {precision_lfe[1]:.2%}")
    print(f"Recall: {recall_lfe[1]:.2%}")
    print(f"F1-Score: {f1_lfe[1]:.2%}")

    # Matja e matrikave të konfuzionit
    y_train_pred_lfe = logistic_fe.predict(X_train)
    conf_matrix_train_lfe = confusion_matrix(y_train, y_train_pred_lfe)
    conf_matrix_test_lfe = confusion_matrix(y_test, y_test_pred_lfe)

    # Vizualizimi i matrikave të konfuzionit
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # Heatmap për Train Set
    sns.heatmap(conf_matrix_train_lfe, annot=True, fmt='d', cmap='Blues', square=True, cbar=False,
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_title('Confusion Matrix\nTrain Set: Featured Engineered Logistic Model')

    # Heatmap për Test Set
    sns.heatmap(conf_matrix_test_lfe, annot=True, fmt='d', cmap='Blues', square=True, cbar=False,
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], ax=axes[1])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_title('Confusion Matrix\nTest Set: Featured Engineered Logistic Model');
    
    # Fitting the Naive Bayes model
    start_time = time.time()
    bayes_fe = GaussianNB()
    # Preforming 10-Fold CV
    bayes_fe_10fold = cross_val_score(bayes_fe, X_train, y_train, cv=10)
    end_time = time.time()
    # Computing Train Time
    bayes_fe_tt = end_time - start_time
    print(f"Training Duration: {bayes_fe_tt} Seconds")

    mean_acc_bfe = bayes_fe_10fold.mean()
    print(f"10-Fold Cross-Validation Accuracy: {mean_acc_bfe}")

    bayes_fe.fit(X_train, y_train)
    # Predict on train and test datasets
    y_test_pred_bfe = bayes_fe.predict(X_test)

    # Test Accuracy
    test_acc_bfe = accuracy_score(y_test, y_test_pred_bfe)
    print(f"Test Accuracy: {test_acc_bfe:.2%}")

    # Compute test probabilities, false positive rate, true positive rate, and auc
    y_test_prob_bfe = bayes_fe.predict_proba(X_test)[:, 1]  # Probability of class 1 (positive)
    fpr_bfe, tpr_bfe, threshold_bfe = roc_curve(y_test, y_test_prob_bfe)
    auc_score_bfe = roc_auc_score(y_test, y_test_prob_bfe)
    precision_bfe, recall_bfe, f1_bfe, _ = precision_recall_fscore_support(y_test, y_test_pred_bfe)
    print(f"AUC Score: {auc_score_bfe:.2%}")
    print(f"Precision: {precision_bfe[1]:.2%}")
    print(f"Recall: {recall_bfe[1]:.2%}")
    print(f"F1-Score: {f1_bfe[1]:.2%}")

    