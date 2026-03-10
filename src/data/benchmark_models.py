import os
import time
import psutil
import threading
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def track_resources(target_func, *args):
    """Monitors CPU and RAM usage of a target function using a background thread."""
    cpu_usage = []
    ram_usage = []
    
    # NOTE: Using a mutable list for 'monitoring' allows the main thread 
    # to update its value and signal the background thread to stop.
    monitoring = [True]

    def monitor_resources():
        p = psutil.Process(os.getpid())
        while monitoring[0]:
            cpu_usage.append(p.cpu_percent(interval=0.1))
            ram_usage.append(p.memory_info().rss / (1024 * 1024))

    t = threading.Thread(target=monitor_resources)
    t.start()

    start = time.time()
    result = target_func(*args)
    end = time.time()

    monitoring[0] = False
    t.join()

    # NOTE: Safeguard against empty lists if target_func executes in < 0.1s.
    avg_cpu = sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0.0
    max_cpu = max(cpu_usage) if cpu_usage else 0.0
    max_ram = max(ram_usage) if ram_usage else psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

    # NOTE: CPU is measured for just 1 core (So the output can be more than 100% if a divice has multiple cores)
    metrics = {
        "Time (s)": round(end - start, 4),
        "Avg CPU (%)": round(avg_cpu, 2),
        "Max CPU (%)": round(max_cpu, 2),
        "Max RAM (MB)": round(max_ram, 2)
    }
    
    return result, metrics

def main():
    df = pd.read_csv("data/processed/processed_data.csv")
    
    X = df.drop(columns=["Label"])
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "k-NN": KNeighborsClassifier(n_neighbors=5),
        "SVM Linear": SVC(kernel="linear", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, tree_method="hist", eval_metric="logloss")
    }

    results_list = []

    for name, model in models.items():        
        print(f"Benchmarking {name}...")
        
        _, train_metrics = track_resources(model.fit, X_train, y_train)
        y_pred, inf_metrics = track_resources(model.predict, X_test)
        
        acc = accuracy_score(y_test, y_pred)
        cr = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        ras = roc_auc_score(y_test,y_pred)
        cv = cross_val_score(model,X_test,y_test,cv=5,scoring="f1_weighted").mean()

        
        combined = {
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(cr["weighted avg"]["precision"], 4),
            "Recall": round(cr["weighted avg"]["recall"], 4),
            "F1-score": round(cr["weighted avg"]["f1-score"], 4),
            "ROC-AUC": round(ras,4),
            "CV-f1": round(cv,4),
            "Train Time (s)": train_metrics["Time (s)"],
            "Train Max CPU (%)": train_metrics["Max CPU (%)"],
            "Train Max RAM (MB)": train_metrics["Max RAM (MB)"],
            "Inf Time (s)": inf_metrics["Time (s)"],
            "Inf Max CPU (%)": inf_metrics["Max CPU (%)"],
            "Inf Max RAM (MB)": inf_metrics["Max RAM (MB)"]
        }
        
        results_list.append(combined)
        
        safe_name = name.replace(' ', '_').lower()
        joblib.dump(model, f"src/models/{safe_name}.joblib")

    comparison_table = pd.DataFrame(results_list).set_index("Model")
    
    comparison_table.to_csv("src/data/resource_usage_report.csv")
    print("\nBenchmarking complete! Report saved.")

if __name__ == "__main__":
    main()