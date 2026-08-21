import gc
import random
import yaml
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def find_hyperparameters():
    print("=== Extended Hyperparameter Search for RandomForestClassifier ===")
    
    # 1. Load Data
    df_train = pd.read_csv('data/train_phase1.csv')
    df_eval = pd.read_csv('data/eval.csv')

    X_train = df_train.drop(columns=['target'])
    y_train = df_train['target']
    X_eval = df_eval.drop(columns=['target'])
    y_eval = df_eval['target']

    # 2. Define Comprehensive Search Space (All standard sklearn RandomForestClassifier hyperparameters)
    n_estimators_opts = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 1000]
    criterion_opts = ['gini', 'entropy', 'log_loss']
    max_depth_opts = [None, 5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 50]
    min_samples_split_opts = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15]
    min_samples_leaf_opts = [1, 2, 3, 4, 5, 6, 8]
    min_weight_fraction_leaf_opts = [0.0, 0.001, 0.01]
    max_features_opts = ['sqrt', 'log2', None, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    max_leaf_nodes_opts = [None, 20, 50, 100, 200, 500]
    min_impurity_decrease_opts = [0.0, 1e-5, 1e-4, 1e-3]
    bootstrap_opts = [True, False]
    class_weight_opts = [None, 'balanced', 'balanced_subsample']
    ccp_alpha_opts = [0.0, 0.0001, 0.001, 0.005, 0.01]
    max_samples_opts = [None, 0.5, 0.7, 0.8, 0.9]

    best_acc = 0.0
    best_params = {}
    successful_configs = []

    num_trials = 500

    print(f"Running {num_trials} trials with strict RAM cleanup...\n")

    for i in range(1, num_trials + 1):
        is_bootstrap = random.choice(bootstrap_opts)
        
        params = {
            'n_estimators': random.choice(n_estimators_opts),
            'criterion': random.choice(criterion_opts),
            'max_depth': random.choice(max_depth_opts),
            'min_samples_split': random.choice(min_samples_split_opts),
            'min_samples_leaf': random.choice(min_samples_leaf_opts),
            'min_weight_fraction_leaf': random.choice(min_weight_fraction_leaf_opts),
            'max_features': random.choice(max_features_opts),
            'max_leaf_nodes': random.choice(max_leaf_nodes_opts),
            'min_impurity_decrease': random.choice(min_impurity_decrease_opts),
            'bootstrap': is_bootstrap,
            'class_weight': random.choice(class_weight_opts),
            'ccp_alpha': random.choice(ccp_alpha_opts)
        }

        if is_bootstrap:
            max_samp = random.choice(max_samples_opts)
            if max_samp is not None:
                params['max_samples'] = max_samp

        # Clean dictionary values where needed while preserving max_depth: None if specified
        clean_params = {k: v for k, v in params.items() if v is not None or k == 'max_depth'}

        try:
            # Train model
            model = RandomForestClassifier(**clean_params, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)

            preds = model.predict(X_eval)
            acc = float(accuracy_score(y_eval, preds))
            f1 = float(f1_score(y_eval, preds, average='weighted'))

            if acc > best_acc:
                best_acc = acc
                best_params = clean_params
                print(f"[Trial {i:03d}/{num_trials}] NEW BEST -> Accuracy: {acc:.4f} | F1: {f1:.4f}")
                print(f"             Params: {best_params}\n")

            if acc >= 0.70:
                successful_configs.append((acc, f1, clean_params))

            del model
        except Exception as e:
            pass
        finally:
            # Memory release
            gc.collect()

    print("=" * 70)
    print(f"SEARCH COMPLETED.")
    print(f"Highest Accuracy Achieved: {best_acc:.4f}")

    if successful_configs:
        print(f"\nFound {len(successful_configs)} configuration(s) achieving accuracy >= 0.70:")
        successful_configs.sort(key=lambda x: (x[0], x[1]), reverse=True)
        top_acc, top_f1, top_params = successful_configs[0]
        
        print(f"\nTop Configuration -> Accuracy: {top_acc:.4f} | F1: {top_f1:.4f}")
        print(f"Params: {top_params}")
        
        with open('params.yaml', 'w') as f:
            yaml.dump(top_params, f, default_flow_style=False)
        print("\n[SUCCESS] Updated params.yaml with the best hyperparameters (Accuracy >= 0.70).")
    else:
        print("\n[WARNING] No configuration reached 0.70 accuracy in this run.")
        if best_params:
            print("Writing current best params to params.yaml...")
            with open('params.yaml', 'w') as f:
                yaml.dump(best_params, f, default_flow_style=False)

if __name__ == "__main__":
    find_hyperparameters()
