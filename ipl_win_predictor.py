import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import os

# ==========================================
# 1. DATA LOADING & CLEANING
# ==========================================
def load_and_clean_data(matches_path, deliveries_path):
    print("Loading datasets...")
    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)
    
    # Filter out matches with no result
    matches = matches[matches['result'] != 'no result']
    matches = matches.dropna(subset=['winner'])
    
    return matches, deliveries

# ==========================================
# 2. FEATURE ENGINEERING & DATASET GENERATION
# ==========================================
def perform_feature_engineering(matches, deliveries):
    print("Performing feature engineering...")
    
    # 1. Calculate the first innings score to determine the target
    first_innings = deliveries[deliveries['inning'] == 1]
    total_scores = first_innings.groupby('match_id')['total_runs'].sum().reset_index()
    total_scores.rename(columns={'total_runs': 'first_innings_total'}, inplace=True)
    total_scores['target'] = total_scores['first_innings_total'] + 1
    
    # 2. Extract only the second innings data
    second_innings = deliveries[deliveries['inning'] == 2].copy()
    
    # Merge the target score with the second innings deliveries
    second_innings = second_innings.merge(total_scores[['match_id', 'target']], on='match_id', how='inner')
    
    # Merge relevant match details (winner, season) with the second innings data
    match_details = matches[['id', 'winner', 'season']]
    second_innings = second_innings.merge(match_details, left_on='match_id', right_on='id', how='inner')
    
    # 3. Calculate cumulative score and wickets
    second_innings['current_score'] = second_innings.groupby('match_id')['total_runs'].cumsum()
    second_innings['is_wicket'] = second_innings['is_wicket'].fillna(0).astype(int)
    second_innings['wickets_lost'] = second_innings.groupby('match_id')['is_wicket'].cumsum()
    
    # 4. Group by over to get the match state AFTER EVERY OVER
    # 'last()' gets the state at the end of the over
    over_data = second_innings.groupby(['match_id', 'over']).last().reset_index()
    
    # 5. Calculate remaining balls, runs left, and run rates
    # Overs are 0-indexed (0 to 19). Balls bowled after over O = (O + 1) * 6
    over_data['balls_bowled'] = (over_data['over'] + 1) * 6
    over_data['balls_left'] = 120 - over_data['balls_bowled']
    
    # Runs left cannot be negative; if negative, chasing team won and runs_left is 0
    over_data['runs_left'] = over_data['target'] - over_data['current_score']
    over_data['runs_left'] = over_data['runs_left'].apply(lambda x: x if x > 0 else 0)
    
    # Current Run Rate (CRR)
    over_data['current_run_rate'] = (over_data['current_score'] / over_data['balls_bowled']) * 6
    
    # Required Run Rate (RRR)
    # If balls_left is 0, set RRR to a high number if runs are left, else 0
    over_data['required_run_rate'] = over_data.apply(
        lambda row: (row['runs_left'] / row['balls_left']) * 6 if row['balls_left'] > 0 else (99 if row['runs_left'] > 0 else 0), 
        axis=1
    )
    
    # 6. Create Target Variable (result): 1 if chasing team wins, 0 otherwise
    over_data['result'] = (over_data['batting_team'] == over_data['winner']).astype(int)
    
    # Select final features
    features_df = over_data[[
        'match_id', 'season', 'batting_team', 'bowling_team', 'over', 'balls_left', 
        'runs_left', 'wickets_lost', 'target', 'current_run_rate', 'required_run_rate', 'result'
    ]]
    
    return features_df, second_innings

# ==========================================
# 3. MODEL TRAINING & EVALUATION
# ==========================================
def split_data_by_season(df):
    print("Splitting data based on seasons...")
    # Find the latest season in the dataset to act as the test set
    latest_season = df['season'].max()
    print(f"Using season '{latest_season}' as the test set.")
    
    train_df = df[df['season'] != latest_season]
    test_df = df[df['season'] == latest_season]
    
    features = ['runs_left', 'balls_left', 'wickets_lost', 'target', 'current_run_rate', 'required_run_rate']
    
    X_train = train_df[features]
    y_train = train_df['result']
    
    X_test = test_df[features]
    y_test = test_df['result']
    
    return X_train, X_test, y_train, y_test, features

def evaluate_model(y_test, y_pred, model_name):
    print(f"\n--- {model_name} Evaluation ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# ==========================================
# 4. VISUALIZATION
# ==========================================
def plot_feature_importance(rf_model, features, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)
    
    plt.figure(figsize=(10, 6))
    plt.title("Random Forest Feature Importances")
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png")
    plt.close()
    print(f"Feature importance graph saved to {output_dir}/feature_importance.png")

def plot_model_comparison(log_acc, rf_acc, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    models = ['Logistic Regression', 'Random Forest']
    accuracies = [log_acc, rf_acc]
    
    plt.figure(figsize=(8, 5))
    plt.bar(models, accuracies, color=['#005B96', '#03396C'])
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 1)
    
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png")
    plt.close()
    print(f"Model comparison graph saved to {output_dir}/model_comparison.png")

# ==========================================
# 5. SPECIAL FEATURE: MATCH PROGRESSION
# ==========================================
def predict_match_progress(match_id, match_data_df, model, features, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    # Extract the data for the specific match
    match_df = match_data_df[match_data_df['match_id'] == match_id].copy()
    
    if match_df.empty:
        print(f"No second innings data found for match_id {match_id}")
        return
    
    # Sort by over to ensure chronological order
    match_df = match_df.sort_values('over')
    
    X = match_df[features]
    
    # Predict probabilities
    # predict_proba returns [prob_loss, prob_win]
    win_probs = model.predict_proba(X)[:, 1]
    loss_probs = model.predict_proba(X)[:, 0]
    
    batting_team = match_df.iloc[0]['batting_team']
    bowling_team = match_df.iloc[0]['bowling_team']
    
    print(f"\nMatch {match_id} - Over-by-Over Win Probability:")
    print(f"Chasing Team: {batting_team} | Defending Team: {bowling_team}")
    for over, p_win, p_loss in zip(match_df['over'], win_probs, loss_probs):
        # over is 0-indexed, so we add 1 for display
        print(f"Over {over + 1:2d} -> {batting_team}: {p_win * 100:3.0f}% | {bowling_team}: {p_loss * 100:3.0f}%")
        
    # Plotting Match Momentum
    plt.figure(figsize=(10, 6))
    plt.plot(match_df['over'] + 1, win_probs * 100, marker='o', linestyle='-', color='#B30000', linewidth=2, label='Win Probability %')
    plt.axhline(50, color='gray', linestyle='--', label='50% Threshold')
    
    plt.title(f"Match Momentum Graph: Match {match_id}")
    plt.xlabel("Over")
    plt.ylabel("Chasing Team Win Probability (%)")
    plt.xticks(range(1, 21))
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    file_path = f"{output_dir}/match_{match_id}_momentum.png"
    plt.savefig(file_path)
    plt.close()
    print(f"Match momentum graph saved to {file_path}")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Ensure output folder exists
    if not os.path.exists('output'):
        os.makedirs('output')

    # 1. Load Data
    matches, deliveries = load_and_clean_data('matches.csv', 'deliveries.csv')
    
    # 2. Feature Engineering
    features_df, raw_second_innings = perform_feature_engineering(matches, deliveries)
    
    # 3. Train-Test Split
    X_train, X_test, y_train, y_test, feature_names = split_data_by_season(features_df)
    
    # 4. Model Training: Logistic Regression
    print("\nTraining Logistic Regression...")
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)
    y_pred_log = log_model.predict(X_test)
    evaluate_model(y_test, y_pred_log, "Logistic Regression")
    
    # 5. Model Training: Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    evaluate_model(y_test, y_pred_rf, "Random Forest")
    
    # 6. Save Models
    with open('output/logistic_regression_model.pkl', 'wb') as f:
        pickle.dump(log_model, f)
    with open('output/random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    print("\nModels saved to 'output' directory.")
    
    # 7. Generate Visualizations
    log_acc = accuracy_score(y_test, y_pred_log)
    rf_acc = accuracy_score(y_test, y_pred_rf)
    plot_model_comparison(log_acc, rf_acc)
    plot_feature_importance(rf_model, feature_names)
    
    # 8. Interactive Match Progression
    print("\n" + "="*50)
    print("MATCH PREDICTOR - INTERACTIVE MODE")
    print("="*50)
    
    test_matches = features_df.loc[X_test.index, 'match_id'].unique()
    if len(test_matches) > 0:
        print(f"Sample test match IDs you can try: {test_matches[0]}, {test_matches[1] if len(test_matches) > 1 else ''} ...")
    
    while True:
        user_input = input("\nEnter a match_id to see progression (or 'q' to quit): ").strip()
        if user_input.lower() == 'q':
            print("Exiting...")
            break
        try:
            match_id_input = int(user_input)
            predict_match_progress(match_id_input, features_df, log_model, feature_names)
        except ValueError:
            print("Please enter a valid numeric match_id.")
