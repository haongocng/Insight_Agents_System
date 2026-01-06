import os
import shutil
import sys
from logger import setup_logger
from load_cfg import WORKING_DIRECTORY, GOOGLE_API_KEY, LANGCHAIN_API_KEY
from agents.manager import ManagerAgent

logger = setup_logger()

class InsightAgentsSystem:
    def __init__(self):
        self.logger = logger
        self.logger.info("Initializing Insight Agents System...")
        
        self.setup_environment()
        
        self.setup_runtime_tools()
        
        self.manager = ManagerAgent()
        self.logger.info("System Initialized Successfully.")

    def setup_environment(self):
        if not os.environ.get("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        
        if not os.environ.get("LANGCHAIN_API_KEY"):
            os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
            os.environ["LANGCHAIN_TRACING"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = "Insight-Agents-Demo"

        if not os.path.exists(WORKING_DIRECTORY):
            os.makedirs(WORKING_DIRECTORY)
            self.logger.info(f"Created working directory: {WORKING_DIRECTORY}")

    def setup_runtime_tools(self):
        try:
            possible_sources = [
                os.path.join("tools", "ml_ops.py"),
                "ml_ops.py"
            ]
            
            source_path = None
            for path in possible_sources:
                if os.path.exists(path):
                    source_path = path
                    break
            
            if not source_path:
                self.logger.error("CRITICAL: Could not find 'ml_ops.py'. Agent will fail to import tools.")
                return

            dest_path = os.path.join(WORKING_DIRECTORY, "ml_tools.py")
            
            shutil.copy(source_path, dest_path)
            self.logger.info(f"Deployed Runtime Tool Library: {source_path} -> {dest_path}")
            
        except Exception as e:
            self.logger.error(f"Error setting up runtime tools: {e}")

    def run(self, user_input: str):
        print(f"\n{'='*50}")
        print(f"USER REQUEST: {user_input}")
        print(f"{'='*50}\n")
        
        try:
            response = self.manager.run(user_input)
            
            print(f"\n{'-'*20} FINAL RESPONSE {'-'*20}")
            print(response)
            print(f"{'-'*56}\n")
            
        except Exception as e:
            self.logger.error(f"System execution failed: {e}", exc_info=True)
            print(f"Error: {str(e)}")

def main():
    ia_system = InsightAgentsSystem()

    # user_input_heart_disease = """
    # datapath: train_heart.csv
    # test_datapath: test_heart.csv
    # task: Binary Classification
    # target_column: HeartDisease
    # Objective: 
    # Build a machine learning model to predict the presence of heart disease (target 'HeartDisease': 1 = Yes, 0 = No).
    # Execution Guidelines (Must use 'ml_tools' library):
    # 1.  **Preprocessing:** - Check for missing values and handle them using `fill_missing_values`.
    #     - Detect outliers in numerical columns (like Cholesterol, RestingBP) using `detect_and_handle_outliers_zscore`.
    #     - Encode categorical variables  using `one_hot_encode` or `label_encode`.
    #     - Scale numerical features using `scale_features`.
    # 2.  **Modeling:** - Use the function `train_and_validation_and_select_the_best_model` from `ml_tools`.
    #     - Candidate models to try: Logistic Regression, Random Forest, XGBoost, SVM.
    #     - Report the models performance: Accuracy, Precision, Recall, F1-Score using from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score with the 'results.md' file.
    # 3.  **Evaluation & Prediction:** - Report the best model's metrics: Accuracy, Precision, Recall, F1-Score using from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score with the 'final_results.md' file.
    #     - Apply the SAME preprocessing steps to the test dataset ('test_heart.csv').
    #     - Generate predictions using the best trained model.
    #     - Save the final predictions to a file named 'predictions.csv' with columns: [id, label_pred].
    # """
    # ia_system.run(user_input_heart_disease)

    user_input_fake_news = """
    datapath: train.csv
    test_datapath: test.csv
    sample_datapath: sample_dataset.csv
    task: Binary Classification (Fake News Detection)
    target_column: label
    Objective: 
    Build a machine learning model to classify news articles as Real (0) or Fake (1).
    Execution Guidelines (Hybrid approach: Python for NLP + 'ml_tools' for Modeling):
    1.  **Data Inspection (CRITICAL CONSTRAINT):** -   **DO NOT** read the full 'train.csv' for initial inspection/EDA because it is too large. 
        -   Use `pd.read_csv('sample_dataset.csv', sep='\t')` to understand the data structure (columns, text field, label).
        -   Only load 'train.csv' (with `sep='\t'`) when you are ready to train.
    2.  **Preprocessing (Text to Features):**
        -   Since `ml_tools` works on numerical data, you must explicitly write Python code to vectorization text:
            -   Handle missing values in text columns (fill with empty string).
            -   Combine text columns (title + author + text) if necessary.
            -   Use `sklearn.feature_extraction.text.TfidfVectorizer` to convert text X into numerical features.
            -   Prepare target y from the 'label' column.
    3.  **Modeling (Must use 'ml_tools'):** -   Once you have the vectorized X and y, pass them into the function `train_and_validation_and_select_the_best_model_4Classification` from `ml_tools`.
        -   Arguments: `X, y, problem_type='binary', selected_models=['LogisticRegression', 'SVM', 'RandomForest']`.
        -   Report performance: Accuracy, Precision, Recall, F1-Score using `from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score` and save to 'results.md' (Print `metrics_df` to see all results.).
    4.  **Evaluation & Prediction:** -   Load 'test.csv' (handle `sep='\t'` if needed).
        -   Apply the **SAME** TfidfVectorizer (transform only) used on training data to the test data.
        -   Generate predictions using the best trained model.
        -   **ID Handling:** If 'id' column is missing in test data, generate it using the DataFrame index.
        -   Save predictions to 'predictions.csv' with columns: [id, label_pred].
        -   Create a 'final_results.md' summary report.
    NOTE THAT: MUST NOT print raw dataframes (e.g., `print(df)` or `print(df.head())`) to the console for avoiding content length errors.
    """
    
    ia_system.run(user_input_fake_news)

    # user_input_education = """
    # datapath: edudata_english.csv
    # target_column: I am willing to share my digital skills with other students
    # Objective: 
    # Analyze student willingness to share digital skills using two approaches: Classification and Regression.
    # Execution Guidelines (use 'ml_tools'):
    # 1.  **Preprocessing (Shared for both tasks):**
    # 2.  **Task 1: Multiclass Classification:**
    #     - Treat the target column as a Categorical variable.
    #     - Use `train_and_validation_and_select_the_best_model_4Classification`:
    #     - Report metrics: Accuracy, precision, recall, f1-Score using `from sklearn.metrics` with .4f save result to 'class_report.md'.
    # 3.  **Task 2: Regression Analysis:**
    #     - Treat the target column as a Numerical variable (reuse the preprocessed X).
    #     - Use `train_and_validation_and_select_the_best_model_4Regression`:
    #         - Arguments: `X, y, problem_type='regression', with selected_models you choose`.
    #     - Report metrics: MSE , RMSE, R2-score. save result to 'reg_report.md'.
    # 4.  **Final Deliverables:**
    #     - Create a summary file 'final_analysis.md' comparing which approach (Classification vs Regression) yielded more interpretable results.
    # """
    
    # ia_system.run(user_input_education)

    #     user_input_education = """
    # datapath: edudata_english.csv
    # target_column: I am willing to share my digital skills with other students
    # Objective: 
    # Analyze student willingness to share digital skills using two approaches: Classification and Regression.
    # Execution Guidelines (use 'ml_tools'):
    # 1.  **Preprocessing (Shared for both tasks):**
    #     - Load 'edudata_english.csv'.
    #     - Handle missing values: `fill_missing_values(df, columns=['...'], method='auto')`.
    #     - Handle outliers: `detect_and_handle_outliers_zscore` for numerical columns (e.g., Age, Time spent...).
    #     - Encode categorical columns: `label_encode` (for ordinal features like 'Education Level') or `one_hot_encode` (for nominal features like 'Gender').
    #     - Scale numerical features: `scale_features`.
    # 2.  **Task 1: Multiclass Classification:**
    #     - Treat the target column as a Categorical variable.
    #     - Use `train_and_validation_and_select_the_best_model`:
    #         - Arguments: `X, y, problem_type='multiclass' with selected_models you choose`.
    #     - Report metrics: Accuracy, precision, recall, f1-Score using `from sklearn.metrics` with .4f save result to 'class_report.md'.
    # 3.  **Task 2: Regression Analysis:**
    #     - Treat the target column as a Numerical variable (reuse the preprocessed X).
    #     - Use `train_and_validation_and_select_the_best_model`:
    #         - Arguments: `X, y, problem_type='regression', with selected_models you choose`.
    #     - Report metrics: MSE , RMSE, R2-score. save result to 'reg_report.md'.
    # 4.  **Final Deliverables:**
    #     - Create a summary file 'final_analysis.md' comparing which approach (Classification vs Regression) yielded more interpretable results.
    # """
    
    # ia_system.run(user_input_education)

if __name__ == "__main__":
    main()