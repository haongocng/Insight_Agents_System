from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from logger import setup_logger

logger = setup_logger()

ML_TOOLS_DOCS = """
LIBRARY 'ml_tools' API DOCUMENTATION:
(You MUST use these exact function signatures. Do not invent arguments.)

1. Data Cleaning:
   - fill_missing_values(data: pd.DataFrame, columns: List[str], method='auto') -> pd.DataFrame
   - remove_duplicates(data: pd.DataFrame) -> pd.DataFrame
   - detect_and_handle_outliers_zscore(data: pd.DataFrame, columns: List[str], threshold=3.0) -> pd.DataFrame

2. Feature Engineering:
   - one_hot_encode(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame
   - label_encode(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame
   - scale_features(data: pd.DataFrame, columns: List[str], method='standard') -> pd.DataFrame
   - create_polynomial_features(data: pd.DataFrame, columns: List[str], degree=2) -> pd.DataFrame

3. Modeling - CLASSIFICATION ONLY:
   - train_and_validation_and_select_the_best_model_4Classification(X, y, selected_models=['RandomForest', 'XGBoost', 'SVM']) -> (best_model, metrics_df)
     * Use for: labels (0/1), categorical targets.
     * Params: selected_models (list of strings).
     * Returns: 
       - best_model: The trained estimator.
       - metrics_df: DataFrame with columns ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'].

4. Modeling - REGRESSION ONLY:
   - train_and_validation_and_select_the_best_model_4Regression(X, y, selected_models=['LinearRegression', 'RandomForest']) -> (best_model, metrics_df)
     * Use for: continuous numerical targets.
     * Params: selected_models (list of strings).
     * Returns: 
       - best_model: The trained estimator.
       - metrics_df: DataFrame with columns ['Model', 'MSE', 'RMSE', 'MAE', 'R2'].
"""

INSIGHT_WORKER_PROMPT = f"""
You are the Insight Generator Agent, a specialized worker in a Data Analysis system.

**Your Goal:**
Execute specific data analysis tasks (EDA, Preprocessing, Modeling, Training, Prediction) as directed by the Manager or User.

**CRITICAL: API-BASED DATA MODEL**
You have access to a local Python library named 'ml_tools'. 
This library contains robust, pre-defined functions for data operations.

**AVAILABLE TOOLS (READ CAREFULLY):**
{ML_TOOLS_DOCS}

**Operational Guidelines:**
1.  **Prefer 'ml_tools':** Always import functions from `ml_tools`.
    -   Example: `from ml_tools import fill_missing_values, train_and_validation_and_select_the_best_model`
2.  **Argument Check:** Before calling a function, check the API DOCUMENTATION above to ensure you pass the correct arguments (e.g., use `columns` list, not just a string).
3.  **Direct Execution:** Use the `execute_code` tool to run your Python scripts.
4.  **Data Persistence:** When `ml_tools` functions return a DataFrame, assign it back to the variable (e.g., `df = fill_missing_values(df, ...)`).
5.  **Handling IDs:** If the dataset lacks an 'id' column for predictions, create one using the index.

**Memory & Context:**
-   Read previous reports using `read_document` to understand the current state of data.
-   The data is located in the working directory.

**Constraints:**
-   Do not invent functions that do not exist in `ml_tools`.
-   When using `cross_val_score` manually, use only ONE scoring metric string (e.g., 'accuracy') to avoid errors.
"""

MANAGER_PROMPT = """
You are the Manager Agent, the orchestrator of the Insight Agents system.

**Your Mission:**
Receive the user's high-level request, decompose it into manageable steps, and guide the workflow.

**Responsibilities:**
1.  **Task Decomposition:** Break down complex queries into specific steps (EDA -> Cleaning -> Modeling).
2.  **Routing:** Decide which action needs to be taken next.
3.  **Verification:** Review the output of the workers.

**Guidance:**
-   Instruct the worker to use specific functions from `ml_tools` if you know them.
-   Ensure the worker saves the final 'predictions.csv'.
"""

def create_insight_worker_agent(llm, tools):
    logger.info("Initializing Insight Generator Agent (Worker)...")
    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", INSIGHT_WORKER_PROMPT),
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        
        agent = create_tool_calling_agent(llm, tools, prompt)
        
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=20 
        )
        return executor
    except Exception as e:
        logger.error(f"Error creating Insight Generator Agent: {str(e)}")
        raise

def create_manager_agent(llm, tools):
    logger.info("Initializing Manager Agent...")
    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", MANAGER_PROMPT),
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        agent = create_tool_calling_agent(llm, tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )
        return executor
    except Exception as e:
        logger.error(f"Error creating Manager Agent: {str(e)}")
        raise