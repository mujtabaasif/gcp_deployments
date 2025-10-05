import pandas as pd
from matplotlib import pyplot as plt
from src.dataloader import load_convfinqa_dataset

def visualize_operators(df):
    # Plot the histogram of operator frequencies as subplots
    # Get all columns that start with 'op_'
    operator_columns = [col for col in df.columns if col.startswith('op_')]

    # Create a figure with subplots for each operator
    num_operators = len(operator_columns)
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 5 * 3))
    axes = axes.flatten()  # Flatten the axes array for easy indexing
    for i, col in enumerate(operator_columns):
        # Plot the histogram for each operator
        df[col].plot(kind='hist', ax=axes[i], bins=20, alpha=0.7)
        axes[i].set_xlim(0, df[col].max() + 1)
        # Set x_ticks for better readability
        axes[i].set_xticks(range(-1, int(df[col].max() + 1), 1))
        axes[i].set_title(f'Frequency of {col}')
        axes[i].set_xlabel('Frequency')
        axes[i].set_ylabel('Count')

    # Compute sum of all operator frequencies
    total_operator_frequencies = df[operator_columns].sum().sort_values(ascending=False)
    print("Total Operator Frequencies:")
    print(total_operator_frequencies)
    # Plot the total operator frequencies as a subplot
    total_operator_frequencies.plot(kind='bar', ax=axes[i + 1], color='skyblue', alpha=0.7)
    axes[i + 1].set_title('Total Operator Frequencies')
    axes[i + 1].set_xlabel('Operators')
    axes[i + 1].set_ylabel('Total Count')
    axes[i + 1].tick_params(axis='x', rotation=45)

    # Sum number of operators across records
    total_num_operators = df[operator_columns].sum(axis=1)
    df['total_num_operators'] = total_num_operators
    # Plot the histogram of total number of operators across records
    df['total_num_operators'].plot(kind='hist', ax=axes[i + 2], bins=20, alpha=0.7, color='orange')
    axes[i + 2].set_title('Total Number of Operators Across Records')
    axes[i + 2].set_xlabel('Total Number of Operators')
    axes[i + 2].set_ylabel('Count')
    # Set x_ticks for better readability
    axes[i + 2].set_xticks(range(0, 20, 1))

    plt.tight_layout()
    # Save the figure
    plt.savefig("plots/operator_frequencies.png")

def visualize_features_stats(df):
    # Plot the statistics of features
    feature_columns = ['num_dialogue_turns', 'has_type2_question', 'has_duplicate_columns', 'has_non_numeric_values', 'num_true_qa_split']

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    for i, col in enumerate(feature_columns):
        # Check is the column is boolean or numeric
        if df[col].dtype == 'bool':
            # Plot a bar chart for boolean features
            df[col].value_counts().plot(kind='bar', ax=axes[i], color='lightgreen', alpha=0.7)
            axes[i].set_title(f'Count of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Count')
        else:
            df[col].plot(kind='hist', ax=axes[i], alpha=0.7)
            axes[i].set_title(f'Statistics of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig("plots/features_statistics.png")

def extract_operators(turn_program):
    operators = []
    for index, op in enumerate(turn_program):
        operator_list = op.split('),')
        for operator in operator_list:
            if '(' in operator:
                operator = operator.split('(')[0]
                operator = operator.strip()
                operators.append(operator)
    return operators

if __name__ == '__main__':
    # Load the ConvFinQA dataset
    train_set, test_set = load_convfinqa_dataset("../data/convfinqa_dataset.json")

    # Convert the training set to a DataFrame for analysis
    train_df = pd.DataFrame([record.model_dump() for record in train_set], columns=['id', 'doc', 'dialogue', 'features'])

    # Extract the pre-text, table, and post-text from the document
    train_df['pre_text'] = train_df['doc'].apply(lambda x: x['pre_text'])
    train_df['table'] = train_df['doc'].apply(lambda x: x['table'])
    train_df['post_text'] = train_df['doc'].apply(lambda x: x['post_text'])

    # Extract the conversation questions and answers, executed_answers, qa_splits, and turn_program from the dialogue
    train_df['conv_questions'] = train_df['dialogue'].apply(lambda x: x['conv_questions'])
    train_df['conv_answers'] = train_df['dialogue'].apply(lambda x: x['conv_answers'])
    train_df['turn_program'] = train_df['dialogue'].apply(lambda x: x['turn_program'])
    train_df['executed_answers'] = train_df['dialogue'].apply(lambda x: x['executed_answers'])
    train_df['qa_split'] = train_df['dialogue'].apply(lambda x: x['qa_split'])
    # Calculate the number 'true' and 'false' values in the qa_split
    train_df['num_true_qa_split'] = train_df['qa_split'].apply(lambda x: sum(x))
    # extract operators from the turn_program
    train_df['operators'] = train_df['turn_program'].apply(lambda x: extract_operators(x))
    # Add a column for operator frequencies for each record
    train_df['operators'] = train_df['operators'].apply(lambda x: pd.Series(x).value_counts().to_dict())
    # Convert the operators dictionary to a DataFrame
    operators_df = pd.json_normalize(train_df['operators'])
    # Fill NaN values with 0 in the operators DataFrame
    operators_df.fillna(0, inplace=True)
    # rename columns to have a prefix 'op_'
    operators_df.columns = [f'op_{col}' for col in operators_df.columns]
    # Concatenate the operators DataFrame with the main DataFrame
    train_df = pd.concat([train_df, operators_df], axis=1)

    train_df['num_dialogue_turns'] = train_df['features'].apply(lambda x: x['num_dialogue_turns'])
    train_df['has_type2_question'] = train_df['features'].apply(lambda x: x['has_type2_question'])
    train_df['has_duplicate_columns'] = train_df['features'].apply(lambda x: x['has_duplicate_columns'])
    train_df['has_non_numeric_values'] = train_df['features'].apply(lambda x: x['has_non_numeric_values'])

    visualize_operators(df=train_df)
    visualize_features_stats(df=train_df)

    # Drop the original 'doc', 'dialogue', and 'features' columns as they are no longer needed
    train_df.drop(columns=['doc', 'dialogue', 'features'], inplace=True)
    # Display basic statistics about the training set
    print("Training Set Statistics:")
    print(train_df.describe(include='all'))

    # Check for missing values in the training set
    print("\nMissing Values in Training Set:")
    print(train_df.isnull().sum())

    # Display the first few records of the training set
    print("\nFirst 5 Records in Training Set:")
    print(train_df.head())

    # Save the processed DataFrame to a CSV file
    train_df.to_csv("data/convfinqa_train_processed.csv", index=False)