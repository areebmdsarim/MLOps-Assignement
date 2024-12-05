from fuzzywuzzy import fuzz
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from typing import List, Dict

def infer_relationships(data1: pd.DataFrame, data2: pd.DataFrame) -> List[Dict[str, float]]:
    relationships = []

    # Clean columns
    data1.columns = data1.columns.str.strip().str.lower()
    data2.columns = data2.columns.str.strip().str.lower()

    # Print columns for debugging
    print("Data1 Columns:", data1.columns)
    print("Data2 Columns:", data2.columns)

    matched_columns = []

    # Fuzzy matching columns
    for column1 in data1.columns:
        for column2 in data2.columns:
            # Compare column names using fuzzy matching
            name_similarity = fuzz.ratio(column1.lower(), column2.lower())
            if name_similarity > 70:  # Adjust threshold if needed
                matched_columns.append((column1, column2))

    print("Matched Columns:", matched_columns)  # Print matched columns for debugging

    if not matched_columns:
        return relationships

    # Loop over each matched pair of columns
    for column1, column2 in matched_columns:
        data1_col = data1[column1]
        data2_col = data2[column2]

        # Ensure we are working with numeric columns
        if not pd.api.types.is_numeric_dtype(data1_col) or not pd.api.types.is_numeric_dtype(data2_col):
            continue  # Skip non-numeric columns

        # Create a DataFrame for training the model
        combined_data = pd.DataFrame({
            data1_col.name: data1_col,
            data2_col.name: data2_col
        })

        # Create the target column 'is_related' based on column matching
        combined_data['is_related'] = combined_data[data1_col.name] == combined_data[data2_col.name]

        # Prepare features (X) and target (y)
        X = combined_data[[data1_col.name, data2_col.name]]
        y = combined_data['is_related']

        # Initialize and train the Random Forest model
        model = RandomForestClassifier()
        model.fit(X, y)

        # Get the confidence score for this pair of columns
        confidence_score = model.score(X, y)

        # Store the relationship and confidence score
        relationships.append({
            'column_1': data1_col.name,
            'column_2': data2_col.name,
            'confidence_score': confidence_score
        })

    return relationships
