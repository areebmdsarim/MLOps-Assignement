import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from typing import List, Dict

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def preprocess(text: str) -> str:
    """
    Preprocesses the column name by tokenizing, removing stopwords, and stemming.
    """
    text = text.lower()  # Lowercase the text
    tokens = word_tokenize(text)  # Tokenization
    stop_words = set(stopwords.words('english'))  # Stopword removal
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]  # Stemming
    return ' '.join(tokens)

def get_similarity_using_bert(col1: str, col2: str) -> float:
    """
    Computes the cosine similarity between two column names using BERT embeddings.
    """
    # Preprocess and encode the column names using Sentence-BERT model
    embeddings = model.encode([col1, col2])
    
    # Calculate cosine similarity between the embeddings
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity

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
            proc_column1 = preprocess(column1)
            proc_column2 = preprocess(column2)
            similarity_score = get_similarity_using_bert(proc_column1, proc_column2)
            if similarity_score > 0.5:
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
