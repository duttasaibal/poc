import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
import math
import hashlib
from scipy import stats
from scipy.stats import ks_2samp
import seaborn as sns
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, mutual_info_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import base64

def add_logo_and_style():
    st.markdown(
        """
        <style>
        .container {
            display: flex;
            justify-content: center;
        }
        .logo-img {
            width: 200px;
            margin-bottom: 20px;
        }
        .stApp {
            background-color: #f0f2f6;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
        .stSelectbox>div>div {
            background-color: #e1e5eb;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Replace 'logo.png' with your actual logo file
    with open("logo.png", "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    
    st.markdown(
        f"""
        <div class="container">
            <img src="data:image/png;base64,{data}" class="logo-img">
        </div>
        """,
        unsafe_allow_html=True
    )



def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return None

def generate_synthetic_data(data, selected_columns, num_samples):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data[selected_columns])
    synthesizer = CTGANSynthesizer(metadata, epochs=3000, batch_size=500, discriminator_lr=2e-4, generator_lr=2e-4)
    synthesizer.fit(data[selected_columns])
    return synthesizer.sample(num_samples)

def anonymize_fields(data, fields_to_anonymize):
    anonymized_data = data.copy()
    for field in fields_to_anonymize:
        anonymized_data[field] = anonymized_data[field].apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest())
    return anonymized_data

def plot_distributions(real_data, synthetic_data, columns):
    num_cols = len(columns)
    rows = math.ceil(num_cols / 2)
    cols = min(2, num_cols)

    fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if num_cols == 1:
        axs = [axs]
    else:
        axs = axs.ravel()

    for i, col in enumerate(columns):
        if np.issubdtype(real_data[col].dtype, np.number):
            axs[i].hist(real_data[col], bins=20, color='blue', alpha=0.5, label='Real Data')
            axs[i].hist(synthetic_data[col], bins=20, color='red', alpha=0.5, label='Synthetic Data')
        else:
            real_counts = real_data[col].value_counts(normalize=True)
            synthetic_counts = synthetic_data[col].value_counts(normalize=True)
            
            all_categories = sorted(set(real_counts.index) | set(synthetic_counts.index))
            
            x = range(len(all_categories))
            real_values = [real_counts.get(cat, 0) for cat in all_categories]
            synthetic_values = [synthetic_counts.get(cat, 0) for cat in all_categories]
            
            axs[i].bar([i - 0.2 for i in x], real_values, width=0.4, color='blue', alpha=0.5, label='Real Data')
            axs[i].bar([i + 0.2 for i in x], synthetic_values, width=0.4, color='red', alpha=0.5, label='Synthetic Data')
            axs[i].set_xticks(x)
            axs[i].set_xticklabels(all_categories, rotation=45, ha='right')
        
        axs[i].set_title(col)
        axs[i].set_xlabel(col)
        axs[i].set_ylabel('Frequency')
        axs[i].legend()

    for i in range(num_cols, len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    return fig

def prepare_data_for_ml(data, target_column):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    return X, y, preprocessor

def regression_test(real_data, synthetic_data, target_column):
    X_real, y_real, preprocessor = prepare_data_for_ml(real_data, target_column)
    X_synthetic, y_synthetic, _ = prepare_data_for_ml(synthetic_data, target_column)

    regressor = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    real_scores = cross_val_score(regressor, X_real, y_real, cv=5, scoring='neg_mean_squared_error')
    synthetic_scores = cross_val_score(regressor, X_synthetic, y_synthetic, cv=5, scoring='neg_mean_squared_error')

    y_real_pred = cross_val_predict(regressor, X_real, y_real, cv=5)
    y_synthetic_pred = cross_val_predict(regressor, X_synthetic, y_synthetic, cv=5)
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].scatter(y_real, y_real_pred, alpha=0.5)
    axs[0].plot([y_real.min(), y_real.max()], [y_real.min(), y_real.max()], 'r--', lw=2)
    axs[0].set_xlabel("Actual Values")
    axs[0].set_ylabel("Predicted Values")
    axs[0].set_title("Real Data")
    
    axs[1].scatter(y_synthetic, y_synthetic_pred, alpha=0.5)
    axs[1].plot([y_synthetic.min(), y_synthetic.max()], [y_synthetic.min(), y_synthetic.max()], 'r--', lw=2)
    axs[1].set_xlabel("Actual Values")
    axs[1].set_ylabel("Predicted Values")
    axs[1].set_title("Synthetic Data")
    
    plt.tight_layout()

    return real_scores, synthetic_scores, fig

def classification_test(real_data, synthetic_data, target_column):
    X_real, y_real, preprocessor = prepare_data_for_ml(real_data, target_column)
    X_synthetic, y_synthetic, _ = prepare_data_for_ml(synthetic_data, target_column)

    classifier = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    real_scores = cross_val_score(classifier, X_real, y_real, cv=5, scoring='accuracy')
    synthetic_scores = cross_val_score(classifier, X_synthetic, y_synthetic, cv=5, scoring='accuracy')

    y_real_pred = cross_val_predict(classifier, X_real, y_real, cv=5)
    y_synthetic_pred = cross_val_predict(classifier, X_synthetic, y_synthetic, cv=5)
    
    cm_real = confusion_matrix(y_real, y_real_pred)
    cm_synthetic = confusion_matrix(y_synthetic, y_synthetic_pred)
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    sns.heatmap(cm_real, annot=True, fmt="d", cmap="YlGnBu", ax=axs[0])
    axs[0].set_title("Confusion Matrix (Real Data)")
    axs[0].set_xlabel("Predicted")
    axs[0].set_ylabel("Actual")
    
    sns.heatmap(cm_synthetic, annot=True, fmt="d", cmap="YlGnBu", ax=axs[1])
    axs[1].set_title("Confusion Matrix (Synthetic Data)")
    axs[1].set_xlabel("Predicted")
    axs[1].set_ylabel("Actual")
    
    plt.tight_layout()

    return real_scores, synthetic_scores, fig

def mutual_information_score(df1, df2):
    scores = []
    numeric_columns_df1 = df1.select_dtypes(include=[np.number])
    numeric_columns_df2 = df2.select_dtypes(include=[np.number])
    numeric_columns_df1, numeric_columns_df2 = numeric_columns_df1.align(numeric_columns_df2, join='inner', axis=0)
    for col in numeric_columns_df1.columns:
        score = mutual_info_score(numeric_columns_df1[col], numeric_columns_df2[col])
        scores.append(score)
    return np.mean(scores)

def correlation_score(df1, df2):
    numeric_columns_df1 = df1.select_dtypes(include=[np.number])
    numeric_columns_df2 = df2.select_dtypes(include=[np.number])
    corr1 = numeric_columns_df1.corr().values.flatten()
    corr2 = numeric_columns_df2.corr().values.flatten()
    return np.corrcoef(corr1, corr2)[0, 1]

def exact_match_score(df1, df2):
    merged_df = df1.merge(df2, indicator=True, how='outer')
    exact_matches = merged_df[merged_df['_merge'] == 'both'].shape[0]
    return exact_matches / min(len(df1), len(df2))

def neighbors_privacy_score(df1, df2, n_neighbors=5):
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    numeric_columns_df1 = df1.select_dtypes(include=[np.number])
    nn.fit(numeric_columns_df1)
    numeric_columns_df2 = df2.select_dtypes(include=[np.number])
    distances, _ = nn.kneighbors(numeric_columns_df2)
    return distances.mean()

def label_encode_data(data, columns_to_encode):
    encoded_data = data.copy()
    label_encoders = {}
    for col in columns_to_encode:
        le = LabelEncoder()
        encoded_data[col] = le.fit_transform(encoded_data[col].astype(str))
        label_encoders[col] = le
    return encoded_data, label_encoders

def reverse_label_encoding(data, label_encoders):
    decoded_data = data.copy()
    for col, le in label_encoders.items():
        decoded_data[col] = le.inverse_transform(decoded_data[col].astype(int))
    return decoded_data

# def sum_of_column_by_combination(data, column_name):
#     total_by_combination = {}

#     # Get unique combinations
#     unique_combinations = data.drop_duplicates()

#     # Iterate over unique combinations
#     for index, combination in unique_combinations.iterrows():
#         combination_key = ', '.join([f"{key}={value}" for key, value in combination.items() if key != column_name])
#         total = data[(data == combination).all(axis=1)][column_name].sum()
#         total_by_combination[combination_key] = total

#     return total_by_combination


# Main Streamlit app
#add_logo_and_style()

st.title("🧬 AXON: AI-based Synthetic Data Generator")
st.markdown("Generate high-quality synthetic data for your machine learning projects.")
#st.title("AXON: AI based Synthetic Data Generator")
st.subheader("📁 Please Choose a CSV or Excel file")
uploaded_file = st.file_uploader("Choose a CSV or Excel file based on your choice", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    data = load_data(uploaded_file)

    if data is not None:
        # st.write("📊 Data Preview:")
        st.subheader("📊 Data Preview")
        st.dataframe(data.head(),use_container_width=True)

        st.subheader("🔍 Select Columns")
        columns = data.columns.tolist()
        selected_columns = st.multiselect("Select columns for synthetic data generation", columns, default=columns)

        st.subheader("🔢 Number of synthetic samples ")
        num_samples = st.number_input("Number of synthetic samples to generate", min_value=1, value=len(data), step=1)

        anonymize = st.checkbox(" 🔐 Anonymize data before generating synthetic data")

        if anonymize:
            fields_to_anonymize = st.multiselect("Select fields to anonymize", selected_columns)

        use_label_encoding = st.checkbox(" 🏷️ Use label encoding for categorical variables")
        if use_label_encoding:
            categorical_columns = data[selected_columns].select_dtypes(include=['object']).columns
            columns_to_encode = st.multiselect("Select categorical columns to encode", categorical_columns)

        st.subheader("🧪 Select Machine learning test type (optional) ")
        test_type = st.selectbox("Select test type ", ["None", "Regression", "Classification"])

        target_column = None
        if test_type != "None":
            if test_type == "Regression":
                target_column = st.selectbox("Select target column for regression test", selected_columns)
            else:
                target_column = st.selectbox("Select target column for classification test", selected_columns)

        if st.button("🚀 Generate Synthetic Data"):
            working_data = data[selected_columns].copy()

            if anonymize and fields_to_anonymize:
                working_data = anonymize_fields(working_data, fields_to_anonymize)
                st.write("Anonymized Data Preview:")
                st.dataframe(working_data.head())

            label_encoders = {}
            if use_label_encoding and columns_to_encode:
                working_data, label_encoders = label_encode_data(working_data, columns_to_encode)
                st.write("Encoded Data Preview:")
                st.dataframe(working_data.head())

            synthetic_data = generate_synthetic_data(working_data, working_data.columns, num_samples)

            if use_label_encoding and columns_to_encode:
                decoded_synthetic_data = reverse_label_encoding(synthetic_data, label_encoders)
                st.write("Decoded Synthetic Data Preview:")
                st.dataframe(decoded_synthetic_data.head())
            else:
                decoded_synthetic_data = synthetic_data

            st.write("Distribution Comparison:")
            fig = plot_distributions(data[selected_columns], decoded_synthetic_data, selected_columns)
            st.pyplot(fig)

            st.write("Additional Metrics:")
            mi_score = mutual_information_score(data[selected_columns], decoded_synthetic_data)
            corr_score = correlation_score(data[selected_columns], decoded_synthetic_data)
            exact_match = exact_match_score(data[selected_columns], decoded_synthetic_data)
            privacy_score = neighbors_privacy_score(data[selected_columns], decoded_synthetic_data)

            st.write(f"Mutual Information Score (score >1): {mi_score:.4f}")
            st.write(f"Correlation Score (score close to 1): {corr_score:.4f}")
            st.write(f"Exact Match Score (score>0): {exact_match:.4f}")
            st.write(f"Neighbors Privacy Score (score >1): {privacy_score:.4f}")

            if test_type == "Regression" and target_column:
                st.write("Regression Test:")
                real_regression_scores, synthetic_regression_scores, fig = regression_test(data[selected_columns], decoded_synthetic_data, target_column)
                st.write("Real Data MSE:", -real_regression_scores.mean())
                st.write("Synthetic Data MSE:", -synthetic_regression_scores.mean())
                st.pyplot(fig)
            elif test_type == "Classification" and target_column:
                st.write("Classification Test:")
                real_classification_scores, synthetic_classification_scores, fig = classification_test(data[selected_columns], decoded_synthetic_data, target_column)
                st.write("Real Data Accuracy:", real_classification_scores.mean())
                st.write("Synthetic Data Accuracy:", synthetic_classification_scores.mean())
                st.pyplot(fig)

                # total_by_combination = sum_of_column_by_combination(decoded_synthetic_data, target_column)
                # sorted_combinations = sorted(total_by_combination.items(), key=lambda x: x[1], reverse=True)

              # # Display top combinations
              #   st.title(f'Top Combinations by Total {column_name}')

              # # Plotting the bar chart for top combinations
              #   plt.figure(figsize=(10, 6))
              #   combinations = [comb for comb, _ in sorted_combinations]
              #   totals = [total for _, total in sorted_combinations[:20]]

              #   plt.barh(combinations[:20], totals, color='skyblue')

              #   # Add values to the bars
              #   for i, total in enumerate(totals):
              #       plt.text(total + 0.05, i, f"{total:.2f}", va='center', fontweight='bold')

              #   plt.xlabel(f'Total {column_name}')
              #   plt.title(f'Top 20 Combinations by Total {column_name}')
              #   plt.gca().invert_yaxis()
              #   st.pyplot(plt)

           
            csv_decoded = decoded_synthetic_data.to_csv(index=False)
            st.download_button(
                label="Download Synthetic Data as CSV",
                data=csv_decoded,
                file_name="synthetic_data_decoded.csv",
                mime="text/csv",
            )


            if use_label_encoding and columns_to_encode:
                category_mapping = {}
                for col, le in label_encoders.items():
                    category_mapping[col] = dict(zip(le.classes_, range(len(le.classes_))))
                
                category_mapping_df = pd.DataFrame()
                for col, mapping in category_mapping.items():
                    df = pd.DataFrame({
                        'Original': mapping.keys(),
                        'Encoded': mapping.values()
                    })
                    df['Column'] = col
                    category_mapping_df = pd.concat([category_mapping_df, df], ignore_index=True)
                
                csv_category_mapping = category_mapping_df.to_csv(index=False)
                st.download_button(
                    label="Download Category Mapping as CSV",
                    data=csv_category_mapping,
                    file_name="category_mapping.csv",
                    mime="text/csv",
                ) 

else:
   


   #page = st.sidebar.radio("Navigate", ["Page 1", "Page 2", "Page 3"])    

  st.write("Please upload a CSV or Excel file to get started.")