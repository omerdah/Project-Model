import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import joblib

# Define the RoundedRandomForestRegressor class
class RoundedRandomForestRegressor(RandomForestRegressor):
    def predict(self, X):
        predictions = super().predict(X)
        rounded_predictions = np.round(predictions)
        return rounded_predictions

# Load the saved model
model = pickle.load(open('rounded_RF_Regressor_model.pkl', 'rb'))

df_mappings = pd.read_excel('feature_mappings.xlsx',index_col=None)

# Define the mapping dictionaries
month_mapping = {
    1: 1,
    2: 1,
    3: 2,
    4: 3,
    5: 1,
    6: 2,
    7: 3,
    8: 1,
    9: 1,
    10: 2,
    11: 3,
    12: 3
}
all_features = ['שלב הזריעה בעונה','עונת גידול','טיפוס תירס','סוג זבל','עיבוד מקדים','אופן הדברה','כרב/גידול קודם','סוג הקרקע','ייעוד','גידול תירס/סורגום שכן','השקיה','משך גידול ממוצע','קרינה ממוצע יום','עננות ממוצע יום','התאדות פנמן ממוצע יום','גשם ממוצע יום','לחות יחסית ממוצע יום','כיוון רוח ממוצע יום','מהירות רוח ממוצע יום','טמפ 2 מ ממוצע יום','קרינה ממוצע לילה','התאדות פנמן ממוצע לילה','לחות יחסית ממוצע לילה','כיוון רוח ממוצע לילה','מהירות רוח ממוצע לילה','טמפ 2 מ ממוצע לילה','קונפידור, קונפידור + טלסטאר בזריעה','מנת גשם עונתי','גובה מפני הים',"מס' דונם",'מועד זריעה']

season_mapping = {'סתיו': 0, 'אביב': 1, 'אביב-קיץ': 2}

# Function to preprocess the input data
def preprocess_input(data):
    data['עונת גידול'] = data['עונת גידול'].map(season_mapping)
    data['שלב הזריעה בעונה'] = data['מועד זריעה'].dt.month.map(month_mapping)
    data['מועד זריעה'] = pd.to_datetime(data['מועד זריעה']).dt.dayofyear
    data = data.astype('float64')
    cat_cols = categorial_feats
    data = pd.get_dummies(data, columns=cat_cols, prefix=cat_cols, prefix_sep='_')
    return data

# Create the Streamlit app
def main():
    st.title('חיזוי מספר הריסוסים כנגד נגעים של גדודנית פולשת בתירס')
    st.write('הזן את כלל הקלטים המופיעים מטה ולחץ "תחזית"')
    
    # Get the feature names that need validation from df_mappings
    input_names = list(df_mappings.columns)

    inputs = []
    for feature_name in all_features:
        if feature_name in input_names:
            valid_values = list(df_mappings[feature_name].dropna().unique())
            if feature_name == 'מועד זריעה':
                min_date = datetime.datetime.now().date()
                max_date = pd.to_datetime('2030-12-31')
                input_value = st.date_input(feature_name, min_value=min_date, max_value=max_date)
            elif isinstance(valid_values, list):
                input_value = st.selectbox(feature_name, valid_values)
            else:
                input_value = st.number_input(feature_name, value=0.0)
        else:
            input_value = st.number_input(feature_name, value=0.0)

        inputs.append(input_value)
    
    # Create a button to trigger the prediction
    if st.button('חיזוי'):
        
        input_data = pd.DataFrame([inputs], columns=all_features)
        
        # Preprocess the input data
        preprocessed_data = preprocess_input(input_data)

        # Make predictions
        prediction = model.predict(preprocessed_data)
    
        # Create a DataFrame from the user inputs
        input_data = pd.DataFrame([inputs], columns=input_names)

        # Display the prediction
        st.write('חיזוי:', prediction)

# Run the app
if __name__ == '__main__':
    main()
