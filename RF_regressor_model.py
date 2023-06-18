import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import joblib
import numpy as np

# Define the RoundedRandomForestRegressor class
class RoundedRandomForestRegressor(RandomForestRegressor):
    def predict(self, X):
        predictions = super().predict(X)
        rounded_predictions = np.round(predictions)
        return rounded_predictions

# Load the saved model
model = pickle.load(open('rounded_RF_Regressor_model.pkl', 'rb'))
stats_per_season_and_area = pd.read_excel('stats_per_season_and_area.xlsx',index_col=None)
df_mappings = pd.read_excel('feature_mappings.xlsx',index_col=None)
X = ['מועד זריעה',
 "מס' דונם",
 'גובה מפני הים',
 'מנת גשם עונתי',
 'ריסוס ראשון ימים מזריעה',
 'טמפ 2 מ ממוצע לילה',
 'מהירות רוח ממוצע לילה',
 'כיוון רוח ממוצע לילה',
 'התאדות פנמן ממוצע לילה',
 'קרינה ממוצע לילה',
 'טמפ 2 מ ממוצע יום',
 'מהירות רוח ממוצע יום',
 'כיוון רוח ממוצע יום',
 'לחות יחסית ממוצע יום',
 'גשם ממוצע יום',
 'התאדות פנמן ממוצע יום',
 'עננות ממוצע יום',
 'קרינה ממוצע יום',
 'משך גידול ממוצע',
 'קונפידור, קונפידור + טלסטאר בזריעה_0.0',
 'קונפידור, קונפידור + טלסטאר בזריעה_1.0',
 'קונפידור, קונפידור + טלסטאר בזריעה_2.0',
 'השקיה_1.0',
 'השקיה_2.0',
 'השקיה_3.0',
 'גידול תירס/סורגום שכן_0.0',
 'גידול תירס/סורגום שכן_1.0',
 'גידול תירס/סורגום שכן_2.0',
 'גידול תירס/סורגום שכן_3.0',
 'גידול תירס/סורגום שכן_4.0',
 'גידול תירס/סורגום שכן_5.0',
 'ייעוד_1.0',
 'ייעוד_2.0',
 'ייעוד_3.0',
 'ייעוד_4.0',
 'סוג הקרקע_2.0',
 'סוג הקרקע_2.5',
 'סוג הקרקע_3.0',
 'כרב/גידול קודם_1.0',
 'כרב/גידול קודם_2.0',
 'כרב/גידול קודם_3.0',
 'כרב/גידול קודם_4.0',
 'אופן הדברה_1.0',
 'אופן הדברה_2.0',
 'אופן הדברה_3.0',
 'עיבוד מקדים_1.0',
 'עיבוד מקדים_2.0',
 'עיבוד מקדים_3.0',
 'סוג זבל_0.0',
 'סוג זבל_1.0',
 'סוג זבל_2.0',
 'סוג זבל_3.0',
 'סוג זבל_4.0',
 'סוג זבל_5.0',
 'טיפוס תירס_1.0',
 'טיפוס תירס_2.0',
 'טיפוס תירס_3.0',
 'טיפוס תירס_4.0',
 'עונת גידול_0.0',
 'עונת גידול_1.0',
 'עונת גידול_2.0',
 'שלב הזריעה בעונה_1.0',
 'שלב הזריעה בעונה_2.0',
 'שלב הזריעה בעונה_3.0']
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
all_features = ['טיפוס תירס','סוג זבל','עיבוד מקדים','אופן הדברה','כרב/גידול קודם','סוג הקרקע','ייעוד','גידול תירס/סורגום שכן','השקיה','משך גידול ממוצע','קרינה ממוצע יום','עננות ממוצע יום','התאדות פנמן ממוצע יום','גשם ממוצע יום','לחות יחסית ממוצע יום','כיוון רוח ממוצע יום','מהירות רוח ממוצע יום','טמפ 2 מ ממוצע יום','קרינה ממוצע לילה','התאדות פנמן ממוצע לילה','לחות יחסית ממוצע לילה','כיוון רוח ממוצע לילה','מהירות רוח ממוצע לילה','טמפ 2 מ ממוצע לילה','קונפידור, קונפידור + טלסטאר בזריעה','מנת גשם עונתי','גובה מפני הים',"מס' דונם",'מועד זריעה','אזור']
categorial_feats = ['קונפידור, קונפידור + טלסטאר בזריעה','השקיה','גידול תירס/סורגום שכן','ייעוד','סוג הקרקע','כרב/גידול קודם','אופן הדברה','עיבוד מקדים','סוג זבל','טיפוס תירס','עונת גידול','שלב הזריעה בעונה']
# season_mapping = {'סתיו': 0, 'אביב': 1, 'אביב-קיץ': 2}
season_mapping = {
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 2,
    6: 2,
    7: 2,
    8: 0,
    9: 0,
    10: 0,
    11: 0,
    12: 0
}
watering = {
   'קונוע' :1,
   'טפטוף' :2,
   'משולב' :3
}
sorgum = {
   'ללא' :0,
   'מזרח' :1,
   'דרום' :2,
   'מערב' :3,
   'צפון' :4,
   'לא ידוע' :5
}
mission = {
   'שוק' :1,
   'תעשייה' :2,
   'תחמיץ' :3,
   'פופקורן' :4
}
soil = {
   'קלה' :1,
   'בינונית' :2,
   'בינונית-כבדה' :2.5,
   'כבדה' :3
}
formar_crop = {
   'קטנית' :1,
   'שושניים' :2,
   'סוככים' :3,
   'דגניים' :4,
   'שחור' :5
}
spray = {
   'קרקע' :1,
   'אוויר' :2,
   'משולב' :3
}
pre_process = {
   'חריש' :1,
   'דיסקוס' :2,
   'קלטור' :3
}

fertile = {
   'ללא' :0,
   'קומפוסט' :1,
   'זבל חצרות' :2,
   'זבל עוף' :3,
   'טריפל' :4,
   'לא ידוע' :5
}
corn_type = {
   'מספוא' :1,
   'מתוק' :2,
   'סופר-מתוק' :3,
   'פופקורן' :4
}
confidor = {
   'ללא' :0,
   'יישום בזריעה' :1,
   'יישום בזריעה ו30 יום לפני אסיף' :2
}
# Function to preprocess the input data
def preprocess_input(data):
    data['עונת גידול'] = pd.to_datetime(data['מועד זריעה']).dt.month.map(season_mapping)
    data['שלב הזריעה בעונה'] = pd.to_datetime(data['מועד זריעה']).dt.month.map(month_mapping)
    data['מועד זריעה'] = pd.to_datetime(data['מועד זריעה']).dt.dayofyear
    
    data['השקיה'] = data['השקיה'].map(watering)
    data['גידול תירס/סורגום שכן'] = data['גידול תירס/סורגום שכן'].map(sorgum)
    data['ייעוד'] = data['ייעוד'].map(mission)
    data['סוג הקרקע'] = data['סוג הקרקע'].map(soil)
    data['כרב/גידול קודם'] = data['כרב/גידול קודם'].map(formar_crop)
    data['אופן הדברה'] = data['אופן הדברה'].map(spray)
    data['עיבוד מקדים'] = data['עיבוד מקדים'].map(pre_process)
    data['סוג זבל'] = data['סוג זבל'].map(fertile)
    data['טיפוס תירס'] = data['טיפוס תירס'].map(corn_type)
    data['קונפידור, קונפידור + טלסטאר בזריעה'] = data['קונפידור, קונפידור + טלסטאר בזריעה'].map(confidor)
    
    data = data.astype('float64')
    data = pd.get_dummies(data, columns=categorial_feats, prefix=categorial_feats, prefix_sep='_')
    # Realign new data columns with training data columns
    missing_cols = set(X) - set(data.columns)
    for col in missing_cols:
        data[col] = 0

    # Ensure the order of columns is the same as in the training data
    data = data[X]
    return data

def procces_meteo(data):
    feats = ['קרינה ממוצע יום','קרינה ממוצע לילה','עננות ממוצע יום','התאדות פנמן ממוצע יום','התאדות פנמן ממוצע לילה','גשם ממוצע יום','לחות יחסית ממוצע יום','כיוון רוח ממוצע יום','כיוון רוח ממוצע לילה','מהירות רוח ממוצע יום','מהירות רוח ממוצע לילה','טמפ 2 ממוצע יום','טמפ 2 ממוצע לילה']
    
    # Extract the season and area values from the first DataFrame
    season = data['עונת גידול'].values[0]
    area = data['אזור'].values[0]

    # Filter the second DataFrame to match the season and area values
    filtered_df2 = stats_per_season_and_area[(stats_per_season_and_area['עונת גידול'] == season) & (stats_per_season_and_area['אזור'] == area)].copy()

    # Get the extra column names from df2
    extra_columns = df2.columns.difference(['עונת גידול', 'אזור'])

    # Add the extra columns to the first DataFrame
    for col in extra_columns:
        df1[col] = filtered_df2[col].values[0]

        
# Create the Streamlit app
def main():
    # Set page layout to center alignment
    st.set_page_config(layout="centered")
    st.title('חיזוי מספר הריסוסים כנגד גדודנית פולשת בתירס')
    st.write('הזן את כלל הקלטים המופיעים מטה ולחץ על כפתור התחזית')

    # Get the feature names that need validation from df_mappings
    input_names = list(df_mappings.columns)

    inputs = []
    for feature_name in all_features:
        if feature_name == 'מועד זריעה':
            min_date = pd.to_datetime('today').date()
            max_date = pd.to_datetime('2030-12-31').date()
            input_value = st.date_input(feature_name, min_value=min_date, max_value=max_date)
        elif feature_name in input_names:
            valid_values = list(df_mappings[feature_name].dropna().unique())

            if isinstance(valid_values, list):
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
        preprocessed_data = procces_meteo(preprocessed_data)
        # Make predictions
        prediction = model.predict(preprocessed_data)

        # Display the prediction
        st.write('חיזוי:', prediction)

# Run the app
if __name__ == '__main__':
    main()