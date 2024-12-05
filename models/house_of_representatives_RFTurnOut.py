import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, r2_score

file_path = 'final_dataframe.csv'  
df = pd.read_csv(file_path)
df['Native People'] = df['Native People'].astype(str).str.replace(',', '').astype(int)
df['Unemployment Rate'] = df['Unemployment Rate'].astype(str).str.replace(',', '').astype(float)
df['Mean household income'] = df['Mean household income'].astype(str).str.replace(',', '').astype(int)
df['Families income below poverty'] = df['Families income below poverty'].astype(str).str.replace(',', '').astype(float)
df['Graduate or professional degree'] = df['Graduate or professional degree'].astype(str).str.replace(',', '').astype(float)
df['Hispanic'] = df['Hispanic'].astype(str).str.replace(',', '').astype(int)
df['Male'] = df['Male'].astype(str).str.replace(',', '').astype(int)
df['Female'] = df['Female'].astype(str).str.replace(',', '').astype(int)

relevant_features = [
    'Native People', 'Unemployment Rate', 'Mean household income', 
    'Families income below poverty', 'Graduate or professional degree', 
    'Hispanic', 'Male', 'Female'
]
df_features = df[relevant_features]

df['turn_out'] = (
    df['house_total_votes']
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_scaled, df['turn_out'], test_size=0.2, random_state=42
)

# ---------- MODELO DE REGRESIÓN (PORCENTAJE DE VOTOS DEMÓCRATAS) ----------
regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train_reg, y_train_reg)

y_pred_reg = regressor.predict(X_test_reg)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

# ---------- PREDICCIÓN PARA UNA MUESTRA ESPECÍFICA ----------
# Están puestos los valores de CA50 del 2023
sample_to_predict = pd.DataFrame([{
    'Native People': 607938,
    'Unemployment Rate': 4.3,
    'Mean household income': 163873, 
    'Families income below poverty': 5.2,
    'Graduate or professional degree': 149338,
    'Hispanic': 189438,
    'Male': 391721,
    'Female': 389530
}])

# Normalizar la muestra de entrada usando el mismo escalador
sample_scaled = scaler.transform(sample_to_predict)

# Predicción del porcentaje de votos demócratas
prediction_percentage = regressor.predict(sample_scaled)

# ---------- RESULTADOS ----------
print("Reporte de turn-out:")
print("\nResultados del modelo de regresión (turnout):")
print(f"Error absoluto medio (MAE): {mae:.2f}")
print(f"Coeficiente de determinación (R²): {r2:.2f}")
print(f"Predicción para la muestra personalizada: Turnout -> {prediction_percentage[0]/648683*100:.2f}%")
