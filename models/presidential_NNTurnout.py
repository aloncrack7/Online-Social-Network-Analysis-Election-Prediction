import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Cargar el dataset
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

# Seleccionar únicamente las columnas relevantes
relevant_features = [
    'Native People', 'Unemployment Rate', 'Mean household income', 
    'Families income below poverty', 'Graduate or professional degree', 
    'Hispanic', 'Male', 'Female'
]
df_features = df[relevant_features]

# Crear la variable objetivo para regresión (porcentaje de votos demócratas)
df['turnout'] = (
    df['house_total_votes']
)
print(f'TURNOUT: {df["turnout"]}')

# Normalizar las características de entrada
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)  # Normalización solo de las características
scaler_target = StandardScaler()
y_scaled = scaler_target.fit_transform(df['turnout'].values.reshape(-1, 1))  # Normalización de la variable objetivo

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_scaled, df['turnout'], test_size=0.2, random_state=42
)

# ---------- RED NEURONAL PARA REGRESIÓN (PORCENTAJE DE VOTOS DEMÓCRATAS) ----------
# Crear el modelo
reg_model = Sequential([
    Dense(8, activation='relu', input_shape=(X_train_reg.shape[1],)),  # Capa de entrada
    Dense(20, activation='relu'),                                      # Primera capa oculta
    Dropout(0.3),                                                      # Dropout del 30%
    Dense(1, activation='linear')                                      # Capa de salida para regresión
])

reg_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Entrenar el modelo
reg_model.fit(X_train_reg, y_train_reg, epochs=500, batch_size=8, verbose=0, validation_split=0.2)

# Evaluar el modelo
reg_loss, reg_mae = reg_model.evaluate(X_test_reg, y_test_reg, verbose=0)
y_pred_reg_nn = reg_model.predict(X_test_reg, verbose=0).flatten()
r2_reg_nn = r2_score(y_test_reg, y_pred_reg_nn)

# ---------- PREDICCIÓN PARA UNA MUESTRA ESPECÍFICA ----------
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
prediction_percentage_nn = scaler_target.inverse_transform(reg_model.predict(sample_scaled, verbose=0).flatten().reshape(-1, 1)).flatten()

# ---------- RESULTADOS ----------
print("\nResultados del modelo de regresión (porcentaje de votos demócratas) - Red Neuronal:")
print(f"Error absoluto medio (MAE): {reg_mae:.2f}")
print(f"Coeficiente de determinación (R²): {r2_reg_nn:.2f}")
print(prediction_percentage_nn)
print(f"Predicción para la muestra personalizada: turnout -> {prediction_percentage_nn[0]/648683*100:.2f}%")
