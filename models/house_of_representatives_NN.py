import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Cargar el dataset
file_path = 'final_df.csv' 
df = pd.read_csv(file_path)

# Seleccionar únicamente las columnas relevantes
relevant_features = [
    'Native People', 'Unemployment Rate', 'Mean household income', 
    'Families income below poverty', 'Graduate or professional degree', 
    'Hispanic', 'Male', 'Female'
]
df_features = df[relevant_features]

# Crear la variable objetivo para clasificación (partido ganador)
df['house_winning_party'] = df.apply(
    lambda row: 'Democrat' if row['house_democrat'] > row['house_republican'] else 'Republican', axis=1
)

# Codificar la variable objetivo de clasificación como números
df['house_winning_party_encoded'] = df['house_winning_party'].map({'Democrat': 0, 'Republican': 1})

# Crear la variable objetivo para regresión (porcentaje de votos demócratas)
df['house_democrat_percentage'] = (
    df['house_democrat'] / df['house_total_votes'] * 100
)

# Normalizar las características de entrada
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)  # Normalización solo de las características

# Dividir en entrenamiento y prueba
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_scaled, df['house_winning_party_encoded'], test_size=0.2, random_state=42, stratify=df['house_winning_party_encoded']
)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_scaled, df['house_democrat_percentage'], test_size=0.2, random_state=42
)

# ---------- RED NEURONAL PARA CLASIFICACIÓN (PARTIDO GANADOR) ----------
# Crear el modelo
clf_model = Sequential([
    Dense(8, activation='relu', input_shape=(X_train_clf.shape[1],)),  # Capa de entrada
    Dense(20, activation='relu'),                                      # Primera capa oculta
    Dropout(0.3),                                                      # Dropout del 30%
    Dense(2, activation='softmax')                                     # Capa de salida para clasificación binaria
])

clf_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
clf_model.fit(X_train_clf, y_train_clf, epochs=50, batch_size=8, verbose=0, validation_split=0.2)

# Evaluar el modelo
clf_loss, clf_accuracy = clf_model.evaluate(X_test_clf, y_test_clf, verbose=0)
y_pred_clf_nn = clf_model.predict(X_test_clf, verbose=0).argmax(axis=1)
report_clf_nn = classification_report(y_test_clf, y_pred_clf_nn)

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
reg_model.fit(X_train_reg, y_train_reg, epochs=50, batch_size=8, verbose=0, validation_split=0.2)

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

# Predicción del partido ganador
prediction_winner_nn = clf_model.predict(sample_scaled, verbose=0).argmax(axis=1)

# Predicción del porcentaje de votos demócratas
prediction_percentage_nn = reg_model.predict(sample_scaled, verbose=0).flatten()

# ---------- RESULTADOS ----------
print("Resultados del modelo de clasificación (partido ganador) - Red Neuronal:")
print(f"Precisión del modelo: {clf_accuracy:.2f}")
print("Reporte de clasificación:")
print(report_clf_nn)
print(f"Predicción para la muestra personalizada: Partido ganador -> {'Democrat' if prediction_winner_nn[0] == 0 else 'Republican'}")

print("\nResultados del modelo de regresión (porcentaje de votos demócratas) - Red Neuronal:")
print(f"Error absoluto medio (MAE): {reg_mae:.2f}")
print(f"Coeficiente de determinación (R²): {r2_reg_nn:.2f}")
print(f"Predicción para la muestra personalizada: Porcentaje de votos demócratas -> {prediction_percentage_nn[0]:.2f}%")
