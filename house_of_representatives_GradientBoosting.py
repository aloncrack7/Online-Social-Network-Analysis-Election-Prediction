import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, r2_score

# Cargar el dataset
file_path = 'final_df.csv'  # Cambia esto por la ruta correcta a tu archivo
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
df['house_winning_party_encoded'] = df['house_winning_party'].map({'Democrat': 0, 'Republican': 1})

# Crear la variable objetivo para regresión (porcentaje de votos demócratas)
df['house_democrat_percentage'] = (
    df['house_democrat'] / df['house_total_votes'] * 100
)

# Dividir en entrenamiento y prueba para clasificación
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    df_features, df['house_winning_party_encoded'], test_size=0.2, random_state=42, stratify=df['house_winning_party_encoded']
)

# Dividir en entrenamiento y prueba para regresión
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    df_features, df['house_democrat_percentage'], test_size=0.2, random_state=42
)

# ---------- MODELO DE CLASIFICACIÓN (PARTIDO GANADOR) ----------
clf = GradientBoostingClassifier(random_state=42)
clf.fit(X_train_clf, y_train_clf)

y_pred_clf = clf.predict(X_test_clf)
accuracy_clf = accuracy_score(y_test_clf, y_pred_clf)
report_clf = classification_report(y_test_clf, y_pred_clf)

# ---------- MODELO DE REGRESIÓN (PORCENTAJE DE VOTOS DEMÓCRATAS) ----------
regressor = GradientBoostingRegressor(random_state=42)
regressor.fit(X_train_reg, y_train_reg)

y_pred_reg = regressor.predict(X_test_reg)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

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

# Predicción del partido ganador
prediction_winner = clf.predict(sample_to_predict)

# Predicción del porcentaje de votos demócratas
prediction_percentage = regressor.predict(sample_to_predict)

# ---------- RESULTADOS ----------
print("Resultados del modelo de clasificación (partido ganador):")
print(f"Precisión del modelo: {accuracy_clf:.2f}")
print("Reporte de clasificación:")
print(report_clf)
print(f"Predicción para la muestra personalizada: Partido ganador -> {'Democrat' if prediction_winner[0] == 0 else 'Republican'}")

print("\nResultados del modelo de regresión (porcentaje de votos demócratas):")
print(f"Error absoluto medio (MAE): {mae:.2f}")
print(f"Coeficiente de determinación (R²): {r2:.2f}")
print(f"Predicción para la muestra personalizada: Porcentaje de votos demócratas -> {prediction_percentage[0]:.2f}%")
