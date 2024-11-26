import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, r2_score

file_path = 'final_df.csv' 
df = pd.read_csv(file_path)

relevant_features = [
    'Native People', 'Unemployment Rate', 'Mean household income', 
    'Families income below poverty', 'Graduate or professional degree', 
    'Hispanic', 'Male', 'Female'
]
df_features = df[relevant_features]

# Crear la variable objetivo para clasificación (partido ganador)
df['winning_party'] = df.apply(
    lambda row: 'Democrat' if row['senate_candidate1'] > row['senate_candidate2'] else 'Republican', axis=1
)

# Crear la variable objetivo para regresión (porcentaje de votos demócratas)
df['democrat_percentage'] = (
    df['senate_candidate1'] / df['senate_total_votes'] * 100
)

# ---------- MODELO DE CLASIFICACIÓN (PARTIDO GANADOR) ----------

X_clf = df_features
y_clf = df['winning_party']

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

clf = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(X_train_clf, y_train_clf)

y_pred_clf = clf.predict(X_test_clf)
accuracy_clf = accuracy_score(y_test_clf, y_pred_clf)
report_clf = classification_report(y_test_clf, y_pred_clf)

# ---------- MODELO DE REGRESIÓN (PORCENTAJE DE VOTOS DEMÓCRATAS) ----------
X_reg = df_features
y_reg = df['democrat_percentage']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

regressor = LinearRegression()
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

prediction_winner = clf.predict(sample_to_predict)

prediction_percentage = regressor.predict(sample_to_predict)

# ---------- RESULTADOS ----------
print("Resultados del modelo de clasificación (partido ganador):")
print(f"Precisión del modelo: {accuracy_clf:.2f}")
print("Reporte de clasificación:")
print(report_clf)
print(f"Predicción para la muestra personalizada: Partido ganador -> {prediction_winner[0]}")

print("\nResultados del modelo de regresión (porcentaje de votos demócratas):")
print(f"Error absoluto medio (MAE): {mae:.2f}")
print(f"Coeficiente de determinación (R²): {r2:.2f}")
print(f"Predicción para la muestra personalizada: Porcentaje de votos demócratas -> {prediction_percentage[0]:.2f}%")
