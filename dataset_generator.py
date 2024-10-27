import pandas as pd
from sklearn.model_selection import train_test_split

# Cargar los datos
citizenship_df = pd.read_csv('./Searched_Datasets/Citizenship.csv')
commuter_df = pd.read_csv('./Searched_Datasets/Commuter Transportation.csv')
education_attainment_df = pd.read_csv('./Searched_Datasets/Educational Attainment.csv')
education_pyramid_df = pd.read_csv('./Searched_Datasets/Educational Pyramid.csv')
employment_sector_df = pd.read_csv('./Searched_Datasets/Employment by Industry Sector.csv')
income_df = pd.read_csv('./Searched_Datasets/Household Income.csv')
earnings_df = pd.read_csv('./Searched_Datasets/Median Earnings by Industry.csv')
poverty_diversity_df = pd.read_csv('./Searched_Datasets/Poverty & Diversity.csv')
vote_history_df = pd.read_csv('./Searched_Datasets/Presidential Popular Vote Over Time.csv')
district_census_df = pd.read_csv('./Searched_Datasets/California_District_50_census_gov.csv')
employment_industries_df = pd.read_csv('./Searched_Datasets/Employment by Industries.csv')
occupations_df = pd.read_csv('./Searched_Datasets/Occupations.csv')
housing_df = pd.read_csv('./Searched_Datasets/Rent vs Own.csv')
house_elections_df = pd.read_csv('./Searched_Datasets/1976-2022-house.csv')

# 1. Limpieza y Procesamiento de los Datos
# Función de limpieza y conversión de tipos de datos, seleccionando columnas relevantes para cada dataset
def limpiar_datos(df, columnas_interes):
    df = df[columnas_interes]  # Seleccionar columnas de interés
    df = df.dropna()  # Eliminar filas con valores nulos
    # Asegurar tipos de datos para columnas relevantes
    for col in df.columns:
        if 'porcentaje' in col.lower() or 'percentage' in col.lower():
            df[col] = pd.to_numeric(df[col], errors='coerce')
        elif 'fecha' in col.lower() or 'date' in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

# Limpiar y seleccionar características clave para cada dataframe
citizenship_df = limpiar_datos(citizenship_df, ['ciudadania', 'porcentaje'])
commuter_df = limpiar_datos(commuter_df, ['transporte', 'porcentaje'])
education_attainment_df = limpiar_datos(education_attainment_df, ['educacion', 'porcentaje'])
education_pyramid_df = limpiar_datos(education_pyramid_df, ['nivel', 'edad', 'porcentaje'])
employment_sector_df = limpiar_datos(employment_sector_df, ['sector', 'empleo'])
income_df = limpiar_datos(income_df, ['ingresos', 'porcentaje_hogar'])
earnings_df = limpiar_datos(earnings_df, ['industria', 'ingresos_medios'])
poverty_diversity_df = limpiar_datos(poverty_diversity_df, ['pobreza', 'diversidad', 'porcentaje'])
vote_history_df = limpiar_datos(vote_history_df, ['año', 'partido', 'votos'])
district_census_df = limpiar_datos(district_census_df, ['caracteristica', 'porcentaje'])
employment_industries_df = limpiar_datos(employment_industries_df, ['industria', 'empleo'])
occupations_df = limpiar_datos(occupations_df, ['ocupacion', 'empleo'])
housing_df = limpiar_datos(housing_df, ['propietarios', 'alquilados'])
house_elections_df = limpiar_datos(house_elections_df, ['year', 'state', 'district', 'candidate', 'votes'])

# 2. Preparación de los datasets finales

# Dataset 1: Predicción de resultados en el Congreso (Clasificación y Regresión)
# Selección de características relevantes para el Congreso
congreso_features = pd.merge(vote_history_df, house_elections_df, left_on='año', right_on='year')
congreso_features = congreso_features[['year', 'party', 'votes', 'state', 'candidate']]

# Crear target para clasificación y regresión
congreso_features['target_candidato'] = congreso_features['party']
congreso_features['target_porcentaje'] = congreso_features['votes'] / congreso_features['votes'].sum() * 100

# Dataset 2: Predicción de representante en CA 50 (Clasificación y Regresión)
# Filtrar datos de CA 50 y características relevantes
ca50_features = house_elections_df[(house_elections_df['state'] == 'CA') & (house_elections_df['district'] == 50)]
ca50_features['target_representante'] = ca50_features['candidate']
ca50_features['target_porcentaje_rep'] = ca50_features['votes'] / ca50_features['votes'].sum() * 100

# Dataset 3: Predicción del número de votantes en CA 50
# Crear un dataset con datos históricos de participación para el distrito CA 50
ca50_votantes = ca50_features.groupby('year').agg({'votes': 'sum'}).reset_index()
ca50_votantes = ca50_votantes.rename(columns={'votes': 'num_votantes'})

# Dividir en conjuntos de entrenamiento y prueba
# División de datos de Congreso
X_congreso = congreso_features.drop(columns=['target_candidato', 'target_porcentaje'])
y_candidato = congreso_features['target_candidato']
y_porcentaje = congreso_features['target_porcentaje']

X_train_congreso, X_test_congreso, y_train_candidato, y_test_candidato = train_test_split(X_congreso, y_candidato, test_size=0.2, random_state=42)
X_train_congreso, X_test_congreso, y_train_porcentaje, y_test_porcentaje = train_test_split(X_congreso, y_porcentaje, test_size=0.2, random_state=42)

# División de datos de CA 50
X_ca50 = ca50_features.drop(columns=['target_representante', 'target_porcentaje_rep'])
y_representante = ca50_features['target_representante']
y_porcentaje_rep = ca50_features['target_porcentaje_rep']

X_train_ca50, X_test_ca50, y_train_rep, y_test_rep = train_test_split(X_ca50, y_representante, test_size=0.2, random_state=42)
X_train_ca50, X_test_ca50, y_train_porcentaje_rep, y_test_porcentaje_rep = train_test_split(X_ca50, y_porcentaje_rep, test_size=0.2, random_state=42)

# División de datos de votantes en CA 50
X_votantes = ca50_votantes[['year']]
y_votantes = ca50_votantes['num_votantes']

X_train_votantes, X_test_votantes, y_train_votantes, y_test_votantes = train_test_split(X_votantes, y_votantes, test_size=0.2, random_state=42)

# Guardar los datasets finales
X_train_congreso.to_csv('X_train_congreso.csv', index=False)
X_test_congreso.to_csv('X_test_congreso.csv', index=False)
y_train_candidato.to_csv('y_train_candidato.csv', index=False)
y_test_candidato.to_csv('y_test_candidato.csv', index=False)
y_train_porcentaje.to_csv('y_train_porcentaje.csv', index=False)
y_test_porcentaje.to_csv('y_test_porcentaje.csv', index=False)

X_train_ca50.to_csv('X_train_ca50.csv', index=False)
X_test_ca50.to_csv('X_test_ca50.csv', index=False)
y_train_rep.to_csv('y_train_rep.csv', index=False)
y_test_rep.to_csv('y_test_rep.csv', index=False)
y_train_porcentaje_rep.to_csv('y_train_porcentaje_rep.csv', index=False)
y_test_porcentaje_rep.to_csv('y_test_porcentaje_rep.csv', index=False)

X_train_votantes.to_csv('X_train_votantes.csv', index=False)
X_test_votantes.to_csv('X_test_votantes.csv', index=False)
y_train_votantes.to_csv('y_train_votantes.csv', index=False)
y_test_votantes.to_csv('y_test_votantes.csv', index=False)

print("Datos procesados y guardados exitosamente.")
