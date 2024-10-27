import pandas as pd

# Cargar los datos
citizenship_df = pd.read_csv('./Searched_Datasets/Citizenship.csv')
commuter_df = pd.read_csv('./Searched_Datasets/Commuter Transportation.csv')
education_attainment_df = pd.read_csv('./Searched_Datasets/Educational Attainment.csv')
education_pyramid_df = pd.read_csv('./Searched_Datasets/Educational Pyramid.csv')
employment_sector_df = pd.read_csv('./Searched_Datasets/Employment by Industry Sector.csv')
income_df = pd.read_csv('./Searched_Datasets/Household Income.csv')
earnings_df = pd.read_csv('./Searched_Datasets/Median Earnings by Industry.csv')
poverty_diversity_df = pd.read_csv('./Searched_Datasets/Poverty &amp; Diversity.csv')
vote_history_df = pd.read_csv('./Searched_Datasets/Presidential Popular Vote Over Time.csv')
district_census_df = pd.read_csv('./Searched_Datasets/California_District_50_census_gov.csv')
employment_industries_df = pd.read_csv('./Searched_Datasets/Employment by Industries.csv')
occupations_df = pd.read_csv('./Searched_Datasets/Occupations.csv')
housing_df = pd.read_csv('./Searched_Datasets/Rent vs Own.csv')
house_elections_df = pd.read_csv('./Searched_Datasets/1976-2022-house.csv')

input_file = './Searched_Datasets/Rent vs Own.csv'
output_files = './Cleaned_CA_50_Datasets/'


# Filtrar el dataset para conservar solo las filas donde Geography es "Congressional District 50, CA"
filtered_df = housing_df[housing_df['Geography'] == 'Congressional District 50, CA']

# Guardar el dataset filtrado en la nueva ubicación
filtered_df.to_csv(f'{output_files}/Rent_vs_Own_CA50', index=False)