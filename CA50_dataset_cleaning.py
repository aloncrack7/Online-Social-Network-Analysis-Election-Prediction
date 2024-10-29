import pandas as pd

# Load Data
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

# Functions:

def drop_constant_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    This function drops those features which all entries have the same value.
    This is because those entries don't give any valuable information.
    '''
    non_constant_features = df.loc[:, (df != df.iloc[0]).any()]
    return non_constant_features

# Clean 1
'''
This dataset has information about Household Ownership
'''
filtered_df = housing_df[housing_df['Geography'] == 'Congressional District 50, CA']
filtered_df = drop_constant_features(filtered_df)
filtered_df.to_csv(f'{output_files}/Rent_vs_Own_CA50.csv', index=False)

# Clean 2
'''
This dataset has information about the winner of each year, and the number of votes and %
Information about other partys that did not win that year has been discarted, but if needed it can be found in the source dataset
'''
vote_history_df['% Votes'] = (vote_history_df['Candidate Votes'] / vote_history_df['Total Votes']) * 100
winners_df = vote_history_df.loc[vote_history_df.groupby('Year')['Candidate Votes'].idxmax()]
winners_df = drop_constant_features(winners_df)
winners_df = winners_df.drop(columns=['Candidate ID', 'Party ID', 'Share'])
winners_df.to_csv(f'{output_files}/Presidential_Popular_Vote_Over_Time.csv', index=False)

# Clean 3
'''
'''
