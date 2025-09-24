#!/usr/bin/env python3
"""
Process IDESCAT data to create a simplified demographic breakdown by province.
Note: population weights had then to be adjusted manually in terms of:
- Unemployed: (since it does not count under-30s, should be lower), reduced by ~3 percentage points
- Bussiness: increased since data is scarce and not only self-employed should count, increased by ~1-2 percentage points
- Middle: adjusted accordingly
"""

import requests
import pandas as pd
import os

def get_idescat_data(url):
    """Helper function to make API requests to IDESCAT"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def get_population_by_age_province_2015():
    """Get population by age groups and province for 2015 from IDESCAT API. We use 2015 so it is relevant from 2012 to 2019."""
    # Try to read from local CSV first
    local_csv = os.path.join(os.path.dirname(__file__), 'idescat_raw', 'age_province_2015.csv')
    if os.path.exists(local_csv):
        print(f"Reading population by age and province data for 2015 from {local_csv}...")
        df = pd.read_csv(local_csv)
        return df

    # If not found, fetch from API
    url = "https://api.idescat.cat/taules/v2/pmh/1180/8078/prov/data?lang=en&YEAR=2015"
    print("Fetching population by age and province data for 2015 from API...")
    data = get_idescat_data(url)
    if not data:
        print("Note: If this URL fails, try exploring nodes at https://api.idescat.cat/taules/v2/pmh/1180")
        return None

    dimensions = data.get('dimension', {})
    values = data.get('value', [])
    size = data.get('size', [])

    age_categories = dimensions.get('AGE', {}).get('category', {}).get('label', {})
    sex_categories = dimensions.get('SEX', {}).get('category', {}).get('label', {})
    province_categories = dimensions.get('PROV', {}).get('category', {}).get('label', {})

    rows = []
    idx = 0

    # Iterate through multidimensional structure
    for _ in range(size[0] if len(size) > 0 else 1):
        for prov_idx in range(size[1] if len(size) > 1 else 1):
            for age_idx in range(size[2] if len(size) > 2 else 1):
                for sex_idx in range(size[3] if len(size) > 3 else 1):
                    for _ in range(size[4] if len(size) > 4 else 1):
                        if idx < len(values) and values[idx] is not None:
                            age_keys = list(age_categories.keys())
                            sex_keys = list(sex_categories.keys())
                            prov_keys = list(province_categories.keys())
                            if age_idx < len(age_keys) and sex_idx < len(sex_keys) and prov_idx < len(prov_keys):
                                age_key = age_keys[age_idx]
                                sex_key = sex_keys[sex_idx]
                                prov_key = prov_keys[prov_idx]
                                rows.append({
                                    'province': province_categories[prov_key],
                                    'age_group': age_categories[age_key],
                                    'sex': sex_categories[sex_key],
                                    'population': values[idx]
                                })
                        idx += 1

    df = pd.DataFrame(rows)

    # Categorize ages matching your script
    def categorize_age(age_str):
        age_str_lower = age_str.lower()
        if 'and over' in age_str_lower or 'i més' in age_str_lower:
            return 'retired'  # ≥65 approximation
        try:
            # Extract numeric age if possible
            age_num = int(''.join(filter(str.isdigit, age_str)))
            if age_num < 18:
                return 'underage'
            elif age_num <= 30:
                return 'young'
            elif age_num >= 65:
                return 'retired'
            else:
                return 'middle'
        except ValueError:
            return 'middle'
    df['category'] = df['age_group'].apply(categorize_age)

    # Save to local CSV for future use
    df.to_csv(local_csv, index=False)
    return df

def working_condition_breakdown(df, total_population_by_province):
    """Calculate working condition breakdown from population data"""

    def categorize_unemployment(df):
        # From middle remove unemployed (from IDESCAT 2012T3 data)
        unemployed_by_province = {
            'barcelona': 0.23,
            'girona': 0.216,
            'lleida': 0.153,
            'tarragona': 0.234,
            'catalunya': 0.225
        }

        # df is a DataFrame with provinces as index, age categories as columns, values are percentages
        df = df.copy()
        if 'total' in df.columns:
            df = df.drop(columns=['total'])

        # For each province, split 'middle' into 'middle' and 'unemployed' columns
        if 'middle' not in df.columns:
            return df  # nothing to do
        df['unemployed'] = 0.0
        for province in df.index:
            unemp_pct = unemployed_by_province.get(province.lower(), 0)
            middle_val = df.at[province, 'middle'] if not pd.isna(df.at[province, 'middle']) else 0.0
            unemployed_val = round(middle_val * unemp_pct, 2)
            remaining_middle_val = round(middle_val - unemployed_val, 2)
            df.at[province, 'middle'] = remaining_middle_val
            df.at[province, 'unemployed'] = unemployed_val
        return df
    
    def categorize_bussiness(df0):
        # Idescat data 2012
        bussiness_pct_by_province = {
            'barcelona': 0.122,
            'girona': 0.14,
            'lleida': 0.064,
            'tarragona': 0.106,
            'catalunya': 0.121
        }
        df0 = df0.copy()
        for province in df0.index:
            bus_pct = bussiness_pct_by_province.get(province.lower(), 0)
            middle_val = df0.at[province, 'middle'] if not pd.isna(df0.at[province, 'middle']) else 0.0
            bussiness_val = round(middle_val * bus_pct, 2)
            remaining_middle_val = round(middle_val - bussiness_val, 2)
            df0.at[province, 'middle'] = remaining_middle_val
            df0.at[province, 'buss'] = bussiness_val
        return df0

    def categorize_rural(df1):
        # Calculate rural share and add that column using total_population_by_province
        urban_csv = os.path.join(os.path.dirname(__file__), 'idescat_raw', 'idescat_urban_2015.csv')
        urban_df = pd.read_csv(urban_csv, sep=';', encoding='utf-8')
        urban_df = urban_df[urban_df['col'] == 'Població']
        prov_map = {
            'Barcelona': 'barcelona',
            'Lleida': 'lleida',
            'Tarragona': 'tarragona',
            'Girona': 'girona',
            'Reus': 'tarragona',
            'Figueres': 'girona',
            'Blanes': 'girona',
            'Lloret de Mar': 'girona',
            'Vendrell, el': 'tarragona',
            'Olot': 'girona',
            'Tortosa': 'tarragona',
            'Cambrils': 'tarragona',
            'Salt': 'girona',
            'Salou': 'tarragona',
            'Valls': 'tarragona',
            'Calafell': 'tarragona',
            'Palafrugell': 'girona',
            'Vila-seca': 'tarragona',
            'Sant Feliu de Guíxols': 'girona',
            'Amposta': 'tarragona',
        }
        def get_province(row):
            name = row['row']
            for prov in prov_map:
                if prov.lower() in name.lower():
                    return prov_map[prov]
            # There's too many Barcelona cities that are relevant, so the default is Barcelona
            return 'barcelona'
        urban_df['province'] = urban_df.apply(get_province, axis=1)
        # Sum urban population by province
        urban_pop = urban_df[urban_df['col'] == 'Població'].groupby('province')['value'].apply(lambda x: x.astype(str).str.replace('.', '').str.replace(',', '').astype(int).sum())
        df1 = df1.copy()
        df1['rural'] = 0.0
        for prov in df1.index:
            if prov.lower() == 'catalunya':
                upop = urban_pop.sum()
            else:
                upop = int(urban_pop[prov.lower()])
            tpop = int(total_population_by_province.get(prov, 0))
            print(f"Province: {prov}, Total Pop: {tpop}, Urban Pop: {upop}")
            if tpop and upop and 'middle' in df1.columns:
                rural_share = 1 - (upop / tpop)
                rural_share = max(0.0, min(1.0, rural_share))
                middle_val = df1.at[prov, 'middle'] if not pd.isna(df1.at[prov, 'middle']) else 0.0
                rural = round(middle_val * rural_share, 2)
                urban = round(middle_val - rural, 2)
                df1.at[prov, 'rural'] = rural
                df1.at[prov, 'middle'] = urban
            else:
                df1.at[prov, 'rural'] = 0.0
        return df1

    def categorize_worktype(df2):
        # Read education data for 2011
        education_csv = os.path.join(os.path.dirname(__file__), 'idescat_raw', 'idescat_education_2011.csv')
        edu_df = pd.read_csv(education_csv, sep=';')
        # Set province as index
        edu_df = edu_df.set_index('province')
        # Convert all columns except 'Total' to numeric
        cols = [c for c in edu_df.columns if c != 'Total']
        edu_df[cols] = edu_df[cols].apply(pd.to_numeric, errors='coerce')
        edu_df['Total'] = pd.to_numeric(edu_df['Total'], errors='coerce')
        # Calculate percentages for each education level (excluding 'Total')
        edu_pct = edu_df[cols].div(edu_df['Total'], axis=0) * 100
        edu_pct = edu_pct.round(2)

        ind_cols = [
            'No sap llegir o escriure',
            'Sense estudis',
            'Educació primària',
            'ESO',
            'FP grau mitjà',
        ]
        middle_cols = [
            'Batxillerat superior',
            'FP grau superior',
            'Diplomatura',
            'Grau universitari',
            'Llicenciatura i doctorat'
        ]
        edu_clean = pd.DataFrame(index=edu_pct.index)
        edu_clean['ind'] = edu_pct[ind_cols].sum(axis=1)
        edu_clean['middle'] = edu_pct[middle_cols].sum(axis=1)

        print("\nEducation level percentages by province (2011):")
        print(edu_clean)

        # Apply education percentages to the middle category of df2
        df2 = df2.copy()
        for prov in df2.index:
            if prov in edu_clean.index and 'middle' in df2.columns:
                middle_val = df2.at[prov, 'middle'] if not pd.isna(df2.at[prov, 'middle']) else 0.0
                ind_pct = edu_clean.at[prov, 'ind'] if not pd.isna(edu_clean.at[prov, 'ind']) else 0.0
                ind_val = round(middle_val * (ind_pct / 100), 2)
                remaining_middle_val = round(middle_val - ind_val, 2)
                df2.at[prov, 'middle'] = remaining_middle_val
                df2.at[prov, 'ind'] = ind_val
        return df2

    
    df0 = categorize_unemployment(df)
    df1 = categorize_bussiness(df0)
    df2 = categorize_rural(df1)
    df3 = categorize_worktype(df2)
    return df3


    

def create_demographic_weights_2015():
    """Main function to create demographic weights by province"""
    
    print("=" * 60)
    print("IDESCAT Population Data Retrieval by Province - 2015")
    print("=" * 60)
    
    # Get age data by province
    age_df = get_population_by_age_province_2015()
    if age_df is None or age_df.empty:
        print("Failed to retrieve age data by province.")
        return

    print("\n✓ Age data by province retrieved successfully")
    print(f"Total records: {len(age_df)}")


    # Total population by province
    # Calculate total population for each province and for all of Catalonia
    filtered_total = age_df[(age_df['sex'].str.lower() == 'total') & (age_df['age_group'].str.lower() != 'total')]
    total_by_province = filtered_total.groupby('province')['population'].sum()
    print("\nTotal population by province (2015):")
    print(total_by_province)

    # Filter to only sex == 'Total' and age_group != 'Total' for weights
    filtered_df = age_df[(age_df['sex'].str.lower() == 'total') & (age_df['age_group'].str.lower() != 'total')]
    # Rename 'age_category' to 'category' for consistency
    filtered_df = filtered_df.rename(columns={'age_category': 'category'})


    # Aggregate by province and age category
    age_weights = filtered_df.groupby(['province', 'category'])['population'].sum().unstack()
    age_weights['total'] = age_weights.sum(axis=1)
    age_weights = (age_weights.div(age_weights['total'], axis=0) * 100).round(2)
    

    print("=" * 60)
    print("IDESCAT Population Occupational breakdown - 2012")
    print("=" * 60)

    data_breakdown = working_condition_breakdown(age_weights, total_by_province)

    data_breakdown.to_csv(os.path.join(os.path.dirname(__file__), 'clean', 'population_weights_2012.csv'))

    print("\nWeights by Province (%):")
    print(data_breakdown)

    print("\n" + "=" * 60)

if __name__ == "__main__":
    create_demographic_weights_2015()
