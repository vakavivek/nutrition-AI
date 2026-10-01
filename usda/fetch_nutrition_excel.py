from usda.load_usda_excel import usda_df

def fetch_usda_nutrition_from_excel(fdc_id:int):
    row= usda_df[usda_df['Food code']==fdc_id]

    if row.empty:
        return None
    
    row= row.iloc[0]
    nutrition={
        "Food code":int(row['Food code']),
        "description":row['Main food description'],
        "per":"100g",
        "calories":float(row['Energy (kcal)']),
        "Protein":float(row['Protein (g)']),
        "Carbohydrates":float(row['Carbohydrate (g)']),
        "Fats":float(row['Total Fat (g)'])
    }

    return nutrition