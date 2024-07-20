import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import matplotlib.pyplot as plt

# hard-coded values
vdf_filepath = r"dataSources/ScrapedOutputFiles/(Roy) data validation.xlsx" 
topN = 3
section = 'Section'
division = 'Division'
group = 'Group'
Class = 'Class'
subclass = 'Sub-class'
level = subclass

pd.set_option('display.max_columns', None)

####################################################################################################
### Select SSIC Hierarchical Level

# 1. 'Section'
# 2. 'Division'
# 3. 'Group'
# 4. 'Class'
# 5. 'Subclass'

# level_input = st.selectbox(
#     "Classification Model Level",
#     ("Division", "Group", 'Class', 'Subclass'))

# level = level_input if level_input else 'Class'
####################################################################################################

df_detailed_def = pd.read_excel(r"dataSources/DoS/ssic2020-detailed-definitions.xlsx", skiprows=4)
df_alpha_index = pd.read_excel(r"dataSources/DoS/ssic2020-alphabetical-index.xlsx", dtype=str, skiprows=5)
df_alpha_index = df_alpha_index.drop(df_alpha_index.columns[2], axis=1).dropna().rename(columns={'SSIC 2020': 'SSIC 2020','SSIC 2020 Alphabetical Index Description': 'Detailed Definitions'})
df_concat = pd.concat([df_detailed_def, df_alpha_index])

####################################################################################################
### Select which fact table to train/transform
# - df_detailed_def
# - df_concat       (concat of df_detailed_def and df_alpha_index)

df_data_dict = df_detailed_def 
####################################################################################################

# Section, 1-alpha 
ssic_1_raw = df_data_dict[df_data_dict['SSIC 2020'].apply(lambda x: len(str(x)) == 1)].reset_index(drop=True).drop(columns=['Detailed Definitions', 'Cross References', 'Examples of Activities Classified Under this Code']) 
ssic_1_raw['Groups Classified Under this Code'] = ssic_1_raw['Groups Classified Under this Code'].str.split('\n•')
ssic_1 = ssic_1_raw.explode('Groups Classified Under this Code').reset_index(drop=True)
ssic_1['Groups Classified Under this Code'] = ssic_1['Groups Classified Under this Code'].str.replace('•', '')
ssic_1['Section, 2 digit code'] = ssic_1['Groups Classified Under this Code'].str[0:2]
ssic_1 = ssic_1.rename(columns={'SSIC 2020': 'Section','SSIC 2020 Title': 'Section Title'})

# Division, 2-digit
ssic_2_raw = df_data_dict[df_data_dict['SSIC 2020'].apply(lambda x: len(str(x)) == 2)].reset_index(drop=True).drop(columns=['Detailed Definitions', 'Cross References', 'Examples of Activities Classified Under this Code'])
ssic_2_raw['Groups Classified Under this Code'] = ssic_2_raw['Groups Classified Under this Code'].str.split('\n•')
ssic_2 = ssic_2_raw.explode('Groups Classified Under this Code').reset_index(drop=True)
ssic_2['Groups Classified Under this Code'] = ssic_2['Groups Classified Under this Code'].str.replace('•', '')
ssic_2 = ssic_2.rename(columns={'SSIC 2020': 'Division','SSIC 2020 Title': 'Division Title'}).drop(columns=['Groups Classified Under this Code']).drop_duplicates()

# Group, 3-digit 
ssic_3_raw = df_data_dict[df_data_dict['SSIC 2020'].apply(lambda x: len(str(x)) == 3)].reset_index(drop=True).drop(columns=['Detailed Definitions', 'Cross References', 'Examples of Activities Classified Under this Code'])
ssic_3_raw['Groups Classified Under this Code'] = ssic_3_raw['Groups Classified Under this Code'].str.split('\n•')
ssic_3 = ssic_3_raw.explode('Groups Classified Under this Code').reset_index(drop=True)
ssic_3['Groups Classified Under this Code'] = ssic_3['Groups Classified Under this Code'].str.replace('•', '')
ssic_3 = ssic_3.rename(columns={'SSIC 2020': 'Group','SSIC 2020 Title': 'Group Title'}).drop(columns=['Groups Classified Under this Code']).drop_duplicates()

# Class, 4-digit
ssic_4_raw = df_data_dict[df_data_dict['SSIC 2020'].apply(lambda x: len(str(x)) == 4)].reset_index(drop=True).drop(columns=['Detailed Definitions', 'Cross References', 'Examples of Activities Classified Under this Code'])
ssic_4_raw['Groups Classified Under this Code'] = ssic_4_raw['Groups Classified Under this Code'].str.split('\n•')
ssic_4 = ssic_4_raw.explode('Groups Classified Under this Code').reset_index(drop=True)
ssic_4['Groups Classified Under this Code'] = ssic_4['Groups Classified Under this Code'].str.replace('•', '')
ssic_4 = ssic_4.rename(columns={'SSIC 2020': 'Class','SSIC 2020 Title': 'Class Title'}).drop(columns=['Groups Classified Under this Code']).drop_duplicates()

# Sub-class, 5-digit
ssic_5 = df_data_dict[df_data_dict['SSIC 2020'].apply(lambda x: len(str(x)) == 5)].reset_index(drop=True).drop(columns=['Groups Classified Under this Code'])
ssic_5.replace('<Blank>', '', inplace=True)
ssic_5.replace('NaN', '', inplace=True)

# prep join columns
ssic_5['Section, 2 digit code'] = ssic_5['SSIC 2020'].astype(str).str[:2]
ssic_5['Division'] = ssic_5['SSIC 2020'].astype(str).str[:2]
ssic_5['Group'] = ssic_5['SSIC 2020'].astype(str).str[:3]
ssic_5['Class'] = ssic_5['SSIC 2020'].astype(str).str[:4]

# join ssic_n Hierarhical Layer Tables (Section, Division, Group, Class, Sub-Class)
ssic_df = pd.merge(ssic_5, ssic_1[['Section', 'Section Title', 'Section, 2 digit code']], on='Section, 2 digit code', how='left')
ssic_df = pd.merge(ssic_df, ssic_2[['Division', 'Division Title']], on='Division', how='left')
ssic_df = pd.merge(ssic_df, ssic_3[['Group', 'Group Title']], on='Group', how='left')
ssic_df = pd.merge(ssic_df, ssic_4[['Class', 'Class Title']], on='Class', how='left')

# mapping
level_map = {
    section: ('Section', ssic_df.iloc[:, [0, 1, 9, 10, 11, 12, 13]].drop_duplicates(), ssic_1.iloc[:, [0, 1]].drop_duplicates().reset_index(drop=True), "nusebacra/ssicsync_section_classifier", ssic_1),
    division: ('Division', ssic_df.iloc[:, [0, 1, 6, 10, 11, 12, 13]].drop_duplicates(), ssic_2.iloc[:, [0, 1]].drop_duplicates().reset_index(drop=True), "nusebacra/ssicsync_division_classifier", ssic_2),
    group: ('Group', ssic_df.iloc[:, [0, 1, 7, 10, 11, 12, 13]].drop_duplicates(), ssic_3.iloc[:, [0, 1]].drop_duplicates().reset_index(drop=True), "nusebacra/ssicsync_group_classifier", ssic_3),
    Class: ('Class', ssic_df.iloc[:, [0, 1, 8, 10, 11, 12, 13]].drop_duplicates(), ssic_4.iloc[:, [0, 1]].drop_duplicates().reset_index(drop=True), "nusebacra/ssicsync_class_classifier", ssic_4),
    subclass: ('SSIC 2020', ssic_df.iloc[:, [0, 1, 9, 10, 11, 12, 13]].drop_duplicates(), ssic_5.iloc[:, [0, 1]].drop_duplicates().reset_index(drop=True), "nusebacra/ssicsync_subclass_classifier", ssic_5)
}

# Get the values for a and b based on the lvl_train
lvl_train, df_streamlit, ssic_n_sl, model, ssic_lvl = level_map.get(level, ('default_a', 'default_b', 'default_c', 'default_d', 'default_e'))
lvl_train_title = lvl_train + " Title"

# prep ssic_n dictionary df_prep
df_prep = ssic_df[[lvl_train, 'Detailed Definitions']]
df_prep['encoded_cat'] = df_prep[lvl_train].astype('category').cat.codes
df_prep = df_prep[[lvl_train, 'encoded_cat']].drop_duplicates()

# load model directly from huggingface
tokenizer = AutoTokenizer.from_pretrained(model)
model = TFAutoModelForSequenceClassification.from_pretrained(model)

# Define the function to predict scores and categories
def predict_text(text, top_n=15):
    # Tokenize the input text
    predict_input = tokenizer.encode(
        text,
        truncation=True,
        padding=True,
        return_tensors="tf"
    )
    
    # Get the model output
    output = model(predict_input)[0]
    output_array = output.numpy()[0]  # Get the first (and only) output for this input

    ############################################################################################################################################### 
    
    # Get the top n scores and their corresponding categories
    top_n_indices = output_array.argsort()[-top_n:][::-1]
    top_n_scores = output_array[top_n_indices]
    top_n_categories = top_n_indices

    # Prepare the merged DataFrame
    lvl_dict = df_prep[[lvl_train, 'encoded_cat']].drop_duplicates()
    lvl_ref = ssic_lvl[[lvl_train, lvl_train_title]].drop_duplicates()
    merged_df = lvl_dict.merge(lvl_ref, on=lvl_train, how='left')

    # Create a DataFrame for the top predictions
    sorted_output_df = pd.DataFrame({
        'encoded_cat': top_n_categories,
        'value': top_n_scores
    })
    
    # Merge the top predictions with the merged DataFrame
    merged_df2 = sorted_output_df.merge(merged_df, on='encoded_cat', how='left')

    # Return the results as a list of dictionaries
    # return final_merged_df[['value', lvl_train, lvl_train_title]].to_dict(orient='records')
    return merged_df2[['value', lvl_train, lvl_train_title]].to_dict(orient='records')

# Validation Data

# Create an empty list to store the predictions
predictions = []

# Iterate over each row of the DataFrame and apply the prediction function
vdf = pd.read_excel(vdf_filepath, dtype = str)
for idx, row in vdf.iterrows():
    text = row['Notes Page Content'] # Specify Notes Page Content2
    result = predict_text(text)
    for pred in result:
        pred.update({
            'UEN': row['UEN'],
            'entity_name': row['entity_name'],
            'ssic_code': row['ssic_code'],
            'ssic_code2': row['ssic_code2'],
            'Notes Page Content': row['Notes Page Content'],
            'Notes Page Content2': text,
            'm_Section': row['Section'],
            'm_Section2': row['Section' + "2"],
            'm_Division': row['Division'],
            'm_Division2': row['Division' + "2"],
            'm_Group': row['Group'],
            'm_Group2': row['Group' + "2"],
            'm_Class': row['Class'],
            'm_Class2': row['Class' + "2"],
            'm_Sub-class': row['Sub-class'],
            'm_Sub-class2': row['Sub-class' + "2"]
        })
        predictions.append(pred)

# Create a DataFrame from the list of predictions
prediction_df = pd.DataFrame(predictions)

# Extracting respective SSIC levels from Predictions
prediction_df['p_Section'] = prediction_df[lvl_train].apply(lambda x: x[:2] if len(x) >= 2 else np.nan)
prediction_df['p_Section'] = prediction_df['p_Section'].map(ssic_1.set_index('Section, 2 digit code')['Section'])
prediction_df['p_Division'] = prediction_df[lvl_train].apply(lambda x: x[:2] if len(x) >= 2 else np.nan)
prediction_df['p_Group'] = prediction_df[lvl_train].apply(lambda x: x[:3] if len(x) >= 3 else np.nan)
prediction_df['p_Class'] = prediction_df[lvl_train].apply(lambda x: x[:4] if len(x) >= 4 else np.nan)
prediction_df['p_Sub-class'] = prediction_df[lvl_train].apply(lambda x: x[:5] if len(x) >= 5 else np.nan)
prediction_df = prediction_df.fillna('NaN')

Section_prediction_df = prediction_df.groupby(['entity_name', 'ssic_code', 'ssic_code2', 'm_Section', 'm_Section2', 'Notes Page Content', 'Notes Page Content2'])['p_Section'].apply(list).reset_index()
Division_prediction_df = prediction_df.groupby(['entity_name', 'ssic_code', 'ssic_code2', 'm_Division', 'm_Division2', 'Notes Page Content', 'Notes Page Content2'])['p_Division'].apply(list).reset_index()
Group_prediction_df = prediction_df.groupby(['entity_name', 'ssic_code', 'ssic_code2', 'm_Group', 'm_Group2', 'Notes Page Content', 'Notes Page Content2'])['p_Group'].apply(list).reset_index()
Class_prediction_df = prediction_df.groupby(['entity_name', 'ssic_code', 'ssic_code2', 'm_Class', 'm_Class2', 'Notes Page Content', 'Notes Page Content2'])['p_Class'].apply(list).reset_index()
Subclass_prediction_df = prediction_df.groupby(['entity_name', 'ssic_code', 'ssic_code2', 'm_Sub-class', 'm_Sub-class2', 'Notes Page Content', 'Notes Page Content2'])['p_Sub-class'].apply(list).reset_index()

def check_alpha_in_list(row, N, m1, m2, p):
    # Check if p contains any NaN values
    if 'NaN' in row[p]:
        return None
    
    # Check if the alpha in the 1st column is in the first N elements of the list in the 3rd column
    if row[m1] in row[p][:N] or row[m2] in row[p][:N]:
        return 'Y'
    else:
        return 'N'

# Section_N = st.text_input('Top N Section:', '')
# Division_N = st.text_input('Top N Division:', '')
# Group_N = st.text_input('Top N Group:', '')
# Class_N = st.text_input('Top N Class:', '')
# Subclass_N = st.text_input('Top N Subclass:', '')

# # Set default values if inputs are blank
# Section_N = int(Section_N) if Section_N else 3
# Division_N = int(Division_N) if Division_N else 3
# Group_N = int(Group_N) if Group_N else 3
# Class_N = int(Class_N) if Class_N else 3
# Subclass_N = int(Subclass_N) if Subclass_N else 3

# Section_prediction_df['Within Top N (Section)'] = Section_prediction_df.apply(check_alpha_in_list, axis=1, N=Section_N, m1 = 'm_Section', m2 = 'm_Section2', p = 'p_Section')
# Division_prediction_df['Within Top N (Division)'] = Division_prediction_df.apply(check_alpha_in_list, axis=1, N=Division_N, m1 = 'm_Division', m2 = 'm_Division2', p = 'p_Division')
# Group_prediction_df['Within Top N (Group)'] = Group_prediction_df.apply(check_alpha_in_list, axis=1, N=Group_N, m1 = 'm_Group', m2 = 'm_Group2', p = 'p_Group')
# Class_prediction_df['Within Top N (Class)'] = Class_prediction_df.apply(check_alpha_in_list, axis=1, N=Class_N, m1 = 'm_Class', m2 = 'm_Class2', p = 'p_Class')
# Subclass_prediction_df['Within Top N (Sub-class)'] = Subclass_prediction_df.apply(check_alpha_in_list, axis=1, N=Subclass_N, m1 = 'm_Sub-class', m2 = 'm_Sub-class2', p = 'p_Sub-class')

Section_prediction_df[section] = Section_prediction_df.apply(check_alpha_in_list, axis=1, N=topN, m1 = 'm_Section', m2 = 'm_Section2', p = 'p_Section')
Division_prediction_df[division] = Division_prediction_df.apply(check_alpha_in_list, axis=1, N=topN, m1 = 'm_Division', m2 = 'm_Division2', p = 'p_Division')
Group_prediction_df[group] = Group_prediction_df.apply(check_alpha_in_list, axis=1, N=topN, m1 = 'm_Group', m2 = 'm_Group2', p = 'p_Group')
Class_prediction_df[Class] = Class_prediction_df.apply(check_alpha_in_list, axis=1, N=topN, m1 = 'm_Class', m2 = 'm_Class2', p = 'p_Class')
Subclass_prediction_df[subclass] = Subclass_prediction_df.apply(check_alpha_in_list, axis=1, N=topN, m1 = 'm_Sub-class', m2 = 'm_Sub-class2', p = 'p_Sub-class')

dfs = [
    Section_prediction_df,
    Division_prediction_df,
    Group_prediction_df,
    Class_prediction_df,
    Subclass_prediction_df
]

# Merge DataFrames on entity_name, ssic_code, ssic_code2
merged_df = dfs[0]  # Start with the first DataFrame
for df in dfs[1:]:
    merged_df = pd.merge(merged_df, df, on=['entity_name', 'ssic_code', 'ssic_code2', 'Notes Page Content', 'Notes Page Content2'], how='outer')

columns_to_include = [section, division, group, Class, subclass]

# Filter the DataFrame to include only the specified columns
filtered_df = merged_df[columns_to_include]

# Calculate percentage of 'Y' and 'N' in each column
percentages = {}
for column in filtered_df.columns:
    counts = filtered_df[column].value_counts(normalize=True) * 100
    percentages[column] = counts.reindex(['Y', 'N']).fillna(0)

# Create a DataFrame from percentages
percentages_df = pd.DataFrame(percentages).round(1).reset_index(drop = True)
percentages_df = percentages_df[percentages_df.index == 0]

# Transpose the DataFrame for vertical plotting
percentages_df_transposed = percentages_df.transpose().reset_index(drop = False)
categories = percentages_df_transposed['index'].to_list()
values = percentages_df_transposed[0].to_list()

# Create horizontal bar chart
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(categories, values, color='skyblue')
# ax.set_xlabel('Percentage')
# ax.set_ylabel('Categories')
ax.set_title('Accuracy')
ax.set_xlim(0, 100)  # Assuming the percentage is between 0 and 100

# Remove right and top spines
ax.spines[['right', 'top']].set_visible(False)

# Adding data labels
for bar in bars:
    ax.annotate(f'{bar.get_width()}%', 
                xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),  # 5 points offset
                textcoords='offset points',
                ha='left', va='center')

# Adjust layout
plt.tight_layout()

# Display plot in Streamlit
st.pyplot(fig)

df_display = {
    section: Section_prediction_df,
    division: Division_prediction_df,
    group: Group_prediction_df,
    Class: Class_prediction_df,
    subclass: Subclass_prediction_df
}

# Streamlit selectbox for user input
level_input = st.selectbox(
    "Level of correct classification",
    (section, division, group, Class, subclass)
)
level = level_input if level_input else section

levelDisplay_df = df_display[level]
levelDisplay_df.rename(columns = {levelDisplay_df.columns[-1]: 'classification'}, inplace = True)
correctWrongClassification_df = levelDisplay_df[['entity_name', 'classification']]

# Display df with text wrapping and no truncation
st.dataframe(
    correctWrongClassification_df.style.set_properties(**{
        'white-space': 'pre-wrap',
        'overflow-wrap': 'break-word',
    })
)

companies_tuple = tuple(levelDisplay_df.entity_name)

companies_input = st.selectbox(
    "List of Companies",
    companies_tuple)

def capitalize_sentence(text):
    # Split the text into sentences
    sentences = text.split('. ')
    # Capitalize the first letter of each sentence
    sentences = [sentence[0].upper() + sentence[1:].lower() if sentence else '' for sentence in sentences]
    # Join the sentences back into a single string
    return '. '.join(sentences)

ssic_input = levelDisplay_df[levelDisplay_df.entity_name == companies_input].reset_index(drop = True).ssic_code[0]
ssic2_input = levelDisplay_df[levelDisplay_df.entity_name == companies_input].reset_index(drop = True).ssic_code2[0]
content_input = capitalize_sentence(levelDisplay_df[levelDisplay_df.entity_name == companies_input].reset_index(drop = True)['Notes Page Content'][0])
content2_input = capitalize_sentence(levelDisplay_df[levelDisplay_df.entity_name == companies_input].reset_index(drop = True)['Notes Page Content2'][0])

sectionTitle_input = capitalize_sentence(ssic_df[ssic_df['SSIC 2020'] == ssic_input].reset_index(drop = True)['Section Title'][0])
divisionTitle_input = capitalize_sentence(ssic_df[ssic_df['SSIC 2020'] == ssic_input].reset_index(drop = True)['Division Title'][0])
groupTitle_input = capitalize_sentence(ssic_df[ssic_df['SSIC 2020'] == ssic_input].reset_index(drop = True)['Group Title'][0])
classTitle_input = capitalize_sentence(ssic_df[ssic_df['SSIC 2020'] == ssic_input].reset_index(drop = True)['Class Title'][0])
subclassTitle_input = capitalize_sentence(ssic_df[ssic_df['SSIC 2020'] == ssic_input].reset_index(drop = True)['Detailed Definitions'][0])
details_display = {
    section: sectionTitle_input,
    division: divisionTitle_input,
    group: groupTitle_input,
    Class: classTitle_input,
    subclass: subclassTitle_input
}
details_input = details_display[level]

sectionTitle2_input = capitalize_sentence(ssic_df[ssic_df['SSIC 2020'] == ssic2_input].reset_index(drop = True)['Section Title'][0])
divisionTitle2_input = capitalize_sentence(ssic_df[ssic_df['SSIC 2020'] == ssic2_input].reset_index(drop = True)['Division Title'][0])
groupTitle2_input = capitalize_sentence(ssic_df[ssic_df['SSIC 2020'] == ssic2_input].reset_index(drop = True)['Group Title'][0])
classTitle2_input = capitalize_sentence(ssic_df[ssic_df['SSIC 2020'] == ssic2_input].reset_index(drop = True)['Class Title'][0])
subclassTitle2_input = capitalize_sentence(ssic_df[ssic_df['SSIC 2020'] == ssic2_input].reset_index(drop = True)['Detailed Definitions'][0])
details_display2 = {
    section: sectionTitle2_input,
    division: divisionTitle2_input,
    group: groupTitle2_input,
    Class: classTitle2_input,
    subclass: subclassTitle2_input
}
details2_input = details_display2[level]

st.header('Company SSIC Details')
st.subheader('Company Name:')
st.text(companies_input)

st.subheader('1st SSIC:')
st.text(ssic_input)
st.subheader('1st SSIC definition:')
st.text(details_input)
st.subheader('Company 1st description:') # TODO how come got 2 descriptions?
st.text(content_input)

st.subheader('2nd SSIC:')
st.text(ssic2_input)
st.subheader('2nd SSIC definition:')
st.text(details2_input)
st.subheader('Company 2nd description:') # TODO how come got 2 descriptions?
st.text(content2_input)

# level_input = st.selectbox(
#     "Classification Model Level",
#     ("Division", "Group", 'Class', 'Subclass'))
# level = level_input if level_input else 'Class'

# # Streamlit selectbox for filter criteria
# filter_criteria = st.selectbox(
#     "Filter last column by:",
#     ("Both", "Y", "N")
# )

# # Get the selected DataFrame
# selected_df = df_display[level_input]

# # Get the name of the last column
# last_column = selected_df.columns[-1]

# # Apply filter based on user selection
# if filter_criteria == "Y":
#     filtered_df = selected_df[selected_df[last_column] == "Y"]
# elif filter_criteria == "N":
#     filtered_df = selected_df[selected_df[last_column] == "N"]
# else:
#     filtered_df = selected_df

# # Display the filtered DataFrame with text wrapping and no truncation
# st.dataframe(
#     filtered_df.style.set_properties(**{
#         'white-space': 'pre-wrap',
#         'overflow-wrap': 'break-word',
#     })
# )

# Visual Effects ### - https://docs.streamlit.io/develop/api-reference/status
st.balloons() 

st.sidebar.success("Explore our pages above ☝️")

import pdb;pdb.set_trace()