import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

# hard-coded values
modelChoice = 'fb_bart_tfidf'
vdf_filepath = r"dataSources/ScrapedOutputFiles/(Roy) data validation.xlsx" 
topN = 3
section = 'Section'
division = 'Division'
group = 'Group'
Class = 'Class'
subclass = 'Sub-class'
level = subclass
DoS = pd.read_csv("C:/Users/Michael/Documents/GitHub/ssicsync/dataSources/ScrapedOutputFiles/(Roy) List of 90 Coy and SSIC.csv")
modelOutputs = pd.read_excel("C:/Users/Michael/Documents/GitHub/ssicsync/vdf.xlsx")

# df_detailed_def = pd.read_excel(r"dataSources/DoS/ssic2020-detailed-definitions.xlsx", skiprows=4)
# df_alpha_index = pd.read_excel(r"dataSources/DoS/ssic2020-alphabetical-index.xlsx", dtype=str, skiprows=5)

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

# df_alpha_index = df_alpha_index.drop(df_alpha_index.columns[2], axis=1).dropna().rename(columns={'SSIC 2020': 'SSIC 2020','SSIC 2020 Alphabetical Index Description': 'Detailed Definitions'})
# df_concat = pd.concat([df_detailed_def, df_alpha_index])

# ####################################################################################################
# ### Select which fact table to train/transform
# # - df_detailed_def
# # - df_concat       (concat of df_detailed_def and df_alpha_index)

# df_data_dict = df_detailed_def 
# ####################################################################################################

# # Section, 1-alpha 
# ssic_1_raw = df_data_dict[df_data_dict['SSIC 2020'].apply(lambda x: len(str(x)) == 1)].reset_index(drop=True).drop(columns=['Detailed Definitions', 'Cross References', 'Examples of Activities Classified Under this Code']) 
# ssic_1_raw['Groups Classified Under this Code'] = ssic_1_raw['Groups Classified Under this Code'].str.split('\n•')
# ssic_1 = ssic_1_raw.explode('Groups Classified Under this Code').reset_index(drop=True)
# ssic_1['Groups Classified Under this Code'] = ssic_1['Groups Classified Under this Code'].str.replace('•', '')
# ssic_1['Section, 2 digit code'] = ssic_1['Groups Classified Under this Code'].str[0:2]
# ssic_1 = ssic_1.rename(columns={'SSIC 2020': 'Section','SSIC 2020 Title': 'Section Title'})

# # Division, 2-digit
# ssic_2_raw = df_data_dict[df_data_dict['SSIC 2020'].apply(lambda x: len(str(x)) == 2)].reset_index(drop=True).drop(columns=['Detailed Definitions', 'Cross References', 'Examples of Activities Classified Under this Code'])
# ssic_2_raw['Groups Classified Under this Code'] = ssic_2_raw['Groups Classified Under this Code'].str.split('\n•')
# ssic_2 = ssic_2_raw.explode('Groups Classified Under this Code').reset_index(drop=True)
# ssic_2['Groups Classified Under this Code'] = ssic_2['Groups Classified Under this Code'].str.replace('•', '')
# ssic_2 = ssic_2.rename(columns={'SSIC 2020': 'Division','SSIC 2020 Title': 'Division Title'}).drop(columns=['Groups Classified Under this Code']).drop_duplicates()

# # Group, 3-digit 
# ssic_3_raw = df_data_dict[df_data_dict['SSIC 2020'].apply(lambda x: len(str(x)) == 3)].reset_index(drop=True).drop(columns=['Detailed Definitions', 'Cross References', 'Examples of Activities Classified Under this Code'])
# ssic_3_raw['Groups Classified Under this Code'] = ssic_3_raw['Groups Classified Under this Code'].str.split('\n•')
# ssic_3 = ssic_3_raw.explode('Groups Classified Under this Code').reset_index(drop=True)
# ssic_3['Groups Classified Under this Code'] = ssic_3['Groups Classified Under this Code'].str.replace('•', '')
# ssic_3 = ssic_3.rename(columns={'SSIC 2020': 'Group','SSIC 2020 Title': 'Group Title'}).drop(columns=['Groups Classified Under this Code']).drop_duplicates()

# # Class, 4-digit
# ssic_4_raw = df_data_dict[df_data_dict['SSIC 2020'].apply(lambda x: len(str(x)) == 4)].reset_index(drop=True).drop(columns=['Detailed Definitions', 'Cross References', 'Examples of Activities Classified Under this Code'])
# ssic_4_raw['Groups Classified Under this Code'] = ssic_4_raw['Groups Classified Under this Code'].str.split('\n•')
# ssic_4 = ssic_4_raw.explode('Groups Classified Under this Code').reset_index(drop=True)
# ssic_4['Groups Classified Under this Code'] = ssic_4['Groups Classified Under this Code'].str.replace('•', '')
# ssic_4 = ssic_4.rename(columns={'SSIC 2020': 'Class','SSIC 2020 Title': 'Class Title'}).drop(columns=['Groups Classified Under this Code']).drop_duplicates()

# # Sub-class, 5-digit
# ssic_5 = df_data_dict[df_data_dict['SSIC 2020'].apply(lambda x: len(str(x)) == 5)].reset_index(drop=True).drop(columns=['Groups Classified Under this Code'])
# ssic_5.replace('<Blank>', '', inplace=True)
# ssic_5.replace('NaN', '', inplace=True)

# # prep join columns
# ssic_5['Section, 2 digit code'] = ssic_5['SSIC 2020'].astype(str).str[:2]
# ssic_5['Division'] = ssic_5['SSIC 2020'].astype(str).str[:2]
# ssic_5['Group'] = ssic_5['SSIC 2020'].astype(str).str[:3]
# ssic_5['Class'] = ssic_5['SSIC 2020'].astype(str).str[:4]

# # join ssic_n Hierarhical Layer Tables (Section, Division, Group, Class, Sub-Class)
# ssic_df = pd.merge(ssic_5, ssic_1[['Section', 'Section Title', 'Section, 2 digit code']], on='Section, 2 digit code', how='left')
# ssic_df = pd.merge(ssic_df, ssic_2[['Division', 'Division Title']], on='Division', how='left')
# ssic_df = pd.merge(ssic_df, ssic_3[['Group', 'Group Title']], on='Group', how='left')
# ssic_df = pd.merge(ssic_df, ssic_4[['Class', 'Class Title']], on='Class', how='left')

# # mapping
# level_map = {
#     section: ('Section', ssic_df.iloc[:, [0, 1, 9, 10, 11, 12, 13]].drop_duplicates(), ssic_1.iloc[:, [0, 1]].drop_duplicates().reset_index(drop=True), "nusebacra/ssicsync_section_classifier", ssic_1),
#     division: ('Division', ssic_df.iloc[:, [0, 1, 6, 10, 11, 12, 13]].drop_duplicates(), ssic_2.iloc[:, [0, 1]].drop_duplicates().reset_index(drop=True), "nusebacra/ssicsync_division_classifier", ssic_2),
#     group: ('Group', ssic_df.iloc[:, [0, 1, 7, 10, 11, 12, 13]].drop_duplicates(), ssic_3.iloc[:, [0, 1]].drop_duplicates().reset_index(drop=True), "nusebacra/ssicsync_group_classifier", ssic_3),
#     Class: ('Class', ssic_df.iloc[:, [0, 1, 8, 10, 11, 12, 13]].drop_duplicates(), ssic_4.iloc[:, [0, 1]].drop_duplicates().reset_index(drop=True), "nusebacra/ssicsync_class_classifier", ssic_4),
#     subclass: ('SSIC 2020', ssic_df.iloc[:, [0, 1, 9, 10, 11, 12, 13]].drop_duplicates(), ssic_5.iloc[:, [0, 1]].drop_duplicates().reset_index(drop=True), "nusebacra/ssicsync_subclass_classifier", ssic_5)
# }

# # Get the values for a and b based on the lvl_train
# lvl_train, df_streamlit, ssic_n_sl, model, ssic_lvl = level_map.get(level, ('default_a', 'default_b', 'default_c', 'default_d', 'default_e'))
# lvl_train_title = lvl_train + " Title"

# # prep ssic_n dictionary df_prep
# df_prep = ssic_df[[lvl_train, 'Detailed Definitions']]
# df_prep['encoded_cat'] = df_prep[lvl_train].astype('category').cat.codes
# df_prep = df_prep[[lvl_train, 'encoded_cat']].drop_duplicates()

values = []
prop_dict = {}
df_display = {}
categories = [section, division, group, Class, "Subclass"]

uenEntity_dict = {"UEN": DoS['UEN'].to_list(),
                  "entity_name": DoS['entity_name'].to_list()}
DoS = pd.DataFrame(uenEntity_dict)
uenEntity_dict = dict(zip(DoS['UEN'], DoS['entity_name']))

for cat in categories:
    prop_dict[cat] = modelOutputs[modelOutputs[f'p_{modelChoice}_{cat}_check'] == 'Y'].shape[0]/modelOutputs.shape[0]
    modelOutputs['entity_name'] = modelOutputs['UEN Number'].map(uenEntity_dict)
    if cat == 'Subclass':
        cat_key = subclass
    else:
        cat_key = cat
    df_display[cat_key] = modelOutputs[['entity_name', f'p_{modelChoice}_{cat}_check']]
    df_display[cat_key].rename(columns = {f'p_{modelChoice}_{cat}_check': 'classification'}, inplace = True)

for level in prop_dict.values():
    values.append(round(level*100, 1))
categories = [section, division, group, Class, subclass]

# sectionprop = modelOutputs[modelOutputs[f'p_{modelChoice}_{"Subclass"}_check == 'Y'].shape[0]/modelOutputs.shape[0]
# divisionprop = modelOutputs[modelOutputs[f'p_{modelChoice}_{"Subclass"}_check == 'Y'].shape[0]/modelOutputs.shape[0]
# groupprop = modelOutputs[modelOutputs[f'p_{modelChoice}_{"Subclass"}_check'] == 'Y'].shape[0]/modelOutputs.shape[0]
# classprop = modelOutputs[modelOutputs[f'p_{modelChoice}_{"Subclass"}_check'] == 'Y'].shape[0]/modelOutputs.shape[0]
# subclassprop = modelOutputs[modelOutputs[f'p_{modelChoice}_{"Subclass"}_check'] == 'Y'].shape[0]/modelOutputs.shape[0]

# values = []
# for level in [sectionprop, divisionprop, groupprop, classprop, subclassprop]:
#     values.append(round(level*100, 1))

# TODO get 'categories' and 'values' from modelOutputs (DONE)

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

# df_display = {
#     section: Section_prediction_df,
#     division: Division_prediction_df,
#     group: Group_prediction_df,
#     Class: Class_prediction_df,
#     subclass: Subclass_prediction_df
# }

# Streamlit selectbox for user input
level_input = st.selectbox(
    "Level of correct classification",
    (section, division, group, Class, subclass)
)
level = level_input if level_input else section

levelDisplay_df = df_display[level]
correctWrongClassification_df = levelDisplay_df[levelDisplay_df.classification.notnull()]
# levelDisplay_df.rename(columns = {levelDisplay_df.columns[-1]: 'classification'}, inplace = True)

# TODO correctWrongClassification_df can get from modelOutputs (DONE)

# Display df with text wrapping and no truncation
st.dataframe(
    correctWrongClassification_df.style.set_properties(**{
        'white-space': 'pre-wrap',
        'overflow-wrap': 'break-word',
    })
)

companies_tuple = tuple(correctWrongClassification_df.entity_name)

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

content_input = capitalize_sentence(modelOutputs[modelOutputs.entity_name == companies_input].reset_index(drop = True)['Notes Page Content'][0])

ssic_input = modelOutputs[modelOutputs.entity_name == companies_input].reset_index(drop = True).ssic_code[0]
ssic2_input = modelOutputs[modelOutputs.entity_name == companies_input].reset_index(drop = True).ssic_code2[0]
ssicDesc_input = modelOutputs[modelOutputs.entity_name == companies_input].reset_index(drop = True)['ssic_code&title'][0]

topNSSIC_input_list = modelOutputs[modelOutputs.entity_name == companies_input].reset_index(drop = True)[f'p_{modelChoice}'][0]
topNSSICDesc_input = modelOutputs[modelOutputs.entity_name == companies_input].reset_index(drop = True)[f'p_{modelChoice}_desc'][0]

st.header('Company SSIC Details')
st.subheader('Company Name:')
st.text(companies_input)
st.subheader('Company Description:')
st.text(content_input)

col1, col2 = st.columns([1,1])
with col1:
    st.subheader('Company SSICs & Descriptions:')
    st.write(ssicDesc_input)
with col2:
    st.subheader(f'Top {topN} Predicted SSICs & Descriptions:')
    st.text(topNSSICDesc_input)

# import pdb;pdb.set_trace()

# Visual Effects ### - https://docs.streamlit.io/develop/api-reference/status
st.balloons() 

st.sidebar.success("Explore our pages above ☝️")