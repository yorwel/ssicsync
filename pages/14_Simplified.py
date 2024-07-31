import streamlit as st
import ast
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
# vdf_filepath = "./dataSources/ScrapedOutputFiles/(Roy) data validation.xlsx"
topN = 3
section = 'Section'
division = 'Division'
group = 'Group'
Class = 'Class'
subclass = 'Sub-class'
# level = subclass
DoS = pd.read_csv("./dataSources/ScrapedOutputFiles/(Roy) List of 90 Coy and SSIC.csv")
modelOutputs = pd.read_excel("./vdf.xlsx")

# functions
def capitalize_sentence(text):
    # Split the text into sentences
    sentences = text.split('. ')
    # Capitalize the first letter of each sentence
    sentences = [sentence[0].upper() + sentence[1:].lower() if sentence else '' for sentence in sentences]
    # Join the sentences back into a single string
    return '. '.join(sentences)

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

# Create horizontal bar chart
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(categories, values, color='skyblue')
# ax.set_xlabel('Percentage')
# ax.set_ylabel('Categories')
ax.set_title('Classification Accuracy',  fontweight='bold')
fig.text(0.525, 0.92, f'Company SSIC(s) Within Top {topN} Predicted SSICs', ha='center', fontsize=10)
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

# Streamlit selectbox for user input
level_input = st.selectbox(
    "Level of Classification:",
    (section, division, group, Class, subclass)
)
level = level_input if level_input else section

levelDisplay_df = df_display[level]
correctWrongClassification_df = levelDisplay_df[levelDisplay_df.classification.notnull()]

correctWrongClassification_df.loc[correctWrongClassification_df.classification == 'N', 'classification'] = 'No'
correctWrongClassification_df.loc[correctWrongClassification_df.classification == 'Y', 'classification'] = 'Yes'
correctWrongClassification_df.rename(columns = {'entity_name': 'Company Name', 'classification': f'Within Top {topN}'}, inplace = True)

# Display df with text wrapping and no truncation
st.dataframe(
    correctWrongClassification_df.style.set_properties(**{
        'white-space': 'pre-wrap',
        'overflow-wrap': 'break-word',
    })
)

companies_tuple = tuple(correctWrongClassification_df['Company Name'])
companies_input = st.selectbox(
    "List of Companies",
    companies_tuple)

content_input = capitalize_sentence(modelOutputs[modelOutputs.entity_name == companies_input].reset_index(drop = True)['Notes Page Content'][0])

ssic_input = modelOutputs[modelOutputs.entity_name == companies_input].reset_index(drop = True).ssic_code[0]
ssic2_input = modelOutputs[modelOutputs.entity_name == companies_input].reset_index(drop = True).ssic_code2[0]
ssicDesc_input = modelOutputs[modelOutputs.entity_name == companies_input].reset_index(drop = True)['ssic_code&title'][0]

topNSSIC_input_list = modelOutputs[modelOutputs.entity_name == companies_input].reset_index(drop = True)[f'p_{modelChoice}'][0]
topNSSICDesc_input = modelOutputs[modelOutputs.entity_name == companies_input].reset_index(drop = True)[f'p_{modelChoice}_desc'][0]

st.header('Company SSIC Details')
st.subheader('Company Name:')
st.write(companies_input)
st.subheader('Company Description:')
st.write(content_input)

###############################################################################################

# TODO TO DELETE EVENTUALLY

df_detailed_def = pd.read_excel("./dataSources/DoS/ssic2020-detailed-definitions.xlsx", skiprows=4)
df_alpha_index = pd.read_excel("./dataSources/DoS/ssic2020-alphabetical-index.xlsx", dtype=str, skiprows=5)
df_alpha_index = df_alpha_index.drop(df_alpha_index.columns[2], axis=1).dropna().rename(columns={'SSIC 2020': 'SSIC 2020','SSIC 2020 Alphabetical Index Description': 'Detailed Definitions'})

df_concat = pd.concat([df_detailed_def, df_alpha_index])

df_data_dict = df_detailed_def 

# prep ssic_n tables for joining/merging and reference
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

###############################################################################################

if pd.isna(ssic2_input):
    ssic2_input = 'NULL'
coySSIC = [ssic_input, ssic2_input]
allSSICs_list = coySSIC + ast.literal_eval(topNSSIC_input_list)

coySSIC_input = []
predictedSSIC_input = []
for index, ssic in enumerate(allSSICs_list):
    if ssic == 'NULL':
        pass
    else:
        if isinstance(ssic, str):
            ssic = ssic
        else:
            ssic = str(int(ssic))
        if level == section:
            ssicCode = ssic[:1]
        elif level == division:
            ssicCode = ssic[:2]
        elif level == group:
            ssicCode = ssic[:3]
        elif level == Class:
            ssicCode = ssic[:4]
        elif level == subclass:
            ssicCode = ssic[:5]

        try:
            sectionTitle_input = capitalize_sentence(ssic_df[ssic_df['SSIC 2020'] == ssic].reset_index(drop = True)['Section Title'][0])
        except:
            sectionTitle_input = 'NULL'
        try:
            divisionTitle_input = capitalize_sentence(ssic_df[ssic_df['SSIC 2020'] == ssic].reset_index(drop = True)['Division Title'][0])
        except:
            divisionTitle_input = 'NULL'
        try:
            groupTitle_input = capitalize_sentence(ssic_df[ssic_df['SSIC 2020'] == ssic].reset_index(drop = True)['Group Title'][0])
        except:
            groupTitle_input = 'NULL'
        try:
            classTitle_input = capitalize_sentence(ssic_df[ssic_df['SSIC 2020'] == ssic].reset_index(drop = True)['Class Title'][0])
        except:
            classTitle_input = 'NULL'
        try:
            subclassTitle_input = capitalize_sentence(ssic_df[ssic_df['SSIC 2020'] == ssic].reset_index(drop = True)['SSIC 2020 Title'][0])
        except:
            subclassTitle_input = 'NULL'

        details_display = {
            section: sectionTitle_input,
            division: divisionTitle_input,
            group: groupTitle_input,
            Class: classTitle_input,
            subclass: subclassTitle_input
        }
        details_input = details_display[level]
        if index <= 1:
            coySSIC_input.append(f"**{ssicCode}**: {details_input}")
        else:
            predictedSSIC_input.append(f"**{ssicCode}**: {details_input}")

col1, col2 = st.columns([1,1])
with col1:
    st.subheader('Company SSICs & Descriptions:')
    coySSICstring_input = '  \n'.join(coySSIC_input)
    st.write(coySSICstring_input)
with col2:
    st.subheader(f'Top {topN} Predicted SSICs & Descriptions:')
    predictedSSICstring_input = '  \n'.join(predictedSSIC_input)
    st.write(predictedSSICstring_input)

# Visual Effects ### - https://docs.streamlit.io/develop/api-reference/status
st.balloons() 

st.sidebar.success("Explore our pages above ☝️")