import random
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from streamlit_ketcher import st_ketcher
from molfeat.calc import FPCalculator

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def hamming_distance(fp1, fp2):
    return np.sum(fp1 != fp2)

def draw_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.Draw.MolToImage(mol)

def canonize_smiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

def check_mol(mol):
    if len(mol.GetAtoms()) < 2:
        st.error("Only compounds with more than 2 atoms are available for input.")
        return False
    else:
        return True

calc = FPCalculator("ecfp")

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

st.set_page_config(page_title='BigSolDBv2.0', layout="wide")

df = pd.read_csv('BigSolDBv2.0.csv')
df['PubChem_CID'] = df['PubChem_CID'].astype('Int64')
compound_names = sorted(df['Compound_Name'].unique().tolist())
fda_compound_names = sorted(df[df['FDA_Approved'] == 'Yes']['Compound_Name'].unique().tolist())

df_smiles = pd.DataFrame({'SMILES_Solute': list(df['SMILES_Solute'].unique())})
df_smiles['mol'] = df_smiles['SMILES_Solute'].apply(Chem.MolFromSmiles)
df_smiles['mol_ecfp'] = df_smiles['mol'].apply(lambda x: calc(x))

solvents = df['Solvent'].value_counts().reset_index().loc[:50]

top_solvents_by_smiles = df.groupby("Solvent")["SMILES_Solute"].nunique().nlargest(50).reset_index()

n_entries = df.shape[0]
n_smiles = df['SMILES_Solute'].nunique()
n_sources = df['Source'].nunique()
n_solvents = df['Solvent'].nunique()
t_min = df['Temperature_K'].min()
t_max = df['Temperature_K'].max()
col1intro, col2intro, col3intro = st.columns([2, 1, 2])
col1intro.markdown(f"""
# BigSolDB 2.0

Download BigSolDB 2.0: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15094979.svg)](https://doi.org/10.5281/zenodo.15094979)
                   
If you are using our database or the App please cite: Krasnov, L., Malikov, D., Kiseleva, M. et al. **BigSolDB 2.0, dataset of solubility values for organic compounds in different solvents at various temperatures**. Sci Data 12, 1236 (2025). https://doi.org/10.1038/s41597-025-05559-8
""")

col2intro.markdown(f"""
# Overall stats:
* **{n_entries}** number of entries
* **{n_smiles}** unique molecules
* **{n_sources}** literature sources
* **{n_solvents}** solvents
* **{t_min}-{t_max}** temperature range
""")

col3intro.markdown(f"""
# Contributing to the dataset:
We encourage researchers to contribute to further development of the dataset either by performing literature screenings in the future or by standardized data contributions from the laboratories from all around the world. 

To supply the data in any format as well as any other suggestions/ideas regarding the BigSolDB project please contact: [lewa.krasnovs@gmail.com](mailto:lewa.krasnovs@gmail.com)
""")

tabs = st.tabs(["Explore", "Search by Compound Name", "Search by Molecular Structure", "Random Solubility🎲"])

with tabs[0]:
    col1fig, col2fig = st.columns([1, 1])
    fig_sol = px.histogram(df, x='Solubility(mole_fraction)', nbins=64, title='Mole fraction solubility distribution in the BigSolDB 2.0')
    fig_sol.update_layout(yaxis_title='Number of entries')
    fig_sol.update_layout(xaxis_title='Solubility(mole fraction)')
    col1fig.plotly_chart(fig_sol)

    fig_log_sol = px.histogram(df, x='LogS(mol/L)', nbins=64, title='Solubility (Mol/L) distribution in the BigSolDB 2.0')
    fig_log_sol.update_layout(yaxis_title='Number of entries')
    fig_log_sol.update_layout(xaxis_title='Log10 Solubility (Mol/L)')
    col2fig.plotly_chart(fig_log_sol)

    fig_solv = px.bar(solvents, x='Solvent', y='count', text='count', title="Most popular solvents by number of entries")
    fig_solv.update_layout(yaxis_title='Number of entries')
    fig_solv.update_layout(xaxis_title='Solvents')
    st.plotly_chart(fig_solv, use_container_width=True)

    fig_solv_smiles = px.bar(top_solvents_by_smiles, x='Solvent', y='SMILES_Solute', text='SMILES_Solute', title="Most popular solvents by number of unique molecules")
    fig_solv_smiles.update_layout(yaxis_title='Number of molecules')
    fig_solv_smiles.update_layout(xaxis_title='Solvents')
    st.plotly_chart(fig_solv_smiles, use_container_width=True)


with tabs[1]:
    fda = st.checkbox("Only FDA Approved molecules")

    if fda:
        selected = st.selectbox(label='Choose molecule', options=fda_compound_names, index=None, placeholder='Paracetamol')
    else:
        selected = st.selectbox(label='Choose molecule', options=compound_names, index=None, placeholder='Paracetamol')
    if selected:
        search_df = df[(df['Compound_Name'] == selected)]
        search_df.reset_index(drop=True, inplace=True)
        col1result, col2result = st.columns([1, 1])
        pubchem = search_df['PubChem_CID'].iloc[0]
        cas = search_df['CAS'].iloc[0]
        if pubchem is not None:
            col1result.markdown(f'PubChem link: **https://pubchem.ncbi.nlm.nih.gov/compound/{pubchem}**')
        if cas is not None:
            col2result.markdown(f'CAS link: **https://commonchemistry.cas.org/detail?cas_rn={cas}**')

        canonize_mol = search_df['SMILES_Solute'].iloc[0]
        col1result, col2result, col3result = st.columns([1, 1, 2])
        col1result.markdown(f'**Molecule from BigSolDB 2.0**')
        col2result.markdown(f'**Source**')
        col3result.markdown(f'**Solubility**')
        dois = list(search_df['Source'].unique())
        for doi in dois:
            col1result, col2result, col3result = st.columns([1, 1, 2])
            df_comp = search_df[(search_df['Source'] == doi) & (search_df['SMILES_Solute'] == canonize_mol)]
            fig_line = px.line(df_comp, x="Temperature_K", y="Solubility(mole_fraction)", color="Solvent", title=f"Dependence of solubility on temperature", markers=True)
            fig_line.update_layout(yaxis_title='Solubility (mole fraction)')
            col1result.image(draw_molecule(canonize_mol), caption=canonize_mol)
            col2result.markdown(f'**https://doi.org/{doi}**')
            col3result.plotly_chart(fig_line)
        st.dataframe(search_df)

with tabs[2]:

    st.markdown("""Draw your molecule to get SMILES and search in the database:""")

    smile_code = st_ketcher(height=400)
    st.markdown(f"""### Your SMILES:""")
    st.markdown(f"``{smile_code}``")
    st.markdown(f"""### Copy and paste this SMILES into the corresponding box below:""")

    smiles = st.text_input(
            "SMILES",
            placeholder='c1ccc2c(c1)[nH]cn2',
            key='smiles')

    if st.button("Search in the database"):
        if smiles:
            mol = Chem.MolFromSmiles(smiles.strip())
            if (mol is not None):
                if check_mol(mol):
                    canonize_mol = Chem.MolToSmiles(mol)
                    search_df = df[(df['SMILES_Solute'] == canonize_mol)]
                    search_df.reset_index(drop=True, inplace=True)
                    if search_df.shape[0] == 0:
                        st.markdown(f'### This compound was not found in BigSolDB 2.0, but similar was found instead:')
                        col1result, col2result, col3result, col4result = st.columns([1, 1, 1, 3])
                        col1result.markdown(f'**Your molecule**')
                        col2result.markdown(f'**Molecule from BigSolDB 2.0**')
                        col3result.markdown(f'**Source**')
                        col4result.markdown(f'**Solubility**')
                        df_smiles['res_dist'] = df_smiles['mol_ecfp'].apply(lambda ecfp: hamming_distance(calc(mol), ecfp))
                        similar_smiles = df_smiles[df_smiles['res_dist'] == df_smiles['res_dist'].min()]['SMILES_Solute'].tolist()
                        for similar_smi in similar_smiles:
                            search_df = df[(df['SMILES_Solute'] == similar_smi)]
                            dois = list(search_df['Source'].unique())
                            for doi in dois:
                                col1result, col2result, col3result, col4result = st.columns([1, 1, 1, 3])
                                df_comp = search_df[(search_df['Source'] == doi) & (search_df['SMILES_Solute'] == similar_smi)]
                                fig_line = px.line(df_comp, x="Temperature_K", y="Solubility(mole_fraction)", color="Solvent", title=f"Dependence of solubility on temperature", markers=True)
                                fig_line.update_layout(yaxis_title='Solubility (mole fraction)')
                                col1result.image(draw_molecule(canonize_mol), caption=canonize_mol)
                                col2result.image(draw_molecule(similar_smi), caption=similar_smi)
                                col3result.markdown(f'**https://doi.org/{doi}**')
                                col4result.plotly_chart(fig_line)
                    else:
                        st.markdown(f'### This compound was found in BigSolDB 2.0:')
                        col1result, col2result = st.columns([1, 1])
                        pubchem = search_df['PubChem_CID'].iloc[0]
                        cas = search_df['CAS'].iloc[0]
                        if pubchem is not None:
                            col1result.markdown(f'PubChem link: **https://pubchem.ncbi.nlm.nih.gov/compound/{pubchem}**')
                        if cas is not None:
                            col2result.markdown(f'CAS link: **https://commonchemistry.cas.org/detail?cas_rn={cas}**')
                        col1result, col2result, col3result = st.columns([1, 1, 2])
                        col1result.markdown(f'**Molecule from BigSolDB 2.0**')
                        col2result.markdown(f'**Source**')
                        col3result.markdown(f'**Solubility**')
                        dois = list(search_df['Source'].unique())
                        for doi in dois:
                            col1result, col2result, col3result = st.columns([1, 1, 2])
                            df_comp = search_df[(search_df['Source'] == doi) & (search_df['SMILES_Solute'] == canonize_mol)]
                            fig_line = px.line(df_comp, x="Temperature_K", y="Solubility(mole_fraction)", color="Solvent", title=f"Dependence of solubility on temperature", markers=True)
                            fig_line.update_layout(yaxis_title='Solubility (mole fraction)')
                            col1result.image(draw_molecule(canonize_mol), caption=smiles)
                            col2result.markdown(f'**https://doi.org/{doi}**')
                            col3result.plotly_chart(fig_line)
                        st.dataframe(search_df)
            else:
                st.error("Incorrect SMILES entered")
        else:
            st.error("Please enter SMILES of the compound")

with tabs[3]:
    if st.button("Get solubility of random molecule🎲"):
        selected = random.choice(compound_names)
        search_df = df[(df['Compound_Name'] == selected)]
        search_df.reset_index(drop=True, inplace=True)
        col1result, col2result = st.columns([1, 1])
        pubchem = search_df['PubChem_CID'].iloc[0]
        cas = search_df['CAS'].iloc[0]
        if pubchem is not None:
            col1result.markdown(f'PubChem link: **https://pubchem.ncbi.nlm.nih.gov/compound/{pubchem}**')
        if cas is not None:
            col2result.markdown(f'CAS link: **https://commonchemistry.cas.org/detail?cas_rn={cas}**')

        canonize_mol = search_df['SMILES_Solute'].iloc[0]
        col1result, col2result, col3result = st.columns([1, 1, 2])
        col1result.markdown(f'**Molecule from BigSolDB 2.0**')
        col2result.markdown(f'**Source**')
        col3result.markdown(f'**Solubility**')
        dois = list(search_df['Source'].unique())
        for doi in dois:
            col1result, col2result, col3result = st.columns([1, 1, 2])
            df_comp = search_df[(search_df['Source'] == doi) & (search_df['SMILES_Solute'] == canonize_mol)]
            fig_line = px.line(df_comp, x="Temperature_K", y="Solubility(mole_fraction)", color="Solvent", title=f"Dependence of solubility on temperature", markers=True)
            fig_line.update_layout(yaxis_title='Solubility (mole fraction)')
            col1result.image(draw_molecule(canonize_mol), caption=f'{canonize_mol}, {selected}')
            col2result.markdown(f'**https://doi.org/{doi}**')
            col3result.plotly_chart(fig_line)
        st.dataframe(search_df)
