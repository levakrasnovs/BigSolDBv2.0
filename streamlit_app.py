import pandas as pd
import numpy as np
from math import log10
import plotly.express as px
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from streamlit_ketcher import st_ketcher
from molfeat.calc import FPCalculator

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
df['Log_Solubility'] = df['Solubility'].apply(lambda x: log10(x))

df_smiles = pd.DataFrame({'SMILES': list(df['SMILES'].unique())})
df_smiles['mol'] = df_smiles['SMILES'].apply(Chem.MolFromSmiles)
df_smiles['mol_ecfp'] = df_smiles['mol'].apply(lambda x: calc(x))

solvents = df['Solvent'].value_counts().reset_index().loc[:50]

top_solvents = df["Solvent"].value_counts().nlargest(10).index
df_top_solvents = df[df["Solvent"].isin(top_solvents)]
top_solvents_by_smiles = df.groupby("Solvent")["SMILES"].nunique().nlargest(50).reset_index()

col1intro, col2intro = st.columns([2, 1])
col1intro.markdown("""
# BigSolDB 2.0

""")


tabs = st.tabs(["Explore", "Search"])

with tabs[0]:
    col1fig, col2fig = st.columns([1, 1])
    fig_sol = px.histogram(df, x='Solubility', nbins=64, title='Mole fraction solubility distribution in the BigSolDB 2.0')
    fig_sol.update_layout(yaxis_title='Number of entries')
    fig_sol.update_layout(xaxis_title='Solubility')
    col1fig.plotly_chart(fig_sol)

    fig_log_sol = px.histogram(df, x='LogS', nbins=64, title='Solubility (Mol/L) distribution in the BigSolDB 2.0')
    fig_log_sol.update_layout(yaxis_title='Number of entries')
    fig_log_sol.update_layout(xaxis_title='Log10 Solubility (Mol/L)')
    col2fig.plotly_chart(fig_log_sol)

    fig_solv = px.bar(solvents, x='Solvent', y='count', text='count', title="Most popular solvents by number of entries")
    fig_solv.update_layout(yaxis_title='Number of entries')
    fig_solv.update_layout(xaxis_title='Solvents')
    st.plotly_chart(fig_solv, use_container_width=True)

    fig_solv_smiles = px.bar(top_solvents_by_smiles, x='Solvent', y='SMILES', text='SMILES', title="Most popular solvents by number of unique molecules")
    fig_solv_smiles.update_layout(yaxis_title='Number of molecules')
    fig_solv_smiles.update_layout(xaxis_title='Solvents')
    st.plotly_chart(fig_solv_smiles, use_container_width=True)

    fig_hist = px.histogram(df_top_solvents, x="LogS", color="Solvent", nbins=30,
                       opacity=0.6, barmode="overlay",
                       title="Log10 Solubility (Mol/L) histograms for top 10 solvents")
    fig_hist.update_layout(yaxis_title='Number of entries')
    fig_hist.update_layout(xaxis_title='Log10 Solubility (Mol/L)')
    st.plotly_chart(fig_hist, use_container_width=True)

with tabs[1]:

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
                    search_df = df[(df['SMILES'] == canonize_mol)]
                    if search_df.shape[0] == 0:
                        st.markdown(f'### This compound was not found in BigSolDB 2.0, but similar was found instead:')
                        col1result, col2result, col3result, col4result = st.columns([1, 1, 1, 3])
                        col1result.markdown(f'**Your molecule**')
                        col2result.markdown(f'**Molecule from BigSolDB 2.0**')
                        col3result.markdown(f'**Source**')
                        col4result.markdown(f'**Solubility**')
                        df_smiles['res_dist'] = df_smiles['mol_ecfp'].apply(lambda ecfp: hamming_distance(calc(mol), ecfp))
                        similar_smiles = df_smiles[df_smiles['res_dist'] == df_smiles['res_dist'].min()]['SMILES'].tolist()
                        for similar_smi in similar_smiles:
                            search_df = df[(df['SMILES'] == similar_smi)]
                            dois = list(search_df['Source'].unique())
                            for doi in dois:
                                col1result, col2result, col3result, col4result = st.columns([1, 1, 1, 3])
                                df_comp = search_df[(search_df['Source'] == doi) & (search_df['SMILES'] == similar_smi)]
                                fig_line = px.line(df_comp, x="T,K", y="Solubility", color="Solvent", title=f"Dependence of solubility on temperature", markers=True)
                                fig_line.update_layout(yaxis_title='Solubility (mole fraction)')
                                col1result.image(draw_molecule(canonize_mol), caption=canonize_mol)
                                col2result.image(draw_molecule(similar_smi), caption=similar_smi)
                                col3result.markdown(f'**https://doi.org/{doi}**')
                                col4result.plotly_chart(fig_line)
                    else:
                        st.markdown(f'### This compound was found in BigSolDB 2.0:')
                        col1result, col2result, col3result = st.columns([1, 1, 2])
                        col1result.markdown(f'**Molecule from BigSolDB 2.0**')
                        col2result.markdown(f'**Source**')
                        col3result.markdown(f'**Solubility**')
                        dois = list(search_df['Source'].unique())
                        for doi in dois:
                            col1result, col2result, col3result = st.columns([1, 1, 2])
                            df_comp = search_df[(search_df['Source'] == doi) & (search_df['SMILES'] == canonize_mol)]
                            fig_line = px.line(df_comp, x="T,K", y="Solubility", color="Solvent", title=f"Dependence of solubility on temperature", markers=True)
                            fig_line.update_layout(yaxis_title='Solubility (mole fraction)')
                            col1result.image(draw_molecule(canonize_mol), caption=smiles)
                            col2result.markdown(f'**https://doi.org/{doi}**')
                            col3result.plotly_chart(fig_line)

            else:
                st.error("Incorrect SMILES entered")
        else:
            st.error("Please enter SMILES of the compound")
