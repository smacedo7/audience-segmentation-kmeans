from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODELS_DIR = Path(__file__).resolve().parent / "notebooks"

encoder = joblib.load(MODELS_DIR / "encoder_sexo.pkl")
scaler = joblib.load(MODELS_DIR / "scaler.pkl")
kmeans = joblib.load(MODELS_DIR / "kmeans.pkl")


st.title("Grupos de interesse para marketing")
st.write("""
         Neste projeto, apliquei o algoritmo de clusterização K-means para identificar e prever agrupamentos de interesses de usuários, com o objetivo de direcionar campanhas de marketing de forma mais eficaz.
         Através dessa análise, consegui segmentar o público em bolhas de interesse, permitindo a criação de campanhas personalizadas e mais assertivas, com base nos padrões de comportamento e preferências de cada grupo.
         """)

up_file = st.file_uploader(
    "Escolha um arquivo CSV para realizar a previsão", type="csv"
)


def processar_prever(df):
    encoded_sexo = encoder.transform(df[["sexo"]])
    encoded_df = pd.DataFrame(
        encoded_sexo, columns=encoder.get_feature_names_out(["sexo"])
    )
    dados = pd.concat([df.drop("sexo", axis=1), encoded_df], axis=1)

    dados_escalados = scaler.transform(dados)

    cluster = kmeans.predict(dados_escalados)

    return cluster


if up_file is not None:

    st.write("""
                    ### Descrição dos Grupos:
                    - **Grupo 0** reúne perfis femininos, com maior interesse em música, cabelo, dança e compras.
                    - **Grupo 1** reúne perfis masculinos, voltados a esportes (futebol americano, basquete e beisebol), banda e rock, com pouco interesse em moda e compras.
                    - **Grupo 2** reúne perfis de sexo não informado (NE), com perfil intermediário: próximo ao Grupo 0 em música, dança, cabelo e compras, porém com menor intensidade.
                """)
    df = pd.read_csv(up_file)
    cluster = processar_prever(df)
    df.insert(0, "grupos", cluster)

    st.write("Visualização dos resultados (10 primeiros registros):")
    st.write(df.head(10))

    csv = df.to_csv(index=False)
    st.download_button(
        label="Baixar resultados completos",
        data=csv,
        file_name="Grupos_interesse.csv",
        mime="text/csv",
    )
