from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "notebooks" / "segmentation_v2.pkl"

NOMES_GRUPOS = {
    "General": "Geral",
    "Sports": "Esportes",
    "Fashion and shopping": "Moda e compras",
    "Animation": "Animação",
}


@st.cache_resource
def carregar_modelo():
    return joblib.load(MODEL_PATH)


artefato = carregar_modelo()
pipeline = artefato["pipeline"]
colunas_interesse = artefato["features"]
nomes = {
    grupo: NOMES_GRUPOS.get(nome, nome) for grupo, nome in artefato["segment_names"].items()
}

st.title("Grupos de interesse para marketing")
st.write("""
         Neste projeto, apliquei o algoritmo de clusterização K-means para identificar e prever agrupamentos de interesses de usuários, com o objetivo de direcionar campanhas de marketing de forma mais eficaz.
         Através dessa análise, consegui segmentar o público em bolhas de interesse, permitindo a criação de campanhas personalizadas e mais assertivas, com base nos padrões de comportamento e preferências de cada grupo.
         """)

up_file = st.file_uploader(
    "Escolha um arquivo CSV para realizar a previsão", type="csv"
)
st.caption(
    "O arquivo precisa ter as colunas de interesse do dataset original "
    "(basquete, futebol_americano, ..., animacao). Colunas como sexo, idade e "
    "numero_de_amigos são opcionais e não entram no modelo."
)


if up_file is not None:
    df = pd.read_csv(up_file)

    faltando = [c for c in colunas_interesse if c not in df.columns]
    if faltando:
        st.error("O arquivo não tem as colunas necessárias: " + ", ".join(faltando))
        st.stop()

    interesses = df[colunas_interesse].apply(pd.to_numeric, errors="coerce")
    if interesses.isna().any().any():
        st.error(
            "As colunas de interesse precisam ser numéricas e não podem ter valores vazios."
        )
        st.stop()

    st.write("""
                    ### Descrição dos Grupos:
                    - **Grupo 0 (Geral)** reúne a maior parte dos perfis, com interesses baixos e dispersos: fica abaixo da média na quase totalidade dos temas.
                    - **Grupo 1 (Esportes)** tem forte interesse em futebol americano, além de beisebol, basquete e esportes em geral.
                    - **Grupo 2 (Moda e compras)** se destaca em vestido, roupas, cabelo, compras e shopping, e tem o maior interesse nas duas marcas.
                    - **Grupo 3 (Animação)** é definido pelo interesse em animação, com compras, cabelo e itens fofos acima da média.
                """)

    grupos = pipeline.predict(interesses)
    df.insert(0, "grupo", grupos)
    df.insert(1, "nome_grupo", [nomes.get(g, str(g)) for g in grupos])

    st.write("Visualização dos resultados (10 primeiros registros):")
    st.write(df.head(10))

    st.write("Perfis por grupo:")
    st.write(df["nome_grupo"].value_counts().rename("perfis"))

    csv = df.to_csv(index=False)
    st.download_button(
        label="Baixar resultados completos",
        data=csv,
        file_name="Grupos_interesse.csv",
        mime="text/csv",
    )
