# Audience Segmentation with K-Means

Unsupervised learning project that groups social media profiles by interests, so marketing campaigns can be targeted at each segment. The trained model is served through a small Streamlit app: upload a CSV of profiles and get each one assigned to a group.

## Dataset

`dados/dados_mkt.csv` contains 12,992 profiles and 27 columns, with no missing values. Column names are in Portuguese.

| Group | Columns |
| --- | --- |
| Demographics | `sexo` (F, M, NE), `idade`, `numero_de_amigos` |
| Sports | `basquete`, `futebol_americano`, `futebol`, `softbol`, `voleibol`, `natacao`, `beisebol`, `tenis`, `esportes` |
| Music and dance | `danca`, `banda`, `marcha`, `musica`, `rock` |
| Appearance and shopping | `fofo`, `cabelo`, `vestido`, `shopping`, `compras`, `roupas` |
| Brands and other | `nossa_marca`, `marca_concorrente`, `bebidas`, `animacao` |

## Approach

The full workflow is in [`notebooks/marketing.ipynb`](notebooks/marketing.ipynb).

1. **Exploration**: correlation heatmap across all numeric features and a closer look at related interests (for example `banda` and `marcha`, correlation of 0.53).
2. **Preprocessing**: one-hot encoding of `sexo` with `OneHotEncoder`, then `MinMaxScaler` on all features.
3. **Choosing k**: elbow method and silhouette score for k from 2 to 20, with and without scaling.
4. **Model**: `KMeans(n_clusters=3, random_state=45)` fitted on the scaled data.
5. **Profiling**: scaled values are converted back to the original units (`inverse_transform`) and averaged per cluster to describe each group.
6. **Deployment**: the encoder, scaler and model are saved with `joblib` and loaded by the Streamlit app.

## Results

Scaling the features made a large difference to cluster quality:

| Features | Best k | Silhouette score |
| --- | --- | --- |
| Raw | 2 | 0.380 |
| Min-Max scaled | 3 | 0.745 |

Profile of the three groups (average interest scores per group):

| Group | Profiles | Highlights |
| --- | --- | --- |
| 0 | 9,893 | Highest on music (8.5), hair (5.6), dance (5.5), shopping (3.3) and purchases (4.9) |
| 1 | 2,160 | Sports and rock oriented: American football (5.3), basketball (3.8), baseball (3.2), band (4.1); lowest on shopping and fashion |
| 2 | 939 | Intermediate profile, close to group 0 in music, dance, hair and shopping, with lower intensity |

Age (26.8 to 27.1) and number of friends (26 to 32) are similar across groups, so the segmentation is driven by interests.

## Notes and next steps

- The three groups line up exactly with the three values of `sexo`: in every cluster, one of the `sexo_*` columns averages 1.0 and the others 0.0. After Min-Max scaling, these binary columns span the full 0 to 1 range while most interest scores stay close to 0, so they dominate the distance calculation. This also explains the high silhouette score.
- A natural next step is to cluster on interests only (or weight the features) and compare the segments with the current ones, then project the result with PCA to inspect the separation.
- Other algorithms such as Gaussian Mixture Models or hierarchical clustering are worth comparing against K-Means.

## Streamlit app

The app (`app.py`) loads the saved encoder, scaler and model, and lets you:

- upload a CSV with the same columns as `dados/dados_mkt.csv`
- preview the first 10 rows with the assigned group
- download the full result as `Grupos_interesse.csv`

## Project structure

```
.
├── app.py                 # Streamlit app
├── dados/
│   └── dados_mkt.csv      # dataset
├── notebooks/
│   ├── marketing.ipynb    # exploration, preprocessing, model selection, profiling
│   ├── encoder_sexo.pkl   # fitted OneHotEncoder
│   ├── scaler.pkl         # fitted MinMaxScaler
│   └── kmeans.pkl         # fitted KMeans (k=3)
├── requirements.txt
└── README.md
```

## Getting started

Requires Python 3.14.

```bash
git clone https://github.com/smacedo7/audience-segmentation-kmeans.git
cd audience-segmentation-kmeans

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

To go through the analysis, open `notebooks/marketing.ipynb` in VS Code or JupyterLab.

The `.pkl` files were generated with scikit-learn 1.9.1, so keep the pinned versions from `requirements.txt` to load them without warnings.

## Tech stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Streamlit, joblib.
