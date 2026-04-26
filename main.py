# %% [markdown]
# # Projet de Prédiction des Matchs de Ligue 1
# 
# L'objectif de ce projet est de prédire le résultat final des matchs de football de la Ligue 1 (Victoire à Domicile (1), Match Nul (0), Victoire à l'Extérieur (-1)) en utilisant des algorithmes de Machine Learning.
# 
# Afin d'éviter tout Data Leakage (fuite de données), notre modèle ne s'appuiera sur aucune statistique se déroulant pendant le match (comme les buts marqués ou les tirs). Nous allons construire des variables basées uniquement sur le contexte d'avant match : puissance financière, forme des équipes, historique des confrontations et indiscipline.

# %%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

from predictor import Predictor

# %% [markdown]
# ## Partie 1 - Préparation des données 
# 
# Nous allons importer nos données brutes. Le fichier "matchs_2013_2024.csv" contient l'historique des rencontres et nous allons l'enrichir avec les informations du fichier "clubs_fr.csv".

# %%
clubs = pd.read_csv("data/clubs_fr.csv")
matchs_2013_2024 = pd.read_csv("data/matchs_2013_2024.csv", index_col=0)

print("Clubs :")
(clubs.head())
print("\nMatchs from 2013 to 2024 :")
(matchs_2013_2024.head())

# %% [markdown]
# ### Nettoyage initial
# Nous supprimons les colonnes textuelles inutiles (comme le nom de l'entraîneur, le nom du stade...) et nous retirons les lignes ne contenant pas de résultat final.

# %%
clubs = clubs.drop(columns=["foreigners_number", "foreigners_percentage", "club_code", "net_transfer_record", "stadium_name", "domestic_competition_id", "name", "coach_name"])
matchs_2013_2024 = matchs_2013_2024.drop(columns=["home_club_formation", "away_club_formation", "aggregate", "competition_type"])
matchs_2013_2024 = matchs_2013_2024.dropna(subset=["results"])

# %% [markdown]
# ### Fusion des données
# Nous joignons les informations des clubs (taille de l'effectif, âge moyen, joueurs en équipe nationale) pour l'équipe à domicile, puis pour l'équipe à l'extérieur.

# %%
matchs_2013_2024 = pd.merge(matchs_2013_2024, clubs, left_on="home_club_id", right_on="club_id")
matchs_2013_2024 = matchs_2013_2024.drop(columns=["club_id"])
matchs_2013_2024 = matchs_2013_2024.rename(columns={"squad_size": "home_club_squad_size", "average_age": "home_club_average_age", "national_team_players": "home_club_national_team_players"})

matchs_2013_2024 = pd.merge(matchs_2013_2024, clubs, left_on="away_club_id", right_on="club_id")
matchs_2013_2024 = matchs_2013_2024.drop(columns=["club_id", "stadium_seats_y"])
matchs_2013_2024 = matchs_2013_2024.rename(columns={"squad_size": "away_club_squad_size", "average_age": "away_club_average_age", "national_team_players": "away_club_national_team_players", "stadium_seats_x": "stadium_seats"})

(matchs_2013_2024.head())

# %% [markdown]
# ### Variable 1 : La Puissance Financière
# Nous allons extraire la valeur marchande totale de chaque club pour l'année du match (grâce aux valeurs individuelles des joueurs dans "player_valuation_before_season.csv"), et calculer la différence financière entre les deux équipes. Une forte valeur positive indique un grand favori à domicile.

# %%
player_valuation_before_season = pd.read_csv("data/player_valuation_before_season.csv")
player_valuation_before_season["date"] = pd.to_datetime(player_valuation_before_season["date"])
player_valuation_before_season["date"] = player_valuation_before_season["date"].dt.year
player_valuation_before_season = player_valuation_before_season.rename(columns={"date": "year"})

club_valueation_per_year = player_valuation_before_season.groupby(["current_club_id", "year"])["market_value_in_eur"].sum().reset_index()

matchs_2013_2024 = pd.merge(matchs_2013_2024, club_valueation_per_year, left_on=["home_club_id", "season"], right_on=["current_club_id", "year"])
matchs_2013_2024 = matchs_2013_2024.drop(columns=["current_club_id", "year"])
matchs_2013_2024 = matchs_2013_2024.rename(columns={"market_value_in_eur": "home_club_value_in_eur"})

matchs_2013_2024 = pd.merge(matchs_2013_2024, club_valueation_per_year, left_on=["away_club_id", "season"], right_on=["current_club_id", "year"])
matchs_2013_2024 = matchs_2013_2024.drop(columns=["current_club_id", "year"])
matchs_2013_2024 = matchs_2013_2024.rename(columns={"market_value_in_eur": "away_club_value_in_eur"})

matchs_2013_2024["value_difference"] = matchs_2013_2024["home_club_value_in_eur"] - matchs_2013_2024["away_club_value_in_eur"]

print("Aperçu de la variable :")
(pd.DataFrame(matchs_2013_2024["value_difference"].head()))

# %% [markdown]
# ### Variable 2 : La forme de l'équipe
# La valeur marchande ne fait pas tout : une équipe chère peut être dans une mauvaise période. Le but est de calculer la somme des buts marqués lors des 3 derniers matchs afin d'évaluer la forme de l'équipe.

# %%
# 1. On prépare les stats de l'équipe à Domicile
home_perf = matchs_2013_2024[["game_id", "date", "home_club_id", "home_club_goals", "away_club_goals", "results"]].copy()
home_perf = home_perf.rename(columns={
    "home_club_id": "club_id", 
    "home_club_goals": "goals_scored",
    "away_club_goals": "goals_conceded"
})
# Calcul des points : 3 pts pour victoire (1), 1 pt pour nul (0), 0 pt pour défaite (-1)
home_perf["points_earned"] = home_perf["results"].map({1: 3, 0: 1, -1: 0})


# 2. On prépare les stats de l'équipe à l'Extérieur
away_perf = matchs_2013_2024[["game_id", "date", "away_club_id", "away_club_goals", "home_club_goals", "results"]].copy()
away_perf = away_perf.rename(columns={
    "away_club_id": "club_id", 
    "away_club_goals": "goals_scored",
    "home_club_goals": "goals_conceded"
})
# Calcul des points : 3 pts pour victoire ext (-1), 1 pt pour nul (0), 0 pt pour défaite ext (1)
away_perf["points_earned"] = away_perf["results"].map({-1: 3, 0: 1, 1: 0})


# 3. On rassemble tout, on trie par équipe et par date chronologique
team_history = pd.concat([home_perf, away_perf]).sort_values(["club_id", "date"])

# 4. LE CALCUL MAGIQUE : Fenêtre glissante (rolling) sur les 3 derniers matchs
# On décale de 1 (.shift(1)) pour ne pas inclure le match actuel (Data Leakage !)
for col in ["goals_scored", "goals_conceded", "points_earned"]:
    team_history[f"last_3_{col}"] = team_history.groupby("club_id")[col].transform(
        lambda x: x.rolling(window=3, min_periods=1).sum().shift(1)
    ).fillna(0)

# 5. On fusionne ces nouvelles pépites dans le dataset principal
form_df = team_history[["game_id", "club_id", "last_3_goals_scored", "last_3_goals_conceded", "last_3_points_earned"]]

# Fusion pour l'équipe à domicile
matchs_2013_2024 = matchs_2013_2024.merge(
    form_df, left_on=["game_id", "home_club_id"], right_on=["game_id", "club_id"], how="left"
).rename(columns={
    "last_3_goals_scored": "home_form_goals_scored",
    "last_3_goals_conceded": "home_form_goals_conceded",
    "last_3_points_earned": "home_form_points"
}).drop(columns=["club_id"])

# Fusion pour l'équipe à l'extérieur
matchs_2013_2024 = matchs_2013_2024.merge(
    form_df, left_on=["game_id", "away_club_id"], right_on=["game_id", "club_id"], how="left"
).rename(columns={
    "last_3_goals_scored": "away_form_goals_scored",
    "last_3_goals_conceded": "away_form_goals_conceded",
    "last_3_points_earned": "away_form_points"
}).drop(columns=["club_id"])

# Différences directes pour mâcher le travail de l'IA
matchs_2013_2024["form_points_difference"] = matchs_2013_2024["home_form_points"] - matchs_2013_2024["away_form_points"]
matchs_2013_2024["form_defense_difference"] = matchs_2013_2024["home_form_goals_conceded"] - matchs_2013_2024["away_form_goals_conceded"]

print("Aperçu des nouvelles variables :")
(matchs_2013_2024[["game_id", "home_form_points", "away_form_points", "form_points_difference"]].tail())


# %% [markdown]
# ### Variable 3 : Historique des confrontations ("Head to Head")
# Nous allons modéliser l'avantage "psychologique" en calculant la moyenne historique des résultats entre ces deux clubs précis.

# %%
matchs_2013_2024 = matchs_2013_2024.sort_values("date")

matchs_2013_2024["h2h_key"] = matchs_2013_2024.apply(lambda r: tuple(sorted((r["home_club_id"], r["away_club_id"]))), axis=1)
matchs_2013_2024["h2h_res"] = matchs_2013_2024.apply(lambda r: r["results"] if r["home_club_id"] == r["h2h_key"][0] else -r["results"], axis=1)
matchs_2013_2024["h2h_history"] = matchs_2013_2024.groupby("h2h_key")["h2h_res"].transform(lambda x: x.expanding().mean().shift(1)).fillna(0)
matchs_2013_2024["home_h2h_advantage"] = matchs_2013_2024.apply(lambda r: r["h2h_history"] if r["home_club_id"] == r["h2h_key"][0] else -r["h2h_history"], axis=1)

matchs_2013_2024 = matchs_2013_2024.drop(columns=["h2h_key", "h2h_res", "h2h_history"])

print("Aperçu de la variable :")
(pd.DataFrame(matchs_2013_2024["home_h2h_advantage"].tail()))

# %% [markdown]
# ### Variable 4 : L'Indiscipline (Cartons Rouges)
# Pour aller plus loin, nous récupérons les données d'apparitions des joueurs (avec "player_appearance.csv") pour calculer la moyenne de cartons rouges sur les 3 derniers matchs. Une équipe très sanctionnée sera désorganisée.

# %%
player_appearance = pd.read_csv("data/player_appearance.csv")

team_match_stats = player_appearance.groupby(["game_id", "player_club_id"]).agg({"yellow_cards": "sum", "red_cards": "sum", "goals": "sum", "assists": "sum"}).reset_index()

team_match_stats = team_match_stats.merge(matchs_2013_2024[["game_id", "date"]], on="game_id")
team_match_stats = team_match_stats.sort_values(["player_club_id", "date"])

team_match_stats["avg_red_cards_3m"] = team_match_stats.groupby("player_club_id")["red_cards"].transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1)).fillna(0)

indiscipline_df = team_match_stats[["game_id", "player_club_id", "avg_red_cards_3m"]]
matchs_2013_2024 = matchs_2013_2024.merge(indiscipline_df, left_on=["game_id", "home_club_id"], right_on=["game_id", "player_club_id"], how="left").rename(columns={"avg_red_cards_3m": "home_red_cards_avg"}).drop(columns=["player_club_id"])
matchs_2013_2024 = matchs_2013_2024.merge(indiscipline_df, left_on=["game_id", "away_club_id"], right_on=["game_id", "player_club_id"], how="left").rename(columns={"avg_red_cards_3m": "away_red_cards_avg"}).drop(columns=["player_club_id"])

matchs_2013_2024["home_red_cards_avg"] = matchs_2013_2024["home_red_cards_avg"].fillna(0)
matchs_2013_2024["away_red_cards_avg"] = matchs_2013_2024["away_red_cards_avg"].fillna(0)

# %% [markdown]
# ### Variable 5 : Forme Spécifique (Domicile / Extérieur) 

# Toujours s'assurer que le tableau est trié dans le temps
matchs_2013_2024 = matchs_2013_2024.sort_values("date")

# 1. On attribue les points du match en cours pour l'équipe à Domicile et à l'Extérieur
matchs_2013_2024["points_a_domicile"] = matchs_2013_2024["results"].map({1: 3, 0: 1, -1: 0})
matchs_2013_2024["points_a_lexterieur"] = matchs_2013_2024["results"].map({-1: 3, 0: 1, 1: 0})

# 2. Pour l'équipe à Domicile : on additionne les 3 derniers matchs JOUÉS À DOMICILE
matchs_2013_2024["home_pts_N1"] = matchs_2013_2024.groupby("home_club_id")["points_a_domicile"].shift(1).fillna(0)
matchs_2013_2024["home_pts_N2"] = matchs_2013_2024.groupby("home_club_id")["points_a_domicile"].shift(2).fillna(0)
matchs_2013_2024["home_pts_N3"] = matchs_2013_2024.groupby("home_club_id")["points_a_domicile"].shift(3).fillna(0)

matchs_2013_2024["home_points_at_home"] = matchs_2013_2024["home_pts_N1"] + matchs_2013_2024["home_pts_N2"] + matchs_2013_2024["home_pts_N3"]

# 3. Pour l'équipe à l'Extérieur : on additionne les 3 derniers matchs JOUÉS À L'EXTÉRIEUR
matchs_2013_2024["away_pts_N1"] = matchs_2013_2024.groupby("away_club_id")["points_a_lexterieur"].shift(1).fillna(0)
matchs_2013_2024["away_pts_N2"] = matchs_2013_2024.groupby("away_club_id")["points_a_lexterieur"].shift(2).fillna(0)
matchs_2013_2024["away_pts_N3"] = matchs_2013_2024.groupby("away_club_id")["points_a_lexterieur"].shift(3).fillna(0)

matchs_2013_2024["away_points_away"] = matchs_2013_2024["away_pts_N1"] + matchs_2013_2024["away_pts_N2"] + matchs_2013_2024["away_pts_N3"]

# 4. On supprime les colonnes de calcul pour garder un tableau propre
matchs_2013_2024 = matchs_2013_2024.drop(columns=[
    "points_a_domicile", "points_a_lexterieur", 
    "home_pts_N1", "home_pts_N2", "home_pts_N3",
    "away_pts_N1", "away_pts_N2", "away_pts_N3"
])


# ### Variable 6 : Différence de Classement (Pour simplifier la vie de l'IA)
matchs_2013_2024["position_difference"] = matchs_2013_2024["home_club_position"] - matchs_2013_2024["away_club_position"]

# %% [markdown]
# ### Création des Dataframes Modèles (A/B Testing des variables)
# Pour prouver notre démarche expérimentale, nous allons comparer les performances des algorithmes sur deux jeux de variables différents :
# * Config A : Les variables fondamentales (Classement, Finance, Forme).
# * Config B : Intégration de données plus fines (H2H, et Indiscipline par cartons rouges).

# %%
features_A = [
    "home_club_position", "away_club_position",
    "home_club_value_in_eur", "away_club_value_in_eur", "value_difference",
    "home_form_goals_scored", "away_form_goals_scored", "results" # <--- Nouveaux noms ici
]

features_B = [
    "home_club_position", "away_club_position",
    "home_club_value_in_eur", "away_club_value_in_eur", "value_difference",
    "home_form_goals_scored", "away_form_goals_scored", # <--- Nouveaux noms ici
    "home_h2h_advantage", "home_red_cards_avg", "away_red_cards_avg", "results"
]

features_C = [
    # 1. Hiérarchie & Puissance Financière
    "home_club_position", "away_club_position",
    "value_difference",
    
    # 2. Talent Pur & Ferveur du public (Issus de clubs_fr.csv)
    "home_club_national_team_players", "away_club_national_team_players", 
    "stadium_seats",
    
    # 3. La Dynamique de l'équipe (Nos nouvelles variables !)
    "form_points_difference",   # Qui gagne le plus en ce moment ?
    "form_defense_difference",  # Qui a la pire défense en ce moment ?
    "home_form_goals_scored", "away_form_goals_scored", # L'attaque
    
    # 4. Le Passé et l'Indiscipline
    "home_h2h_advantage",
    "home_red_cards_avg", "away_red_cards_avg",
    
    # Cible
    "results"
]

features_D = [
    # 1. Hiérarchie & Puissance Financière
    "position_difference",  # Remplacement de l'ancien système (plus clair pour le SVM !)
    "value_difference",
    
    # 2. Talent Pur & Stade
    "home_club_national_team_players", "away_club_national_team_players", 
    "stadium_seats",
    
    # 3. La Dynamique de l'équipe (Globale ET Spécifique !)
    "form_points_difference",   
    "form_defense_difference",  
    "home_points_at_home", "away_points_away", # Les deux nouvelles pépites !
    
    # 4. Le Passé et l'Indiscipline
    "home_h2h_advantage",
    "home_red_cards_avg", "away_red_cards_avg",
    
    # Cible
    "results"
]


df_model_A = matchs_2013_2024[features_A].dropna()
df_model_B = matchs_2013_2024[features_B].dropna()
df_model_C = matchs_2013_2024[features_C].dropna()
df_model_D = matchs_2013_2024[features_D].dropna()


# %% [markdown]
# ## Partie 2 - Exploration Qualitative (EDA)
# 
# L'exploration visuelle permet de s'assurer de la pertinence des données avant l'apprentissage de l'IA.

# %% [markdown]
# ### L'avantage à domicile

# %%
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df_model_A, x="results", palette=["#974339", "#bfb296", "#7ab191"])
plt.title("Distribution des résultats des matchs (Avantage à domicile)", fontsize=14)
plt.xlabel("Résultat (-1: Victoire Extérieur, 0: Nul, 1: Victoire Domicile)", fontsize=12)

total = len(df_model_A)
for p in ax.patches:
    percentage = f"{100 * p.get_height() / total:.1f}%"
    x = p.get_x() + p.get_width() / 2 - 0.1
    y = p.get_height() + 20
    ax.annotate(percentage, (x, y), fontweight="bold")
plt.show()

# %% [markdown]
# Comme le montre le graphique ci-dessous, le hasard total (33%) est un mythe dans le football professionnel. L'équipe qui reçoit gagne dans **45,7%** des cas. Notre objectif réel pour le Machine Learning est donc de battre ce seuil critique de ~46%.

# %% [markdown]
# ### La Preuve Statistique (Corrélations)

# %%
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

mask_A = np.triu(np.ones_like(df_model_A.corr(), dtype=bool))
sns.heatmap(df_model_A.corr(), mask=mask_A, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[0])
axes[0].set_title("Corrélations (Config A)")

mask_B = np.triu(np.ones_like(df_model_B.corr(), dtype=bool))
sns.heatmap(df_model_B.corr(), mask=mask_B, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1])
axes[1].set_title("Corrélations (Config B)")

plt.tight_layout()
plt.show()

# %% [markdown]
# Les matrices de corrélations valident nos choix : 
# La "value_difference" possède la plus forte corrélation positive avec la cible, prouvant que l'écart financier est une variable prédictive importante.

# %% [markdown]
# ### Analyse d'impact Visuel

# %%
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

sns.boxplot(data=df_model_A, x="results", y=df_model_A["value_difference"] / 1_000_000, palette=["#e74c3c", "#95a5a6", "#2ecc71"], showfliers=False, ax=axes[0])
axes[0].set_title("Impact de la différence de valeur")
axes[0].set_ylabel("Différence de Valeur (Millions d'euros)")
axes[0].axhline(0, color="black", linestyle="--", alpha=0.5)

sns.kdeplot(data=df_model_B, x="home_red_cards_avg", hue="results", fill=True, palette=["#e74c3c", "#b5b77d", "#2ecc71"], ax=axes[1])
axes[1].set_title("Impact des Cartons Rouges (Domicile)")
# plt.tight_layout()
plt.show()

# %% [markdown]
# Ce Boxplot et ce Graphique de Densité montrent comment l'argent et l'indiscipline influencent le jeu :
# * Boxplot : En cas de victoire à domicile, la différence de valeur est quasi toujours positive. Vice versa pour un défaite
# * Kdeplot : Une moyenne élevée de cartons rouges augmente mécaniquement les chances de victoire de l'adversaire ou de match nul.

# %% [markdown]
# ## Partie 3 - Prédictions (avec plusieurs algorithmes)
# 
# La classe "Predictor" permet de faire le train/test Split (80/20), la standardisation des variables pour chaque modèle testé et l'evaluation finale. Le football étant très incertain, un score plafonnant autour de 60% est un indicateur réaliste qui démontre que notre modèle a trouvé de vraies règles sans tricher.

# %%
print("\n--- TEST 1 : RÉGRESSION LOGISTIQUE ---")
algo_log_reg = LogisticRegression(max_iter=1000, random_state=42)
predictor_log_reg = Predictor(dataframe=df_model_A, model=algo_log_reg)
predictor_log_reg.prepare_data()
predictor_log_reg.train()
predictor_log_reg.evaluate()

print("\n--- TEST 1 : RÉGRESSION LOGISTIQUE (Config B - Cartons) ---")
algo_log_reg_red_cards = LogisticRegression(max_iter=1000, random_state=42)
predictor_log_reg_red_cards = Predictor(dataframe=df_model_B, model=algo_log_reg_red_cards)
predictor_log_reg_red_cards.prepare_data()
predictor_log_reg_red_cards.train()
predictor_log_reg_red_cards.evaluate()


# %% [markdown]
# ### Test 2 : Support Vector Machine (SVM)
# 
# L'algorithme SVM tente de tracer une frontière (hyperplan) la plus large possible entre nos classes. Il obtient généralement les meilleures performances globales sur des problèmes linéaires.

# %%
print("\n--- TEST 2 : SVM ---")
algo_svm = SVC(kernel="linear", random_state=42)
predictor_svm = Predictor(dataframe=df_model_A, model=algo_svm)
predictor_svm.prepare_data()
predictor_svm.train()
predictor_svm.evaluate()

print("\n--- TEST 2 : SVM (Config B - Cartons) ---")
algo_svm_red_cards = SVC(kernel="linear", random_state=42)
predictor_svm_red_cards = Predictor(dataframe=df_model_B, model=algo_svm_red_cards)
predictor_svm_red_cards.prepare_data()
predictor_svm_red_cards.train()
predictor_svm_red_cards.evaluate()

# Analyse visuelle des erreurs (Matrice de Confusion)
cm = confusion_matrix(predictor_svm.y_test, predictor_svm.model.predict(predictor_svm.X_test_scaled))
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Extérieur", "Nul", "Domicile"], yticklabels=["Extérieur", "Nul", "Domicile"])
plt.ylabel("Réel")
plt.xlabel("Prédit")
plt.title("Matrice de Confusion - Modèle SVM Champion (Config A)")
plt.show()

# %% [markdown]
# ### Test 3 & 4 : Arbres de décisions et Random Forest
# 
# Les modèles basés sur les arbres permettent de découvrir des règles non linéaires complexes. Nous créons également une version *Balanced* du Random Forest pour compenser le fait que l'IA a beaucoup de mal à détecter les matchs nuls.

# %%
print("\n--- TEST 3 : ARBRE DE DÉCISION ---")
algo_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
predictor_tree = Predictor(dataframe=df_model_A, model=algo_tree)
predictor_tree.prepare_data()
predictor_tree.train()
predictor_tree.evaluate()

print("\n--- TEST 3 : ARBRE DE DÉCISION (Config B - Cartons) ---")
algo_tree_red_cards = DecisionTreeClassifier(max_depth=5, random_state=42)
predictor_tree_red_cards = Predictor(dataframe=df_model_B, model=algo_tree_red_cards)
predictor_tree_red_cards.prepare_data()
predictor_tree_red_cards.train()
predictor_tree_red_cards.evaluate()

# %%
print("\n--- TEST 4 : RANDOM FOREST ---")
algo_rf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
predictor_rf = Predictor(dataframe=df_model_A, model=algo_rf)
predictor_rf.prepare_data()
predictor_rf.train()
predictor_rf.evaluate()

print("\n--- TEST 4 : RANDOM FOREST (Config B - Cartons) ---")
algo_rf_red_cards = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
predictor_rf_red_cards = Predictor(dataframe=df_model_B, model=algo_rf_red_cards)
predictor_rf_red_cards.prepare_data()
predictor_rf_red_cards.train()
predictor_rf_red_cards.evaluate()

# %%
print("\n--- TEST 4 bis : RANDOM FOREST (ÉQUILIBRÉ - Poids Ajustés) ---")
algo_rf_bal = RandomForestClassifier(n_estimators=100, max_depth=7, class_weight="balanced", random_state=42)
predictor_rf_bal = Predictor(dataframe=df_model_A, model=algo_rf_bal)
predictor_rf_bal.prepare_data()
predictor_rf_bal.train()
predictor_rf_bal.evaluate()

print("\n--- TEST 4 bis : RANDOM FOREST (ÉQUILIBRÉ) (Config B - Cartons) ---")
algo_rf_bal_red_cards = RandomForestClassifier(n_estimators=100, max_depth=7, class_weight="balanced", random_state=42)
predictor_rf_bal_red_cards = Predictor(dataframe=df_model_B, model=algo_rf_bal_red_cards)
predictor_rf_bal_red_cards.prepare_data()
predictor_rf_bal_red_cards.train()
predictor_rf_bal_red_cards.evaluate()

# %% [markdown]
# ### Analyse du comportement : "Feature Importance"
# 
# Grâce au Random Forest, nous pouvons demander au modèle ce qu'il a regardé en priorité pour prendre sa décision. L'argent (différence de valeur) et les positions au classement restent les reines indiscutables de la prédiction.

# %%
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

importances_A = algo_rf.feature_importances_
features_A_plot = df_model_A.drop(columns=["results"]).columns
indices_A = np.argsort(importances_A)
axes[0].barh(range(len(indices_A)), importances_A[indices_A], color="b", align="center")
axes[0].set_yticks(range(len(indices_A)), [features_A_plot[i] for i in indices_A])
axes[0].set_title("Feature Importance (Config A)")

importances_B = algo_rf_red_cards.feature_importances_
features_B_plot = df_model_B.drop(columns=["results"]).columns
indices_B = np.argsort(importances_B)
axes[1].barh(range(len(indices_B)), importances_B[indices_B], color="r", align="center")
axes[1].set_yticks(range(len(indices_B)), [features_B_plot[i] for i in indices_B])
axes[1].set_title("Feature Importance (Config B - Indiscipline)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Test 5 à 9 : Benchmark de la concurrence
# 
# Nous finissons notre panorama en testant des modèles de boosting, de calcul de distance (KNN), de réseaux de neurones et de probabilités Bayesiennes.

# %%
print("\n--- TEST 5 : GRADIENT BOOSTING ---")
algo_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
predictor_gb = Predictor(dataframe=df_model_A, model=algo_gb)
predictor_gb.prepare_data()
predictor_gb.train()
predictor_gb.evaluate()

print("\n--- TEST 5 : GRADIENT BOOSTING (Config B - Cartons) ---")
algo_gb_red_cards = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
predictor_gb_red_cards = Predictor(dataframe=df_model_B, model=algo_gb_red_cards)
predictor_gb_red_cards.prepare_data()
predictor_gb_red_cards.train()
predictor_gb_red_cards.evaluate()

# %%
print("\n--- TEST 6 : LINEAR SVC ---")
algo_lsvc = LinearSVC(max_iter=1000, random_state=42)
predictor_lsvc = Predictor(dataframe=df_model_A, model=algo_lsvc)
predictor_lsvc.prepare_data()
predictor_lsvc.train()
predictor_lsvc.evaluate()

print("\n--- TEST 6 : LINEAR SVC (Config B - Cartons) ---")
algo_lsvc_red_cards = LinearSVC(max_iter=1000, random_state=42)
predictor_lsvc_red_cards = Predictor(dataframe=df_model_B, model=algo_lsvc_red_cards)
predictor_lsvc_red_cards.prepare_data()
predictor_lsvc_red_cards.train()
predictor_lsvc_red_cards.evaluate()

# %%
print("\n--- TEST 7 : KNN (K-Nearest Neighbors) ---")
algo_knn = KNeighborsClassifier(n_neighbors=15)
predictor_knn = Predictor(dataframe=df_model_A, model=algo_knn)
predictor_knn.prepare_data()
predictor_knn.train()
predictor_knn.evaluate()

print("\n--- TEST 7 : KNN (K-Nearest Neighbors) (Config B - Cartons) ---")
algo_knn_red_cards = KNeighborsClassifier(n_neighbors=15)
predictor_knn_red_cards = Predictor(dataframe=df_model_B, model=algo_knn_red_cards)
predictor_knn_red_cards.prepare_data()
predictor_knn_red_cards.train()
predictor_knn_red_cards.evaluate()

# %%
print("\n--- TEST 8 : RÉSEAU DE NEURONES (MLP) ---")
algo_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
predictor_mlp = Predictor(dataframe=df_model_A, model=algo_mlp)
predictor_mlp.prepare_data()
predictor_mlp.train()
predictor_mlp.evaluate()

print("\n--- TEST 8 : RÉSEAU DE NEURONES (MLP) (Config B - Cartons) ---")
algo_mlp_red_cards = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
predictor_mlp_red_cards = Predictor(dataframe=df_model_B, model=algo_mlp_red_cards)
predictor_mlp_red_cards.prepare_data()
predictor_mlp_red_cards.train()
predictor_mlp_red_cards.evaluate()

# %%
print("\n--- TEST 9 : NAIVE BAYES ---")
algo_nb = GaussianNB()
predictor_nb = Predictor(dataframe=df_model_A, model=algo_nb)
predictor_nb.prepare_data()
predictor_nb.train()
predictor_nb.evaluate()

print("\n--- TEST 9 : NAIVE BAYES (Config B - Cartons) ---")
algo_nb_red_cards = GaussianNB()
predictor_nb_red_cards = Predictor(dataframe=df_model_B, model=algo_nb_red_cards)
predictor_nb_red_cards.prepare_data()
predictor_nb_red_cards.train()
predictor_nb_red_cards.evaluate()

for i in range(50):
    print("=", end="")

# %% [markdown]
# ### Benchmark sur la Configuration C (Configuration Optimisée)
# Nous allons maintenant tester nos 9 algorithmes sur notre dataset optimisé qui intègre la forme en points, la solidité défensive, les joueurs internationaux et la pression du stade.

# %%
print("\n" + "="*50)
print("DÉMARRAGE DES TESTS SUR LA CONFIG C (Optimisée)")
print("="*50)

# --- TEST 1 : RÉGRESSION LOGISTIQUE ---
print("\n--- TEST 1 : RÉGRESSION LOGISTIQUE (Config C - Optimisée) ---")
algo_log_reg_C = LogisticRegression(max_iter=1000, random_state=42)
predictor_log_reg_C = Predictor(dataframe=df_model_C, model=algo_log_reg_C)
predictor_log_reg_C.prepare_data()
predictor_log_reg_C.train()
predictor_log_reg_C.evaluate()

# --- TEST 2 : SVM ---
print("\n--- TEST 2 : SVM (Config C - Optimisée) ---")
algo_svm_C = SVC(kernel="linear", random_state=42)
predictor_svm_C = Predictor(dataframe=df_model_C, model=algo_svm_C)
predictor_svm_C.prepare_data()
predictor_svm_C.train()
predictor_svm_C.evaluate()

# --- TEST 3 : ARBRE DE DÉCISION ---
print("\n--- TEST 3 : ARBRE DE DÉCISION (Config C - Optimisée) ---")
algo_tree_C = DecisionTreeClassifier(max_depth=5, random_state=42)
predictor_tree_C = Predictor(dataframe=df_model_C, model=algo_tree_C)
predictor_tree_C.prepare_data()
predictor_tree_C.train()
predictor_tree_C.evaluate()

# --- TEST 4 : RANDOM FOREST ---
print("\n--- TEST 4 : RANDOM FOREST (Config C - Optimisée) ---")
algo_rf_C = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
predictor_rf_C = Predictor(dataframe=df_model_C, model=algo_rf_C)
predictor_rf_C.prepare_data()
predictor_rf_C.train()
predictor_rf_C.evaluate()

# --- TEST 4 bis : RANDOM FOREST (ÉQUILIBRÉ) ---
print("\n--- TEST 4 bis : RANDOM FOREST ÉQUILIBRÉ (Config C - Optimisée) ---")
algo_rf_bal_C = RandomForestClassifier(n_estimators=100, max_depth=7, class_weight="balanced", random_state=42)
predictor_rf_bal_C = Predictor(dataframe=df_model_C, model=algo_rf_bal_C)
predictor_rf_bal_C.prepare_data()
predictor_rf_bal_C.train()
predictor_rf_bal_C.evaluate()

# %% [markdown]
# **Vérification du modèle : Quelles sont les variables qui ont gagné ?**
# Regardons si nos nouvelles variables (Défense, Points, Stades) ont pris le pas sur l'attaque.

# %%
# Analyse de l'importance des variables pour le Random Forest de la Config C
# On utilise algo_rf_C qui vient d'être entraîné
importances_C = algo_rf_C.feature_importances_
features_C_plot = df_model_C.drop(columns=["results"]).columns
indices_C = np.argsort(importances_C)

plt.figure(figsize=(10, 6))
plt.title("Feature Importance (Config C - Optimisée)", fontsize=14)
plt.barh(range(len(indices_C)), importances_C[indices_C], color="green", align="center")
plt.yticks(range(len(indices_C)), [features_C_plot[i] for i in indices_C])
plt.xlabel("Importance relative", fontsize=12)
plt.show()

# %%
# --- TEST 5 : GRADIENT BOOSTING ---
print("\n--- TEST 5 : GRADIENT BOOSTING (Config C - Optimisée) ---")
algo_gb_C = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
predictor_gb_C = Predictor(dataframe=df_model_C, model=algo_gb_C)
predictor_gb_C.prepare_data()
predictor_gb_C.train()
predictor_gb_C.evaluate()

# --- TEST 6 : LINEAR SVC ---
print("\n--- TEST 6 : LINEAR SVC (Config C - Optimisée) ---")
algo_lsvc_C = LinearSVC(max_iter=1000, random_state=42)
predictor_lsvc_C = Predictor(dataframe=df_model_C, model=algo_lsvc_C)
predictor_lsvc_C.prepare_data()
predictor_lsvc_C.train()
predictor_lsvc_C.evaluate()

# --- TEST 7 : KNN ---
print("\n--- TEST 7 : KNN (K-Nearest Neighbors) (Config C - Optimisée) ---")
algo_knn_C = KNeighborsClassifier(n_neighbors=15)
predictor_knn_C = Predictor(dataframe=df_model_C, model=algo_knn_C)
predictor_knn_C.prepare_data()
predictor_knn_C.train()
predictor_knn_C.evaluate()

# --- TEST 8 : RÉSEAU DE NEURONES (MLP) ---
print("\n--- TEST 8 : RÉSEAU DE NEURONES (MLP) (Config C - Optimisée) ---")
algo_mlp_C = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
predictor_mlp_C = Predictor(dataframe=df_model_C, model=algo_mlp_C)
predictor_mlp_C.prepare_data()
predictor_mlp_C.train()
predictor_mlp_C.evaluate()

# --- TEST 9 : NAIVE BAYES ---
print("\n--- TEST 9 : NAIVE BAYES (Config C - Optimisée) ---")
algo_nb_C = GaussianNB()
predictor_nb_C = Predictor(dataframe=df_model_C, model=algo_nb_C)
predictor_nb_C.prepare_data()
predictor_nb_C.train()
predictor_nb_C.evaluate()

print("\n" + "="*50)
print("DÉMARRAGE DES TESTS SUR LA CONFIG D (Spécifique)")
print("="*50)

# --- TEST 1 : LOGISTIC REGRESSION ---
print("\n--- TEST 1 : RÉGRESSION LOGISTIQUE (Config D) ---")
algo_log_reg_D = LogisticRegression(max_iter=1000, random_state=42)
predictor_log_reg_D = Predictor(dataframe=df_model_D, model=algo_log_reg_D)
predictor_log_reg_D.prepare_data()
predictor_log_reg_D.train()
predictor_log_reg_D.evaluate()

# --- TEST 2 : SVM ---
print("\n--- TEST 2 : SVM (Config D) ---")
algo_svm_D = SVC(kernel="linear", random_state=42)
predictor_svm_D = Predictor(dataframe=df_model_D, model=algo_svm_D)
predictor_svm_D.prepare_data()
predictor_svm_D.train()
predictor_svm_D.evaluate()

# --- TEST 3 : DECISION TREE ---
print("\n--- TEST 3 : ARBRE DE DÉCISION (Config D) ---")
algo_tree_D = DecisionTreeClassifier(max_depth=5, random_state=42)
predictor_tree_D = Predictor(dataframe=df_model_D, model=algo_tree_D)
predictor_tree_D.prepare_data()
predictor_tree_D.train()
predictor_tree_D.evaluate()

# --- TEST 4 : RANDOM FOREST ---
print("\n--- TEST 4 : RANDOM FOREST (Config D) ---")
algo_rf_D = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
predictor_rf_D = Predictor(dataframe=df_model_D, model=algo_rf_D)
predictor_rf_D.prepare_data()
predictor_rf_D.train()
predictor_rf_D.evaluate()

# --- TEST 4 bis : RANDOM FOREST (ÉQUILIBRÉ) ---
print("\n--- TEST 4 bis : RANDOM FOREST ÉQUILIBRÉ (Config D) ---")
algo_rf_bal_D = RandomForestClassifier(n_estimators=100, max_depth=7, class_weight="balanced", random_state=42)
predictor_rf_bal_D = Predictor(dataframe=df_model_D, model=algo_rf_bal_D)
predictor_rf_bal_D.prepare_data()
predictor_rf_bal_D.train()
predictor_rf_bal_D.evaluate()

# --- TEST 5 : GRADIENT BOOSTING ---
print("\n--- TEST 5 : GRADIENT BOOSTING (Config D) ---")
algo_gb_D = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
predictor_gb_D = Predictor(dataframe=df_model_D, model=algo_gb_D)
predictor_gb_D.prepare_data()
predictor_gb_D.train()
predictor_gb_D.evaluate()

# --- TEST 6 : LINEAR SVC ---
print("\n--- TEST 6 : LINEAR SVC (Config D) ---")
algo_lsvc_D = LinearSVC(max_iter=1000, random_state=42)
predictor_lsvc_D = Predictor(dataframe=df_model_D, model=algo_lsvc_D)
predictor_lsvc_D.prepare_data()
predictor_lsvc_D.train()
predictor_lsvc_D.evaluate()

# --- TEST 7 : KNN ---
print("\n--- TEST 7 : KNN (Config D) ---")
algo_knn_D = KNeighborsClassifier(n_neighbors=15)
predictor_knn_D = Predictor(dataframe=df_model_D, model=algo_knn_D)
predictor_knn_D.prepare_data()
predictor_knn_D.train()
predictor_knn_D.evaluate()

# --- TEST 8 : MLP (NEURAL NETWORK) ---
print("\n--- TEST 8 : RÉSEAU DE NEURONES (Config D) ---")
algo_mlp_D = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
predictor_mlp_D = Predictor(dataframe=df_model_D, model=algo_mlp_D)
predictor_mlp_D.prepare_data()
predictor_mlp_D.train()
predictor_mlp_D.evaluate()

# --- TEST 9 : NAIVE BAYES ---
print("\n--- TEST 9 : NAIVE BAYES (Config D) ---")
algo_nb_D = GaussianNB()
# MODIFIE ICI : df_model_D au lieu de df_model_C
predictor_nb_D = Predictor(dataframe=df_model_D, model=algo_nb_D) 
predictor_nb_D.prepare_data()
predictor_nb_D.train()
predictor_nb_D.evaluate()

# %%
importances_D = algo_rf_D.feature_importances_
features_D_plot = df_model_D.drop(columns=["results"]).columns
indices_D = np.argsort(importances_D)

plt.figure(figsize=(10, 6))
plt.title("Importance des Variables (Config D - Spécifique)", fontsize=14)
plt.barh(range(len(indices_D)), importances_D[indices_D], color="purple", align="center")
plt.yticks(range(len(indices_D)), [features_D_plot[i] for i in indices_D])
plt.xlabel("Importance relative")
plt.show()

# %% [markdown]
# ### Conclusion de l'Évaluation
# Au terme de ces tests, le **Support Vector Machine (Test 2) et la Régression Logistique (Test 1)** restent nos meilleurs modèles. Ils prouvent que la différence de budget et le classement sont des relations linéaires fortes.
# 
# On note également que l'ajout des cartons rouges (Config B) aide particulièrement les modèles probabilistes (Naive Bayes), mais a tendance à ajouter un très léger bruit sur les modèles plus complexes.


