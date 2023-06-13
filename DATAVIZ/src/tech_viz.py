#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
from plotly.offline import plot
from distinctipy import distinctipy


# Exercice 1 stack area + stack bar plot

# In[2]:


# Lecture du fichier CSV PhD_v3
phd = pd.read_csv("PhD_v3.csv", low_memory=False)


# In[3]:


phd.info()


# In[4]:


# changement de types des variables indiquant des dates
phd["Date de soutenance"] = pd.to_datetime(phd["Date de soutenance"])


# In[5]:


# Sélection des disciplines sur la période 1985-2018
discipline = phd[["Date de soutenance", "Discipline"]].sort_values("Date de soutenance")
discipline = discipline[(discipline["Date de soutenance"] >= "1985") & (discipline["Date de soutenance"] < "2019")]


# In[6]:


# Indication des disciplines par année de soutenance
discipline["annee"] = discipline["Date de soutenance"].dt.year

# Nom des disciplines en majuscule afin de réduire les doublons au regroupement
discipline["Discipline"] = discipline["Discipline"].str.upper()

# distribution des disciplines par année sur la période 1985-2018
discipline_dist = discipline.groupby(["annee", "Discipline"])["Discipline"].agg(["count"]).reset_index()


# In[7]:


discipline_dist


# In[8]:


# Distribution de l'ensemble des disciplines par annee
total_dist = discipline_dist.groupby("annee", as_index=False).agg("sum")


# In[9]:


# Correspondance de la distribution des soutenances par discipline et par année avec la distribution de chaque discipline par année
disciplines = discipline_dist.merge(total_dist, how="inner", on="annee", suffixes=("_discipline", "_total"))


# In[10]:


# Calcul du poucentage de soutenances pour chaque discipline par année
disciplines["pourcentage"] = round(disciplines["count_discipline"] / disciplines["count_total"] * 100, 2) 


# In[11]:


disciplines.sort_values(by="pourcentage", ascending=False, inplace=True)


# In[12]:


disciplines


# In[13]:


#disciplines[(disciplines["Discipline"].str.contains('^BIOLOGIE')) & (disciplines["count_discipline"] >= 10)].sort_values(by="count_discipline", ascending=False)


# In[14]:


# Création d'un dataframe dont l'index sont les années de la période souhaitée et les colonnes sont les disciplines ayant pour valeurs leur pourcentage par année 
# Ceci afin de mieux indiquer le pourcentage par année pour chaque discipline 
df = disciplines.pivot_table(values="count_discipline", index="annee", columns="Discipline")


# In[15]:


# calcul du nombre de soutenances pour chaque discipline sur toute la période et inversion des indexes et des variables
sum_year = df.agg(["sum"]).T


# In[16]:


sum_year


# In[17]:


# Calcul du nombre de soutenances de l'ensemble des disciplines sur toute la période
total_disciplines = sum_year["sum"].sum()


# In[18]:


total_disciplines 


# In[19]:


# pourcentage de chaque discipline sur toute la période
sum_year["percent"] = round(sum_year["sum"] / total_disciplines * 100, 2) 


# In[20]:


sum_year = sum_year.sort_values(by="percent", ascending=True)


# In[21]:


sum_year.sort_values(by="sum", ascending=False).head(20)


# In[22]:


# selection des disciplines dont la moyenne des soutenances est supérieure à 1% afin de conserver les disciplines les plus productives
disciplines_principales = sum_year[sum_year["percent"] > 1].index


# In[23]:


# selection des autres disciplines
disciplines_autres = sum_year[sum_year["percent"] <= 1].index


# In[24]:


disciplines_principales = disciplines_principales.tolist()


# In[25]:


disciplines_principales


# In[26]:


# selection du nombre de soutenances par année des disciplines principales
d_principales = disciplines[disciplines["Discipline"].isin(disciplines_principales)] 


# In[27]:


d_principales.isna().sum()


# In[28]:


# selection du nombre de soutenances par année des autres disciplines
d_autres = disciplines[disciplines["Discipline"].isin(disciplines_autres)]


# In[29]:


# Sélection des valeurs pour les disciplines principales par année pour l'axe des ordonnées du stackplot 
df_principales = d_principales.pivot_table(values="pourcentage", index="annee", columns="Discipline", fill_value=0)


# In[30]:


# Sélection des valeurs pour les autres disciplines par année pour l'axe des ordonnées du stackplot
df_autres =  d_autres.pivot_table(values="pourcentage", index="annee", columns="Discipline")


# In[31]:


df_principales.head()


# In[32]:


# On aggrège les disciplines voisines
df_principales["DROIT"] = df_principales[["DROIT", "DROIT PUBLIC", "DROIT PRIVE"]].sum(axis=1)


# In[33]:


# Suppression des variables inutiles
df_principales.drop(["DROIT PUBLIC", "DROIT PRIVE", "SCIENCES APPLIQUEES"], axis=1, inplace=True)


# In[34]:


# Renommage de certaines variables
df_principales.rename(columns = {"SCIENCES BIOLOGIQUES ET FONDAMENTALES APPLIQUEES. PSYCHOLOGIE": "BIOLOGIE", 
                                 "SCIENCES DE GESTION": "GESTION",
                                 "SCIENCES ECONOMIQUES": "ECONOMIE"}, inplace=True)


# In[35]:


df_principales.head()


# In[36]:


# Réordonnencement des disciplines selon leur valeur maximale
df_principales.max().sort_values()


# In[37]:


# Réordonnencement des disciplines selon leur valeur maximale
labels = list(df_principales.max().sort_values().index)


# In[38]:


labels


# In[39]:


# axe des abscisses
x_annee = df_principales.index


# In[40]:


# axes des ordonnées. Récupération des valeurs de chaque discipline depuis un dictionnaire
y_principales = df_principales[labels].to_dict('list').values()


# In[41]:


# création des couleurs (générées par la lib distinctipy)
#colors = distinctipy.get_colors(len(y_principales))
colors = [(0.41335106753762413, 0.44408083891209305, 0.6879146368696584), 
         (0.5174478458571609, 0.9757668812017093, 0.4697456183440195), 
         (0.9817670041302922, 0.6052580570996697, 0.5264877571428185), 
         (0.41183998769266056, 0.8430909818179251, 0.9808086428907623), 
         (0.9788561341833234, 0.4176052673381056, 0.9972051585194968), 
         (0.9867886192752237, 0.9931617750822137, 0.5116269381539859), 
         (0.7892876501984007, 0.736175853660061, 0.8926095107002004), 
         (0.5912669383839205, 0.6425500603581105, 0.4142264973742901), 
         (0.6793490501911816, 0.9899852884440983, 0.8027245515611241), 
         (0.6314668171537415, 0.4411339814942379, 0.976339543343846), 
         (0.7471452309327387, 0.4163644416960017, 0.6296326128700039), 
         (0.4355402368887255, 0.7694742828299236, 0.6500800650028454), 
         (0.8101725688058676, 0.8086224066809281, 0.4671447002518019), 
         (0.41496825330785264, 0.6154111143337324, 0.8976818120719426), 
         (0.9861221523414294, 0.8207839003257499, 0.7272629295046426)]


# In[42]:


# Création du stackplot
fig, ax = plt.subplots(figsize=(10,10))
ax.stackplot(x_annee, y_principales, labels=labels, colors=colors)
ax.legend()
plt.xlabel("Année")
plt.ylabel("Pourcentage")
plt.title("Proportion des disciplines des soutenances en thèse de 1985 à 2018")
#plt.savefig("/home/mertes/Documents/DU_Data_Analyst/projet_2/proportion_disciplines.png")
plt.show()


# In[43]:


# Création du stacked Barplot
fig, ax = plt.subplots(figsize=(10,10))

# liste des pourcentages de toutes les disciplines pour chaque année
for n, height in enumerate(y_principales):
    if n > 0:
        # création du stack courant avec pour base le sommet du précedent
        ax.bar(x_annee, height, width=1, bottom=bottom, edgecolor='white', label=labels[n], color=colors[n])
        # la base du nouveau stack est le sommet du précédent stack 
        bottom = np.add(bottom, height).tolist()
    else:
        # Création de la première barre
        ax.bar(x_annee, height, width=1, edgecolor='white', label=labels[n], color=colors[n])
        # initialisation de la base du 1er stack
        bottom = height
        
fig.legend(bbox_to_anchor=(1.09, 0.85))
plt.xlabel("Année")
plt.ylabel("Pourcentage")
plt.title("Proportion des disciplines des soutenances en thèse de 1985 à 2018")
plt.show()


# Exercice 2: grille + transparence

# In[44]:


fig, ax = plt.subplots(figsize=(10,10))
ax.stackplot(x_annee, y_principales, labels=labels, colors=colors)
ax.legend(loc="upper right")
ax.legend(loc="right", bbox_to_anchor=(1.25, 0.7), ncol=1)
plt.grid()
plt.xlabel("Année")
plt.ylabel("Pourcentage")
plt.title("Proportion des disciplines des soutenances en thèse de 1985 à 2018")
plt.show()


# In[45]:


fig, ax = plt.subplots(figsize=(10,10))
ax.stackplot(x_annee, y_principales, labels=labels, alpha=0.3, colors=colors)
ax.legend(loc="upper right")
ax.legend(loc="right", bbox_to_anchor=(1.25, 0.7), ncol=1)
plt.grid()
plt.xlabel("Année")
plt.ylabel("Pourcentage")
plt.title("Proportion des disciplines des soutenances en thèse de 1985 à 2018")
plt.show()


# In[46]:


fig, ax = plt.subplots(figsize=(10,10))
ax.stackplot(x_annee, y_principales, labels=labels, alpha=0.5, colors=colors)
ax.legend(loc="upper right")
ax.legend(loc="right", bbox_to_anchor=(1.25, 0.7), ncol=1)
plt.grid()
plt.xlabel("Année")
plt.ylabel("Pourcentage")
plt.title("Proportion des disciplines des soutenances en thèse de 1985 à 2018")
plt.show()


# Exercice 3: Définition de la distance des labels sur les 2 axes

# In[47]:


fig, ax = plt.subplots(figsize=(10,10))
ax.stackplot(x_annee, y_principales, labels=labels, colors=colors)
ax.legend(loc="upper right")
ax.legend(loc="right", bbox_to_anchor=(1.25, 0.7), ncol=1)
plt.grid()
# définition de la distance des labels de l'axe X
plt.tick_params(axis='x', which='major', pad=15)
plt.xlabel("Année")
plt.ylabel("Pourcentage")
plt.title("Proportion des disciplines des soutenances en thèse de 1985 à 2018")
plt.show()


# In[48]:


fig, ax = plt.subplots(figsize=(10,10))
ax.stackplot(x_annee, y_principales, labels=labels, colors=colors)
ax.legend(loc="upper right")
ax.legend(loc="right", bbox_to_anchor=(1.25, 0.7), ncol=1)
# définition de la distance des labels de l'axe X
plt.tick_params(axis='x', which='major', pad=15, labelrotation=45)
plt.grid()
plt.xlabel("Année")
plt.ylabel("Pourcentage")
plt.title("Proportion des disciplines des soutenances en thèse de 1985 à 2018")
plt.show()


# Exercice 4 changement de la police (+taille) du label des axes + titre et config. marges

# In[49]:


fig, ax = plt.subplots(figsize=(10,10))
ax.stackplot(x_annee, y_principales, labels=labels, colors=colors)
ax.legend(loc="upper right")
ax.legend(loc="right", bbox_to_anchor=(1.25, 0.7), ncol=1)
ax.set_title("Proportion des disciplines des soutenances en thèse de 1985 à 2018", 
             fontsize=20, 
             fontname="Times New Roman")
ax.set_xlabel("Année", fontsize=16, fontname="Times New Roman")
ax.set_ylabel("Pourcentage", fontsize=16, fontname="Times New Roman")
# définition de la distance des labels de l'axe X
plt.tick_params(axis='x', which='major', pad=15, labelrotation=45)
plt.grid()
plt.xlabel("Année")
plt.ylabel("Pourcentage")
plt.show()


# Ajustement des marges:

# In[50]:


fig, ax = plt.subplots(figsize=(10,10))
fig.subplots_adjust(left=0.20, right=0.8, bottom=0.25, top=0.75) # Ajustement des marges de la figure
ax.stackplot(x_annee, y_principales, labels=labels, colors=colors)
#ax.legend(loc="upper right")
ax.legend(loc="best", ncol=1, fontsize=5)
ax.set_title("Proportion des disciplines des soutenances en thèse de 1985 à 2018", fontsize=20, fontname="Times New Roman")
ax.set_xlabel("Année", fontsize=16, fontname="Times New Roman")
ax.set_ylabel("Pourcentage", fontsize=16, fontname="Times New Roman")
# définition de la distance des labels de l'axe X
plt.tick_params(axis='x', which='major', pad=15, labelrotation=45)
plt.grid()
plt.xlabel("Année")
plt.ylabel("Pourcentage")
plt.show()


# Exercice 5: Echelle logarithmique en ordonnée

# In[51]:


fig, ax = plt.subplots(figsize=(10,10))
ax.stackplot(x_annee, y_principales, labels=labels, colors=colors)
ax.legend(loc="upper right")
ax.set_yscale("log")
ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=None))
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(0.2, 0.4, 0.6, 0.8), numticks=None))
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
ax.legend(loc="right", bbox_to_anchor=(1.25, 0.7), ncol=1)
plt.xlabel("Année")
plt.ylabel("Pourcentage")
plt.title("Proportion des disciplines des soutenances en thèse de 1985 à 2018")
plt.show()


# Exercice 6: Position de la légende

# In[52]:


fig, ax = plt.subplots(figsize=(10,10))
ax.stackplot(x_annee, y_principales, labels=labels, colors=colors)
ax.legend(loc="upper right")
plt.xlabel("Année")
plt.ylabel("Pourcentage")
plt.title("Proportion des disciplines des soutenances en thèse de 1985 à 2018")
plt.show()


# In[53]:


fig, ax = plt.subplots(figsize=(10,10))
ax.stackplot(x_annee, y_principales, labels=labels, colors=colors)
ax.legend(loc="upper right")
plt.xlabel("Année")
plt.ylabel("Pourcentage")
plt.title("Proportion des disciplines des soutenances en thèse de 1985 à 2018")
ax.legend(loc="right", bbox_to_anchor=(1.25, 0.7), ncol=1)
plt.show()


# Exercice 7: Palette des couleurs

# In[54]:


#Création du stackplot
fig, ax = plt.subplots(figsize=(10,10))
ax.stackplot(x_annee, y_principales, labels=labels, colors=colors)
ax.legend(loc="upper right")
plt.xlabel("Année")
plt.ylabel("Pourcentage")
plt.title("Proportion des disciplines des soutenances en thèse de 1985 à 2018")
plt.show()


# Exercice 8: Changement de l'ordre des disciplines

# In[55]:


# inversion des éléments de la liste comportant les valeurs pour chaque displicine
y_inverse = list(y_principales)[::-1]

# inversion des disciplines dans la legende
lab_inv = labels[::-1]


# In[56]:


fig, ax = plt.subplots(figsize=(10,10))
ax.stackplot(x_annee, y_inverse, labels=lab_inv, colors=colors)
ax.legend(loc="upper right")
plt.xlabel("Année")
plt.ylabel("Pourcentage")
plt.title("Proportion des disciplines des soutenances en thèse de 1985 à 2018")
ax.legend(loc="right", bbox_to_anchor=(1.25, 0.7), ncol=1)
plt.show()


# Exercice 9:

# In[57]:


# Distribution des soutenances pour chaque mois sur toute la période
d_theses = pd.DataFrame({"date": phd[phd["Statut"] == "soutenue"]["Date de soutenance"]}).dropna()
dates_year_month = pd.DataFrame({"annee": d_theses["date"].dt.year, 
                                 "mois": d_theses["date"].dt.month, 
                                 "jour": d_theses["date"].dt.day})
dist_year_month = dates_year_month.groupby(["annee", "mois"], as_index=False)["annee"].value_counts()


# In[58]:


# Calcul du nombre de thèses soutenues par mois de chaque année sur toute la période
theses_month = dist_year_month.groupby(["annee", "mois"])["count"].agg(["sum"]).reset_index()

# Calcul du nombre total de thèses pour chaque année sur toute la période
theses_year = theses_month.groupby("annee")["sum"].sum().reset_index()

# Fusion du nombre total de thèses de chaque année avec le nombre de thèses par mois pour chaque année
total_theses = theses_month.merge(theses_year, on="annee", how="inner", suffixes=["_mois", "_annee"]).set_index(["annee", "mois"])


# In[59]:


# Calcul du pourcentage de theses par mois sur une année sur toute la période
total_theses["pourcentage"] = total_theses["sum_mois"] / total_theses["sum_annee"] * 100


# In[60]:


df_theses = total_theses.reset_index()


# In[61]:


df_year = df_theses[(df_theses["annee"] > 2005) & (df_theses["annee"] < 2019)].sort_values(by="mois")


# In[62]:


# Conversion des mois
map_mois = {1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril", 5: "Mai", 6: "Juin", 
           7: "Juillet", 8: "Août", 9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"}


# In[63]:


num_mois = df_year["mois"].unique()


# In[64]:


num_mois


# In[65]:


df_year["map_mois"] = df_year["mois"].map(map_mois)


# In[66]:


df_year["map_mois"]


# In[67]:


df_year


# In[68]:


# Initialisation des valeurs des mois maquants à 0
df_missing = df_year.pivot_table(index=["mois", "map_mois"], columns=["annee"], values="pourcentage", fill_value=0)


# In[69]:


df_missing


# In[70]:


df_missing = df_missing.reset_index().melt(id_vars=["mois", "map_mois"], value_name="pourcentage").sort_values(by=["annee", "mois"])


# In[71]:


df_missing


# In[72]:


# labels par année du slider
years = df_missing.annee.unique()


# In[73]:


years


# In[83]:


fig = go.Figure()
for year in years:
    df = df_missing[df_missing["annee"] == year]
    fig.add_trace(go.Bar(x=df['map_mois'], 
                         y=df['pourcentage']))
    
#  Paramétres du slider
slider = {
    'steps': [{'label': str(year), 'method': 'update', 'args': [{'visible': [year == y for y in years]}, {'title': f"Pourcentage de soutenances par mois pour l'année {year}"}]} for year in years],
    'active': 0,
    'y': -0.1,
}

# Paramètres de la figure
fig.update_layout(
    width=800,
    height=600,
    sliders=[slider],
    title=f"Pourcentage de soutenances par mois pour l'année {years[0]}",
    xaxis_title="Mois",
    yaxis_title="Pourcentage"
)

fig.show()


# In[84]:


# Création du widget html 
widget_html = plot(fig, output_type='div')
file = open("widget_slider.html", "w")
file.write(str(widget_html))
file.close()


# In[81]:


fig = go.Figure()
for year in years:
    df = df_missing[df_missing["annee"] == year]
    fig.add_trace(go.Bar(x=df['map_mois'], 
                         y=df['pourcentage']))
    
# Paramétres du slider
selector = {
    'buttons': [{'label': str(year), 'method': 'update', 'args': [{'visible': [year == y for y in years]}, {'title': f"Pourcentage de soutenances par mois pour l'année {year}"}]} for year in years],
    'active': 0
}

# Paramètres de la figure
fig.update_layout(
    width=800,
    height=600,
    updatemenus=[dict(selector, x=1.05, xanchor='left')],
    title=f"Pourcentage de soutenances par mois pour l'année {years[0]}",
    xaxis_title="Mois",
    yaxis_title="Pourcentage"
)
fig.show()


# In[82]:


# Création du widget html 
widget_html = plot(fig, output_type='div')
file = open("widget_selector.html", "w")
file.write(str(widget_html))
file.close()

