---
title: "Prédiction de la taille à l’aide des génotypes individuels"
author: "Gaby Maroun"
date: "18/5/2020"
output:
  rmdformats::readthedown:
  # prettydoc::html_pretty:
    highlight: kate
    df_print: paged
    
    # number_sections: yes
  #   theme: cayman
  #   kramdown:
  #     toc_levels: 2..3
    # toc: yes
    # toc_float:
    #   collapsed: yes
    #   smooth_scroll: yes
  # pdf_document:
  #   toc: yes
resource_files: DATA_PROJET1.RData
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
library(ggplot2 )
library(caret)
library(caretEnsemble)
library(C50)
library(tidyverse)
library(reshape2)
library(glmnet)
library(randomForest)
library(ranger)
library(gbm)
library(fastICA)
library(nnet)
library(skimr)
```

# Introduction

  Dans ce document, on explorera plusieurs types différents de modèles prédictifs: `glmnet`, `rainforest`, `gmb`, `evtree` et `neural network`, et proposera même une autre solution technique adaptée à la réalisation d'un apprentissage supervisé afin de produire un modèle capable de prédire la taille d'un individu, variable continue qui nécessite donc une étude de régression à partir de données à notre disposition issues de 6 études sur 6 centres européens concernant 120 patients et leurs 1000 allèles (il n'existe que deux allèles différents dans la population)..
  
* La valeur 0 = AA : l’individu a hérité de l’allèle majeur A de chacun de ses deux parents.
  
* La valeur 1 = AG : l’individu a hérité d’un allèle A, et d’un allèle G (l’information de transmission de l’allèle par le père ou la mère est inconnue).

* La valeur 2 = GG : l’individu a hérité de l’allèle mineur G de chacun de ses deux parents.

Vous pouvez voir les données en utilisations dans le `dataframe` suivant:

```{r df, echo=FALSE}
# Charger les données du fichier DATA_projet1.RData
load(file = "C:/Users/Gaby\'s/Desktop/Semestre 2 UPPA/Machine Learning/Projet 2/DATA_PROJET2.RData")

# Mettant les données en un table de type data.frame
df<-data.frame(DATA_projet2)
df
```


On s'assure s'il y a des valeurs manquantes avec `is.na ()` pour tourner notre concentration sur le `preprocessing`, ce qui n'est pas le cas ici:
```{r na, echo=FALSE}
# Vérifier s'il y a des cases de données N/a ou vide pour utiliser preprocessing
sum(is.na(df))

```

# Création d'une partition de données

En créant une partition de données adaptée à l'apprentissage, nous aurons une table de données plus lisible et facilement accessible.

Donc, pour ce faire, on prend les 6 dernières colonnes du `df` qui représentent les tailles des 120 patients dans chaque étude et on les met les uns au-dessus des autres dans la même colonne` Taille` dans notre nouveau tableau `aEtudier`.
```{r aEtudier, echo = FALSE, results = 'hide'}
# Construire une table de données comme notre base d'étude 
aEtudier <- NULL
for(i in 1:5){
  aEtudier <- (Taille =c(aEtudier,df[,6000+i]))
}

aEtudier <-data.frame(Taille = aEtudier)

```

Dans une nouvelle table de données `aPredire` qui représente la liste dont les résultats prévus seront mis, on remplisse une colonne qui ne représente que «0» initialement.

```{r aPredire, echo = FALSE, results = 'hide'}
# Construire une table de données à prédire
aPredire<- NULL
for(i in 1:120){
  aPredire<-append(aPredire,0)
}
# aPredire <- as.factor(aPredire)
aPredire <- data.frame(Taille = aPredire)

```

Pour terminer cette étape, nous joignons les valeurs génotypiques de chaque patient avec son propre taille.

```{r aEtudieraPredire, echo = FALSE, results = 'hide'}
# Remplir la table aEtudier par les données genotypiques de chaque patients
# des premiers 5 études:
for (i in 1:1000) {
  j <- 0
  k <- 1
  for (genotypes in DATA_projet2$Genotypes[1:5]) {
    j <- j + 120
    aEtudier[k:j,paste("x",as.character(i),sep="")]<-genotypes[,i]
    k <- k + 120
  }
}

# Remplir la table aPredire par les données genotypiques de chaque patients
# du 6eme études:
for (i in 1:1000) {
  for (genotypes in DATA_projet2$Genotypes[6]) {
    aPredire[,paste("x",as.character(i),sep="")]<-genotypes[,i]
  }
}
```


Avant de passer à la création des `traincontrol` et au prétraitement des fonctionnalités, observons les statistiques descriptives de chaque colonne du jeu de données d'apprentissage `aEtudier`.

Le package `skimr` fournit une bonne solution pour afficher les statistiques descriptives clés pour chaque colonne.

La sortie de la trame de données comprend un joli histogramme dessiné sans aucune aide de traçage:
 
```{r descript, echo=FALSE, results = 'hide'}
# description

skimmed <- skim(aEtudier)
```

```{r descript1, echo=FALSE}
head(skimmed)

```
 

 
# Création d'indices de train / test personnalisés

La première chose à faire est donc de créer un objet `trainControl` réutilisable que on peut utiliser pour les comparer de manière fiable.

On définisse également une «répartition aléatoire» avec `set.seed` afin que notre travail soit reproductible et on obtient la même distribution aléatoire chaque fois qu'on exécute notre script.

Nous utilisons `createFolds ()` pour faire 10 `Cross-Validation` sur `Taille`, notre variable cible pour cet exercice qui a été choisie après plusieurs essais et en déduisons que c'est la valeur optimale qui nous donne des prédictions plus réalistes comme nous voyons à la fin de cet article.

```{r myFolds, echo = FALSE, results = 'hide'}
# Définir une graine aléatoire afin que notre travail soit reproductible:
set.seed(42)

# Création des plis
myFolds <- createFolds(aEtudier$Taille, k = 10)
```

On les passe à `trainControl ()` pour créer un trainControl réutilisable pour comparer les modèles comme déjà décrit.

```{r myControl, echo = FALSE, results = 'hide'}
# Création du reUsable trainControl object
objControl <- trainControl(method='cv', 
                           index = myFolds,
                           returnResamp='final')
```


La plupart des algorithmes ML sont capables de déterminer quelles fonctionnalités sont importantes pour prédire le Y. Mais dans certains scénarios, il faut peut-être faire attention à n'inclure que les variables qui peuvent être significativement importantes et qui ont un fort sens commercial.

C'est assez courant dans les institutions bancaires, économiques et financières. Ou on peut simplement effectuer une analyse exploratoire pour déterminer les prédicteurs importants et les signaler en tant que métrique dans notre tableau de bord d'analyse.

L'élimination récursive des fonctionnalités `RFE` est un bon choix pour sélectionner les fonctionnalités importantes possible avec la fonction `rfe()`.

Pour les fonctions de `randomForest`, les plus importantes variables sont:

``` {r rfe, echo = FALSE}
# RFE 
options(warn=-1)

subsets <- c(#1:lm
  100,200.300,400,500,600,700,800,900,1000)

ctrl <- rfeControl(functions = rfFuncs,#lmFuncs
                   method = "cv",
                   index = myFolds,
                   verbose = FALSE)

lmProfile <- rfe(Taille ~., aEtudier,
                 sizes = subsets,
                 rfeControl = ctrl)

lmProfile
```

Alors que pour les fonctions linéaires, les plus importantes variables sont:

``` {r rfe1, echo = FALSE}

# RFE 
ctrl1 <- rfeControl(functions = lmFuncs,
                   method = "cv",
                   index = myFolds,
                   verbose = FALSE)

lmProfile1 <- rfe(Taille ~., aEtudier,
                 sizes = subsets,
                 rfeControl = ctrl1)

lmProfile1
```

La taille finale du sous-ensemble de modèles sélectionné est marquée d'un  * dans la colonne sélectionnée la plus à droite.

D'après la sortie ci-dessus, une taille de modèle semble atteindre la précision optimale.

Étant donné que les algorithmes ML ont leur propre façon d'apprendre la relation entre les prédicteurs et la cible, il n'est pas sage de négliger les autres prédicteurs, en particulier lorsqu'il existe des preuves qu'il existe des informations contenues dans les autres variables pour expliquer la relation.

# Adapter les modèles au `train`

On a choisi d'essayer 5 modèles différents, afin d'avoir plusieurs points de vue et d'être sûr d'avoir des résultats plus exacts et précis.

Dans ce qui suit, on décrit notre approche pour tourner un modèle avec les résultats optimaux.

L'erreur quadratique moyenne (RMSE) est une mesure fréquemment utilisée des différences entre les valeurs (valeurs d'échantillon ou de population) prédites par un modèle ou un estimateur et les valeurs observées. 

Alors que le coefficient de détermination, noté `Rsquared`, est la proportion de la variance de la variable dépendante qui est prévisible à partir des variables indépendantes.

Un modèle dit efficace s'il a un petit `RMSE` et un grand `Rsquared`.

En plus, on a essayé plusieurs types de `preProcess`, `range` (normaliser les valeurs pour qu'elle soit comprise entre 0 et 1), `pca` (remplacer par des composants principaux et qui fait une analyse globale) et `ica` (remplacer par des composants indépendants qui va plus en détail que `pca`), sur les données pour essayer d'être plus précis en diminuant la `RMSE` autant que possible et augmentant le `Rsquared`.

## Generalized Linear Model

Les modèles linéaires généralisés (GLM) sont une extension des modèles de régression linéaire «simples», qui prédisent la variable de réponse en fonction de plusieurs variables prédictives. Les modèles de régression linéaire fonctionnent sur quelques hypothèses, telles que l'hypothèse selon laquelle nous pouvons utiliser une ligne droite pour décrire la relation entre la réponse et les variables prédictives.

On a choisi `glmnet` car il est simple, rapide, facile à interpréter et pénalise les modèles de régression linéaire et logistique sur la taille et le nombre de coefficients pour éviter le `overfitting.`

Pour cela, on a spécifié l'alpha ("Ridge regression" (ou ` alpha = 0`), "Lasso regression" (ou` alpha = 1`)) et `lambda`  (taille de la pénalité).

Ensuite, on a commencé à adapter `glmnet` à notre ensemble de données sur les tailles et à évaluer sa précision prédictive à l'aide de `myControl` déjà créé.

Pour ce modèle, on a trouvé que le `preprocess` optimale est `pca`.

```{r model_glmnet, echo = FALSE, results = 'hide'}
# Adaptez le modèle glmnet à nos données:
glmnet_grid <- expand.grid(alpha = 1,
            lambda = 0.5359083)

model_glmnet <- train(
  Taille~., aEtudier,
  tuneGrid= glmnet_grid,
  metric = "RMSE",
  method = "glmnet",
  preProcess ='pca',
  trControl = objControl
)

```

On peut voir le moyen d'erreur `RMSE` obtenu à partir de `glmnet` sur nos données et le `rsquared` :

```{r model_glmnet5, echo = FALSE}
model_glmnet
```

Dans le graphique suivant, nous pouvons voir les allèles les plus importants ou les plus influents sur les prédictions de ce modèle:

```{r plotmodel_glmnet}
#Les 15 variables qui influencent le plus la décision de 
#prédiction de ce modèle:
plot(varImp(model_glmnet,scale=F),top = 15)
```

### AVANTAGES

* La variable de réponse peut avoir n'importe quelle forme de type de distribution exponentielle,
* Capable de gérer des prédicteurs catégoriques,
* Relativement facile à interpréter et permet de comprendre clairement comment chacun des prédicteurs influence le résultat,
* Moins sensible au `overfitting`.

### LIMITES

* Nécessite des ensembles de données relativement volumineux. Plus il y a de variables prédictives, plus la taille d'échantillon requise est grande. En règle générale, le nombre de variables prédictives doit être inférieur à N / 10 (5000/10=600>120 comme dans notre cas ce qui est bon),
* Sensible aux valeurs aberrantes.

## Rainforest

Les forêts aléatoires (`Rainforest`) sont une extension d'arbres de classification uniques dans lesquels plusieurs arbres de décision sont construits avec des sous-ensembles aléatoires des données. Il combine un ensemble d'arbres de décision non linéaires en un modèle très flexible et généralement assez précis.

Les `rainforest` sont un peu plus difficiles à interpréter que les modèles linéaires, bien qu'il soit toujours possible de les comprendre.

On utilise le paquet `ranger` qui est une réimplémentation de` randomForest` qui produit presque exactement les mêmes résultats, mais qui est plus rapide, plus stable et utilise moins de mémoire

Le `tuneGrid` nous donne un contrôle plus fin sur les paramètres de réglage qui sont explorés et ensuite une vitesse pour tourner le modèle sur les points qui nous importent. Les `mtry` ont été choisis après plusieurs exécutions et une grande` tuneLength.`

Et ensuite, nous commencerons à adapter `ranger` à notre ensemble de données sur les tailles des individus et à évaluer sa précision prédictive à l'aide de `trainControl` déjà créé.

Pour ce modèle, on a trouvé que le `preprocess` optimale est `range`.

```{r model_ranger, echo = FALSE, results = 'hide'}
#Adaptez le modèle ranger représentant le type de modèle RainForest à nos données:

#Création d'un tuneGrid pour le modèle
rangerGrid <- data.frame(.mtry =2,
                       .splitrule = "variance",
                       .min.node.size = 5)
model_ranger <- train(
  Taille~., aEtudier,
  tuneGrid = rangerGrid,
  metric = "RMSE",
  method = "ranger",
  preProcess ='range',
  # tuneLength=5,
  trControl = objControl
)

```

Le moyen d'erreur `RMSE` obtenu à partir de `ranger` sur nos données et le `rsquared` :

```{r plotmodel_ranger1, echo = FALSE}
model_ranger
```

### AVANTAGES

* L'un des algorithmes d'apprentissage les plus précis disponibles,
* Il peut gérer de nombreuses variables prédictives,
* Fournit des estimations de l'importance de différentes variables prédictives,
* Maintient la précision même lorsqu'une grande partie des données est manquante(ce qui n'est pas le cas là).

### LIMITES

* Peut surcharger des ensembles de données particulièrement bruyants,
* Pour les données comprenant des variables prédictives catégorielles avec différents nombres de niveaux, les forêts aléatoires sont biaisées en faveur des prédicteurs avec plus de niveaux. Par conséquent, les scores d'importance variable de la forêt aléatoire ne sont pas toujours fiables pour ce type de données.

## Generalized Boosting Model

Ces modèles sont une combinaison de deux techniques: les algorithmes d'arbre de décision et les méthodes de boosting. Les modèles de boosting généralisés s'adaptent à plusieurs reprises à de nombreux arbres de décision pour améliorer la précision du modèle. Alors que les forêts aléatoires construisent un ensemble d'arbres indépendants profonds, les GBM construisent un ensemble d'arbres successifs peu profonds et faibles, chaque arbre apprenant et améliorant le précédent. 

Pour chaque nouvelle arborescence du modèle, un sous-ensemble aléatoire de toutes les données est sélectionné à l'aide de la méthode de boosting. Pour chaque nouvel arbre du modèle, les données d'entrée sont pondérées de telle sorte que les données mal modélisées par les arbres précédents ont une probabilité plus élevée d'être sélectionnées dans le nouvel arbre. 

Cela signifie qu'après l'ajustement du premier arbre, le modèle tiendra compte de l'erreur de prédiction de cet arbre pour l'ajustement de l'arbre suivant, etc. Cette approche séquentielle est unique au boosting.

On a choisi ce modèle en raison de sa bonne réputation dans le monde ML.

Pour ce modèle, on a constaté que le `preprocess` n'est pas réalisable, nous avons donc tourné le modèle sans `preprocess`.

```{r model_gbm, echo = FALSE, results = 'hide'}
gbmGrid <- data.frame(n.trees = 50,
                      interaction.depth = 1,
                      shrinkage = 0.1 ,
                      n.minobsinnode = 10
                      )

model_gbm <- train(
  Taille~., aEtudier,
  tuneGrid = gbmGrid,
  metric = "RMSE",
  method = "gbm",
  # preProcess ='scale', #ne marche pas
  # tuneLength=5,
  trControl = objControl,
)

```

Le moyen d'erreur `RMSE` obtenu à partir de `gbm` sur nos données et le `rsquared` :

```{r plotmodel_gbm1, echo = FALSE}
model_gbm
```

Après plusieurs exécutions, nous avons pu choisir ce `tuneGrid` qui diminue l'influence des prédicteurs non utiles jusqu'à -50 pour 1000.

```{r plotmodel_gbm2, echo = FALSE}
model_gbm$finalModel
```

Nous pouvons voir dans ce graphique, les variables les plus influentes:

```{r plotmodel_gbm3, echo = FALSE}
#Les 15 variables qui influencent le plus la décision de 
#prédiction de ce modèle:
plot(varImp(model_gbm,scale=F),top = 15)
```

### AVANTAGES

* Peut être utilisé avec une variété de types de réponses (binôme, gaussien, poisson),
Stochastique, qui améliore les performances prédictives
* Le meilleur ajustement est automatiquement détecté par l'algorithme,
* Le modèle représente l'effet de chaque prédicteur après avoir pris en compte les effets des autres prédicteurs,
* Robuste aux valeurs manquantes et aux valeurs aberrantes.

### LIMITES

* Nécessite au moins 2 variables prédictives pour s'exécuter

## Evolutionnary Trees

Les modèles arborescents répartissent les données en groupes de présence ou d'absence de plus en plus homogènes en fonction de leur relation avec un ensemble de variables environnementales, les variables prédictives. L'arbre de classification unique est la forme la plus élémentaire d'un modèle d'arbre de décision. Comme son nom l'indique, les arbres de classification ressemblent à un arbre et se composent de trois types de nœuds différents, reliés par des bords dirigés (branches).

On a choisi ce modèle pour plus de clarté, sa certitude de prédiction et l'occasion de voir la forme d'arbre réalisée.

Pour ce modèle, on a trouvé que le `preprocess` optimale est `ica`.

```{r model_evtree, echo = FALSE, results = 'hide'}
# Adapter le modèle d'arbres à partir d'algorithmes génétiques evtree
# à notre données: 
evtreeGrid <- data.frame(alpha =1)
model_evtree <- train(
  Taille~., aEtudier,
  metric = "RMSE",
  tuneGrid= evtreeGrid,
  method = "evtree",
  preProcess ='ica',
  trControl = objControl
)
```

On peut voir cet arbre à la `root` remplacé par un `ica` ici:

```{r plotmodel_evtree1, echo = FALSE}
#La forme de l'arbre:
plot(model_evtree$finalModel)
```


Le moyen d'erreur `RMSE` obtenu à partir de `evtree` sur nos données et le `rsquared` :

```{r plotmodel_evtree2,echo = FALSE}
model_evtree
```

On peut voir dans ce graphique, les variables qui influencent le plus la décision:

```{r plotmodel_evtree3}
#Les 15 variables qui influencent le plus la décision de 
#prédiction de ce modèle:
plot(varImp(model_evtree,scale=F),top = 15)
```

### AVANTAGES

* Simple à comprendre et à interpréter,
* Peut gérer à la fois des données numériques et catégoriques,
* Identifier les interactions hiérarchiques entre les prédicteurs,
* Caractériser les effets de seuil des prédicteurs sur la présence d'espèces,
* Robuste aux valeurs manquantes et aux valeurs aberrantes.

### LIMITES

* Moins efficace pour les réponses d'espèces linéaires ou lisses en raison de l'approche par étapes,
* Nécessite de grands ensembles de données pour détecter les modèles, en particulier avec de nombreux prédicteurs,
* Très instable: de petits changements dans les données peuvent changer considérablement l'arbre,
* Prend beaucoup de temps pour tourner.

## Neural Network

Les réseaux de neurones sont un ensemble d'algorithmes, modélisés librement d'après le cerveau humain, qui sont conçus pour reconnaître les modèles. Ils interprètent les données sensorielles à travers une sorte de perception machine, d'étiquetage ou de regroupement des entrées brutes. Les motifs qu'ils reconnaissent sont numériques, contenus dans des vecteurs, dans lesquels toutes les données du monde réel, que ce soit des images, du son, du texte ou des séries temporelles, doivent être traduites.

On a choisi ce modèle par curiosité pour l'essayer sur nos données après avoir passé beaucoup de temps avec lui en python et remarqué sa précision. Mais on n'a pas eu le temps d'entrer dans les détails et de spécifier les types de couches.

Pour ce modèle, on a trouvé que le `preprocess` optimale est `pca` et comme méthode `nnet` représentant les modèles du réseaux de neurones.

```{r model_nnet, echo = FALSE, results = 'hide'}
# nnetGrid <- model_nnet$bestTune
nnetGrid <- expand.grid(size = 1,
                         decay = 1e-04)

model_nnet <- train(
  Taille~., aEtudier,
  method="nnet",
  trControl=objControl,
  MaxNWts = 10000,
  tuneGrid = nnetGrid,
  preProcess ='pca',
  trace=TRUE,
  maxit=50,
  linout = TRUE,
  allowParallel = TRUE
  )
```

Le moyen d'erreur `RMSE` obtenu à partir de `nnet` sur nos données et le `rsquared` :

```{r plotmodel_nnet2, echo = FALSE}
#Montrez le modèle sur un graphique:
model_nnet
```


On peut voir dans ce graphique, les variables qui influencent le plus la décision:

```{r plotmodel_nnet3}
#Les 15 variables qui influencent le plus la décision de 
#prédiction de ce modèle:
plot(varImp(model_nnet,scale=F),top = 15)
```

### AVANTAGES
* Stockage d'informations sur l'ensemble du réseau.
* Capacité à travailler avec des connaissances incomplètes.
* Avoir une tolérance aux pannes.


### LIMITES
* Dépendance du hardware
* Comportement inexpliqué du réseau (couches invisibles)


# Création d'un objet `resamples`

Maintenant qu'on a ajusté les modèles à notre ensemble de données, il est temps de comparer leurs prédictions hors échantillon et de choisir celle qui est la meilleure dans notre cas.

Nous pouvons le faire en utilisant, et selon `caret`, la fonction` resamples () `:

```{r resamps, echo = FALSE, results = 'hide'}
# Création d'une liste de modèles:
model_list <- list(
  glmnet = model_glmnet,
  ranger = model_ranger,
  evtree = model_evtree,
  gbm = model_gbm,
  neuralnetwork = model_nnet
)
```

```{r resampsL, echo = FALSE, results = 'hide'}
# Insérez la liste des modèles dans les resamples ():
resamps <- resamples(model_list)
resamps
```

On peut voir dans ce qui suit la distribution des modèles par rapport à leur valeur de `RMSE`, `MAE` et `Rsquared`:
```{r sumresamps, echo = FALSE}
# Résumez les résultats
summary(resamps)
```

# Comparaison graphiques

Pour que la comparaison entre les modèles soit facilement visible, on affiche les distributions de précision prédictive dans des diagrammes:

## Boîte à Moustaches

```{r bwplot, echo = FALSE}
# Création d'une boîte à moustache de points RMSE et Rsquared:
bwplot(resamps, metric = "RMSE")
bwplot(resamps, metric = "Rsquared")
```

On peut voir sur ce diagramme de boîte à moustaches des `RMSE` que `ranger` a une médiane plus petites que celle des 4 autres modèles alors qu'une médiane plus grande des `rsquared`.

## DotPlot

```{r dotplot, echo = FALSE}
# Creation du DotPlot des points du RMSE et Rsquared:
dotplot(resamps, metric = "RMSE")
```

Ce diagramme nous montre les mêmes informations de la boîte à moustaches mais on ne voit que la moyenne des `RMSE` qui semble plus petite pour `ranger` que les autres.

## Nuage de Points

```{r xyplot2, echo = FALSE}
resampsF <- resamples(list(ranger=model_ranger,glmnet=model_glmnet))
xyplot(resampsF, metric = "RMSE")
xyplot(resampsF, metric = "Rsquared")
```

Enfin, selon le `scattertrot` ci-dessus, nous pouvons comparer directement les `RMSE` sur les 10 plis `cv`. On voit donc que la plupart du temps, `ranger` avait le plus petit` RMSE` que `glmnet` et le plus grand `Rsquared`. En plus, les 2 modèles sont totalement différents, puis comparables.

# Prédiction

On peut voir sur les diagrammes et les résultats ci-dessus, que le modèle `ranger` est le plus approprié de ces 4 modèles étudiés et donc, nous choisirons ce modèle pour la prédiction des résultats de la 6ème étude:

```{r predicted, echo = FALSE}
## Prédiction1
# Prédire la taille des patients du 6eme étude avec le 
# modèle choisit
Prediction_s6 <- data.frame(Predicted = predict(model_ranger,aPredire))

#Affichage des prédictions
Prediction_s6

```


# Modèle Proposer
Après être parvenu à cette conclusion et prédire les tailles des individus avec `ranger` comme modèle en raison de son `RMSE` plus petit que les autres modèles.

On va faire des graphiques discutant la distribution des prédictions de ces modèles et leur certitude:

## Generalized Linear Model

```{r plotmodel_glmnet4, echo = FALSE}
# glmnet
model_glmnet <- train(
  Taille~., aEtudier,
  # tuneGrid= glmnet_grid,
  metric = "RMSE",
  method = "glmnet",
  preProcess ='range',
  trControl = objControl
)
Prediction_glmnet <-predict(model_glmnet,aPredire)
boxplot(Prediction_glmnet)

```

De ce graphique, on peut déduire que ce modèle est réaliste en utilisant `range` comme `preprocess` (qui s'est avéré être le prétraitement le plus distribué du modèle) parce que les prédictions sont réparties entre un intervalle plus grand. 

```{r occglm, echo = FALSE}
occurences<-table(unlist(Prediction_glmnet))
head(occurences)
```

Bien que nous puissions également voir que les prédictions sont distinctes les unes des autres.

## RainForest

```{r plotmodel_ranger4, echo = FALSE}
# rainforest
Prediction_ranger <-predict(model_ranger,aPredire)
boxplot(Prediction_ranger)

```

De ce graphique, on peut déduire que ce modèle a une distribution dans un très petit intervalle ce qui nous donne l'impression de ne pas être sûr de ce modèle.

```{r occrf, echo = FALSE}
occurences<-table(unlist(Prediction_ranger))
head(occurences)
```

Bien que nous puissions également voir que les prédictions sont distinctes les unes des autres.

## Generalized Boosting Model

```{r plotmodel_gbm4, echo = FALSE}
Prediction_gbm <-predict(model_gbm,aPredire)
boxplot(Prediction_gbm)
```

De ce graphique, on peut déduire que ce modèle a une distribution dans un petit intervalle, mais plus grand que celui de `ranger`, donc plus réaliste.

```{r occgbm, echo = FALSE}
occurences<-table(unlist(Prediction_gbm))
head(occurences)
```

Les prédictions sont distinctes les unes des autres.

## Evolutionnary Trees

```{r plotmodel_evtree4, echo = FALSE}
# evtree
Prediction_evtree <-predict(model_evtree,aPredire)
boxplot(Prediction_evtree)

```

De ce graphique, on peut déduire que ce modèle a une très mauvaise distribution des prédictions de valeurs continues avec une seule valeur comme prédictions pour tous les patients.

```{r occevtree, echo = FALSE}
occurences<-table(unlist(Prediction_evtree))
head(occurences)
```


## Neural Network

```{r plotmodel_nnet4, echo = FALSE}
#neural network
Prediction_nnet <-predict(model_nnet,aPredire)
boxplot(Prediction_nnet)

```

De ce graphique, on peut déduire que ce modèle a une très mauvaise distribution des prédictions de valeurs continues.

```{r occnnet, echo = FALSE}
occurences<-table(unlist(Prediction_nnet))
head(occurences)
```

Bien qu'il existe des prédictions qui ont plusieurs occurrences, ce qui nous donne l'impression de ne pas être sûr de ce modèle.

Je n'ai peut-être pas bien utilisé ce modèle et je n'ai peut-être pas eu le temps d'essayer plus en connaissant son potentiel, je suis sûr qu'il peut faire mieux.

## Prédiction 2

En regardant la distribution des prédictions, on peut voir que `glmnet` et `gbm` sont les plus appropriés et les plus fiables de ces résultats car il ont une distribution très realistes des données et si on revient à la comparaison des modèles on peut voir que `glmnet` a un `RMSE` plus petit que `gbm` .

Et, par conséquent, je suggère `glmnet` pour calculer la prédiction des tailles comme une deuxième option.

```{r predicted2, echo = FALSE}
## prediction 2
# Prédire la taille des patients de la 6e étude 
#avec le modèle choisi
Prediction2_s6 <- data.frame(Predicted2 = predict(model_glmnet,aPredire))

#Affichage des prédictions:
Prediction2_s6

```

À mon avis, les modèles arborescents ont prédit à petits intervalles, parfois avec des valeurs similaires, et c'est peut-être une faiblesse des modèles arborescents à prédiction de type régression.

# Limites Possibles

D'un point de vue biologique, Il n'est pas facile de prédire l'avenir et ce qui limite la prédiction est l'ambiguïté sur le future et l'incertitude. On peut également voir que les `RMSE` sont un peu gros (à peu près égaux à 6) tandis que les `rsquared` sont très petits avec des différences négligeables entre les modèles. Cela peut être dû à des prédictions concernant l'être humain qui ne peut être prédit.

De plus, 600 patients ne constituent pas un grand échantillon et ne ciblent pas une grande majorité avec seulement 1000 allèles des milliers trouvés dans le corps humain. Nous n'avons même pas de description de ces allèles pour savoir s'il s'agit des allèles responsables d'affecter la taille d'un être humain. 

En n'oubliant pas les maladies qui provoquent l'arrêt de la croissance comme par exemple celle d'un cas rare de carence en hormone de croissance dont Lionel Messi a souffert. Mais ce sont des cas particuliers.

L'`ICA` comme `preprocess` est notablement plus fort que les autres la plupart du temps mais il est parfois in-interprétable.

Peut-être que si on a eu plus de temps et moins de projets à delivrer cette dernière semaine, on aura pu essayer différents ensembles avec différents modèles génétiques.

# Conclusion

Pour conclure, j'espère avoir été clair et avoir bien discuté et interprété les problèmes et les procédures pour arriver à ces 2 prédictions choisies après avoir bien compris ce que je fais et ce que j'ai écrit.

J'ai vraiment aimé travailler avec ces outils (ce qui est évident je pense). Je voudrais avoir des remarques, qui comptent plus pour moi que les notes, pour m'améliorer et rester sur la bonne voie et je serai à votre disposition pour toutes sortes de questions.

                                          La fin.
