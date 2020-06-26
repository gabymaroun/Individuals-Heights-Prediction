####### Projet 2 : Prédiction de la taille à l’aide des génotypes individuels #######
####### Gaby Maroun #######

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

# Chargeant les données du fichier DATA_projet2.RData
load(file = "C:/Users/Gaby\'s/Desktop/Semestre 2 UPPA/Machine Learning/Projet 2/DATA_PROJET2.RData")

# Mettant les données en un table de type data.frame
df<-data.frame(DATA_projet2)
# View(df)

# Vérifier s'il y a des cases de données N/a ou vide 
#pour utiliser preprocessing
sum(is.na(df))

# Construire une table de données comme notre base d'étude en remplissant la première
# colonne des statuts de maladies des patients de chaques études,
# l'une au-dessus de l'autre
aEtudier <- NULL
for(i in 1:5){
  aEtudier <- (Taille =c(aEtudier,df[,6000+i]))
}

aEtudier <-data.frame(Taille = aEtudier)

# Construire une table de données en remplissant la première colonne en 0
# initialement comme taille des patients de la 6eme étude:
aPredire<- NULL
for(i in 1:120){
  aPredire<-append(aPredire,0)
}
aPredire <- data.frame(Taille = aPredire)


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


#description du aEtudier
skimmed <- skim_to_wide(aEtudier)
head(skimmed)

# # Définir une graine aléatoire afin que notre travail soit reproductible:
set.seed(42)

# Création des 10 plis
myFolds <- createFolds(aEtudier$Taille, k = 10)
# summary(myFolds)


# Création du reUsable trainControl object pour regression
objControl <- trainControl(method='cv', 
                           index = myFolds,
                           returnResamp='final')

# RFE 
# set.seed(100)
options(warn=-1)

# décomposition des subsets des variables à comparer
subsets <- c(#1:lm
  100,200.300,400,500,600,700,800,900,1000)

# dans functions, on choisit la methode
ctrl <- rfeControl(functions = lmFuncs,#rfFuncs
                   method = "cv",
                   index = myFolds,
                   verbose = FALSE)

lmProfile <- rfe(Taille ~., aEtudier,
                 sizes = subsets,
                 rfeControl = ctrl)

lmProfile

# Dans ce qui suit, on va utiliser que les choix optimaux


# Adaptez le modèle glmnet à nos données:
# Précision des alpha (Ridge regression(alpha = 0),Lasso regression(or alpha = 1))
# et lambda (taille du pénalité) pour empêcher l'overfitting.
# glmnet_grid = model_glmnet$bestTune
glmnet_grid <- data.frame(alpha = 1,
                           lambda = 0.5359083)

model_glmnet <- train(
  Taille~., aEtudier,
  tuneGrid= glmnet_grid,
  metric = "RMSE",
  method = "glmnet",
  preProcess ='pca',
  trControl = objControl
)
model_glmnet
# plot(model_glmnet)


#Montrez le modèle sur un graphique:
plot(model_glmnet$finalModel)

#Les 15 variables qui influencent le plus la décision de 
#prédiction de ce modèle:
plot(varImp(model_glmnet,scale=F),top = 15)




#Adaptez le modèle ranger représentant le type de modèle RainForest à nos données:
# Création d'un tuneGrid pour le modèle
# rangerGrid <- model_ranger$bestTune
rangerGrid <- data.frame(.mtry =2, #c(2,3,5,7,10,11,14,19,27,37,52,73,101,140,194,270,374,519,721,1000),
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
model_ranger
# model_ranger$finalModel


# Adapter le modèle d'arbres à partir d'algorithmes génétiques evtree
# à notre données: 
# evtreeGrid <- model_evtree$bestTune
evtreeGrid <- data.frame(alpha =1)
model_evtree <- train(
  Taille~., aEtudier,
  metric = "RMSE",
  tuneGrid= evtreeGrid,
  method = "evtree",
  preProcess ='ica',
  trControl = objControl
)
model_evtree


summary(model_evtree$finalModel)

#La forme de l'arbre:
plot(model_evtree$finalModel)

#Les 15 variables qui influencent le plus la décision de 
#prédiction de ce modèle:
plot(varImp(model_evtree,scale=F),top = 15)


# Adaptez le modèle gbm à nos données:
# gbmGrid <- model_gbm$bestTune
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
model_gbm
# summary(model_gbm)


#Montrez le modèle sur un graphique:
model_gbm$finalModel


#Les 15 variables qui influencent le plus la décision de 
#prédiction de ce modèle:
plot(varImp(model_gbm,scale=F),top = 15)

# Neural Network

# nnetGrid <- model_nnet$bestTune
nnetGrid <- expand.grid(size = 1,
                         decay = 1e-04)

# maxNWts est fixé et à 10000 pour rapidifier l'execution du modèle
# une iteration max de 50 est suffisante 
# linout pour regression 
# allowparallel  pour le calcule en parallele des processus

model_nnet <- train(
  Taille~., aEtudier,
  method = "nnet",
  trControl=objControl,
  MaxNWts = 10000,
  tuneGrid = nnetGrid,
  preProcess ='pca',
  trace=TRUE,
  maxit=50,
  linout = TRUE,
  allowParallel = TRUE
  )
model_nnet
# plot(model_nnet$finalModel)
plot(varImp(model_nnet,scale=F),top = 15)


# Création d'une liste de modèles:
model_list <- list(
  glmnet = model_glmnet,
  ranger = model_ranger,
  evtree = model_evtree,
  gbm = model_gbm,
  neuralnetwork = model_nnet
)

# Insérez la liste des modèles dans les resamples ():
resamps <- resamples(model_list)
resamps

# Résumez les résultats
summary(resamps)

# Création d'une boîte à moustache des points du RMSE et Rsquared :
bwplot(resamps, metric = "RMSE")
bwplot(resamps, metric = "Rsquared")

# Création du DotPlot des points du RMSE et Rsquared:
dotplot(resamps, metric = "RMSE")
dotplot(resamps, metric = "Rsquared")

# # Création du diagramme des densité de points du RMSE et Rsquared:
# densityplot(resamps, metric = "RMSE")
# densityplot(resamps, metric = "Rsquared")

# Création du nuage des points du RMSE et Rsquared:
resampsF <- resamples(list(glmnet=model_glmnet,gbm=model_gbm))
xyplot(resampsF, metric = "RMSE")
xyplot(resampsF, metric = "Rsquared")


resampsF2 <- resamples(list(ranger=model_ranger,nnet=model_nnet))
xyplot(resampsF, metric = "RMSE")
xyplot(resampsF, metric = "Rsquared")

## Prédiction1
# Prédire le statut des maladies des patients du 6eme étude avec le 
# modèle choisit
Prediction_s6 <- data.frame(Predicted = predict(model_ranger,aPredire))


#Affichage des prédictions
View(Prediction_s6)

# Sauvegarder les prédictions dans un fichier .RData
save(Prediction_s6, file = "Projet2_Maroun_Gaby_prediction.RData")

## prediction 2
# Distribution des prédictions de chaque modèle et leur certitude:
# Distribution des probabilités des 'Non' après prédictions 

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
occurences<-table(unlist(Prediction_glmnet))
occurences

# rainforest
Prediction_ranger <-predict(model_ranger,aPredire)
boxplot(Prediction_ranger)
occurences<-table(unlist(Prediction_ranger))
head(occurences)

# evtree
Prediction_evtree <-predict(model_evtree,aPredire)
boxplot(Prediction_evtree)
occurences<-table(unlist(Prediction_evtree))
occurences

# gbm
Prediction_gbm <-predict(model_gbm,aPredire)
boxplot(Prediction_gbm)
occurences<-table(unlist(Prediction_gbm))
occurences

#neural network
Prediction_nnet <-predict(model_nnet,aPredire)
boxplot(Prediction_nnet)
occurences<-table(unlist(Prediction_nnet))
occurences

#prediction
# Prédire l'état de maladie des patients de la 6e étude 
#avec le modèle choisi
Prediction2_s6 <- data.frame(Predicted2 = predict(model_glmnet,aPredire))

#Affichage des prédiction
View(Prediction2_s6)

#Comparaison visuel entre les prédictions des 2 modèles:
Comparaison <-data.frame(c(Prediction_s6,Prediction2_s6))

View(Comparaison)

#qui assure notre choix 
# Predicted2_Prob = predict(model_glmnet,aPredire)
# # , type='class')
# plot(Predicted2_Prob)

# Sauvegarder les prédictions dans un fichier .RData
save(Prediction2_s6, file = "Projet2_Maroun_Gaby_prediction2.RData")

