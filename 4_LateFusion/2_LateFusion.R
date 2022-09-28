########################################################
### TRAIN LATE FUSION MODEL (FFPE+RNA)
#########################################################
### This script trains the multimodal late fusion model 
### - input: 
###          - FFPE model scores (from 3_HistoPath_savescore.py)
###          - RNA model scores (from 2_GeneExpress_savescore.py)
### - output:  Concatenated FFPE and RNA model scores for Late Fusion model
###############################################################################
###############################################################################
### Example command
### $ Rscript 2_LateFusion.R
###################################################
###################################################

### Set Environment
####################
suppressMessages(library(data.table))
suppressMessages(library(survival))
suppressMessages(library(glmnet))
suppressMessages(library(survcomp))
suppressMessages(library(tictoc))

### Train Model
################
### Read in data
score_train <- read.csv('combined_score_train.csv')
score_val <- read.csv('combined_score_val.csv')

### Train late Fusion Model
obj_train <- Surv(score_train$survival_months, score_train$vital_status)
x_train <- score_train[,c(2,6)]
cv.fit <- cv.glmnet(as.matrix(x_train), obj_train, family="cox", maxit = 4242)

## Run on Training Set
x_train <- score_train[,c(2,6)]
pred_train <- predict(cv.fit, newx = as.matrix(x_train),s='lambda.min')   

## Run on Validation Set
x_val <- score_val[,c(2,6)]
pred_val <- predict(cv.fit, newx = as.matrix(x_val),s='lambda.min')

## Export scores
score_train$score <- pred_train
score_val$score <- pred_val

write.csv(score_train, "model_late_train.csv", row.names = FALSE)
write.csv(score_val, "model_late_val.csv", row.names = FALSE) 