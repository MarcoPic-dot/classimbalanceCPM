

########################################################## Description of the data

# Data used in this analysis are from the International Stroke Trial database (https://datashare.ed.ac.uk/handle/10283/124) which is made available 
# under the ODC Attribution License (https://datashare.ed.ac.uk/bitstream/handle/10283/124/license_text?sequence=12&isAllowed=y). The file used
# is "IST_corrected.csv".
  
# Variable description from Sandercock et al. 2011 (https://trialsjournal.biomedcentral.com/articles/10.1186/1745-6215-12-101)

# We will use the following variables:

# Predictors
#   -   RDELAY Delay between stroke and randomisation in hours
#   -   RCONSC Conscious state at randomisation (F - fully alert, D - drowsy, U - unconscious)
#   -   SEX M = male; F = female
#   -   AGE Age in years
#   -   RSBP Systolic blood pressure at randomisation (mmHg)
#   -   Score obtained as sum of several variables with this rule "Y=1, C=0, N=0"
#         -   RDEF1 Face deficit (Y/N/C=can't assess)
#         -   RDEF2 Arm/hand deficit (Y/N/C=can't assess)
#         -   RDEF3 Leg/foot deficit (Y/N/C=can't assess)
#         -   RDEF4 Dysphasia (Y/N/C=can't assess)
#         -   RDEF5 Hemianopia (Y/N/C=can't assess)
#         -   RDEF6 Visuospatial disorder (Y/N/C=can't assess)
#         -   RDEF7 Brainstem/cerebellar signs (Y/N/C=can't assess)
#         -   RDEF8 Other deficit (Y/N/C=can't assess)
#   -   RXASP Trial aspirin allocated (Y/N)
#   -   RXHEP Trial heparin allocated (M/L/N). The terminology for the allocated dose of unfractioned heparin changed slightly from the pilot to the main study. Patients were allocated either 12500 units subcutaneously twice daily (coded as H in the pilot and M in the main trial), 5000 units twice daily (coded as L throughout) or to 'avoid heparin' (coded as N throughout).
#   Outcome
#   -   Death within 14 days, according to
#         -   SET14D Know to be dead or alive at 14 days (1 = Yes, 0 = No); this does not necessarily mean that we know outcome at 6 months
#         -   ID14 Indicator of death at 14 days (1 = Yes, 0 = No)


########################################################## Import data

# Load packages
library(PRROC)
library(pROC)
library(boot)
library(arsenal)
library(tidyverse)
library(caret)

# Load dataset
setwd(choose.dir())
dat <- read_csv("IST_corr_31012024.csv")

########################################################## Data management

# Only for two individuals the death/alive status was unknown at 14 days.
table(dat$SET14D)
# We will assume they were alive, and use ID14 as outcome variable
dat$ID14 <- factor(ifelse(dat$ID14==1,"Yes","No"), levels=c("No","Yes"), ordered = T)

# Build the deficit score
dat$DEF_SCORE <- (dat$RDEF1=="Y") + (dat$RDEF2=="Y") + (dat$RDEF3=="Y") + (dat$RDEF4=="Y") + (dat$RDEF5=="Y") + (dat$RDEF6=="Y") + (dat$RDEF7=="Y") + (dat$RDEF8=="Y")

# Merge H and M in the RXHEP variable
table(dat$RXHEP)
dat$RXHEP[dat$RXHEP=="H"] <- "M"

# Create an ID variable
dat$ID <- 1:nrow(dat)

# Keep only relevant columns
dat <- dat %>% select(ID, RDELAY, RCONSC, SEX, AGE, RSBP, DEF_SCORE, RXASP, RXHEP, ID14)

# Split the dataset into training and test datasets (2:1)
set.seed(02022024)
dat$TEST <- rbinom(nrow(dat), size=1, prob=1/3)

dat_train <- dat %>% filter(TEST==0)
dat_test <- dat %>% filter(TEST==1)

# Create vector with predictors
predictors <- c("RDELAY", "RCONSC", "SEX", "AGE", "RSBP", "DEF_SCORE", "RXASP", "RXHEP")

########################################################## Descriptive statistics

tableby(TEST ~ RDELAY + RCONSC + SEX + AGE + RSBP + DEF_SCORE + RXASP + RXHEP + ID14, data=dat, numeric.stats=c("Nmiss","meansd","median","range","q1q3")) %>% summary(test=FALSE, total=TRUE, digits=0)

########################################################## Train the "original" models

# Fit the logistic regression on the training data
formula <- as.formula(paste("ID14", paste(predictors ,collapse="+"),sep="~"))
glm_o <- glm(formula, family=binomial, data=dat_train)
knitr::kable(broom::tidy(glm_o))

# Train and tune the classification tree
train_control <- trainControl(method = "cv",
                              number = 10,
                              summaryFunction = multiClassSummary,
                              classProbs = TRUE,
                              verboseIter = FALSE,
                              selectionFunction = "best")

# Change all parameters to ensure that the tree grows
rpart_control <- rpart::rpart.control(minbucket = 5, minsplit = 2, cp=0)

# Change maxdepth during tuning
tune_grid <- data.frame(maxdepth = 5:20)

# Use logLoss as metric
cart = train(x=as.data.frame(dat_train[,predictors]), 
             y=dat_train$ID14, 
             method="rpart2", trControl=train_control, metric="logLoss", control= rpart_control, tuneGrid=tune_grid)

# Choose the best model
tr_o <- cart$finalModel
rattle::fancyRpartPlot(tr_o,cex=0.5)

########################################################## Train the "naive" models (oversampling)

# Oversampling
index_cases <- which(dat_train$ID14=="Yes")
index_controls <- which(dat_train$ID14=="No")
index_s <- c(index_controls, sample(index_cases, size = sum(dat_train$ID14=="No"), replace=TRUE))
dat_s <-dat_train[index_s,]
table(dat_s$ID14)

# Fit the logistic regression on the balanced dataset
glm_s <- glm(formula, family=binomial, data=dat_s)
knitr::kable(broom::tidy(glm_s))

# Train and tune the classification tree on the balanced dataset
# Increase minbucket to avoid overfitting since one individual now counts for several units, all other settings stay the same
rpart_control_s <- rpart::rpart.control(minbucket = 100, minsplit = 2, cp=0)

cart_s = train(x=as.data.frame(dat_s[,predictors]), 
               y=dat_s$ID14, 
               method="rpart2", trControl=train_control, metric="logLoss", control= rpart_control_s, tuneGrid=tune_grid)
tr_s <- cart_s$finalModel
rattle::fancyRpartPlot(tr_s,cex=0.5)

########################################################## Create "corrected" predictions with the plugin estimator

# Plugin estimator function
Pr_Y <- mean(dat_train$ID14=="Yes")
plugin <- function(pred_s) {inv.logit( logit(pred_s) + logit(Pr_Y) )}

# Visualize the function
x=seq(0,1,by=0.001)
plot(x, plugin(x), xlab="Probability from the naive model", ylab="Corrected probabilities")

########################################################## Obtain predictions in the test dataset

# Obtain predictions from original models
dat_test$pred_glm_o <- predict(glm_o, newdata = dat_test, type="response")
dat_test$pred_tr_o <- predict(tr_o, type="prob", newdata = dat_test)[,2]
dat_test$pred_tr_o[dat_test$pred_tr_o==0] <- 0.5/(nrow(dat_train))
dat_test$pred_tr_o[dat_test$pred_tr_o==1] <- 1 - 0.5/(nrow(dat_train))

# Obtain predictions from naive models
dat_test$pred_glm_s <- predict(glm_s, newdata = dat_test, type="response")
dat_test$pred_tr_s <- predict(tr_s, type="prob", newdata = dat_test)[,2]
dat_test$pred_tr_s[dat_test$pred_tr_s==0] <- 0.5/(nrow(dat_s))
dat_test$pred_tr_s[dat_test$pred_tr_s==1] <- 1 - 0.5/(nrow(dat_s))

# Obtain predictions correcting the naive predictions with the plugin function
dat_test$pred_glm_plugin <- plugin(dat_test$pred_glm_s)
dat_test$pred_tr_plugin <- plugin(dat_test$pred_tr_s)

########################################################## Assess performance in the test dataset

# Compare AUC
dat_test %>% select(starts_with("pred_")) %>% 
  rename("original logistic"="pred_glm_o", "original class. tree"="pred_tr_o", "naive logistic"="pred_glm_s", "naive class. tree"="pred_tr_s", "corrected logistic"="pred_glm_plugin", "corrected class. tree"="pred_tr_plugin") %>% 
  summarise_all(function (x) ci.auc(roc(dat_test$ID14=="Yes",x,direction="<"))) %>% mutate(term=c("lower_ci","point_estimate","upper_ci")) %>% 
  relocate(term) %>% knitr::kable()

# Define Integrated Calibration Index
ICI <- function(outcome, prediction, main, plot = FALSE) {
  loess <- predict (loess(outcome ~ prediction), se=T)
  P.calibrate <- loess$fit
  lci <-  loess$fit - qt(0.975,loess$df)*loess$se.fit
  uci <-  loess$fit + qt(0.975,loess$df)*loess$se.fit
  if(plot == TRUE){
    plot(outcome ~ prediction, main = main, xlim=c(0,1), ylim=c(0,1), xlab="Prediction", ylab="Outcome", pch=0.2, cex=0.5)
    polygon(x = c(prediction[order(prediction)], rev(prediction[order(prediction)])),
            y = c(lci[order(prediction)], 
                  rev(uci[order(prediction)])), col ='grey70', border = NA)
    lines(prediction[order(prediction)],P.calibrate[order(prediction)],col="red",lwd=1.5)
    abline(0,1, lty=2)
  }
  ( ICI <- mean (abs(P.calibrate - prediction)) )
}

# Compare ICI
par(mfrow=c(3,2))
dat_test %>% select(starts_with("pred_")) %>% 
  rename("original logistic"="pred_glm_o", "original class. tree"="pred_tr_o", "naive logistic"="pred_glm_s", "naive class. tree"="pred_tr_s", "corrected logistic"="pred_glm_plugin", "corrected class. tree"="pred_tr_plugin") %>% 
  summarise_all(function (x) ICI(dat_test$ID14=="Yes", x, main=deparse(substitute(x)), plot=TRUE)) %>% knitr::kable()

# Compare ROC and Precision-Recall curves
par(mfrow=c(2,2))
roc.curve(scores.class0=dat_test$pred_glm_o, weights.class0=dat_test$ID14=="Yes", curve=T) %>% plot(add=F, col="deeppink", main="ROC curves logistic regressions", auc.main=F, xlab="1-Specificity")
roc.curve(scores.class0=dat_test$pred_glm_s, weights.class0=dat_test$ID14=="Yes", curve=T) %>% plot(add=T, col="goldenrod")
roc.curve(scores.class0=dat_test$pred_glm_plugin, weights.class0=dat_test$ID14=="Yes", curve=T) %>% plot(add=T, col="dodgerblue4", lty=3)

roc.curve(scores.class0=dat_test$pred_tr_o, weights.class0=dat_test$ID14=="Yes", curve=T) %>% plot(add=F, col="deeppink", main="ROC curves class. trees", auc.main=F, xlab="1-Specificity")
roc.curve(scores.class0=dat_test$pred_tr_s, weights.class0=dat_test$ID14=="Yes", curve=T) %>% plot(add=T, col="goldenrod")
roc.curve(scores.class0=dat_test$pred_tr_plugin, weights.class0=dat_test$ID14=="Yes", curve=T) %>% plot(add=T, col="dodgerblue4", lty=3)

pr.curve(scores.class0=dat_test$pred_glm_o, weights.class0=dat_test$ID14=="Yes", curve=T) %>% plot(add=F, col="deeppink", main="Precision-Recall curves logistic regressions", auc.main=F, xlab="Recall (or Sensitivity)", ylab="Precision (or Positive Predictive Value)")
pr.curve(scores.class0=dat_test$pred_glm_s, weights.class0=dat_test$ID14=="Yes", curve=T) %>% plot(add=T, col="goldenrod")
pr.curve(scores.class0=dat_test$pred_glm_plugin, weights.class0=dat_test$ID14=="Yes", curve=T) %>% plot(add=T, col="dodgerblue4", lty=3)

pr.curve(scores.class0=dat_test$pred_tr_o, weights.class0=dat_test$ID14=="Yes", curve=T) %>% plot(add=F, col="deeppink", main="Precision-Recall curves class. trees", auc.main=F, xlab="Recall (or Sensitivity)", ylab="Precision (or Positive Predictive Value)")
pr.curve(scores.class0=dat_test$pred_tr_s, weights.class0=dat_test$ID14=="Yes", curve=T) %>% plot(add=T, col="goldenrod")
pr.curve(scores.class0=dat_test$pred_tr_plugin, weights.class0=dat_test$ID14=="Yes", curve=T) %>% plot(add=T, col="dodgerblue4", lty=3)

