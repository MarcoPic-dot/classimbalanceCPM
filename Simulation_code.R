library(pROC)
library(boot)
library(rpart)
library(tidyverse)

Nsim <- 500
low_prob <-  function(data) {0.5/(nrow(data))}
high_prob <- function(data) {1-0.5/(nrow(data))}

models=c("pred_glm_o","pred_tr_o","pred_glm_s","pred_tr_s","pred_glm_plugin","pred_tr_plugin")  

report <- expand.grid(N=c(1000,4000,8000), Nval=10000, p=6, corr_p=0.2, b0=-1.7, out_gen=c("tree","logistic"), balance=c("over","under"))
report$report_row <- 1:nrow(report)
report$Pr_Y_avg <- NA_real_

temp <- data.frame()

set.seed(1234)

for (i in 1:nrow(report)) { 

report_i <- expand.grid(sim=1:Nsim, models=models, metric=c("MAE","Bias","AUC")) %>% mutate(value=NA)
Pr_Y_i <- rep(NA_real_, Nsim)

for (sim in 1:Nsim) {

# create variance matrix for predictors
sigma_p <- matrix(report[i,"corr_p"], nrow= report[i,"p"], ncol= report[i,"p"])
diag(sigma_p) <- 1

# generate predictors
dat <- data.frame(id=1:report[i,"N"], mvtnorm::rmvnorm(n= report[i,"N"], mean= rep(0, report[i,"p"]) , sigma= sigma_p))
predictors <- paste("X",1:report[i,"p"],sep="")
# GGally::ggpairs(dat %>% select(-id))

# generate outcome
if (report[i, "out_gen"]=="logistic") {
  if (report[i,"p"]==6) { coeffs_out <- c(report[i,"b0"], 0.4, 0.2, 0.1, -0.2, -0.4, -0.6) }

  dat$true_prob <- inv.logit( as.matrix(cbind(1,dat[,predictors,drop=FALSE])) %*% coeffs_out )
}

if (report[i, "out_gen"]=="tree") {
  dat2 <- dat %>% mutate_at(vars(contains("X")), .funs = function(x) as.numeric(x>0)) 
  if (report[i,"p"]==6) {
   dat$true_prob <- inv.logit((dat2$X1==0)*(report[i,"b0"]-2.9)+dat2$X1*dat2$X2*(report[i,"b0"]+1.3)+dat2$X1*(dat2$X2==0)*(dat2$X3==0)*(report[i,"b0"]-0.75)+dat2$X1*(dat2$X2==0)*dat2$X3*(dat2$X4==0)*(report[i,"b0"])+dat2$X1*(dat2$X2==0)*dat2$X3*dat2$X4*dat2$X5*dat2$X6*(report[i,"b0"]+0.85)+dat2$X1*(dat2$X2==0)*dat2$X3*dat2$X4*dat2$X5*(dat2$X6==0)*(report[i,"b0"]+0.6)+dat2$X1*(dat2$X2==0)*dat2$X3*dat2$X4*(dat2$X5==0)*dat2$X6*(report[i,"b0"]+0.5)+dat2$X1*(dat2$X2==0)*dat2$X3*dat2$X4*(dat2$X5==0)*(dat2$X6==0)*(report[i,"b0"]+0.18))}
}
dat$Y <- rbinom(report[i,"N"], 1, prob=dat$true_prob)
# ggplot(dat,aes(x=true_prob, y=Y))+geom_point()+geom_smooth(method="loess")+xlim(0,1)+ylim(0,1)+geom_abline(intercept=0,slope=1,col="red")

# class imbalance correction
index_cases <- dat[dat$Y==1, "id"]
index_controls <- dat[dat$Y==0, "id"]
if (report[i,"balance"]=="over") { index_s <- c(index_controls, sample(index_cases, size = sum(dat$Y==0), replace=TRUE)) }
if (report[i,"balance"]=="under") { index_s <- c(index_cases, sample(index_controls, size = sum(dat$Y), replace=FALSE)) }
dat_s <-dat[index_s,]
# table(dat_s$Y)

# calculate Pr(Y=1)
Pr_Y_i[sim] <- mean(dat$Y)

# fit logistic regressions
formula <- as.formula(paste("Y", paste(predictors ,collapse="+"),sep="~"))
glm_o <- glm(formula, family=binomial, data=dat)
glm_s <- glm(formula, family=binomial, data=dat_s)

# train classification trees
control_tree <- rpart.control(minbucket = 10,minsplit = 2,cp=0.0001,maxdepth = 6,xval=0)
tr_o <- rpart(formula, data=dat, method="class", control=control_tree)
if (report[i,"balance"]=="over") {control_tree <- rpart.control(minbucket = 100,minsplit = 2,cp=0.0001,maxdepth = 6,xval=0)}
tr_s <- rpart(formula, data=dat_s, method="class", control=control_tree)
# rattle::fancyRpartPlot(tr_s,cex=0.3)

# plugin recalibration
plugin <- function(pred_s) {inv.logit( logit(pred_s) + logit(Pr_Y_i[sim]) )}
  
# generate validation dataset
dat_v <- data.frame(id=1:report[i,"Nval"], mvtnorm::rmvnorm(n= report[i,"Nval"], mean= rep(0, report[i,"p"]) , sigma= sigma_p))
# GGally::ggpairs(dat_v %>% select(-id))
if (report[i, "out_gen"]=="logistic") {
  dat_v$true_prob <- inv.logit( as.matrix(cbind(1,dat_v[,predictors,drop=FALSE])) %*% coeffs_out )
}
if (report[i, "out_gen"]=="tree") {
  dat_v2 <- dat_v %>% mutate_at(vars(contains("X")), .funs = function(x) as.numeric(x>0)) 
  if (report[i,"p"]==6) {
    dat_v$true_prob <- inv.logit((dat_v2$X1==0)*(report[i,"b0"]-2.9)+dat_v2$X1*dat_v2$X2*(report[i,"b0"]+1.3)+dat_v2$X1*(dat_v2$X2==0)*(dat_v2$X3==0)*(report[i,"b0"]-0.75)+dat_v2$X1*(dat_v2$X2==0)*dat_v2$X3*(dat_v2$X4==0)*(report[i,"b0"])+dat_v2$X1*(dat_v2$X2==0)*dat_v2$X3*dat_v2$X4*dat_v2$X5*dat_v2$X6*(report[i,"b0"]+0.85)+dat_v2$X1*(dat_v2$X2==0)*dat_v2$X3*dat_v2$X4*dat_v2$X5*(dat_v2$X6==0)*(report[i,"b0"]+0.6)+dat_v2$X1*(dat_v2$X2==0)*dat_v2$X3*dat_v2$X4*(dat_v2$X5==0)*dat_v2$X6*(report[i,"b0"]+0.5)+dat_v2$X1*(dat_v2$X2==0)*dat_v2$X3*dat_v2$X4*(dat_v2$X5==0)*(dat_v2$X6==0)*(report[i,"b0"]+0.18))}
}
dat_v$Y <- rbinom(report[i,"Nval"], 1, prob=dat_v$true_prob)
# ggplot(dat_v,aes(x=true_prob, y=Y))+geom_point()+geom_smooth(method="loess")+xlim(0,1)+ylim(0,1)+geom_abline(intercept=0, slope=1, col="red")

# get all predictions
dat_v$pred_glm_o <- predict(glm_o, type="response", newdata = dat_v) 
dat_v$pred_tr_o <- predict(tr_o, type="prob", newdata = dat_v)[,2]
dat_v$pred_tr_o[dat_v$pred_tr_o==1] <- high_prob(dat)
dat_v$pred_tr_o[dat_v$pred_tr_o==0] <- low_prob(dat)
dat_v$pred_glm_s <- predict(glm_s, type="response", newdata = dat_v) 
dat_v$pred_tr_s <- predict(tr_s, type="prob", newdata = dat_v)[,2]
dat_v$pred_tr_s[dat_v$pred_tr_s==1] <- high_prob(dat_s)
dat_v$pred_tr_s[dat_v$pred_tr_s==0] <- low_prob(dat_s)
dat_v$pred_glm_plugin <- plugin(dat_v$pred_glm_s)
dat_v$pred_tr_plugin <- plugin(dat_v$pred_tr_s)

MAE <- dat_v %>% select(starts_with("pred_")) %>% mutate_all(function (x) abs(x-dat_v$true_prob)) %>% 
  summarise_all(mean)
Bias <- dat_v %>% select(starts_with("pred_")) %>% mutate_all(function (x) logit(x)-logit(dat_v$true_prob)) %>% 
  summarise_all(mean)
AUC <- dat_v %>% select(starts_with("pred_")) %>% summarise_all(function (x) auc(roc(dat_v$Y,x,direction="<")))
if (!identical(names(Bias),models)) stop("models do no match")
if (!identical(names(MAE),models)) stop("models do no match")
if (!identical(names(AUC),models)) stop("models do no match")
report_i[report_i$sim==sim & report_i$metric=="MAE","value"] <- MAE %>% unlist() %>% unname()
report_i[report_i$sim==sim & report_i$metric=="Bias","value"] <- Bias %>% unlist() %>% unname()
report_i[report_i$sim==sim & report_i$metric=="AUC","value"] <- AUC %>% unlist() %>% unname()

rm(sigma_p,dat,predictors,index_cases,index_controls,index_s,dat_s,formula,glm_o,tr_o,glm_s,tr_s,plugin,dat_v,MAE,Bias,AUC,control_tree)
if (report[i, "out_gen"]=="tree") {rm(dat2, dat_v2)}else{rm(coeffs_out)}
}

temp <- temp %>% bind_rows(report_i %>% mutate(report_row=i))
report[i,"Pr_Y_avg"] <- mean(Pr_Y_i)

rm(report_i, Pr_Y_i)
}

report <- report %>% right_join(temp, by="report_row")

#rename for consistency with the manuscript
report$metric <- recode(report$metric, MAE="MAE",Bias="ME_logit",AUC="AUC")
report$out_gen <- ifelse(report$out_gen=="tree","Outcome generation mechanism 2 (tree)", "Outcome generation mechanism 1 (logistic)")
report$N <- paste("N development dataset = ", report$N)
report$models <- as.character(report$models)
report[report$models=="pred_glm_o", "models"] <- "original logistic"
report[report$models=="pred_tr_o", "models"] <- "original class. tree"
report[report$models=="pred_glm_s", "models"] <- "naive logistic"
report[report$models=="pred_tr_s", "models"] <- "naive class. tree"
report[report$models=="pred_glm_plugin", "models"] <- "corrected logistic"
report[report$models=="pred_tr_plugin", "models"] <- "corrected class. tree"
report$models <- factor(report$models, levels=c("naive logistic","corrected logistic","original logistic","naive class. tree","corrected class. tree","original class. tree"))

cols <- c("chocolate","brown4","coral1","blue","deepskyblue4","cadetblue1")


# graphs 

ggplot(report %>% filter(metric=="MAE", balance=="under"), aes(x=models, y=value, fill=models)) + geom_violin() + 
  facet_grid(out_gen~N) + ylab("MAE") + xlab("Models") + scale_fill_manual("Models",values=cols) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + ylim(0,0.4)
ggsave("Fig2.pdf",device="pdf", width=9,height=7)

ggplot(report %>% filter(metric=="ME_logit", balance=="under"), aes(x=models, y=value, fill=models)) + geom_violin() + 
  facet_grid(out_gen~N) + ylab("ME-logit") + xlab("Models") + scale_fill_manual("Models",values=cols) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + ylim(-3,3)
ggsave("Fig3.pdf",device="pdf", width=9,height=7)

ggplot(report %>% filter(metric=="AUC", balance=="under"), aes(x=models, y=value, fill=models)) + geom_violin() + 
  facet_grid(out_gen~N) + ylab("AUC") + xlab("Models") + scale_fill_manual("Models",values=cols) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+ ylim(0.5,0.9)
ggsave("Fig4.pdf",device="pdf", width=9,height=7)

ggplot(report %>% filter(metric=="MAE", balance=="over"), aes(x=models, y=value, fill=models)) + geom_violin() + 
  facet_grid(out_gen~N) + ylab("MAE") + xlab("Models") + scale_fill_manual("Models",values=cols) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+ ylim(0,0.4)
ggsave("FigS1.pdf",device="pdf", width=9,height=7) 

ggplot(report %>% filter(metric=="ME_logit", balance=="over"), aes(x=models, y=value, fill=models)) + geom_violin() + 
  facet_grid(out_gen~N) + ylab("ME-logit") + xlab("Models") + scale_fill_manual("Models",values=cols) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + ylim(-3,3)
ggsave("FigS2.pdf",device="pdf", width=9,height=7)

ggplot(report %>% filter(metric=="AUC", balance=="over"), aes(x=models, y=value, fill=models)) + geom_violin() + 
  facet_grid(out_gen~N) + ylab("AUC") + xlab("Models") + scale_fill_manual("Models",values=cols) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+ ylim(0.5,0.9)
ggsave("FigS3.pdf",device="pdf", width=9,height=7)

# statistics for text

report %>% filter(N=="N development dataset =  8000",sim==1) %>% group_by(out_gen) %>% summarise(mean(Pr_Y_avg)) %>% as.vector()

report %>% filter(out_gen=="Outcome generation mechanism 1 (logistic)", metric=="MAE",balance=="under") %>% group_by(models,N) %>% summarise(round(mean(value),3))
report %>% filter(out_gen=="Outcome generation mechanism 1 (logistic)", metric=="MAE",balance=="under") %>% group_by(N,models) %>% summarise(min(value)<0.20)

report %>% filter(out_gen=="Outcome generation mechanism 2 (tree)", metric=="MAE",balance=="under") %>% group_by(models,N) %>% summarise(round(mean(value),3))
report %>% filter(out_gen=="Outcome generation mechanism 2 (tree)", metric=="MAE",balance=="under") %>% group_by(N,models) %>% summarise(min(value)<0.15)

report %>% filter(out_gen=="Outcome generation mechanism 1 (logistic)", metric=="ME_logit",balance=="under") %>% group_by(models,N) %>% summarise(round(mean(value),3))
report %>% filter(out_gen=="Outcome generation mechanism 2 (tree)", metric=="ME_logit",balance=="under") %>% group_by(models,N) %>% summarise(round(mean(value),3))

report %>% filter(out_gen=="Outcome generation mechanism 1 (logistic)", metric=="AUC",balance=="under") %>% group_by(models,N) %>% summarise(round(mean(value),3))
report %>% filter(out_gen=="Outcome generation mechanism 2 (tree)", metric=="AUC",balance=="under") %>% group_by(models,N) %>% summarise(round(mean(value),3))

## oversampling

report %>% filter(out_gen=="Outcome generation mechanism 1 (logistic)", metric=="MAE",balance=="over") %>% group_by(models,N) %>% summarise(round(mean(value),3))
report %>% filter(out_gen=="Outcome generation mechanism 1 (logistic)", metric=="MAE",balance=="over") %>% group_by(N,models) %>% summarise(min(value)<0.20)

report %>% filter(out_gen=="Outcome generation mechanism 2 (tree)", metric=="MAE",balance=="over") %>% group_by(models,N) %>% summarise(round(mean(value),3))
report %>% filter(out_gen=="Outcome generation mechanism 2 (tree)", metric=="MAE",balance=="over") %>% group_by(N,models) %>% summarise(min(value)<0.15)

report %>% filter(out_gen=="Outcome generation mechanism 1 (logistic)", metric=="ME_logit",balance=="over") %>% group_by(models,N) %>% summarise(round(mean(value),3))
report %>% filter(out_gen=="Outcome generation mechanism 2 (tree)", metric=="ME_logit",balance=="over") %>% group_by(models,N) %>% summarise(round(mean(value),3))

report %>% filter(out_gen=="Outcome generation mechanism 1 (logistic)", metric=="AUC",balance=="over") %>% group_by(models,N) %>% summarise(round(mean(value),3))
report %>% filter(out_gen=="Outcome generation mechanism 2 (tree)", metric=="AUC",balance=="over") %>% group_by(models,N) %>% summarise(round(mean(value),3))


# table

report %>% group_by(N,out_gen,balance,models,metric) %>% 
  summarise(mean=round(mean(value),3),low=round(quantile(value,probs=0.025),3),high=round(quantile(value,probs=0.975),3)) %>% 
  ungroup() %>% mutate(value=paste(mean," \n (",low,"-",high,")"),gr=paste(models,metric)) %>% select(-mean,-high,-low,-models,-metric) %>% 
  spread(gr,value) %>% relocate(balance,out_gen,N,
                                "naive logistic AUC","corrected logistic AUC","original logistic AUC","naive class. tree AUC","corrected class. tree AUC","original class. tree AUC",
                                "naive logistic MAE","corrected logistic MAE","original logistic MAE","naive class. tree MAE","corrected class. tree MAE","original class. tree MAE",
                                "naive logistic ME_logit","corrected logistic ME_logit","original logistic ME_logit","naive class. tree ME_logit","corrected class. tree ME_logit","original class. tree ME_logit"
                                ) %>% arrange(out_gen, balance) %>% write.csv("Table.csv")
