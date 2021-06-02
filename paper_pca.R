#######################  PCA - GRADIENT BOOSTING ##################################################

rm(list = ls())
wd<-"C:/Users/bereu/OneDrive/Documents/Frankfurt/SCR1S21"
setwd(wd)
xsell <- get(load("xsell_loan.RData"))


# replacing missing values by 0....
xsell$inflows[is.na(xsell$inflows)] <- 0
xsell$outflows[is.na(xsell$outflows)] <- 0
xsell$volume_debit_6months[is.na(xsell$volume_debit_6months)] <- 0

# replacing missing values by mean
xsell$ext_city_size[is.na(xsell$ext_city_size)] <- round(mean(xsell$ext_city_size, na.rm = TRUE))
xsell$ext_house_size[is.na(xsell$ext_house_size)] <- round(mean(xsell$ext_house_size, na.rm = TRUE))
xsell$ext_purchase_power[is.na(xsell$ext_purchase_power)] <- round(mean(xsell$ext_purchase_power, na.rm = TRUE))
xsell$ext_car_power[is.na(xsell$ext_car_power)] <- round(mean(xsell$ext_car_power, na.rm = TRUE))
xsell$ext_share_new_cars[is.na(xsell$ext_share_new_cars)] <- round(mean(xsell$ext_share_new_cars, na.rm = TRUE))
xsell$ext_share_new_houses[is.na(xsell$ext_share_new_houses)] <- round(mean(xsell$ext_share_new_houses, na.rm = TRUE))
xsell$ext_living_duration[is.na(xsell$ext_living_duration)] <- round(mean(xsell$ext_living_duration, na.rm = TRUE))


# select only the numeric variables
xsell_numeric<-xsell[sapply(xsell, is.numeric)] 

# PCA Principal Components Analysis helps to reduce dimensions and to generate variables that are not correlated

pca <- princomp(xsell_numeric[, c("acad_title","age","calls","complaints","customer_tenure_months","directmails","inflows", 
                                  "last_acc_opening_days","logins_desktop","logins_mobile","member_get_member_active","member_get_member_passive",
                                  "nr_products","outflows","prod_mortgages","prod_brokerage","prod_pensionplan","prod_savings","relocations",
                                  "volume_debit","volume_debit_6months","ext_city_size","ext_purchase_power","ext_share_new_houses","ext_share_new_cars",
                                  "ext_car_power","ext_living_duration")], scores=TRUE)

#  "prod_loan" and "giro_mailing" deleted

# Add loadings to the churn dataset
xsell<- cbind(xsell,pca$scores)


### Build Gradient Boosting Model------------------------------------------ ####
library(gbm)

set.seed(12345) # fix random number generator seed for reproducibility
xsell_random <- xsell[order(runif(100000)),] #sort the data set randomly
xsell_valid <- xsell_random[1:20000, ]       # 20% / 2000 observations in the validation dataset
xsell_train <- xsell_random[20001:100000, ]   # 80% / 8000 in the training data set

model_orig <- xsell ~ acad_title + age  + nr_products  + inflows + logins_mobile + volume_debit + customer_tenure_months + logins_desktop

set.seed(1234)
gbm_orig <- gbm(formula=model_orig,		             # Model 
           distribution="bernoulli",     # Option needed for binary response variable
           data=xsell_train, 	           # data set
           n.trees=600,		               # Number trees / iterations
           interaction.depth=3,          # max depth of each tree
           shrinkage=0.01, 	             # Learning (shrinkage) rate
           n.minobsinnode=50,	           # Minimum number of obs per node
           bag.fraction = 0.5, 	           # size of random samples (if <1 you're also doing bagging)
           verbose=FALSE)	               # show in-between-steps TRUE/FALSE


#check results
gbm_orig
summary(gbm_orig)


# Generate predictions orig
xsell_valid$pred_gbm_orig <- predict(gbm_orig, newdata=xsell_valid, 
                                type="response", na.action=na.pass)

xsell_valid$pred_gbm_orig_factor <- factor(ifelse(xsell_valid$pred_gbm_orig>.1,1,0),labels=c("No Xsell","Xsell"))



### Check accuracy with the confusion matrix #### orig
library(caret)
conf_matrix_gbm_orig <- confusionMatrix(xsell_valid$pred_gbm_orig_factor,factor(xsell_valid$xsell,labels=c("No Xsell","Xsell")), 
                                   positive="Xsell",dnn = c("Prediction", "Actual Data"))
conf_matrix_gbm_orig

conf_matrix_gbm_orig$byClass["F1"]


### gbm with model pca

model_pca <- xsell ~ Comp.1 + Comp.2  + Comp.3  + Comp.4 + Comp.5 + Comp.6 + Comp.7 + Comp.8

set.seed(1234)
gbm_pca <- gbm(formula=model_pca,		             # Model 
               distribution="bernoulli",     # Option needed for binary response variable
               data=xsell_train, 	           # data set
               n.trees=600,		               # Number trees / iterations
               interaction.depth=3,          # max depth of each tree
               shrinkage=0.01, 	             # Learning (shrinkage) rate
               n.minobsinnode=50,	           # Minimum number of obs per node
               bag.fraction = 0.5, 	           # size of random samples (if <1 you're also doing bagging)
               verbose=FALSE)	               # show in-between-steps TRUE/FALSE
summary(gbm_pca)

# Generate predictions pca
xsell_valid$pred_gbm_pca <- predict(gbm_pca, newdata=xsell_valid, 
                                    type="response", na.action=na.pass)

xsell_valid$pred_gbm_pca_factor <- factor(ifelse(xsell_valid$pred_gbm_pca>.1,1,0),labels=c("No Xsell","Xsell"))

### Check accuracy with the confusion matrix #### pca
conf_matrix_gbm_pca <- confusionMatrix(xsell_valid$pred_gbm_pca_factor,factor(xsell_valid$xsell,labels=c("No Xsell","Xsell")), 
                                       positive="Xsell",dnn = c("Prediction", "Actual Data"))
conf_matrix_gbm_pca

conf_matrix_gbm_pca$byClass["F1"]



#### Algorithm XGBOOST ####
library(xgboost)
library(Matrix)

set.seed(12345) # fix random number generator seed for reproducibility
xsell_random <- xsell[order(runif(100000)),] #sort the data set randomly
xsell_valid <- xsell_random[1:20000, ]       # 20% / 2000 observations in the validation dataset
xsell_train <- xsell_random[20001:100000, ]   # 80% / 8000 in the training data set

# Prepare the data (sparse matrix format)
train_m <-sparse.model.matrix(model_orig, data = xsell_train)
valid_m <-sparse.model.matrix(model_orig, data = xsell_valid)

train_label <-ifelse(xsell_train$xsell==1,1,0)
valid_label <-ifelse(xsell_valid$xsell==1,1,0)

# Run the model
set.seed(1234)
xgb_orig <- xgboost(data=train_m,label=train_label,max.depth=5,eta=0.01,subsample=0.5,
               nrounds=600,objective="binary:logistic", verbose=0)

print(xgb_orig,verbose=TRUE)
xgb_orig


# Make predictions
xsell_valid$pred_xgb <- predict(xgb_orig, newdata = valid_m)
xsell_valid$pred_xgb_orig_factor <- factor(ifelse(xsell_valid$pred_xgb>.1,1,0))

summary(xsell_valid$pred_xgb_orig_factor)


### Check accuracy with the confusion matrix ####
conf_matrix_xgb_orig<-confusionMatrix(as.factor(xsell_valid$pred_xgb_orig_factor),as.factor(xsell_valid$xsell), 
                                 positive="1", # which value is what we're trying to predict? Here, 1 (xsell)
                                 dnn = c("Prediction", "True Data"))

conf_matrix_xgb_orig
conf_matrix_xgb_orig$byClass["F1"]


###XGB with pca

set.seed(12345) # fix random number generator seed for reproducibility
xsell_random <- xsell[order(runif(100000)),] #sort the data set randomly
xsell_valid <- xsell_random[1:20000, ]       # 20% / 2000 observations in the validation dataset
xsell_train <- xsell_random[20001:100000, ]   # 80% / 8000 in the training data set


# Prepare the data (sparse matrix format)
train_m <-sparse.model.matrix(model_pca, data = xsell_train)
valid_m <-sparse.model.matrix(model_pca, data = xsell_valid)# Run the model
set.seed(1234)
xgb_pca <- xgboost(data=train_m,label=train_label,max.depth=5,eta=0.01,subsample=0.5,
                    nrounds=600,objective="binary:logistic", verbose=0)

print(xgb_pca,verbose=TRUE)
xgb_pca


# Make predictions
xsell_valid$pred_xgb_pca <- predict(xgb_pca, newdata = valid_m)
xsell_valid$pred_xgb_pca_factor <- factor(ifelse(xsell_valid$pred_xgb_pca>.1,1,0))

summary(xsell_valid$pred_xgb_pca_factor)


### Check accuracy with the confusion matrix ####
conf_matrix_xgb_pca<-confusionMatrix(as.factor(xsell_valid$pred_xgb_pca_factor),as.factor(xsell_valid$xsell), 
                                      positive="1", # which value is what we're trying to predict? Here, 1 (xsell)
                                      dnn = c("Prediction", "True Data"))

conf_matrix_xgb_pca
conf_matrix_xgb_pca$byClass["F1"]



conf_matrix_gbm_orig
conf_matrix_gbm_orig$byClass["F1"]
conf_matrix_gbm_pca
conf_matrix_gbm_pca$byClass["F1"]
conf_matrix_xgb_orig
conf_matrix_xgb_orig$byClass["F1"]
conf_matrix_xgb_pca
conf_matrix_xgb_pca$byClass["F1"]
