library(MASS)
library(randomForest)
library(missForest)
library(tidyr)
library(dplyr)
library(nnet)
#install.packages("gbm", dependencies=TRUE)
library(gbm)

setwd("C:/Users/Cornelius von Lenthe/Box/Machine Learning/Group Project")
load("citizens_officials_ideology.RData")     
#this will convert all columns to binary which have the following properties: 
#1. the data type is not numeric
#2. the values are 1,2,3,4
convert_to_binary = function(dataframe) {
  dataframe$fem[dataframe$fem == 1] = 3
  dataframe$fem[dataframe$fem == 0] = 1
  for (i in 1:ncol(dataframe)) { 
    cn = colnames(dataframe)[i]
    
    #if we only want to convert q19 to binary and keep the rest as is
    
    #if (cn == "q19") (
    #   dataframe[, cn][dataframe[, cn] <= 2] = 1
    #   dataframe[, cn][dataframe[, cn] > 2] = 2
    #   )
    
    dataframe[, cn][dataframe[, cn] <= 2] = 0
    dataframe[, cn][dataframe[, cn] > 2] = 1
  } 
  return(dataframe)
}



# Omit missing
citizens <- na.omit(d.citizens)

# Drop variables
citizens <- citizens %>% dplyr::select(-q1, -q2, -q4, -q5, -q7, -q16, -q34, -q37, -q45)

#to try models on small dataset
# citizens = citizens[1:500, ]

# Drop non-China
citizens <- citizens %>% filter(country == "China")

# Make into dataframe
citizens <- data.frame(citizens)
citizens$region <- droplevels(citizens$region)
#citizens <- citizens %>% filter(!grepl("Liaoning", region) & !grepl("Jilin", region) & !grepl("Guangdong", region) & !grepl("Jiangxi", region) & !grepl("Mongolia", region) & !grepl("Heilongjiang", region) & !grepl("Heilongjiang", region) & !grepl("Hebei", region)) 
# Age and Factors
citizens.nonfactors <- citizens %>% dplyr::select(age) 
citizens <- citizens %>%  dplyr::select(-region, -city, -age, -country)
#added binary conversion here
citizens <- convert_to_binary (citizens)
citizens <- sapply(citizens, as.factor)
citizens <- cbind(citizens, citizens.nonfactors)

#Test
a <- citizens
a.subset <- sample(seq(nrow(a)), round(nrow(a) * 0.1))
a.small <- a[a.subset , 1:20]
citizens <- a.small

##BEST MODEL CODE ##

# keep track of the best model
best_model_score = 0
best_model = NA
best_model_name = ''

##F1 Score Function##
f1_score <- function(predicted_y, true_y) {
  library(dplyr)
  num_unique_y      <- length(unique(true_y))
  scores            <- vector(length = num_unique_y, mode = "double")
  
  for (i in 1:num_unique_y) {
    trans_pred      <- as.numeric(predicted_y == i)
    trans_true      <- as.numeric(true_y == i)
    df              <- cbind.data.frame(trans_pred, trans_true)
    colnames(df)    <- c("pred", "true")
    df              <- df %>%
      mutate(true_pos = ((pred == 1) & (true == 1)),
             true_neg = ((pred == 0) & (true == 0)),
             false_pos = ((pred == 1) & (true == 0)),
             false_neg = ((pred == 0) & (true == 1))) %>%
      summarise(true_pos = sum(true_pos),
                false_pos = sum(false_pos),
                false_neg = sum(false_neg))
    scores[i]       <- 2 * df$true_pos / (2 * df$true_pos + 
                                            df$false_neg + 
                                            df$false_pos)
    
  }
  F1                <- mean(scores)
  return(F1)
}

## CREATE TEST/TRAIN ##
set.seed(222)
train               <- sample(seq(nrow(citizens)),
                              floor(nrow(citizens) * 0.8))
train               <- sort(train)
test                <- which(!(seq(nrow(citizens)) %in% train))

## CREATE DATA FRAMES ##
# This is for Q19
citizens.train= citizens[train, ]
citizens.test=citizens[-train, -c(14)]
citizens.check=citizens[-train, c(14)]
names(citizens)

## END STATIC CLEANING CODE ##

################
# Boosting
################
##This will check from 1-20 interaction depth:
set.seed(222)
max_boost_score <- 0
interaction.depth_sequence <- seq(20)
boost2_f1_scores <- matrix(0, nrow = length(interaction.depth_sequence), ncol=1)
for (interaction.depth in interaction.depth_sequence) {
  cvboost.citizens <- gbm(q19~., data=data.frame(citizens[-test,]), distribution='multinomial', 
                          n.trees=1000, interaction.depth=interaction.depth, shrinkage = 0.01)
  yhat.cvboost <- predict(cvboost.citizens, newdata=citizens.test, n.trees=1000, type='response')
  p.yhat.cvboost <- apply(yhat.cvboost,1, which.max)
  p.yhat.cvboost <- as.numeric(p.yhat.cvboost)
  citizens.check <- as.numeric(citizens.check)
  f1score_boost2 <- f1_score(p.yhat.cvboost, citizens.check)
  boost2_f1_scores[interaction.depth] <- c(f1score_boost2)
  if (f1score_boost2 > max_boost_score) {
    max_boost_score = f1score_boost2;
  }
}
print(boost2_f1_scores)

##########Input best interaction depth depending on CV^#########
boost.citizens.best <- gbm(q19~., data=data.frame(citizens[-test,]), distribution='multinomial', 
                           n.trees=1000, interaction.depth=20, shrinkage = 0.01)
yhat.cvboost.best <- predict(boost.citizens.best, newdata=citizens.test, n.trees=1000, type='response')
p.yhat.cvboost.best <- apply(yhat.cvboost.best,1, which.max)
f1score_boost.best <- f1_score(p.yhat.cvboost.best, citizens.check)
print(f1score_boost.best)

# Which is the model, Cornelius?
if (max_boost_score > best_model_score){
  best_model_score = max_boost_score;
  best_model = cvboost.citizens;
  best_model_name = "Boosting";
}

## TEST F SCORES 500 Trees:
## Interaction 1: 0.3702375
## Interaction 2: 0.3921297
## Interaction 3: 0.4012710
## Interaction 4: 0.4046753
## Interaction 5: 0.4083283

################
# Random Forest
################

# Question 19
# Bagging
#sqrt(43) 6.5 so mtry should be 6 or 7
bag.citizens <- randomForest(q19~., data = data.frame(citizens[-test,]), 
                             mtry=7, importance =TRUE)
bag.citizens$importance

# Question 19
##  predictions 

#citizens.test <- citizens[-train,"q19"]
# this is citizens.check

yhat.citizens.bag <- predict(bag.citizens, newdata=data.frame(citizens.test))
yhat.citizens.bag <- as.numeric(yhat.citizens.bag)
citizens.check <- as.numeric(citizens.check)
f1_score_rf <- f1_score(yhat.citizens.bag, citizens.check)

# Old f1 score = 0.5088178
# New f1 score = 0.3690354

if (f1_score_rf > best_model_score){
  best_model_score = f1_score_rf;
  best_model = bag.citizens;
  best_model_name = "RandomF";
}

# Question 29
varImpPlot(bag.citizens)

################
# Logistic
################

#logistic regression model

run_logistic_reg = function(tdata, vdata) {
  
  #if multinomial prediction is required uncomment the next two lines and comment out the "if binary predictions" code 
  # log_reg <- multinom(q19 ~ ., data = tdata) # multinom Model
  # preds = predict(log_reg, newdata = vdata)
  # modelname = "Logistic Regression (multinomial)"
  
  #if binary predictions 
  log_reg <- glm(q19 ~ ., data = tdata, family = binomial)
  val_prediction_probabilities <- predict(log_reg, vdata, type = "response")
  preds = as.numeric(val_prediction_probabilities > 0.5) + 1
  modelname = "Logistic Regression (binary)"
  
  #keep this for both binary or multinomial
  returnList = list(model = log_reg, val_predictions = preds, name= modelname)
  return(returnList)
  
}

#run log_reg model
model_log_reg = run_logistic_reg(citizens.train[, ], citizens.test[, ])

#f1 score
log_reg_score = f1_score(as.numeric(model_log_reg$val_predictions), citizens.check)

if (log_reg_score > best_model_score){
  best_model_score = log_reg_score;
  best_model = model_log_reg$model;
  best_model_name = model_log_reg$name;
}


##OUTPUT CODE ##

print("*********************************")
print("Best Model Chosen:")
print(best_model_name)
print("Best Model Score:")
print(best_model_score)
print("*********************************")
print("*********************************")


#####Apply to Officials Dataset
#test data predictions using chosen model

predict_test_data = function(best_model, test_data) {
  
  output_df = data.frame(matrix(ncol = 2, nrow = nrow(test_data)))
  test_data$predictions = predict(best_model, newdata = test_data)
  return(test_data)
  
}
convert_factor = function(x){as.numeric(as.character(x))}

# Creating matching officials dataset
officials <- na.omit(d.officials)

# Age and Factors
officials.nonfactors <- officials %>% dplyr::select(age) 
officials <- officials %>%  dplyr::select(-city, -age)
#added binary conversion here
officials <- convert_to_binary (officials)
officials <- sapply(officials, as.factor)
officials <- cbind(officials, officials.nonfactors)
#not sure about this but if this is here then maybe include it before converting to binary two lines above
officials$q19 <- convert_factor(officials$q19)

# Create test and check datasets
officials.check <- officials %>% dplyr::select(q19)
officials.check <- data.frame(officials.check)
officials.test <- officials %>% dplyr::select(-q19)

# Predict officials
#RF
yhat.officials.bag <- predict(bag.citizens, newdata=data.frame(officials.test))
yhat.officials.bag <- as.numeric(yhat.officials.bag)
#officials.check <- as.numeric(officials.check)
f1_score_rf_officials <- f1_score(yhat.officials.bag, officials.check)
print(f1_score_rf_officials)


#f1 score log
official_preds = predict(model_log_reg$model, newdata = officials.test)
multilog_score = f1_score(as.numeric(official_preds), officials.check)

log_reg_official_preds_probs = predict(model_log_reg$model, officials.test, type = "response")
log_reg_official_preds = as.numeric(log_reg_official_preds_probs > 0.5) + 1

f1_score_log_reg_officials <- f1_score(log_reg_official_preds, officials.check)
print(f1_score_log_reg_officials)

#f1 score boosting

officials_boost <- predict(boost.citizens.best, newdata=officials.test, n.trees=1000, type='response')
p.yhat.official <- apply(officials_boost,1, which.max)
p.yhat.official <- as.numeric(p.yhat.official)
f1score_officials_boost <- f1_score(p.yhat.official, officials.check)
print(f1score_officials_boost)

print("RF")
print(f1_score_rf)
print(f1_score_rf_officials)
print("Log")
print(log_reg_score)
print(f1_score_log_reg_officials)
print("Boost")
print(f1score_boost.best)
print(f1score_officials_boost)

###LEFT OFF HERE
########
# Here we should apply our best model to the officials dataset (we have q19 so we want to compare f-scores)

# Writing output
#pred_d.officials = predict_test_data(best_model, officials.test)
#f1score_officials <- f1_score(p.yhat.cvboost, officials.check)

#pred_df = predict_test_data(best_model, test_data)

#print("Official Answers csv.")
#write.csv(pred_df, output_filename, row.names=FALSE);
#print("*********************************")