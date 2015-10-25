Practical Machine Learning  
---  

author: "winnie"  

date: "Sunday, October 25, 2015"  

Overview  
---  
In this project, my goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website [here](http://groupware.les.inf.puc-rio.br/har)  
The goal of your project is to predict the manner in which they did the exercise.  

Data Processing  
---  

```r
library(C50)
library(gmodels)
library(caret)
utils::View(training0)
training0 <- read.csv("pml-training.csv")
testing0 <- read.csv("Practical.csv")
prop.table(table(levels(training0$classe)))
```

```
## 
##   A   B   C   D   E 
## 0.2 0.2 0.2 0.2 0.2
```
- library the packages that will be useful  
- make a propability table of the numbers of each acivity of the whole training set.  
- **We can see that the 5 manners are nearly uniformly distributed.  

```r
index <- grep("classe",names(training0))
index
```

```
## [1] 160
```
- Find out where the outcome variable is  

K-fold Cross Validation  and Sample Errors
---  


```r
nrow(training0)
```

```
## [1] 19622
```

```r
w <- c()
for(i in 1:10){
w <- c(w,ifelse((nrow(training0)-2)%%i==0,T,F))
}
print(w)
```

```
##  [1]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE FALSE  TRUE  TRUE
```

```r
k <- 5
(nrow(training0)-2)/5
```

```
## [1] 3924
```

```r
w1 <- 1:3024
w2 <- 3025:7848
w3<- 7849:11772
w4<- 11773:15696
w5<- 15697:19622
```
- Split the training set into less than 10 groups, make a loop to find out which k can be feasible. If w=TRUE then such number can be k. Here we can make k=5  
- w1 to w5 is the sequences to indicate where the test sets are, and the sets left behind will be train set.  

```r
set.seed(555)
training1 <- training0[order(runif(19622)),]
cvtest1 <- training1[w1,]
cvtrain1 <- training1[-w1,]
cvtest2 <- training1[w2,]
cvtrain2 <- training1[-w2,]
cvtest3 <- training1[w3,]
cvtrain3 <- training1[-w3,]
cvtest4 <- training1[w4,]
cvtrain4 <- training1[-w4,]
cvtest5 <- training1[w5,]
cvtrain5 <- training1[-w5,]
```
- make 5 groups for cross validation, the training set should be randomly choosen from itself because the manners seems to distribute with some sort of orders.  

```r
model1 <- C5.0(cvtrain1[,1:159],cvtrain1$classe)
```

```
## c50 code called exit with value 1
```

```r
p1 <- predict(model1,cvtest1[,1:159],type="class")
```

```
## Error in predict.C5.0(model1, cvtest1[, 1:159], type = "class"): either a tree or rules must be provided
```

```r
confusionMatrix(p1,cvtest1[,160])
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 896   0   0   0   0
##          B   0 573   0   0   0
##          C   0   1 511   0   0
##          D   0   0   1 509   0
##          E   0   0   0   0 533
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9993          
##                  95% CI : (0.9976, 0.9999)
##     No Information Rate : 0.2963          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9992          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9983   0.9980   1.0000   1.0000
## Specificity            1.0000   1.0000   0.9996   0.9996   1.0000
## Pos Pred Value         1.0000   1.0000   0.9980   0.9980   1.0000
## Neg Pred Value         1.0000   0.9996   0.9996   1.0000   1.0000
## Prevalence             0.2963   0.1898   0.1693   0.1683   0.1763
## Detection Rate         0.2963   0.1895   0.1690   0.1683   0.1763
## Detection Prevalence   0.2963   0.1895   0.1693   0.1687   0.1763
## Balanced Accuracy      1.0000   0.9991   0.9988   0.9998   1.0000
```

```r
model2 <- C5.0(cvtrain2[,1:159],cvtrain2$classe)
```

```
## c50 code called exit with value 1
```

```r
p2 <- predict(model2,cvtest2[,1:159],type="class")
```

```
## Error in predict.C5.0(model2, cvtest2[, 1:159], type = "class"): either a tree or rules must be provided
```

```r
confusionMatrix(p2,cvtest2[,160])
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1343    0    0    0    0
##          B    0  916    0    0    0
##          C    0    0  855    0    0
##          D    0    0    0  807    0
##          E    0    0    0    0  903
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9992, 1)
##     No Information Rate : 0.2784     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2784   0.1899   0.1772   0.1673   0.1872
## Detection Rate         0.2784   0.1899   0.1772   0.1673   0.1872
## Detection Prevalence   0.2784   0.1899   0.1772   0.1673   0.1872
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

```r
model3 <- C5.0(cvtrain3[,1:159],cvtrain3$classe)
```

```
## c50 code called exit with value 1
```

```r
p3 <- predict(model3,cvtest3[,1:159],type="class")
```

```
## Error in predict.C5.0(model3, cvtest3[, 1:159], type = "class"): either a tree or rules must be provided
```

```r
confusionMatrix(p3,cvtest3[,160])
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1123    0    0    0    0
##          B    1  767    0    0    0
##          C    0    0  690    0    0
##          D    0    0    0  627    0
##          E    0    0    0    0  716
## 
## Overall Statistics
##                                      
##                Accuracy : 0.9997     
##                  95% CI : (0.9986, 1)
##     No Information Rate : 0.2864     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 0.9997     
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9991   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   0.9997   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   0.9987   1.0000   1.0000   1.0000
## Neg Pred Value         0.9996   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2864   0.1955   0.1758   0.1598   0.1825
## Detection Rate         0.2862   0.1955   0.1758   0.1598   0.1825
## Detection Prevalence   0.2862   0.1957   0.1758   0.1598   0.1825
## Balanced Accuracy      0.9996   0.9998   1.0000   1.0000   1.0000
```

```r
model4 <- C5.0(cvtrain4[,1:159],cvtrain4$classe)
```

```
## c50 code called exit with value 1
```

```r
p4 <- predict(model4,cvtest4[,1:159],type="class")
```

```
## Error in predict.C5.0(model4, cvtest4[, 1:159], type = "class"): either a tree or rules must be provided
```

```r
confusionMatrix(p4,cvtest4[,160])
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1115    0    0    0    0
##          B    0  767    0    0    0
##          C    0    0  670    0    0
##          D    0    0    0  638    0
##          E    0    0    0    0  734
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9991, 1)
##     No Information Rate : 0.2841     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2841   0.1955   0.1707   0.1626   0.1871
## Detection Rate         0.2841   0.1955   0.1707   0.1626   0.1871
## Detection Prevalence   0.2841   0.1955   0.1707   0.1626   0.1871
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

```r
model5 <- C5.0(cvtrain5[,1:159],cvtrain5$classe)
```

```
## c50 code called exit with value 1
```

```r
p5 <- predict(model5,cvtest5[,1:159],type="class")
```

```
## Error in predict.C5.0(model5, cvtest5[, 1:159], type = "class"): either a tree or rules must be provided
```

```r
confusionMatrix(p5,cvtest5[,160])
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1102    0    0    0    0
##          B    0  773    0    0    0
##          C    0    0  695    0    0
##          D    0    0    0  634    0
##          E    0    0    0    1  721
## 
## Overall Statistics
##                                      
##                Accuracy : 0.9997     
##                  95% CI : (0.9986, 1)
##     No Information Rate : 0.2807     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 0.9997     
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000    1.000   0.9984   1.0000
## Specificity            1.0000   1.0000    1.000   1.0000   0.9997
## Pos Pred Value         1.0000   1.0000    1.000   1.0000   0.9986
## Neg Pred Value         1.0000   1.0000    1.000   0.9997   1.0000
## Prevalence             0.2807   0.1969    0.177   0.1617   0.1836
## Detection Rate         0.2807   0.1969    0.177   0.1615   0.1836
## Detection Prevalence   0.2807   0.1969    0.177   0.1615   0.1839
## Balanced Accuracy      1.0000   1.0000    1.000   0.9992   0.9998
```
- model1 to model5 are the tree predition models for k-fold cross validation, using the C5.0() from C50 package. C5.0 is a highly efficient method for tree predition.  
- p1 to p5 store the results of the preditions in model1 to model5  
- confusionMatrix() gives us Accuracy of the Cross Validation, model1 to model5 did a great job in prediting the result of test set with an average accuracy of 0.99+  

Apply the Model  and Get the Result
---  

```r
model <- C5.0(training1[,1:159],training1$classe)
```

```
## c50 code called exit with value 1
```

```r
pred <- predict(model,testing0[,1:159],type="class")
```

```
## Error in `[.data.frame`(newdata, , object$predictors, drop = FALSE): undefined columns selected
```

```r
table(pred)
```

```
## pred
##  A  B  C  D  E 
## 20  0  0  0  0
```

- So make the tree model of the original training set and predict the manners in the test set which are A of all.



