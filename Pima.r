##########################################################################################################################
# HarvardX PH125.9x Data Science Capstone CYO Project
#
# Student: Gideon Vos (gideonvos@icloud.com) www.gideonvos.com (LinkedIn) https://gideonvos.wordpress.com (Blog)
#
# All the code, report, rmd and dataset can be found here: https://github.com/gideonvos/pima
#
##########################################################################################################################

##########################################################################################################################
# The following packages are required. Please set your default repository to CRAN prior to installing.
# We use the neuralnet package and training takes 2 or 3 minutes on any machine
# There are no specific CPU or RAM requirements
##########################################################################################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(neuralnet)) install.packages("neuralnet", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(ggthemes)) install.packages("ggthemes", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(corrplot)
library(gridExtra)
library(ggplot2)
library(ggthemes)
library(neuralnet)


##########################################################################################################################
# Start by reading in the small dataset and setting column names
#
# Please download the small 23K dataset from here: https://github.com/gideonvos/pima/blob/master/pima-indians-diabetes.csv
#
##########################################################################################################################
pima <- read.csv("~/Documents/Harvard_Capstone/Part2/pima-indians-diabetes.csv", header=FALSE)
colnames(pima)<- c("pregnant", "glucose", "pressure", "triceps", "insulin", "mass", "pedigree", "age", "diabetic")

##########################################################################################################################
# Exploratory Analysis
##########################################################################################################################
# Let's have a quick look at the first few records
head(pima)

# Review a summary of the dataset
summary(pima)

# We use “sapply”" to check the number of missing values in each columns.
sapply(pima, function(x) sum(is.na(x)))
# There are no missing values, so this makes our lives easier
# The dataset is clean and widely used, so no need for additional wrangling

# We need to predict the diabetic indicator
# We can see the data is a near 50/50 split between
# diabetic and non-diabetic
ggplot(pima,aes(diabetic,fill = as.factor(diabetic))) +
  geom_bar(colour="black") + 
  scale_fill_manual(values=c("light green", "light blue")) +
  guides(fill=FALSE) +
  ggtitle("Distribution of Outcome variable")

# The data contains a fair distribution of age groups, most being under 30
# with a second large group between 30 and 50 years of age
pima %>%
  ggplot(aes(age)) + 
  geom_histogram(bins = 30, binwidth=0.2, color="black", show.legend = FALSE, aes(fill = cut(age, 100))) + 
  scale_x_log10() + 
  ggtitle("Age Distribution")

# Similarly we can review the BMI index with a significant grouping over 20 upwards
pima %>%
  ggplot(aes(mass)) + 
  geom_histogram(bins = 30, color="black", show.legend = FALSE, aes(fill = cut(mass, 30))) + 
  ggtitle("BMI Distribution")

# Plotting the ages of the diabetic group we see a clear indicator showing
# more frequency at lower age groups
# Correlation does not equal causation, however diabetes is often associated with
# lifestyle habits, even in the West, and if younger Pima people are adopting a 
# Western lifestyle and eating habits this could explain the effect on the 
# younger population versus folks aged over 45
p <- pima[pima$diabetic==1,] %>%
  select(diabetic, age) %>% # select columns we need
  group_by(age) %>% # group by age
  summarise(count = n())  %>% # count per age group
  arrange(age)
p %>%
  ggplot(aes(x = age, y = count)) +
  geom_line(color="blue")

# Review possible corelation with glucose levels
# A corelation is visible here
p2 <- ggplot(pima, aes(x = glucose, color = diabetic, fill = as.factor(diabetic))) +
  geom_density(alpha = 0.8) +
  theme(legend.position = "bottom") +
  scale_fill_manual(values=c("light green", "light blue")) +
  labs(x = "Glucose", y = "Density", title = "Density plot of glucose")

p1 <- ggplot(pima, aes(x = diabetic, y = glucose,fill = as.factor(diabetic))) +
  geom_boxplot() +
  theme(legend.position = "bottom") +
  scale_fill_manual(values=c("light green", "light blue")) +
  ggtitle("Variation of glucose Vs Diabetes")

gridExtra::grid.arrange(p1, p2, ncol = 2)

# Review possible corelation with Blood Pressure 
# no real corelation
p2 <- ggplot(pima, aes(x = pressure, color = diabetic, fill = as.factor(diabetic))) +
  geom_density(alpha = 0.8) +
  theme(legend.position = "bottom") +
  scale_fill_manual(values=c("light green", "light blue")) +
  labs(x = "Blood pressure", y = "Density", title = "Density plot of Blood pressure")

p1 <- ggplot(pima, aes(x = diabetic, y = pressure,fill = as.factor(diabetic))) +
  geom_boxplot() +
  theme(legend.position = "bottom") +
  scale_fill_manual(values=c("light green", "light blue")) +
  ggtitle("Variation of blood pressure Vs Diabetes")

gridExtra::grid.arrange(p1, p2, ncol = 2)

# Review number of pregancies over time against age
boxplot(pregnant ~ age, data=pima, outline = TRUE, names, plot = TRUE, 
        col= 'light blue', xlab = "Age", ylab = "Pregnancies")

# Review BMI over time. Remains fairly steady
boxplot(mass ~ age, data=pima, outline = TRUE, names, plot = TRUE, 
        col= 'light green', xlab = "Age", ylab = "BMI")

# Clear distinction is seen in the distribution of the ‘Age’ variable for those that have Diabetes versus those who don’t.
g <- ggplot(pima, aes(age))
g + geom_density(aes(fill=factor(diabetic)), alpha=0.8) + 
  labs(title="Density plot", 
       subtitle="Age Grouped by Diabetic Indicator",
       x="Age",
       fill="Diabetic")

# It’s clear in the plot below that diabetic patients are associated with more number of pregnancies
g <- ggplot(pima, aes(pregnant))
g + geom_density(aes(fill=factor(diabetic)), alpha=0.8) + 
  labs(title="Density plot", 
       subtitle="# Pregnancies Grouped by Diabetic Indicator",
       x="Pregnancies",
       fill="Diabetic")

# Finally, I want to try to implement some “basic-level clustering”. 
# This is not model-based clustering; rather, it is simply using a scatterplot and a few nice
# plotting parameters in ggplot2() to make some things pop right out at the viewer - again, with l
# ittle room for ambiguity. What I like most here is the boxes that we can draw nicely to showcase the “clusters” 
# a little better, along-with the multi-layered information, e.g., age, BMI, glucose, etc.
# Notice the cluster on the right, showing a clear grouping of how high BMI
# correlates with being diabetic when combined with glucose
d<-pima
d$age <- ifelse(d$age < 30, "<30 yrs", ">= 30 yrs")

ggplot(d, aes(x = glucose, y = mass)) +
  geom_rect(aes(linetype = "High BMI - Diabetic"), xmin = 160, ymax = 40, fill = NA, xmax = 200, 
            ymin = 25, col = "black") + 
  geom_rect(aes(linetype = "Low BMI - Not Diabetic"), xmin = 0, ymax = 25, fill = NA, xmax = 120, 
            ymin = 10, col = "black") + 
  geom_point(aes(col = factor(diabetic), shape = factor(age)), size = 3) +
  scale_color_brewer(name = "Type", palette = "Set1") +
  scale_shape(name = "Age") +
  scale_linetype_manual(values = c("High BMI - Diabetic" = "dotted", "Low BMI - Not Diabetic" = "dashed"),
                        name = "Segment") + theme_minimal()

# A correlation plot was drawn between all the numerical variables to establish the linear association between each other. 
# As observed in the bivariate associations, Insulin and Glucose, BMI and Skin Thickness had a moderate – high linear correlation.
corMat = cor (pima[, -9])
diag (corMat) = 0 #Remove self correlations
corrplot.mixed(corMat,tl.pos = "lt")

# This is all very interesting and useful for other models, but nothing shouts out as main components
# We can instead feed all 8 variables into a neural network as-is and see how we go.

##########################################################################################################################
# Model Training - Neural Network
##########################################################################################################################

# Let's use a Neural Network for this problem
# to build a deep learning model and see if we can improve on
# what has been reported with other models
# We aim to score at least 75% (low) and get to 77% (excellent)

# Split into a 90/10 training/validation split

set.seed(42)
test_index <- createDataPartition(y = pima$diabetic, times = 1, p = 0.1, list = FALSE)
train <- pima[-test_index,]
validation <- pima[test_index,]

# The neuralnet package requires a string formula
n <- names(train)
f <- as.formula(paste("diabetic ~", paste(n[!n %in% "diabetic"], collapse = " + ")))

# We train our deep model using a single hidden node of 8 neurons
nn <- neuralnet(f,data=train,hidden=c(8),linear.output=FALSE, threshold = 0.06)

# Compute predictions off our validation set
predicted.nn.values <- compute(nn,validation[,1:8])

# Review what our deep network looks like visually
plot(nn)

# Let's review our accuracy
summarization = confusionMatrix(as.factor(round(predicted.nn.values$net.result)), as.factor(validation[,9]))
summarization #77.9

# Compare the predicted rating with real rating using visualization. Looks fantastic.
plot(validation$diabetic, predicted.nn.values$net.result, col='blue', pch=16, ylab = "predicted rating NN", xlab = "real rating")
abline(0,1)

# Calculate Root Mean Square Error (RMSE)
rmse = (sum((validation$diabetic - predicted.nn.values$net.result)^2) / nrow(validation)) ^ 0.5
rmse # 0.40

# Accuracy is spot-on for this problem and compares well with other models published before.
# While the neuralnet package in R is not as sophisticated as other Python libraries
# we managed to achieve great, if not better results with a few lines of R.
#
# Thanks for reviewing my code! Much appreciated!
#
##########################################################################################################################


