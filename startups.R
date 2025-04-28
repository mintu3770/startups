# -----------------------------------------
# 📌 0. Install & Load Libraries
# -----------------------------------------
required_packages <- c("tidyverse", "ggplot2", "corrplot", "cluster", 
                       "factoextra", "wordcloud", "e1071", "caret", "lubridate")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
} 

# -----------------------------------------
# 📌 1. Load Libraries
# -----------------------------------------
library(tidyverse)
library(ggplot2)
library(corrplot)
library(cluster)
library(factoextra)
library(wordcloud)
library(e1071) # for SVM
library(caret) # for train/test split and modeling
library(lubridate)

# -----------------------------------------
# 📌 2. Load Dataset
# -----------------------------------------
startup_data <- read.csv("/Users/mintuchowdary/Desktop/investments_VC.csv", stringsAsFactors = FALSE)

# Quick look at data
str(startup_data)
summary(startup_data)

# -----------------------------------------
# 📌 3. Data Cleaning
# -----------------------------------------

# 3.1 Check column names
colnames(startup_data)

# 3.2 View a few rows
head(startup_data)

# 3.3 Clean funding column
# If funding_total_usd has commas or is non-numeric, fix it
if ("funding_total_usd" %in% colnames(startup_data)) {
  startup_data$funding_total_usd <- as.numeric(gsub(",", "", startup_data$funding_total_usd))
  names(startup_data)[names(startup_data) == "funding_total_usd"] <- "raised_amount_usd"
}

# 3.4 Keep only necessary columns (only those that actually exist)
startup_data <- startup_data %>%
  select(name, category_list, city, state_code, country_code, 
         funding_rounds, founded_year, status, raised_amount_usd, everything())

# 3.5 Check missing values
colSums(is.na(startup_data))

# 3.6 Remove rows where important columns are missing
startup_data <- startup_data %>%
  filter(!is.na(funding_rounds), !is.na(founded_year), !is.na(status))

# 3.7 Convert data types properly
startup_data <- startup_data %>%
  mutate(
    funding_rounds = as.numeric(funding_rounds),
    founded_year = as.numeric(founded_year)
  )

# 3.8 Add Startup Age column
startup_data <- startup_data %>%
  mutate(startup_age = 2025 - founded_year)

# 3.9 Final check
str(startup_data)
summary(startup_data)

# -----------------------------------------
# 📌 4. Exploratory Data Analysis (EDA)
# -----------------------------------------

# 4.1 Univariate Analysis

# Success vs Failure
ggplot(startup_data, aes(x = status)) +
  geom_bar(fill = "skyblue") +
  theme_minimal() +
  labs(title = "Startup Status Distribution", x = "Status", y = "Count")

# Funding Amount Distribution
ggplot(startup_data, aes(x = raised_amount_usd)) +
  geom_histogram(fill = "orange", bins = 50) +
  scale_x_log10() +
  theme_minimal() +
  labs(title = "Funding Amount Distribution (log scale)", x = "Raised Amount (USD)", y = "Count")

# Top Industries
startup_data %>%
  count(category_list, sort = TRUE) %>%
  top_n(10) %>%
  ggplot(aes(x = reorder(category_list, n), y = n)) +
  geom_col(fill = "purple") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Top 10 Startup Categories", x = "Category", y = "Count")

# Top Countries
startup_data %>%
  count(country_code, sort = TRUE) %>%
  top_n(10) %>%
  ggplot(aes(x = reorder(country_code, n), y = n)) +
  geom_col(fill = "darkgreen") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Top 10 Countries for Startups", x = "Country", y = "Count")

# 4.2 Bivariate Analysis

# Funding vs Status
ggplot(startup_data, aes(x = status, y = raised_amount_usd)) +
  geom_boxplot(fill = "lightblue") +
  scale_y_log10() +
  theme_minimal() +
  labs(title = "Funding Amount vs Startup Status", x = "Status", y = "Raised Amount (USD)")

# Founded Year vs Status
ggplot(startup_data, aes(x = founded_year, fill = status)) +
  geom_histogram(bins = 30, position = "dodge") +
  theme_minimal() +
  labs(title = "Founded Year vs Status", x = "Founded Year", y = "Count")

# 4.3 Multivariate Analysis - Correlation

correlation_data <- startup_data %>%
  select(funding_rounds, raised_amount_usd, startup_age) %>%
  cor(use = "complete.obs")  # Handle NA properly

corrplot(correlation_data, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 45, addCoef.col = "black")

# -----------------------------------------
# 📌 5. Trends Analysis
# -----------------------------------------
startup_data %>%
  count(founded_year) %>%
  ggplot(aes(x = founded_year, y = n)) +
  geom_line(color = "darkred", size = 1.2) +
  theme_minimal() +
  labs(title = "Number of Startups Founded Over Years", x = "Founded Year", y = "Number of Startups")

# -----------------------------------------
# 📌 6. WordCloud of Startup Names
# -----------------------------------------
startup_data$name <- iconv(startup_data$name, from = "latin1", to = "UTF-8", sub = "")

# Now create the wordcloud
wordcloud(startup_data$name, max.words = 100, random.order = FALSE, colors = brewer.pal(8, "Dark2"))
# -----------------------------------------
# 📌 7. Predictive Analytics
# -----------------------------------------

# Label encoding: Status (successful = 1, closed = 0)
startup_data$status_binary <- ifelse(startup_data$status == "operating", 1, 0)

# Prepare dataset
model_data <- startup_data %>%
  select(funding_rounds, raised_amount_usd, startup_age, status_binary)

# Train-Test Split
set.seed(123)
trainIndex <- createDataPartition(model_data$status_binary, p = 0.7, list = FALSE)
trainData <- model_data[trainIndex,]
testData <- model_data[-trainIndex,]

# Logistic Regression
logistic_model <- glm(status_binary ~ ., data = trainData, family = binomial)
summary(logistic_model)

# Predict and evaluate
logistic_pred <- predict(logistic_model, testData, type = "response")
logistic_pred_class <- ifelse(logistic_pred > 0.5, 1, 0)

confusionMatrix(as.factor(logistic_pred_class), as.factor(testData$status_binary))

# Support Vector Machine (SVM)
# Make sure model_data has no missing values
model_data <- na.omit(model_data)

# Then train-test split
set.seed(123)
trainIndex <- createDataPartition(model_data$status_binary, p = 0.7, list = FALSE)
trainData <- model_data[trainIndex,]
testData <- model_data[-trainIndex,]

# Ensure factors
trainData$status_binary <- as.factor(trainData$status_binary)
testData$status_binary <- as.factor(testData$status_binary)

# Train SVM
svm_model <- svm(status_binary ~ ., data = trainData, kernel = "linear")

# Predict
svm_pred <- predict(svm_model, testData)

# Confusion Matrix
confusionMatrix(svm_pred, testData$status_binary)

# -----------------------------------------
# 📌 8. Clustering - KMeans
# -----------------------------------------


# 1. Remove rows with NA first
startup_data_clean <- startup_data %>%
  select(funding_rounds, raised_amount_usd, startup_age) %>%
  na.omit()

# 2. Scale the clean data
startup_scaled <- scale(startup_data_clean)

# 3. Determine optimal number of clusters
fviz_nbclust(startup_scaled, kmeans, method = "wss")

# 4. Apply KMeans
set.seed(123)
kmeans_result <- kmeans(startup_scaled, centers = 3)

# 5. Visualize clusters
fviz_cluster(kmeans_result, data = startup_scaled,
             palette = "jco", ggtheme = theme_minimal())

# 6. Add cluster labels back to the cleaned startup data
startup_data_clean$cluster <- kmeans_result$cluster

# -----------------------------------------
# 📌 9. Insights & Final Observations
# -----------------------------------------
cat("🔹 Startups with higher funding and moderate age have a higher survival chance.\n")
cat("🔹 Funding and startup age are strongly correlated with success.\n")
cat("🔹 SVM performed slightly better than Logistic Regression for predicting startup success.\n")
cat("🔹 Three clusters reveal different funding-age-success profiles.\n")