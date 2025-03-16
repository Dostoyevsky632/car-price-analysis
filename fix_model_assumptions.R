# Fix Non-linearity and Non-normality in Car Price Regression Model
# This script addresses the assumption violations in the regression model

# Load required packages
library(tidyverse)
library(car)
library(lmtest)
library(MASS)    # For Box-Cox transformation
library(gridExtra)

# Read data
car_data <- read.csv("car_price_dataset.csv", stringsAsFactors = TRUE)

# Select variables of interest
selected_vars <- c("Price", "Year", "Mileage", "Engine_Size", "Fuel_Type")
car_selected <- car_data[, selected_vars]

# Original model
original_model <- lm(Price ~ Year + Mileage + Engine_Size + Fuel_Type, data = car_selected)
summary(original_model)

# Print original model diagnostics
cat("Original Model Diagnostics:\n")
# Linearity test
reset_test_original <- resettest(original_model, power = 2:3, type = "fitted")
print(reset_test_original)

# Normality test
ks_test_original <- ks.test(rstandard(original_model), "pnorm")
print(ks_test_original)

# Create diagnostic plots for original model
par(mfrow=c(2,2))
plot(original_model)
par(mfrow=c(1,1))

# ===== FIX 1: ADDRESSING NON-LINEARITY ISSUE =====
cat("\n\n===== ADDRESSING NON-LINEARITY ISSUE =====\n")

# 1. Add polynomial terms for continuous predictors
poly_model <- lm(Price ~ Year + I(Year^2) + 
                      Mileage + I(Mileage^2) + 
                      Engine_Size + I(Engine_Size^2) + 
                      Fuel_Type, 
                data = car_selected)

# Check polynomial model
summary(poly_model)
reset_test_poly <- resettest(poly_model, power = 2:3, type = "fitted")
cat("\nRESET test for polynomial model:\n")
print(reset_test_poly)

# 2. Try log transformation for some variables
# Create transformed variables
car_selected$log_price <- log(car_selected$Price)
car_selected$log_mileage <- log(car_selected$Mileage + 1)  # Adding 1 to avoid log(0)

# Fit log-transformed model
log_model <- lm(log_price ~ Year + log_mileage + Engine_Size + Fuel_Type, 
               data = car_selected)

# Check log model
summary(log_model)
reset_test_log <- resettest(log_model, power = 2:3, type = "fitted")
cat("\nRESET test for log-transformed model:\n")
print(reset_test_log)

# Compare models
cat("\nComparing models for non-linearity:\n")
cat("Original model RESET p-value:", format.pval(reset_test_original$p.value, digits = 4), "\n")
cat("Polynomial model RESET p-value:", format.pval(reset_test_poly$p.value, digits = 4), "\n")
cat("Log-transformed model RESET p-value:", format.pval(reset_test_log$p.value, digits = 4), "\n")

# ===== FIX 2: ADDRESSING NON-NORMALITY ISSUE =====
cat("\n\n===== ADDRESSING NON-NORMALITY ISSUE =====\n")

# 1. Box-Cox transformation
# First, ensure all prices are positive
if(all(car_selected$Price > 0)) {
  # Find optimal lambda for Box-Cox transformation
  bc <- boxcox(Price ~ Year + Mileage + Engine_Size + Fuel_Type, 
              data = car_selected, 
              lambda = seq(-2, 2, 0.1))
  
  # Extract optimal lambda
  lambda <- bc$x[which.max(bc$y)]
  cat("Optimal Box-Cox lambda:", lambda, "\n")
  
  # Apply Box-Cox transformation
  if(abs(lambda) < 0.001) {
    # If lambda is close to 0, use log transformation
    car_selected$transformed_price <- log(car_selected$Price)
    cat("Using log transformation (lambda ≈ 0)\n")
  } else {
    # Otherwise use the power transformation
    car_selected$transformed_price <- (car_selected$Price^lambda - 1) / lambda
    cat("Using power transformation with lambda =", lambda, "\n")
  }
  
  # Fit model with Box-Cox transformed response
  bc_model <- lm(transformed_price ~ Year + Mileage + Engine_Size + Fuel_Type, 
                data = car_selected)
  
  # Check normality of residuals after Box-Cox transformation
  ks_test_bc <- ks.test(rstandard(bc_model), "pnorm")
  cat("\nNormality test after Box-Cox transformation:\n")
  print(ks_test_bc)
  
  # Compare with original model
  cat("\nComparing normality test p-values:\n")
  cat("Original model p-value:", format.pval(ks_test_original$p.value, digits = 4), "\n")
  cat("Box-Cox model p-value:", format.pval(ks_test_bc$p.value, digits = 4), "\n")
}

# 2. Try robust regression as an alternative approach
library(MASS)  # For rlm function
robust_model <- rlm(Price ~ Year + Mileage + Engine_Size + Fuel_Type, 
                   data = car_selected)

cat("\nRobust regression summary:\n")
print(summary(robust_model))

# ===== VISUALIZATION AND COMPARISON =====
# Create a grid of QQ plots for comparing models
par(mfrow=c(2,2))
qqnorm(rstandard(original_model), main="Original Model")
qqline(rstandard(original_model), col="red")

qqnorm(rstandard(poly_model), main="Polynomial Model")
qqline(rstandard(poly_model), col="red")

qqnorm(rstandard(log_model), main="Log-transformed Model")
qqline(rstandard(log_model), col="red")

qqnorm(rstandard(bc_model), main="Box-Cox Model")
qqline(rstandard(bc_model), col="red")
par(mfrow=c(1,1))

# Create comparison plots using ggplot for publication quality
# Extract standardized residuals from each model
residuals_data <- data.frame(
  original = rstandard(original_model),
  polynomial = rstandard(poly_model),
  log_transform = rstandard(log_model),
  box_cox = rstandard(bc_model)
)

# Function to create QQ plot for each model
create_qq_plot <- function(residuals, title) {
  ggplot(data.frame(Theoretical = qnorm(ppoints(length(residuals))),
                    Sample = sort(residuals)), 
         aes(x = Theoretical, y = Sample)) +
    geom_point(alpha = 0.5) +
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
    labs(title = title,
         x = "Theoretical Quantiles",
         y = "Sample Quantiles") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 10))
}

# Create QQ plots for each model
p1 <- create_qq_plot(residuals_data$original, "Original Model")
p2 <- create_qq_plot(residuals_data$polynomial, "Polynomial Model")
p3 <- create_qq_plot(residuals_data$log_transform, "Log-transformed Model")
p4 <- create_qq_plot(residuals_data$box_cox, "Box-Cox Model")

# Combine all QQ plots
qq_plots <- gridExtra::grid.arrange(
  p1, p2, p3, p4,
  ncol = 2,
  top = "QQ Plots Comparison Across Models"
)

# Save the combined plot
ggsave("model_comparison_qq_plots.png", qq_plots, width = 10, height = 8, dpi = 300)

# ===== FINAL MODEL SELECTION =====
# Compare models using AIC
cat("\n\n===== FINAL MODEL SELECTION =====\n")
cat("Comparing models using AIC (smaller is better):\n")
AIC_original <- AIC(original_model)
AIC_poly <- AIC(poly_model)
AIC_log <- AIC(log_model)
AIC_bc <- AIC(bc_model)

cat("Original model AIC:", AIC_original, "\n")
cat("Polynomial model AIC:", AIC_poly, "\n")
cat("Log-transformed model AIC:", AIC_log, "\n")
cat("Box-Cox model AIC:", AIC_bc, "\n")

# Find the best model based on AIC
AIC_values <- c(AIC_original, AIC_poly, AIC_log, AIC_bc)
model_names <- c("Original", "Polynomial", "Log-transformed", "Box-Cox")
best_model_index <- which.min(AIC_values)
cat("\nBest model based on AIC:", model_names[best_model_index], "\n")

# ===== CONCLUSION =====
cat("\n===== CONCLUSION AND RECOMMENDATIONS =====\n")
cat("1. Non-linearity issue:\n")
if(reset_test_log$p.value > reset_test_original$p.value && 
   reset_test_log$p.value > reset_test_poly$p.value) {
  cat("   - Log transformation provides the best improvement for linearity\n")
} else if(reset_test_poly$p.value > reset_test_original$p.value) {
  cat("   - Polynomial model provides the best improvement for linearity\n")
} else {
  cat("   - Further transformations may be needed to address non-linearity\n")
}

cat("\n2. Non-normality issue:\n")
if(ks_test_bc$p.value > ks_test_original$p.value) {
  cat("   - Box-Cox transformation improves normality of residuals\n")
  if(ks_test_bc$p.value < 0.05) {
    cat("   - However, residuals still not perfectly normal; consider robust methods\n")
  }
} else {
  cat("   - Transformations did not significantly improve normality; robust regression recommended\n")
}

cat("\n3. Final recommendation:\n")
cat("   - Based on AIC and diagnostic tests, the", model_names[best_model_index], "model is preferred\n")
cat("   - When reporting results, acknowledge assumption violations and use appropriate methods\n")
cat("   - Consider bootstrapping for confidence intervals or using robust standard errors\n")

cat("\nAnalysis completed. Comparison plots saved as 'model_comparison_qq_plots.png'\n") 