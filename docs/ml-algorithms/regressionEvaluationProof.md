# Regression Evaluation Metric Types

We will cover 3 types of Regression Evaluation Metric
- [Mean Squared Error (MSE)](#mean-squared-error)
- [Mean Absolute Error (MAE)](#mean-absolute-error)
- [R Squared ($R^2$)](#r-squared)

And then covering on how to read the score and the academic evidence for each metric.

## Mean Squared Error

Range: $0$ to $+\infty$

Lower is better.

Best Score: $0$

### Academic Evidence

About Outliers (Chicco et al., 2021)

Quote: "MSE can be used if there are outliers that need to be detected. In fact, MSE is great for attributing larger weights to such points..."

About Context / Scalability (Chicco et al., 2021)

Quote: "...it is difficult to interpret sole values of MSE, RMSE, MAE and MAPE, since they have $+\infty$ as upper bound. An MSE = $0.7$, for example does not say much about the overall quality of a regression model : the value could mean both an excellent regression model and a poor regression model."

## Mean Absolute Error

Range: $0$ to $+\infty$

Lower is better.

Best Score: $0$

### Academic Evidence

About Interpretation Difficulty (Chicco et al., 2021)

Quote: "Even without digging into the mathematical properties of the aforementioned statistical rates, it is clear that it is difficult to interpret sole values of MSE, RMSE, MAE and MAPE, since they have $+\infty$ as upper bound."

About Robustness (Willmot & Matsuura, 2005)

Quote: "Our findings indicate that MAE is a more natural measure of average error, and (unlike RMSE) is unambiguous."

## R-squared

Range: $0$ to $1$

Can be negative if the model is worse than simply predicting the average.

Best Score: $1$

### Academic Evidence

About Upper Bound (Chicco et al., 2021)

Quote: "The positive values of the coefficient of determination range in the [0, 1] interval, with 1 meaning perfect prediction."

About Zero Baseline (Chicco et al., 2021)

Quote: "R-squared can have value 0 when the regression model explains none of the variability of the response data around its mean."

About Negative Scores (Chicco et al., 2021)

Quote: "R-squared can have negative values, which mean that the regression performed poorly."

About General Interpretability (Chicco et al., 2021)

Quote: "A positive value of R-squared can be considered similar to percentage of correctness obtained by the regression."

# References

Chicco, D., Warrens, M.J., & Jurman, G. (2021). The coefficient of determination R-squared is more informative than SMAPE, MAE, MAPE, MSE and RMSE in regression analysis evaluation. https://peerj.com/articles/cs-623

Willmott, C. J., Matsuura, K. (2005). Advantages of the mean absolute error (MAE) over the root mean square error (RMSE) in assessing average model performance. https://www.int-res.com/articles/cr2005/30/c030p079.pdf