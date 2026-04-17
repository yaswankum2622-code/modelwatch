# ModelWatch Problem Statement

ModelWatch exists to solve the production failure that most model demos ignore: a model can stay online, keep producing predictions, and still become operationally wrong because the world changed underneath it.

The clearest business example is Zillow. In 2021 Zillow shut down its iBuying division and wrote off **$569 million** after its home-price prediction systems failed in a market reshaped by COVID-era volatility. The model did not stop running. The environment changed faster than the monitoring around it.

Credit-risk models face the same pattern. Repayment behavior, outstanding balances, and customer stress signals move with interest rates, inflation pressure, and macro shocks. If those input distributions drift away from the baseline training period, a model that once separated risky and safe customers well can lose precision, change its decision logic, and quietly create bad approvals or unnecessary declines.

ModelWatch treats this as an MLOps monitoring problem rather than a one-time training problem. It loads a real UCI credit default dataset, creates four monitoring windows, injects progressive drift, tracks statistical shift, measures model quality decay, and shows when retraining is justified instead of guessed.
