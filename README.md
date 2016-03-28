# Credit-Default-analytics
Uses genetic algorithms to identify users who are most likely to default
Logistic regression is used to classify users as defaulters or non-defaulters
GA is then used to evaluate the accuracy of the model, and improve upon it by tweaking certain parameters
This step is repeatedly run till convergence.

Uses 30 chromosomes (or models) per population, and keeps the best performing 15.
These models are evolved using cross-over (rate = 0.95) and mutation (rate = 0.05)
At each stage, the average accuracy of the models improves due to pruning of the worst-performing models.
