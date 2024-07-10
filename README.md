# Language models are better than humans at next-token prediction

This folder contains code to measure human performance at next token prediction used in [Language models are better than humans at next-token prediction](https://arxiv.org/pdf/2212.11281).

It contains:

- The samples used for the experiments (`docs` and `multi_comparisons`, compressed in the `data.zip` file)
- The data collected from the human experiments, anonymized (`results_raw`)
- The code used to analyze the results and check the validity of the method to compute perplexity (`perplexity_results_analysis.py` and `accuracy_results_analysis.py`). This requires the installation of the modules present in `requirements.txt`.
- The code for the web interface used to collect the human experiments (`website`). This requires setting up a website on Heroku and the database tables corresponding to the schemas defined in `app.py`.
