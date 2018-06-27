# Checking version consistency of Python data science libraries

This is the companion code for the [blogpost on model consistency](https://medium.com/@tinawenzel/how-stable-is-model-performance-across-different-versions-in-python-data-science-libraries-a1936b13b3e5)


## To reproduce the analysis


1. Clone this repo
2. cd into the cloned local copy
3. Generate data: `python prepare_data_set.py`
4. Run analysis `./control_freak.sh` from Terminal


This will run a classification example, looping through available versions of scikit-learn, xgboost, catboost, h2o and lightgbm. The output will be saved to `results_clf.txt`.


## Notes
* nix* and OSX only :)
* By default `./control_freak.sh` will call the `run_models.py` file which runs the classification analysis. Change it to `run_models_reg.py` if you want to run the regression analysis instead.
* By default `./control_freak.sh` will save the output to `results_clf.txt`. Change the name in `./control_freak.sh` as needed when running the regression analysis, e.g. by commenting in `echo "library,version,f1_score,timeit" > results_reg.txt`
* Requirements: We're looping through tons of old versions. See `./control_freak.sh` for details.
* Warning: If you test very old versions of scikit-learn, `train_test_split` may not be available. Generate the data before running the analysis to avoid this problem.
