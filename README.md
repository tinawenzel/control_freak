# Checking version consistency of Python gradient boosting libraries

This is the companion code for the [blogpost on model consistency](https://medium.com/@tinawenzel/how-stable-is-model-performance-across-different-versions-in-python-data-science-libraries-a1936b13b3e5)


## To reproduce the analysis


1. clone this repo
2. cd into the cloned local copy
3. generate data: `python3 prepare_data_set.py`
4. run analysis `./control_freak.sh` from Terminal
5. at the prompt, enter "reg" or "clf" to run the regression or classification task. This will loop through available versions of scikit-learn, xgboost, catboost, h2o and lightgbm and save the output to `results_clf.txt` or `results_reg.txt` depending on the task.
6. wait ... for 10k rows, `./control_freak.sh` takes about a hour to install/ uninstall all versions and run regression & classification tasks.
7. graphing functions are run at the end and saved as .png in the local dir. Alternatively run `python3 plot.py`


## Notes
* nix* and OSX only :)
* Python 3 recommended
* Requirements: We're looping through tons of old versions. See `./control_freak.sh` for details.
* Warning: If you test very old versions of scikit-learn, `train_test_split` may not be available. Generate the data before running the analysis to avoid this problem.

## Improvements
* package versions currently harcoded. Should catch the error from e.g. `pip install scikit-learn==xxx` and extract the versions from there
