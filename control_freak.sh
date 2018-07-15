#!/bin/bash

# check if data directory exists, or generate it
if [ ! -d "./data" ]; then
    echo "No data directory found. Generating it now  \n `python3 prepare_data_set.py`"
fi

declare -a sklearn=("0.10" "0.11" "0.12" "0.12.1" "0.13" "0.13.1" "0.14" "0.14.1" "0.15.0b1" "0.15.0b2" "0.15.0" "0.15.1" "0.15.2" "0.16b1" "0.16.0" "0.16.1" "0.17b1" "0.17" "0.17.1" "0.18" "0.18.1" "0.18.2" "0.19b2" "0.19.0" "0.19.1")
declare -a xgboost=("0.4a12" "0.4a13" "0.4a14" "0.4a15" "0.4a18" "0.4a19" "0.4a20" "0.4a21" "0.4a22" "0.4a23" "0.4a24" "0.4a25" "0.4a26" "0.4a27" "0.4a28" "0.4a29" "0.4a30" "0.6a1" "0.6a2" "0.7.post3" "0.7.post4" "0.71" "0.72.1")
declare -a lightgbm=("2.0.2" "2.0.3" "2.0.4" "2.0.5" "2.0.6" "2.0.7" "2.0.10" "2.0.11" "2.0.12" "2.1.0" "2.1.1" "2.1.2")
declare -a h2o=("3.10.0.8" "3.10.0.10" "3.10.3.3" "3.10.3.4" "3.10.4.1" "3.10.4.2" "3.10.4.3" "3.10.4.4" "3.10.4.6" "3.10.4.8" "3.16.0.0" "3.16.0.1" "3.16.0.2" "3.16.0.3" "3.16.0.4" "3.18.0.1" "3.18.0.2" "3.18.0.3" "3.18.0.4" "3.18.0.5" "3.18.0.6" "3.18.0.7" "3.18.0.8" "3.18.0.9" "3.18.0.10" "3.18.0.11")
declare -a catboost=("0.1.1" "0.1.1.2" "0.1.1.3" "0.1.1.5" "0.1.1.6" "0.1.1.7" "0.1.1.8" "0.1.1.9" "0.2" "0.2.1" "0.2.2" "0.2.4" "0.2.5" "0.3.0" "0.3.1" "0.4.1" "0.5" "0.5.1" "0.5.2" "0.5.2.1" "0.6" "0.6.1" "0.6.1.1" "0.6.2" "0.6.3" "0.7" "0.7.1" "0.7.2" "0.7.2.1" "0.8.1.1" "0.9.1" "0.9.1.1")

# Define whether regression or classification case should be run. Else run both.
read -p "Enter 'reg' or 'clf' to run the regression or classification case. Press enter to run both.: " case

if [ "$case" = '' ]; then
   echo "library,version,mae_score,time" > results_reg.txt
   echo "library,version,f1_score,time" > results_clf.txt
   declare -a files_to_run=("run_models_clf.py" "run_models_reg.py")

elif [ "$case" = 'reg' ]; then
   echo "library,version,mae_score,time" > results_reg.txt
    declare -a files_to_run=("run_models_reg.py")
   #file_to_run="run_models_reg.py"

elif [ "$case" = 'clf' ]; then
   echo "library,version,f1_score,time" > results_clf.txt
   declare -a files_to_run=("run_models_clf.py")
   #file_to_run="run_models_clf.py"

else
   echo "Unsupported objective - available options are 'reg' or 'clf'."
   exit
fi

# Run objectiv
for v in "${xgboost[@]}"
do
   if echo "$v"
      pip install xgboost=="$v" ; then
      for f in "${files_to_run[@]}"
      do           
        python3 "$f" xgboost "$v"
      done
   else
       echo "Version $v not installed. Trying next version"
   fi
done

for v in "${sklearn[@]}"
do
   if echo "$v"
      pip install scikit-learn=="$v" ; then
      for f in "${files_to_run[@]}"
      do  
        python3 "$f" sklearn "$v"
      done
   else
       echo "Version $v not installed. Trying next version"
   fi
done

for v in "${lightgbm[@]}"
do
   if echo "$v"
      pip install lightgbm=="$v" ; then
      for f in "${files_to_run[@]}"
      do  
        python3 "$f" lightgbm "$v"
      done
   else
       echo "Version $v not installed. Trying next version"
   fi
done

for v in "${h2o[@]}"
do
   if echo "$v"
      pip install h2o=="$v" ; then
      for f in "${files_to_run[@]}"
      do  
        python3 "$f" h2o "$v"
      done
   else
       echo "Version $v not installed. Trying next version"
   fi
done

for v in "${catboost[@]}"
do
   if echo "$v"
      pip install catboost=="$v" ; then
      for f in "${files_to_run[@]}"
      do  
        python3 "$f" catboost "$v"
        rm -rf catboost_info
        rm -rf train
        rm -rf test
        rm -rf learn
        rm -rf catboost_info
      done
   else
       echo "Version $v not installed. Trying next version"
   fi
done

# plot the results
python3 plot.py
