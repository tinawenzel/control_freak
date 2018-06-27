#!/bin/bash

declare -a sklearn=("0.10" "0.11" "0.12" "0.12.1" "0.13" "0.13.1" "0.14" "0.14.1" "0.15.0b1" "0.15.0b2" "0.15.0" "0.15.1" "0.15.2" "0.16b1" "0.16.0" "0.16.1" "0.17b1" "0.17" "0.17.1" "0.18" "0.18.1" "0.18.2" "0.19b2" "0.19.0" "0.19.1")
declare -a xgboost=("0.4a12" "0.4a13" "0.4a14" "0.4a15" "0.4a18" "0.4a19" "0.4a20" "0.4a21" "0.4a22" "0.4a23" "0.4a24" "0.4a25" "0.4a26" "0.4a27" "0.4a28" "0.4a29" "0.4a30" "0.6a1" "0.6a2" "0.7.post3" "0.7.post4")
declare -a lightgbm=("2.0.2" "2.0.3" "2.0.4" "2.0.5" "2.0.6" "2.0.7" "2.0.10" "2.0.11" "2.0.12" "2.1.0")
declare -a h2o=("0.0.0a2" "0.0.0a3" "0.0.0a4" "0.0.0a5" "0.2.2.9" "0.2.2.13" "0.2.2.14" "0.2.3.6" "3.0.0.8" "3.0.0.16" "3.0.0.18" "3.0.0.22" "3.0.0.25" "3.0.0.26" "3.0.1.2" "3.0.1.3" "3.0.1.4" "3.2.0.1" "3.2.0.3" "3.2.0.5" "3.2.0.8" "3.2.0.9" "3.6.0.3" "3.6.0.8" "3.8.0.3" "3.8.1.4" "3.8.2.6.post1" "3.8.3.3" "3.10.0.3" "3.10.0.6" "3.10.0.7" "3.10.0.8" "3.10.0.10" "3.10.3.3" "3.10.3.4" "3.10.4.1" "3.10.4.2" "3.10.4.3" "3.10.4.4" "3.10.4.6" "3.10.4.8" "3.16.0.0" "3.16.0.1" "3.16.0.2" "3.16.0.3" "3.16.0.4" "3.18.0.1" "3.18.0.2" "3.18.0.3" "3.18.0.4" "3.18.0.5")
declare -a catboost=("0.1.1" "0.1.1.2" "0.1.1.3" "0.1.1.5" "0.1.1.6" "0.1.1.7" "0.1.1.8" "0.1.1.9" "0.2" "0.2.1" "0.2.2" "0.2.4" "0.2.5" "0.3.0" "0.3.1" "0.4.1" "0.5" "0.5.1" "0.5.2" "0.5.2.1" "0.6" "0.6.1" "0.6.1.1" "0.6.2" "0.6.3" "0.7" "0.7.1" "0.7.2" "0.7.2.1")


# echo "library,version,f1_score,timeit" > results_reg.txt
echo "library,version,f1_score,timeit" > results_clf.txt

for v in "${xgboost[@]}"
do
   if echo "$v"
      pip install xgboost=="$v" ; then
      python run_models.py xgboost "$v"
   else
       echo "Version $v not installed. Trying next version"
   fi
done

for v in "${sklearn[@]}"
do
   if echo "$v"
      pip install scikit-learn=="$v" ; then
      python run_models.py sklearn "$v"
   else
       echo "Version $v not installed. Trying next version"
   fi
done

for v in "${lightgbm[@]}"
do
   if echo "$v"
      pip install lightgbm=="$v" ; then
      python run_models.py lightgbm "$v"
   else
       echo "Version $v not installed. Trying next version"
   fi
done

for v in "${h2o[@]}"
do
   if echo "$v"
      pip install h2o=="$v" ; then
      python run_models.py h2o "$v"
   else
       echo "Version $v not installed. Trying next version"
   fi
done

for v in "${catboost[@]}"
do
   if echo "$v"
      pip install catboost=="$v" ; then
      python run_models.py catboost "$v"
      rm -rf train
      rm -rf test
      rm -rf learn
      rm *.tsv
      rm catboost_training.json
   else
       echo "Version $v not installed. Trying next version"
   fi
done