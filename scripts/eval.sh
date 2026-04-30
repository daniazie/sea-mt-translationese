model=$1

bash scripts/run_metrics.sh
bash t_index/scripts/run/synthetic.sh ${model} synthetic_enms alt
# bash t_index/scripts/run/synthetic.sh ${model} wild ntrex-128
# bash t_index/scripts/run/teval_test.sh ${model} synthetic_enms alt
# bash t_index/scripts/run/teval_test.sh ${model} wild ntrex-128