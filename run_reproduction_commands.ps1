# Run from reproducibility_package/

python code/benchmark_attackseqbench_openai.py --dataset-dir data/AttackSeqBench/dataset --tasks AttackSeq-Tactic --max-questions-per-task 100 --model gpt-4o-mini --request-max-tokens 128 --temperature 0.0

python code/benchmark_attackseqbench_openai.py --dataset-dir data/AttackSeqBench/dataset --tasks AttackSeq-Tactic --max-questions-per-task 100 --model gpt-4o-mini --enable-mitre-kb --mitre-kb-path data/AttackSeqBench/mitre_kb/mitre.json --mitre-top-k 3 --mitre-max-chars 900 --request-max-tokens 128 --temperature 0.0

python code/benchmark_attackseqbench_openai.py --dataset-dir data/AttackSeqBench/dataset --tasks AttackSeq-Tactic --max-questions-per-task 100 --model gpt-4o-mini --enable-cascade --cascade-model gpt-4o --cascade-max-escalation-rate 0.25 --cascade-self-consistency-votes 3 --cascade-self-consistency-temperature 0.2 --cascade-min-confidence 90 --cascade-min-margin 20 --cascade-hard-confidence 80 --cascade-hard-margin 10 --cascade-min-vote-share 1.0 --cascade-confusion-threshold 0.35 --cascade-enable-verifier --request-max-tokens 128 --temperature 0.0

python code/benchmark_attackseqbench_openai.py --dataset-dir data/AttackSeqBench/dataset --tasks AttackSeq-Tactic --max-questions-per-task 100 --model gpt-4o-mini --enable-mitre-kb --mitre-kb-path data/AttackSeqBench/mitre_kb/mitre.json --mitre-top-k 3 --mitre-max-chars 900 --enable-cascade --cascade-model gpt-4o --cascade-max-escalation-rate 0.25 --cascade-self-consistency-votes 3 --cascade-self-consistency-temperature 0.2 --cascade-min-confidence 90 --cascade-min-margin 20 --cascade-hard-confidence 80 --cascade-hard-margin 10 --cascade-min-vote-share 1.0 --cascade-confusion-threshold 0.35 --cascade-enable-verifier --request-max-tokens 128 --temperature 0.0

python code/run_autogen_cyber_eval.py --dataset-csv data/AttackSeqBench/dataset/AttackSeq-Tactic.csv --max-questions 500 --model gpt-4o-mini --mode both
python code/run_autoprune_cyber_eval.py --dataset-csv data/AttackSeqBench/dataset/AttackSeq-Tactic.csv --max-questions 100 --model gpt-4o-mini --mode all
python code/run_gmemory_cyber_eval.py --dataset-csv data/AttackSeqBench/dataset/AttackSeq-Tactic.csv --max-questions 100 --model gpt-4o-mini --mode all

python code/run_cybermetric_autogen_agentprune_eval.py --dataset-json data/CyberMetric/CyberMetric-500-v1.json --mode all --max-questions 500 --model gpt-4o-mini
