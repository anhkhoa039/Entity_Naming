
sh run_script.sh 0,1 ai True False conll2003 Train

sh run_script.sh 0,1 bionlp13cg True False conll2003 Train

sh run_script.sh 0,1 literature True False conll2003 Train

sh run_script.sh 0,1 music True False conll2003 Train

sh run_script.sh 0,1 science True False conll2003 Train

sh run_script.sh 0,1 twitter True False conll2003 Train

# Inference Dev
sh run_script.sh 0,1 politics False True conll2003 dev
sh run_script.sh 0,1 politics False True pile_ner dev


sh run_script.sh 0,1 ai False True conll2003 dev

sh run_script.sh 0,1 bionlp13cg False True conll2003 dev

sh run_script.sh 0,1 literature False True conll2003 dev

sh run_script.sh 0,1 music False True conll2003 dev

sh run_script.sh 0,1 science False True conll2003 dev

sh run_script.sh 0,1 twitter False True conll2003 dev


# Inference Test
sh run_script.sh 0,1 politics False True conll2003 test

sh run_script.sh 0,1 ai False True conll2003 test

sh run_script.sh 0,1 bionlp13cg False True conll2003 test

sh run_script.sh 0,1 literature False True conll2003 test

sh run_script.sh 0,1 music False True conll2003 test

sh run_script.sh 0,1 science False True conll2003 test

sh run_script.sh 0,1 twitter False True conll2003 test