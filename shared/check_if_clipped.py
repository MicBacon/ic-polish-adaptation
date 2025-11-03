import jsonlines

INPUT_FILES = [#'../Qwen2_5-VL/eval_results/raw_pl_test_std/predictions_nb_e1_128.jsonl',
               #'../Qwen2_5-VL/eval_results/raw_pl_test_hq/predictions_nb_e1_128.jsonl',
              #'../Qwen2_5-VL/eval_results/raw_pl_test_std/predictions.jsonl',
              #'../Qwen2_5-VL/eval_results/raw_pl_test_hq/predictions.jsonl',
              #'../Qwen2_5-VL/eval_results/raw_pl_test_hq/predictions_nb_e1_128_update.jsonl',
              #'../Qwen2_5-VL/eval_results/raw_pl_test_std/predictions_nb_e1_96_update.jsonl',
              #'../Qwen2_5-VL/eval_results/raw_ext_pl_test_std/predictions_nb_e2_512.jsonl',
              #'../Qwen2_5-VL/eval_results/raw_ext_pl_test_hq/predictions_nb_e2_512.jsonl',
              #'../Qwen2_5-VL/eval_results/raw_ext_pl_test_hq/predictions_nb_e2_512_update.jsonl',
               '../Qwen2_5-VL/eval_results/raw_ft_pl_test_hq/predictions_nb_e4.jsonl',
               '../Qwen2_5-VL/eval_results/raw_ft_pl_test_std/predictions_nb_e4.jsonl',
              ]

count_not_end = 0
for input_file in INPUT_FILES:
    with jsonlines.open(input_file) as input:
        for line in input:
            pred = line.get('prediction')
            if(not pred.endswith('.')):
                count_not_end += 1
                id = line.get('id')
                print(id, pred)
        

print('- Nie skończone:', count_not_end)