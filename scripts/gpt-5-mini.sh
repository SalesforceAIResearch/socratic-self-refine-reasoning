model="gpt-5-mini"
for method in cot self-refine debate atom mctsr ssr ssr-adaptive ssr-planning; do
    for dataset in math_level5 aime24 aime25 zebra_puzzles mini_sudoku; do
        if [ $dataset == "math_level5" ]; then
            end=681
        elif [ $dataset == "aime24" ]; then
            end=30
        elif [ $dataset == "aime25" ]; then
            end=30
        elif [ $dataset == "zebra_puzzles" ]; then
            end=100
        elif [ $dataset == "mini_sudoku" ]; then
            end=100
        fi
        start_index=1
        for ((i=${start_index}; i<=1; i++)); do
            cot_path="log/${model}/${dataset}/cot/0-${end}/${i}.json"
            case $method in
                cot)
                    python main.py --dataset $dataset --start 0 --end $end --model $model --max-tokens 16384 --max-concurrent-tasks 30 --method cot &
                    ;;
                self-refine)
                    python main.py --dataset $dataset --start 0 --end $end --model $model --max-tokens 16384 --max-concurrent-tasks 30 --method self-refine --max-iter 3 &
                    ;;
                debate)
                    python main.py --dataset $dataset --start 0 --end $end --model $model --max-tokens 16384 --max-concurrent-tasks 30 --method debate --max-iter 3 --num-agents 2 &
                    ;;
                atom)
                    python main.py --dataset $dataset --start 0 --end $end --model $model --max-tokens 16384 --max-concurrent-tasks 30 --method atom --max-iter 3 &
                    ;;
                mctsr)
                    python main.py --dataset $dataset --start 0 --end $end --model $model --max-tokens 16384 --max-concurrent-tasks 30 --method mctsr --max-iter 4 &
                    ;;
                ssr)
                    python main.py --dataset $dataset --start 0 --end $end --model $model --max-tokens 16384 --max-concurrent-tasks 30 --method ssr --max-iter 3 &
                    ;;
                ssr-adaptive)
                    python main.py --dataset $dataset --start 0 --end $end --model $model --max-tokens 16384 --max-concurrent-tasks 30 --method ssr-adaptive --max-iter 3 &
                    ;;
                ssr-planning)
                    python main.py --dataset $dataset --start 0 --end $end --model $model --max-tokens 16384 --max-concurrent-tasks 30 --method ssr-planning --max-iter 3 &
                    ;;
            esac
        done
        wait
    done
done