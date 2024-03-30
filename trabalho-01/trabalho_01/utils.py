from numpy import ndarray, unique

def print_result(score: float,  predict: ndarray) -> None:
    print('Score: ', score)

    pred_unique, pred_count = unique(predict, return_counts=True)
    count_true = dict(zip(pred_unique, pred_count)).get(True, 0)
    print(f'Predict: {count_true} / {len(predict)} ' )
