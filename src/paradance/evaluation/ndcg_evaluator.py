from typing import TYPE_CHECKING, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import ndcg_score

if TYPE_CHECKING:
    from .calculator import Calculator


def calculate_ndcg(
        calculator: "Calculator",
        groupby: Optional[str] = None,
        top_n: Optional[int] = None,
        label_column: str = "label",
        pd_column='overall_score',
) -> float:
    """Calculate the rank ratio of negative target
    :param label_column: target column, its values must be 0 or 1
    :param top_n: The percentage of the top rows to consider. Defaults to 100.
    :param groupby: groupby column name.

    """
    if top_n is None:
        top_n = 100

    df_data = calculator.df
    df_data = df_data[[groupby, label_column, pd_column]]
    df_group = df_data.groupby(groupby)

    ndcgs = []
    for name, group in df_group:
        lab = group[label_column].tolist()
        score = group[pd_column].tolist()
        ndcg = ndcg_score(np.asarray([lab]), np.asarray([score]), k=top_n)
        ndcgs.append(ndcg)

    return np.mean(ndcgs)

