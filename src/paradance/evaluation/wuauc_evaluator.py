from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

if TYPE_CHECKING:
    from .calculator import Calculator

from .base_evaluator import evaluation_preprocessor


@evaluation_preprocessor
def calculate_wuauc(
    calculator: "Calculator",
    target_column: str,
    mask_column: Optional[str] = None,
    groupby: Optional[str] = None,
    weights_for_groups: Optional[pd.Series] = None,
    auc: bool = False,
    pd_column='overall_score',
) -> float:
    """Calculate weighted user AUC.

    :param groupby: groupby column
    :param weights_for_groups: weights for group
    :param target_column: label column
    :param auc: bool, optional, default: False
    :return: AUC/WUAUC/UAUC
    """
    df = calculator.evaluated_dataframe
    if auc:
        result = float(roc_auc_score(df[target_column].values, df[pd_column]))
    else:
        if groupby is not None:
            '''
            # 校验 分组内是否只有一个 lab ，删除只有一个lab 的数据
            temp_pd = df.groupby(groupby).agg( {target_column: ['count',  'sum'] } ).reset_index()
            temp_pd.columns = [ groupby, 'count', 'lab_count']
            # remove_pd = temp_pd[(temp_pd.lab_count == 0) | (temp_pd.count == temp_pd.lab_count)]
            remove_pd = temp_pd[ (temp_pd['lab_count'] == 0) | (temp_pd['count'] == temp_pd['lab_count'])]
            print( f'remove single lab data {target_column}-{groupby} size {'======'*10}', len( remove_pd ))

            df_temp = df
            for v in remove_pd.itertuples():
                print( 'remove data ======'*10, v.order_id, v.count, v.lab_count)
                df_temp=df_temp[df_temp[groupby] != v.order_id]

            print( f'remove after data size {len(df_temp)} {'======'*10}' )
            grouped = df_temp.groupby(groupby).apply(
                lambda x: float(roc_auc_score(x[target_column], x[pd_column]))
            )

            '''
            # print( groupby, target_column, pd_column)
            grouped = df.groupby(groupby).apply(
                lambda x: float(roc_auc_score(x[target_column], x[pd_column]))
            )

            if weights_for_groups is not None:
                counts_sorted = weights_for_groups.loc[grouped.index]
                result = float(np.average(grouped, weights=counts_sorted.values))
            else:
                result = float(np.mean(grouped))
    return result
