from typing import TYPE_CHECKING, Optional
import numpy as np
import pandas as pd

from .base_evaluator import evaluation_preprocessor

if TYPE_CHECKING:
    from .calculator import Calculator


@evaluation_preprocessor
def calculate_top_coverage(
    calculator: "Calculator",
    target_column: str,
    mask_column: Optional[str] = None,
    head_percentage: Optional[float] = None,
    pd_column='overall_score',
) -> float:
    """Calculates the top coverage ratio of a specified column in a DataFrame.

    Args:
        calculator (Calculator): An object that provides access to the evaluated DataFrame.
        target_column (str): The name of the column to calculate the coverage for.
        mask_column (Optional[str], optional): The name of the column to apply a mask on. Defaults to None.
        head_percentage (Optional[float], optional): The percentage of the top rows to consider. Defaults to 0.05.

    Returns:
        float: The ratio of the sum of the top `head_percentage` rows to the total sum of the `target_column`.
    """
    if head_percentage is None:
        head_percentage = 0.05

    df = calculator.evaluated_dataframe
    df = df.sort_values(pd_column, ascending=False)
    total_sum = df[target_column].sum()
    top_rows_count = int(len(df) * head_percentage)
    top_sum = df.iloc[:top_rows_count][target_column].sum()
    top_coverage_ratio = top_sum / total_sum

    return float(top_coverage_ratio)


@evaluation_preprocessor
def calculate_distinct_top_coverage(
    calculator: "Calculator",
    target_column: str,
    mask_column: Optional[str] = None,
    head_percentage: Optional[float] = None,
    pd_column='overall_score',
) -> float:
    """Calculates the distinct top coverage ratio of a specified column in a DataFrame.

    Args:
        calculator (Calculator): An object that provides access to the evaluated DataFrame.
        target_column (str): The name of the column to calculate the distinct coverage for.
        mask_column (Optional[str], optional): The name of the column to apply a mask on. Defaults to None.
        head_percentage (Optional[float], optional): The percentage of the top rows to consider. Defaults to 0.05.

    Returns:
        float: The ratio of the number of unique top `head_percentage` rows to the total number of unique `target_column` values.
    """
    if head_percentage is None:
        head_percentage = 0.05

    df = calculator.evaluated_dataframe
    total_ids = df[target_column].nunique()

    df_sorted = df.sort_values(by=pd_column, ascending=False).reset_index(
        drop=True
    )
    df_sorted["is_unique"] = ~df_sorted[target_column].duplicated()
    top_rows_count = int(len(df) * head_percentage)
    top_sum = df_sorted.iloc[:top_rows_count]["is_unique"].sum()

    top_coverage_ratio = top_sum / total_ids

    return float(top_coverage_ratio)



def pd_data_group_top_n(data: pd.DataFrame, group_cols: list, val_cols: list, ascending: bool = False, k: int = 1):
    """
    自定义获取数据框topN
    :param data: pd.DataFrame类型
    :param group_cols: list, 需要聚合的列名
    :param val_cols: list, 需要排序的列名
    :param ascending: 排序方式，默认`False`，倒序排序，接收bool或这个列表里全部为bool的列表
    :param k: 取前k项值
    :return: 返回topN数据框
    """
    # 为了能返回传入数据框的原index，将index保存至values中
    datac = data.reset_index().copy()
    index_colname = datac.columns[0]
    # 对原数据框进行排序
    datac.sort_values(group_cols + val_cols, ascending=ascending, inplace=True)
    # 主要代码：分组对组内进行编号
    rank0 = np.hstack(datac.value_counts(group_cols, sort=False).map(lambda x: range(x)).values)
    # 取topN值
    datac = datac[rank0 < k]
    # 取出原index重置为index值
    datac.index = datac[index_colname].values
    # 删除额外生成的index值的列
    del datac[index_colname]
    return datac


@evaluation_preprocessor
def calculate_top_n_coverage(
    calculator: "Calculator",
    target_column: str,
    mask_column: Optional[str] = None,
    groupby: Optional[str] = None,
    top_n: Optional[int] = None,
    pd_column='overall_score',
) -> float:
    """Calculates the top coverage ratio of a specified column in a DataFrame.

    Args:
        calculator (Calculator): An object that provides access to the evaluated DataFrame.
        target_column (str): The name of the column to calculate the coverage for.
        mask_column (Optional[str], optional): The name of the column to apply a mask on. Defaults to None.
        top_n (Optional[int], optional): The percentage of the top rows to consider. Defaults to 100.

    Returns:
        float: The ratio of the sum of the top `top_n` rows to the total sum of the `target_column`.
    """
    if top_n is None:
        top_n = 100

    df = calculator.evaluated_dataframe
    total_sum = df[target_column].sum()
    # print( '***'*10 + 'calculate_top_n_coverage total_sum', total_sum)
    group_top_df = pd_data_group_top_n(data=df, group_cols=[groupby], val_cols=[pd_column], ascending=False, k=top_n)
    # print( '***'*10 + 'calculate_top_n_coverage group_top_df', len(group_top_df))
    top_sum = group_top_df[target_column].sum()
    # print( '***'*10 + 'calculate_top_n_coverage top_sum', top_sum)
    top_coverage_ratio = top_sum / total_sum

    return float(top_coverage_ratio)
