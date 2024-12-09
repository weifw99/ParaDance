from typing import Dict, Any


def col_tex(column_val, fo_weight, p_weight):
    return ( 1 + fo_weight*column_val)**p_weight


def sort_dict_values_by_key(input_dict: Dict[str, Any]) -> list[Any]:
    # 使用 sorted 函数对字典按键进行排序，获取排序后的键列表
    sorted_keys = sorted(input_dict.keys())
    # 根据排序后的键列表，构建对应的值列表
    sorted_values = [input_dict[key] for key in sorted_keys]
    return sorted_values
