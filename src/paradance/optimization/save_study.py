import csv
import pickle
import sys
from itertools import zip_longest

from paradance.evaluation.calculator import Calculator
from .multiple_objective import MultipleObjective


def convert_trials_log(multiple_objective: MultipleObjective) -> None:
    """
    Extracts and saves the best trials from the provided log content.
    """
    ob = multiple_objective
    file_path = f"{ob.full_path}/paradance.log"
    output_path = f"{ob.full_path}/paradance_full_trials_info.csv"

    with open(file_path, "r") as file:
        lines = file.readlines()

    extracted_data = []
    sys.stdout.write(f"\nFormula:\t{ob.formula}\n")
    sys.stdout.write(f"Evaluators:\t{ob.evaluator_flags}\n")
    sys.stdout.write(f"Features:\t{ob.calculator.selected_columns}\n")
    for idx, line in enumerate(lines):
        # 解析日志源数据
        # Trial 0 finished with value: [0.843672536284416, 0.8657909813490523] and parameters: [23.987171796642915, 6.634986242117158] and targets: [0.843672536284416, 0.8663505926767384, 0.806573261754342, 0.6387309293607156, 0.03843802862950801, 0.03577642453240539, 0.720284480273136, 0.7222572611202347, 0.6632637592066641, 0.7940226951110784, 0.5401175080825665, 0.8657909813490523, 0.8807738994702624, 0.8291146088872814, 0.21522094263607938, 0.0018108057697090276]. Best is trial 4 with value: [0.8476278141630739, 0.7931699227594781]

        if "Best is trial" in line:
            first_split = line.split('finished with value:')
            trial_number = int(first_split[0].split(" ")[1].strip())

            sec_split = first_split[1].split("and parameters:")
            results_line = sec_split[0].strip()
            th_split = sec_split[1].split('and targets:')
            weights_line = th_split[0].strip()

            targets_line = th_split[1].split('. Best is trial ')[0].strip()

            import json
            targets = json.loads(targets_line)
            weights = json.loads(weights_line)

            results = json.loads(results_line)

            extracted_data.append((results, trial_number, targets, weights))

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        evaluator_flags = []
        for (target_column, flag, groupby, hyperparameter, ) in zip_longest( ob.target_columns, ob.evaluator_flags, ob.groupbys, ob.hyperparameters, fillvalue=None):
            evaluator_flags.append(f'{target_column}_{flag}_{groupby}_{hyperparameter}')

        header = ["Trial",] + ob.formula + evaluator_flags + [f'w_{s_c}' for s_c in ob.calculator.selected_columns]
        writer.writerow(header)

        extracted_data = sorted(extracted_data, key=lambda x: x[1])
        for (results, trial_number, targets, weights) in extracted_data:
            rows = [trial_number]
            rows += results
            rows += targets
            rows += weights
            writer.writerow(rows)


def save_multiple_objective_info(ob: MultipleObjective, filename: str) -> None:
    """Save the parameters and evaluator info of a MultipleObjective object to a txt file.

    Args:
        ob (MultipleObjective): The object whose info needs to be saved.
        filename (str): The name of the txt file.
    """

    with open(filename, "w") as file:
        file.write(f"Study Name: {ob.study_name}\n")
        file.write("-" * 50 + "\n")
        file.write(f"Formula: {ob.formula}\n")
        file.write(f"Selected Columns: {ob.calculator.selected_columns}\n")
        file.write(f"Direction: {ob.direction}\n")
        file.write(f"Weights Number: {ob.weights_num}\n")
        file.write(f"equation_type: {ob.calculator.equation_type}\n")

        file.write("\nEvaluators Info:\n")
        file.write("-" * 50 + "\n")
        for flag, target_column, hyperparameter, evaluator_property, groupby in zip(
            ob.evaluator_flags,
            ob.target_columns,
            ob.hyperparameters,
            ob.evaluator_propertys,
            ob.groupbys,
        ):
            file.write(f"Flag: {flag}\n")
            file.write(f"Target Column: {target_column}\n")
            file.write(f"Hyperparameter: {hyperparameter}\n")
            file.write(f"Evaluator Property: {evaluator_property}\n")
            file.write(f"Groupby: {groupby}\n")
            file.write("\n")


def save_study(multiple_objective: MultipleObjective) -> None:
    """Save the study results of the given multiple objective optimization.

    Args:
        multiple_objective (MultipleObjective): An instance of the MultipleObjective class containing the study to be saved.
    """
    ob = multiple_objective

    if isinstance(ob.calculator, Calculator) and hasattr(
        ob.calculator, "equation_json"
    ):
        ob.export_completed_formulas()

    save_multiple_objective_info(ob, f"{ob.full_path}/objective_info.txt")
    ob.study.trials_dataframe().to_csv(f"{ob.full_path}/paradance_full_trials.csv")
    with open(f"{ob.full_path}/study.pkl", "wb") as f:
        pickle.dump(ob.study, f)
    convert_trials_log(ob)
