import logging
from abc import ABCMeta, abstractmethod
from itertools import zip_longest
from typing import Dict, Optional, Union

import pandas as pd

from ..dataloader import CSVLoader, ExcelLoader, load_config
from ..evaluation import Calculator, LogarithmPCACalculator
from ..optimization import MultipleObjective, optimize_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class BasePipeline(metaclass=ABCMeta):
    """Abstract base class for implementing processing pipelines.

    This class provides a structured way to define a sequence of operations
    for data processing and optimization tasks. It is designed to be subclassed
    with specific implementations of the abstract methods provided.

    Attributes:
        config (Dict): Configuration settings loaded from a configuration file.
        n_trials (int): The number of optimization trials to perform.

    """

    def __init__(
        self,
        dataframe: Optional[pd.DataFrame] = None,
        config_path: Optional[str] = None,
        n_trials: int = 200,
    ) -> None:
        self.dataframe = dataframe
        self.config: Dict = load_config(config_path)
        self.n_trials = n_trials

    def _load_dataset(self) -> None:
        """Loads the dataset based on the file type specified in the configuration.

        Supports loading from CSV and Excel files.
        """
        if self.dataframe is None:
            self.file_type = self.config["DataLoader"].get("file_type", "csv")
            selected_columns = self.config["Calculator"].get("selected_columns", None)
            config = self.config["DataLoader"]
            # config["clean_zero_columns"] = selected_columns
            config["clean_zero_columns"] = self.config["DataLoader"].get("clean_zero_columns", None)
            config["clean_gauc_lab_columns"] = self.config["DataLoader"].get("clean_gauc_lab_columns", None)
            logger.info(f"_load_dataset config {config}")
            if self.dataframe is None:
                if self.file_type == "csv":
                    logger.info(f"_load_dataset csv config {config}")
                    self.dataframe = CSVLoader(
                        config=config,
                    ).df
                elif self.file_type == "xlsx":
                    self.dataframe = ExcelLoader(
                        config=config,
                    ).df
            logger.info(f"_load_dataset data size {len(self.dataframe)}")

    @abstractmethod
    def _load_calculator(self) -> Union[Calculator, LogarithmPCACalculator]:
        """Load or define the calculator for the pipeline operations.

        This method should be implemented to load or define the calculator, which
        might be a model or any computational tool needed for the pipeline.
        """
        pass

    def _add_objective(
        self, calculator: Union[Calculator, LogarithmPCACalculator]
    ) -> None:
        """Defines the optimization objective for PCA."""
        self.objective = MultipleObjective(
            calculator=calculator,
            config=self.config["Objective"],
        )

    def _add_evaluators(self) -> None:
        """Adds evaluators for optimization based on configuration settings."""
        flags = self.config["Evaluator"].get("flags", None)
        target_columns = self.config["Evaluator"].get("target_columns", [])
        mask_columns = self.config["Evaluator"].get("mask_columns", [])
        hyperparameters = self.config["Evaluator"].get("hyperparameters", [])
        evaluator_propertys = self.config["Evaluator"].get("evaluator_propertys", [])
        groupbys = self.config["Evaluator"].get("groupbys", [])
        for (
            flag,
            target_column,
            mask_column,
            hyperparameter,
            evaluator_property,
            groupby,
        ) in zip_longest(
            flags,
            target_columns,
            mask_columns,
            hyperparameters,
            evaluator_propertys,
            groupbys,
        ):
            self.objective.add_evaluator(
                flag=flag,
                target_column=target_column,
                mask_column=mask_column,
                hyperparameter=hyperparameter,
                evaluator_property=evaluator_property,
                groupby=groupby,
            )

    def _optimize(self) -> None:
        """Runs the optimization process for the defined objective and evaluators."""
        optimize_run(
            multiple_objective=self.objective,
            n_trials=self.n_trials,
            parallel= self.config["Objective"].get("parallel", True)
        )

    @abstractmethod
    def show_results(self) -> None:
        """Displays the results of the optimization process."""
        pass

    def _pre_run(self) -> None:
        """
        Execute the preliminary setup tasks for the pipeline.

        This method includes steps such as loading the dataset, initializing the
        calculator, and adding objectives and evaluators to the calculator.
        """
        logger.info("Running pipeline...")
        self._load_dataset()
        logger.info(f"dataset size {len(self.dataframe)}")
        calculator = self._load_calculator()
        self._add_objective(calculator)
        self._add_evaluators()

    def run(self) -> None:
        """
        Run the main execution flow of the pipeline.

        This method handles the entire flow of running the optimization after
        performing all pre-run setup tasks. It concludes with displaying the results.
        """
        self._pre_run()
        self._optimize()
        self.show_results()
