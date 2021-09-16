import os
import argparse
import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from modeling.info import data_info
from modeling.ou_model_trainer import OUModelTrainer
from modeling.type import OpUnit
from modeling.util import logging_util

logging_util.init_logging('info')


class AbstractModel(ABC):
    """
    Interface for all the models
    """

    def __init__(self) -> None:
        # Model cache that maps from the model path on disk to the model
        self.model_cache = dict()

    @abstractmethod
    def train(self, data: Dict) -> Tuple[bool, str]:
        """
        Perform fitting.
        Should be overloaded by a specific model implementation.
        :param data: data used for training
        :return: if training succeeds, {True and empty string}, else {False, error message}
        """
        raise NotImplementedError("Should be implemented by child classes")

    @abstractmethod
    def infer(self, data: Dict) -> Tuple[Any, bool, str]:
        """
        Do inference on the model, give the data file, and the model_map_path
        :param data: data used for inference
        :return: {List of predictions, if inference succeeds, error message}
        """
        raise NotImplementedError("Should be implemented by child classes")

    def _load_model(self, save_path: str):
        """
        Check if a trained model exists at the path.
        Load the model into cache if it is not.
        :param save_path: path to model to load
        :return: None if no model exists at path, or Model map saved at path
        """
        save_path = Path(save_path)

        # Check model exists
        if not save_path.exists():
            return None

        # use the path string as the key of the cache
        save_path_str = str(save_path)

        # Load from cache
        if self.model_cache.get(save_path_str, None) is not None:
            return self.model_cache[save_path_str]

        # Load into cache
        model = self._load_model_from_disk(save_path)

        self.model_cache[save_path_str] = model
        return model

    @abstractmethod
    def _load_model_from_disk(self, save_path: Path):
        """
        Load model from the path on disk (invoked when missing model cache)
        :param save_path: model path on disk
        :return: model for the child class' specific model type
        """
        raise NotImplementedError("Should be implemented by child classes")


class OUModel(AbstractModel):
    """
    OUModel that handles training and inference for OU models
    """

    # Training parameters
    TEST_RATIO = 0.2
    TRIM_RATIO = 0.2
    EXPOSE_ALL = True
    TXN_SAMPLE_RATE = 2

    def __init__(self) -> None:
        # Initialize the infer cache
        # TODO(lin): add mechanism to invalidate the infer cache for outdated models
        self.infer_cache = dict()
        AbstractModel.__init__(self)

    def train(self, data: Dict) -> Tuple[bool, str]:
        """
        Train a model with the given model name and seq_files directory
        :param data: {
            methods: [lr, XXX, ...],
            input_path: PATH_TO_SEQ_FILES_FOLDER, or None
            save_path: PATH_TO_SAVE_MODEL_MAP
        }
        :return: if training succeeds, {True and empty string}, else {False, error message}
        """
        ml_models = data["methods"]
        seq_files_dir = data["input_path"]
        save_path = data["save_path"]

        # Do path checking up-front
        save_path = Path(save_path)
        save_dir = save_path.parent
        try:
            # Exist ok, and Creates parent if ok
            save_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            return False, "FAIL_PERMISSION_ERROR"

        # Create result model metrics in the same directory
        save_file_name = save_path.stem
        result_path = save_path.with_name(
            str(save_file_name) + "_metric_results")
        result_path.mkdir(parents=True, exist_ok=True)

        test_ratio = OUModel.TEST_RATIO
        trim = OUModel.TRIM_RATIO
        expose_all = OUModel.EXPOSE_ALL
        txn_sample_rate = OUModel.TXN_SAMPLE_RATE

        trainer = OUModelTrainer(seq_files_dir, result_path, ml_models,
                                 test_ratio, trim, expose_all, txn_sample_rate)
        # Perform training from OUModelTrainer and input files directory
        model_map = trainer.train()

        # Pickle dump the model
        with open(args.save_path + '/ou_model_map.pickle', 'wb') as file:
            pickle.dump((model_map, data_info.instance), file)

        return True, ""

    def infer(self, data: Dict) -> Tuple[Any, bool, str]:
        """
        Do inference on the model, give the data file, and the model_map_path
        :param data: {
            features: 2D float arrays [[float]],
            opunit: Opunit integer for the model
            model_path: model path
        }
        :return: {List of predictions, if inference succeeds, error message}
        """
        features = data["features"]
        opunit = data["opunit"]
        model_path = data["model_path"]

        # Load the model map
        model_map = self._load_model(model_path)
        if model_map is None:
            logging.error(
                f"Model map at {str(model_path)} has not been trained")
            return [], False, "MODEL_MAP_NOT_TRAINED"

        # Parameter validation
        if not isinstance(opunit, str):
            return [], False, "INVALID_OPUNIT"
        try:
            opunit = OpUnit[opunit]
        except KeyError as e:
            logging.error(f"{opunit} is not a valid Opunit name")
            return [], False, "INVALID_OPUNIT"

        features = np.array(features)
        logging.debug(f"Using model on {opunit}")

        model = model_map[opunit]
        if model is None:
            logging.error(f"Model for {opunit} doesn't exist")
            return [], False, "MODEL_NOT_FOUND"

        y_pred = model.predict(features)
        return y_pred.tolist(), True, ""

    def _load_model_from_disk(self, save_path: Path) -> Dict:
        """
        Load model from the path on disk (invoked when missing model cache)
        :param save_path: model path on disk
        :return: OU model map
        """
        with save_path.open(mode='rb') as f:
            model, data_info.instance = pickle.load(f)
        return model


# ==============================================
# main
# ==============================================
if __name__ == '__main__':
    aparser = argparse.ArgumentParser(description="Model Trainer")
    aparser.add_argument('--input_path', default='data/ou_runner_input',
                         help='Input file path for the ou runners')
    aparser.add_argument('--model_results_path', default='data/ou_results',
                         help='Prediction results of the ou models')
    aparser.add_argument('--save_path', default='data/trained_model',
                         help='Path to save the ou models')
    aparser.add_argument('--ml_models', nargs='*', type=str,
                         default=["lr", "rf", "gbm"],
                         help='ML models for the ou trainer to evaluate')
    args = aparser.parse_args()

    # Make output directory if not exists
    os.makedirs(args.save_path, exist_ok=True)

    data = {"input_path": args.input_path,
            "save_path": args.save_path, "methods":  args.ml_models}

    ok,res = OUModel().train(data)
    if ok:
        logging.info(f"Model successfully trained")
    else:
        logging.error(f"Error in training model : {res}")

