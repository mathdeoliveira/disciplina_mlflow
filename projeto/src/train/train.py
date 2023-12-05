import os
import sys

import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import mlflow
import pandas as pd
import structlog
from feature_engine.discretisation import (EqualFrequencyDiscretiser,
                                           EqualWidthDiscretiser)
from feature_engine.imputation import MeanMedianImputer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = structlog.getLogger()
from evaluation.classifier_eval import ModelEvaluation
from utils.utils import load_config_file

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("prob_loan")


class TrainModels:
    def __init__(self, dados_X: pd.DataFrame, dados_y: pd.DataFrame):
        self.dados_X = dados_X
        self.dados_y = dados_y
        self.model_name = load_config_file().get("model_name")

    def get_best_model(self):
        logger.info("Obtendo o melhor modelo do MLFlow")
        df_mlflow = mlflow.search_runs(
            filter_string="metrics.valid_roc_auc < 1"
        ).sort_values("metrics.valid_roc_auc", ascending=False)
        run_id = df_mlflow.loc[df_mlflow["metrics.valid_roc_auc"].idxmax()]["run_id"]
        df_best_params = df_mlflow.loc[df_mlflow["run_id"] == run_id][
            [
                "params.class_weight",
                "params.discretizer",
                "params.warm_start",
                "params.imputer",
                "params.solver",
                "params.scaler",
                "params.max_iter",
                "params.fit_intercept",
                "params.tol",
                "params.multi_class",
                "params.C",
            ]
        ]
        best_roc_auc = df_mlflow.loc[df_mlflow["metrics.valid_roc_auc"].idxmax()][
            "metrics.valid_roc_auc"
        ]
        return best_roc_auc, df_best_params

    def run(self):
        _, df_best_params = self.get_best_model()
        logger.info(f"Iniciando o treinamendo do modelo: {self.model_name}")

        with mlflow.start_run(run_name="final_model"):
            mlflow.set_tag("model_name", self.model_name)

            model = LogisticRegression(
                warm_start=eval(df_best_params["params.warm_start"].values[0]),
                multi_class=df_best_params["params.multi_class"].values[0],
                class_weight=df_best_params["params.class_weight"].values[0],
                max_iter=int(df_best_params["params.max_iter"].values[0]),
                C=float(df_best_params["params.C"].values[0]),
                solver=df_best_params["params.solver"].values[0],
                tol=float(df_best_params["params.tol"].values[0]),
            )

            pipe = Pipeline(
                [
                    ("imputer", eval(df_best_params["params.imputer"].values[0])),
                    (
                        "discretizer",
                        eval(df_best_params["params.discretizer"].values[0]),
                    ),
                    ("scaler", eval(df_best_params["params.scaler"].values[0])),
                    ("model", model),
                ]
            )
            pipe.fit(self.dados_X, self.dados_y)

            # logar metricas a de avaliacao
            y_val_preds = pipe.predict_proba(self.dados_X)[:, 1]
            model_eval = ModelEvaluation(model, self.dados_X, self.dados_y)
            val_roc_auc = model_eval.evaluate_predictions(self.dados_y, y_val_preds)
            mlflow.log_metric("valid_roc_auc", val_roc_auc)

            # registrar o modelo
            mlflow.sklearn.log_model(
                pipe,
                self.model_name,
                pyfunc_predict_fn="predict_proba",
                input_example=self.dados_X.iloc[[0]],
                registered_model_name=self.model_name,
            )
