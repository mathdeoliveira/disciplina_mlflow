import os 
import sys 

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

import sqlite3
from evidently.report import Report
from evidently.metrics import *
from evidently.metric_preset import DataDriftPreset
from evidently.test_preset import DataDriftTestPreset
import pandas as pd 

from data.data_load import DataLoad

class ModelMonitoring():
    def __init__(self):
        self.query = "SELECT * FROM predictions"
    
    def get_pred_data(self):
        conn = sqlite3.connect("/home/mathdeoliveira/repos/disciplina_mlflow/preds.db")
        df_pred = pd.read_sql_query(self.query, conn)
        conn.close()
        return df_pred 
    
    def get_training_data(self):
        dl = DataLoad()
        df_train = dl.load_data('train_dataset_name')
        return df_train 
    
    def run(self):
        df_cur = self.get_pred_data() # dados atuais 
        df_ref = self.get_training_data().drop('target', axis=1) # dados referencia
        
        model_card = Report(metrics=[
            DatasetSummaryMetric(),
            DataDriftPreset(),
            DatasetMissingValuesMetric()
        ])
        
        model_card.run(reference_data=df_ref,
                       current_data=df_cur)
        model_card.save_html("../../docs/model_monitoring_report.html")
        
mm = ModelMonitoring()
mm.run()