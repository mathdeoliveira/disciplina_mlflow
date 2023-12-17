import json 

import pandas as pd 
from azure.ai.ml import MLClient 
from azure.identity import DefaultAzureCredential

workspace_name = "prob-loan-ws"
workspace_location = "East US"
resource_group = "azure-mlops"
subscription_id = "SUBSCRIPTION_ID"
endpoint_name = "prob-loan-endpoint"

ml_client = MLClient(DefaultAzureCredential(),
                     subscription_id,
                     resource_group,
                     workspace_name)

df_test = pd.read_csv('CAMINHO_DADOS_DE_TESTE')
data = {"input_data": df_test.iloc[[0]].to_dict(orient='split')}
print(data)

with open("file.json", "w") as f:
    json.dump(data, f)
    
response = ml_client.online_endpoints.invoke(
    endpoint_name=endpoint_name,
    request_file="file.json"
)

print(response)
