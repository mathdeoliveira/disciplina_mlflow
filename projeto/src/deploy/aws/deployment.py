from mlflow.deployments import get_deploy_client

app_name = "prob-loan-sagemaker"
arn = "ARN_CRIADO" # ROLE
image_ecr_uri = "URI_IMAGEM_DOCKER" # ECR
region = "us-east-1" 

model_uri = "CAMINHO_MODELO_TREINADO_MLFLOW"


config = dict(
                execution_role_arn=arn,
                bucket_name="New-s3-bucket",
                image_url=image_ecr_uri,
                region_name=region,
                archive=False,
                instance_type="ml.m4.xlarge",
                instance_count=1,
                synchronous=True,
                timeout_seconds=3600,
                variant_name="prod-variant-3",
                tags={"training_timestamp": "2023-12-05"},
 )

client = get_deploy_client("sagemaker")
deploy_client = client.create_deployment(app_name,
                                         model_uri=model_uri,
                                         flavor='python_function',
                                         config=config)

print(f'deploy_client: {deploy_client}')
