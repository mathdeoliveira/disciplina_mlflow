from mlflow.deployments import get_deploy_client

app_name = "prob-loan-sagemaker"
arn = "arn:aws:iam::452416756053:role/aws_sagemaker_for_deploy" # ROLE
image_ecr_uri = "452416756053.dkr.ecr.us-east-1.amazonaws.com/mlflow-pyfunc:2.8.1" # ECR
region = "us-east-1" 

model_uri = "/home/mathdeoliveira/repos/disciplina_mlflow/projeto/mlartifacts/1/ea4e68e210c34563b272a1819c36beab/artifacts/modelo.joblib"


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