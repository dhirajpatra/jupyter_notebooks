import azure.cli.core

# This will fail if you are not logged in
cli = azure.cli.core.AzureCli()
# This will list all storage accounts
cli.invoke(["storage", "account", "list"])


from azureml.core import Workspace
# This will fail if you are not logged in
ws = Workspace.create(name='myworkspace', 
                     subscription_id='...', 
                     resource_group='myresourcegroup', 
                     location='eastus')


from transformers import pipeline
# This will download a pre-trained model
classifier = pipeline("sentiment-analysis")
result = classifier("I love this course!")
print(result)


from datasets import load_dataset

dataset = load_dataset("glue", "mrpc")
train_dataset = dataset["train"]
print(len(train_dataset))


from azureml.opendatasets import PublicHolidays

holidays = PublicHolidays()
holidays_df = holidays.to_pandas_dataframe()
print(holidays_df.head())

