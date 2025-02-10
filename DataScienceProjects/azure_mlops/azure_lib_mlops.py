import azure.cli.core

cli = azure.cli.core.AzureCli()
cli.invoke(["storage", "account", "list"])


from azureml.core import Workspace

ws = Workspace.create(name='myworkspace', 
                     subscription_id='...', 
                     resource_group='myresourcegroup', 
                     location='eastus')


from transformers import pipeline

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

