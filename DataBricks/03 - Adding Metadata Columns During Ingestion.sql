-- Databricks notebook source
-- MAGIC %md
-- MAGIC
-- MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
-- MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
-- MAGIC </div>
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # 3 - Adding Metadata Columns During Ingestion
-- MAGIC
-- MAGIC In this demonstration, we'll explore how to add metadata columns during data ingestion. 
-- MAGIC
-- MAGIC This process will include adding metadata, converting Unix timestamps to standard `DATE` format, and row  ingestion times.
-- MAGIC
-- MAGIC ### Learning Objectives
-- MAGIC
-- MAGIC By the end of this lesson, you should be able to:
-- MAGIC
-- MAGIC - Modify columns during data ingestion from cloud storage to your bronze table.
-- MAGIC - Add the current ingestion timestamp to the bronze.
-- MAGIC - Use the `_metadata` column to extract file-level metadata (e.g., file name, modification time) during ingestion.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## REQUIRED - SELECT CLASSIC COMPUTE
-- MAGIC
-- MAGIC Before executing cells in this notebook, please select your classic compute cluster in the lab. Be aware that **Serverless** is enabled by default and you have a Shared SQL warehouse.
-- MAGIC
-- MAGIC <!-- ![Select Cluster](./Includes/images/selecting_cluster_info.png) -->
-- MAGIC
-- MAGIC Follow these steps to select the classic compute cluster:
-- MAGIC
-- MAGIC
-- MAGIC 1. Navigate to the top-right of this notebook and click the drop-down menu to select your cluster. By default, the notebook will use **Serverless**.
-- MAGIC
-- MAGIC 2. If your cluster is available, select it and continue to the next cell. If the cluster is not shown:
-- MAGIC
-- MAGIC    - Click **More** in the drop-down.
-- MAGIC
-- MAGIC    - In the **Attach to an existing compute resource** window, use the first drop-down to select your unique cluster.
-- MAGIC
-- MAGIC **NOTE:** If your cluster has terminated, you might need to restart it in order to select it. To do this:
-- MAGIC
-- MAGIC 1. Right-click on **Compute** in the left navigation pane and select *Open in new tab*.
-- MAGIC
-- MAGIC 2. Find the triangle icon to the right of your compute cluster name and click it.
-- MAGIC
-- MAGIC 3. Wait a few minutes for the cluster to start.
-- MAGIC
-- MAGIC 4. Once the cluster is running, complete the steps above to select your cluster.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC ## A. Classroom Setup
-- MAGIC
-- MAGIC Run the following cell to configure your working environment for this notebook.
-- MAGIC
-- MAGIC **NOTE:** The `DA` object is only used in Databricks Academy courses and is not available outside of these courses. It will dynamically reference the information needed to run the course in the lab environment.

-- COMMAND ----------

-- MAGIC %run ./Includes/Classroom-Setup-03

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Run the cell below to view your default catalog and schema. Notice that your default catalog is **dbacademy** and your default schema is your unique **labuser** schema.
-- MAGIC
-- MAGIC **NOTE:** The default catalog and schema are pre-configured for you to avoid the need to specify the three-level name when writing your tables to your **labuser** schema (i.e., catalog.schema.table).

-- COMMAND ----------

-- DBTITLE 1,View current catalog and schema
SELECT current_catalog(), current_schema()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## B. Explore the Data Source Files
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 1. We'll create a table containing historical user data from Parquet files stored in the volume  
-- MAGIC    `'/Volumes/dbacademy_ecommerce/v01/raw/users-historical'` within Unity Catalog.
-- MAGIC
-- MAGIC    Use the `LIST` statement to view the files in this volume. Run the cell and review the results.
-- MAGIC
-- MAGIC    View the values in the **name** column that begin with **part-**. This shows that this volume contains multiple **Parquet** files.

-- COMMAND ----------

-- DBTITLE 1,List files in a raw/users-historical volume
LIST '/Volumes/dbacademy_ecommerce/v01/raw/users-historical'

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## C. Adding Metadata Columns to the Bronze Table During Ingestion
-- MAGIC
-- MAGIC When ingesting data into the Bronze layer, you can apply transformations during ingestion and also retrieve metadata about the input files using the **_metadata** column.
-- MAGIC
-- MAGIC The **_metadata** column is a hidden column available for all supported file formats. To include it in the returned data, you must explicitly select it in the read query that specifies the source.
-- MAGIC
-- MAGIC
-- MAGIC ### Ingestion Requirements
-- MAGIC
-- MAGIC During data ingestion, we'll perform the following actions:
-- MAGIC
-- MAGIC 1. Convert the parquet Unix timestamp to a `DATE` column.
-- MAGIC
-- MAGIC 2. Include the **input file name** to indicate the data raw source.
-- MAGIC
-- MAGIC 3. Include the **last modification** timestamp of the input file.
-- MAGIC
-- MAGIC 4. Add the **file ingestion time** to the Bronze table.
-- MAGIC
-- MAGIC **Note:** The `_metadata` column is available across all supported input file formats.
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 1. Run the cell below to display the parquet data in the `"/Volumes/dbacademy_ecommerce/v01/raw/users-historical"` volume and view the results.
-- MAGIC
-- MAGIC     Notice that the **user_first_touch_timestamp** column has a Unix timestamp.

-- COMMAND ----------

SELECT *
FROM read_files(
  "/Volumes/dbacademy_ecommerce/v01/raw/users-historical",
  format => 'parquet')
LIMIT 10;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC ### C1. Convert the Unix Time on Ingestion to Bronze
-- MAGIC
-- MAGIC The Unix timestamp column **user_first_touch_timestamp** values represent the time in microseconds since the Unix epoch (January 1, 1970).
-- MAGIC
-- MAGIC To create a readable date column, use the [`from_unixtime()`](https://docs.databricks.com/en/sql/language-manual/functions/from_unixtime.html) function, converting the **user_first_touch_timestamp** from microseconds to seconds by dividing by 1,000,000.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 1. Run the query and review the results. The query generates a new column, **first_touch_date**, by converting the Unix timestamp into a human-readable date column.
-- MAGIC
-- MAGIC    Run the cell and view the **first_touch_date** column. Notice the **first_touch_date** column is cast to a data type of **DATE**.

-- COMMAND ----------

-- DBTITLE 1,Convert UNIX timestamp
SELECT
  *,
  cast(from_unixtime(user_first_touch_timestamp/1000000) AS DATE) AS first_touch_date
FROM read_files(
  "/Volumes/dbacademy_ecommerce/v01/raw/users-historical",
  format => 'parquet')
LIMIT 10;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC ### C2. Adding Column Metadata on Ingestion
-- MAGIC
-- MAGIC
-- MAGIC
-- MAGIC
-- MAGIC The following metadata can be added to the bronze table:
-- MAGIC
-- MAGIC - `_metadata.file_modification_time`: Adds the last modification time of the input file.
-- MAGIC
-- MAGIC - `_metadata.file_name`: Adds the input file name.
-- MAGIC
-- MAGIC - [`current_timestamp()`](https://docs.databricks.com/aws/en/sql/language-manual/functions/current_timestamp): Returns the current timestamp (`TIMESTAMP` data type) when the query starts, useful for tracking ingestion time.
-- MAGIC
-- MAGIC You can read more about the `_metadata` column in the [Databricks documentation](https://docs.databricks.com/en/ingestion/file-metadata-column.html).

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 1. Run the query below to add the following columns:
-- MAGIC
-- MAGIC    - **file_modification_time** and **file_name**, using the **_metadata** column to capture input file details.  
-- MAGIC    
-- MAGIC    - **ingestion_time**, which records the exact time the data was ingested.
-- MAGIC
-- MAGIC    Review the results. You should see the new columns **file_modification_time**, **source_file**, and **ingestion_time** added to the output.

-- COMMAND ----------

-- DBTITLE 1,Add metadata columns
SELECT
  *,
  cast(from_unixtime(user_first_touch_timestamp / 1000000) AS DATE) AS first_touch_date,
  _metadata.file_modification_time AS file_modification_time,      -- Last data source file modification time
  _metadata.file_name AS source_file,                              -- Ingest data source file name
  current_timestamp() as ingestion_time                            -- Ingestion timestamp
FROM read_files(
  "/Volumes/dbacademy_ecommerce/v01/raw/users-historical",
  format => 'parquet')
LIMIT 10;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### C3. Creating the Final Bronze Table
-- MAGIC 1. Put it all together with the `CTAS` statement to create the Delta table.
-- MAGIC
-- MAGIC     Run the cell to create and view the new table **historical_users_bronze**.
-- MAGIC     
-- MAGIC     Confirm that the new columns **first_touch_date**, **file_modification_time**, **source_file** and **ingestion_time** were created successfully in the bronze table.

-- COMMAND ----------

-- DBTITLE 1,Create the final bronze table
-- Drop the table if it exists for demonstration purposes
DROP TABLE IF EXISTS historical_users_bronze;


-- Create an empty table
CREATE TABLE historical_users_bronze AS
SELECT
  *,
  cast(from_unixtime(user_first_touch_timestamp / 1000000) AS DATE) AS first_touch_date,
  _metadata.file_modification_time AS file_modification_time,      -- Last data source file modification time
  _metadata.file_name AS source_file,                              -- Ingest data source file name
  current_timestamp() as ingestion_time                            -- Ingestion timestamp
FROM read_files(
  "/Volumes/dbacademy_ecommerce/v01/raw/users-historical",
  format => 'parquet');


-- View the final bronze table
SELECT * 
FROM historical_users_bronze
LIMIT 10;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### C4. Exploring the Final Bronze Table

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 1. With the additional metadata columns added to the bronze table, you can explore metadata information from the input files. For example, you can execute a query to see how many rows came from each Parquet file by querying the **source_file** column.

-- COMMAND ----------

-- DBTITLE 1,Count rows by parquet file
SELECT 
  source_file, 
  count(*) as total
FROM historical_users_bronze
GROUP BY source_file
ORDER BY source_file;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## D. (BONUS) Python Equivalent

-- COMMAND ----------

-- MAGIC %python
-- MAGIC
-- MAGIC from pyspark.sql.functions import col, from_unixtime, current_timestamp
-- MAGIC from pyspark.sql.types import DateType
-- MAGIC
-- MAGIC # 1. Read parquet files in cloud storage into a Spark DataFrame
-- MAGIC df = (spark
-- MAGIC       .read
-- MAGIC       .format("parquet")
-- MAGIC       .load("/Volumes/dbacademy_ecommerce/v01/raw/users-historical")
-- MAGIC     )
-- MAGIC
-- MAGIC
-- MAGIC # 2. Add metadata columns
-- MAGIC df_with_metadata = (
-- MAGIC     df.withColumn("first_touch_date", from_unixtime(col("user_first_touch_timestamp") / 1_000_000).cast(DateType()))
-- MAGIC       .withColumn("file_modification_time", col("_metadata.file_modification_time"))
-- MAGIC       .withColumn("source_file", col("_metadata.file_name"))
-- MAGIC       .withColumn("ingestion_time", current_timestamp())
-- MAGIC )
-- MAGIC
-- MAGIC
-- MAGIC # 3. Save as a Delta table
-- MAGIC (df_with_metadata
-- MAGIC  .write
-- MAGIC  .format("delta")
-- MAGIC  .mode("overwrite")
-- MAGIC  .saveAsTable(f"dbacademy.{DA.schema_name}.historical_users_bronze_python_metadata")
-- MAGIC )
-- MAGIC
-- MAGIC
-- MAGIC # 4. Read and display the table
-- MAGIC historical_users_bronze_python_metadata = spark.table(f"dbacademy.{DA.schema_name}.historical_users_bronze_python_metadata")
-- MAGIC
-- MAGIC display(historical_users_bronze_python_metadata)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC &copy; 2025 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="blank">Apache Software Foundation</a>.<br/>
-- MAGIC <br/><a href="https://databricks.com/privacy-policy" target="blank">Privacy Policy</a> | 
-- MAGIC <a href="https://databricks.com/terms-of-use" target="blank">Terms of Use</a> | 
-- MAGIC <a href="https://help.databricks.com/" target="blank">Support</a>