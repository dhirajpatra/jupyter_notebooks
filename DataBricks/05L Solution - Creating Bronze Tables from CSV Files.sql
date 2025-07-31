-- Databricks notebook source
-- MAGIC %md
-- MAGIC
-- MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
-- MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
-- MAGIC </div>
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Lab - Creating Bronze Tables from CSV Files
-- MAGIC ### Duration: ~15-20 minutes
-- MAGIC
-- MAGIC This lab is divided into two sections: the **course lab** and an **optional challenge**. 
-- MAGIC
-- MAGIC In a live class, if you finish the main lab early, feel free to attempt the challenge. You can also complete the challenge after class. 
-- MAGIC
-- MAGIC **NOTE:** The challenge will require you to use resources outside the scope of this course.
-- MAGIC
-- MAGIC ### Learning Objectives
-- MAGIC   - Inspect CSV files.
-- MAGIC   - Read in CSV files as a Delta table and append with metadata columns.
-- MAGIC   - Record malformed data types using the `_rescued_data` column.  
-- MAGIC
-- MAGIC ### Challenge Learning Objectives:
-- MAGIC   - Clean malformed data types during ingestion by leveraging the `_rescued_data` column.

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
-- MAGIC # Course Lab - Creating Bronze Tables from CSV Files
-- MAGIC
-- MAGIC ### Scenario
-- MAGIC
-- MAGIC You are working with your data team on ingesting a CSV file into Databricks. However, you notice there is a malformed row in your CSV file. Your job is to ingest the file and rescue the malformed data and write as a Delta table.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC ## A. Classroom Setup
-- MAGIC
-- MAGIC Run the following cell to configure your working environment for this notebook.
-- MAGIC
-- MAGIC **NOTE:** The `DA` object is only used in Databricks Academy courses and is not available outside of these courses. It will dynamically reference the information needed to run the course in the lab environment.

-- COMMAND ----------

-- MAGIC %run ./Includes/Classroom-Setup-05L

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Run the cell below to view your default catalog and schema. Notice that your default catalog is **dbacademy** and your default schema is your unique **labuser** schema.
-- MAGIC
-- MAGIC **NOTE:** The default catalog and schema are pre-configured for you to avoid the need to specify the three-level name when writing your tables (i.e., catalog.schema.table).

-- COMMAND ----------

SELECT current_catalog(), current_schema()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## B. Lab - CSV File Ingestion
-- MAGIC Ingest the CSV file as a Delta table.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### B1. Inspect the Dataset
-- MAGIC
-- MAGIC 1. In the cell below, view the value of the SQL variable `DA.paths_working_dir`. This variable will reference your **labuser** volume, as each user has a different source volume. This variable is created within the classroom setup script to dynamically reference your unique volume.
-- MAGIC
-- MAGIC    Run the cell and review the results. You’ll notice that the `DA.paths_working_dir` variable points to your `/Volumes/dbacademy/ops/labuser` volume.
-- MAGIC
-- MAGIC **Note:** Instead of using the `DA.paths_working_dir` variable, you could also specify the path name directly by right clicking on your volume and selecting **Copy volume path**.

-- COMMAND ----------

values(DA.paths_working_dir)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 2. You can concatenate the `DA.paths_working_dir` SQL variable with a string to specify a specific subdirectory in your volume.
-- MAGIC
-- MAGIC    Run the cell below and review the results. You’ll notice that it returns the path to your **lab_malformed_data.csv** file. This method will be used when referencing your volume within the `read_files` function.

-- COMMAND ----------

values(DA.paths_working_dir || '/csv_demo_files/lab_malformed_data.csv')

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 3. Next, let's take a look at our CSV file. 
-- MAGIC
-- MAGIC     Copy the path from the output below and paste it within the backticks in the query to reference the **lab_malformed_data.csv** file.
-- MAGIC
-- MAGIC     The query should use `text.<path_from_above>` to return the headers and rows from the CSV file. 
-- MAGIC
-- MAGIC     Run the cell and view row 4. Notice that the value for the price contains a `$`.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC spark.sql(f'''
-- MAGIC SELECT *
-- MAGIC FROM text.`{DA.paths.working_dir}/csv_demo_files/lab_malformed_data.csv`
-- MAGIC '''
-- MAGIC ).display()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### B2. Ingesting and Rescuing Malformed Data
-- MAGIC
-- MAGIC Begin developing your query to ingest the CSV file in the specified path and view malformed records using the **_rescued_data** column.
-- MAGIC
-- MAGIC #### Requirements
-- MAGIC Your final SQL query should ingest the CSV file using CTAS and `read_files`. **In the cell below, do not create a table yet. Simply start developing your query to ingest and create the table**:
-- MAGIC
-- MAGIC 1. Select all columns from the raw CSV file.
-- MAGIC
-- MAGIC 2. Use the `read_files()` function with appropriate options to read the CSV file. 
-- MAGIC    - **HINT:** Note that the delimiter is a comma (`,`) not a pipe (`|`).
-- MAGIC
-- MAGIC 3. Explicitly define the schema for ingestion. The schema is defined as follows:  
-- MAGIC    - `item_id` (STRING)  
-- MAGIC    - `name` (STRING)  
-- MAGIC    - `price` (DOUBLE)
-- MAGIC
-- MAGIC 4. Use the correct option to include the rescued data column and name it **_rescued_data** to capture malformed rows.
-- MAGIC
-- MAGIC    - **HINT**: If you define a schema you must [use the rescuedDataColumn option](https://docs.databricks.com/aws/en/sql/language-manual/functions/read_files#csv-options) to add the **_rescued_data** column.
-- MAGIC
-- MAGIC **SOLUTION OUTPUT**
-- MAGIC
-- MAGIC Your output result should look like the following:
-- MAGIC | item_id (STRING)   | name (STRING)                   | price (DOUBLE)| _rescued_data (STRING)                                                                                         |
-- MAGIC |-----------|-------------------------|-------|-----------------------------------------------------------------------------------------------------------|
-- MAGIC | M_PREM_Q  | Premium Queen Mattress   | 1795  | null                                                                                                      |
-- MAGIC | M_STAN_F  | Standard Full Mattress   | 945   | null                                                                                                      |
-- MAGIC | M_PREM_A  | Premium Queen Mattress   | null  | {"price":"$100.00","_file_path":"dbfs:/Volumes/dbacademy/ops/peter_s@databricks_com/csv_demo_files/lab_malformed_data.csv"} |
-- MAGIC

-- COMMAND ----------

SELECT * 
FROM read_files(
        DA.paths_working_dir || '/csv_demo_files/lab_malformed_data.csv',
        format => "csv",
        sep => ",",
        header => true,
        schema => '''
              item_id STRING, 
              name STRING, 
              price DOUBLE
        ''',
        rescueddatacolumn => "_rescued_data"
      )

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### B3. Add Additional Metadata Columns During Ingestion
-- MAGIC
-- MAGIC Next, you can create the final bronze table named **05_lab_bronze** that contains the additional metadata columns. Use the query you created above as the starting point.
-- MAGIC
-- MAGIC ### Final Table Requirements
-- MAGIC
-- MAGIC Incorporate the SQL query you created in the previous section and complete the following:
-- MAGIC
-- MAGIC 1. Use a CTAS statement to create the final bronze Delta table named **05_lab_bronze**. 
-- MAGIC
-- MAGIC 1. Ingest the same CSV file `/Volumes/dbacademy/ops/<username>/csv_demo_files/lab_malformed_data.csv`
-- MAGIC
-- MAGIC 1. Use the same defined schema:  
-- MAGIC    - `item_id` (STRING)  
-- MAGIC    - `name` (STRING)  
-- MAGIC    - `price` (DOUBLE)
-- MAGIC
-- MAGIC 1. Use the `_metadata` column to create two new columns named **file_modification_time** and **source_file**  within your SELECT statement.
-- MAGIC    - **HINT:** [_metadata](https://docs.databricks.com/en/ingestion/file-metadata-column.html)
-- MAGIC
-- MAGIC 1. Add a column named **ingestion_time** that provides a timestamp for ingestion. 
-- MAGIC    - **HINT:** Use the [current_timestamp()](https://docs.databricks.com/aws/en/sql/language-manual/functions/current_timestamp) to record the current timestamp at the start of the query evaluation. 
-- MAGIC
-- MAGIC
-- MAGIC **NOTE:** If you need to see how your final table should look, run the next cell to view the solution table **05_lab_solution**.

-- COMMAND ----------

-- Run if you want to see the final table
SELECT * 
FROM 05_lab_solution

-- COMMAND ----------

---- Drop the table if it exists for demonstration purposes
DROP TABLE IF EXISTS 05_lab_bronze;

---- Create the Delta table
CREATE TABLE 05_lab_bronze 
AS
SELECT
  *,
  _metadata.file_modification_time AS file_modification_time,
  _metadata.file_name AS source_file, 
  current_timestamp() as ingestion_time
FROM read_files(
        DA.paths_working_dir || '/csv_demo_files/lab_malformed_data.csv',
        format => "csv",
        sep => ",",
        header => true,
        schema => 'item_id STRING, name STRING, price DOUBLE', 
        rescueddatacolumn => "_rescued_data"
      );

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Run the cell below to view your final table **05_lab_bronze** and compare it with the solution table.

-- COMMAND ----------

-- View the final table
SELECT * 
FROM 05_lab_bronze

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## C. (Optional) Challenge: Rescuing Data
-- MAGIC The challenge is optional during a live class if you have time. This challenge may require you to use documentation. If you run out of time during a live class, try and complete this after class.
-- MAGIC
-- MAGIC #### Optional Challenge Scenario
-- MAGIC You report back to your data team and everyone agrees to clean up any values in the rescued data column that contain a `$` when ingesting as a bronze table. To fix this issue, you agree to handle these edge case during ingestion by leveraging the `_rescued_data` column.
-- MAGIC
-- MAGIC Your team has decided to strip the `$` in the price and simply store the numeric value during ingestion.
-- MAGIC
-- MAGIC
-- MAGIC #### Requirements
-- MAGIC
-- MAGIC - Complete the SQL query below to modify the **_rescued_data** column and correctly fixes the malformed value.
-- MAGIC
-- MAGIC - Your final table should return the following columns:
-- MAGIC   - **item_id**
-- MAGIC   - **name**
-- MAGIC   - **price** (the original price column)
-- MAGIC   - **TO DO**: **price_fixed** (fix the malformed row in the rescued data column `$100` and return all prices a numeric values)
-- MAGIC   - **_rescued_data**
-- MAGIC   - **source_file**
-- MAGIC   - **file_modification_time**
-- MAGIC   - **ingestion_timestamp**
-- MAGIC
-- MAGIC - Use `CREATE TABLE AS` with `read_files()` to read the CSV file and create a table named .
-- MAGIC
-- MAGIC **HINT:** One solution you can use is the [`COALESCE()`](https://docs.databricks.com/aws/en/sql/language-manual/functions/coalesce) function along with the [`REPLACE()`](https://docs.databricks.com/aws/en/sql/language-manual/functions/replace) function to replace the malformed string price (`$100`) with `100`.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC You can view the final table by running the next cell if you would like a hint for building your query.

-- COMMAND ----------

-- Run if you want to see the final table
SELECT * 
FROM 05_lab_challenge_solution;

-- COMMAND ----------

CREATE OR REPLACE TABLE 05_lab_challenge
SELECT
  item_id,
  name,
  price,
  coalesce(price, replace(_rescued_data:price,'$','')) AS price_fixed,
  _rescued_data,
  _metadata.file_modification_time AS file_modification_time,
  _metadata.file_name AS source_file, 
  current_timestamp() as ingestion_timestamp
FROM read_files(
        DA.paths_working_dir || '/csv_demo_files/lab_malformed_data.csv',
        format => "csv",
        sep => ",",
        header => true,
        schema => 'item_id STRING, name STRING, price DOUBLE', 
        rescueddatacolumn => "_rescued_data"
      );

---- Display the table
SELECT *
FROM 05_lab_challenge

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC &copy; 2025 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="blank">Apache Software Foundation</a>.<br/>
-- MAGIC <br/><a href="https://databricks.com/privacy-policy" target="blank">Privacy Policy</a> | 
-- MAGIC <a href="https://databricks.com/terms-of-use" target="blank">Terms of Use</a> | 
-- MAGIC <a href="https://help.databricks.com/" target="blank">Support</a>