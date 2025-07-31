-- Databricks notebook source
-- MAGIC %md
-- MAGIC
-- MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
-- MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
-- MAGIC </div>
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Lab - Creating Bronze Tables from JSON Files
-- MAGIC ### Duration: ~ 15 minutes
-- MAGIC
-- MAGIC In this lab you will ingest a JSON file as Delta table and then flatten the JSON formatted string column.
-- MAGIC
-- MAGIC ### Learning Objectives
-- MAGIC   - Inspect a raw JSON file.
-- MAGIC   - Read in JSON files to a Delta table and flatten the JSON formatted string column.

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

-- MAGIC %run ./Includes/Classroom-Setup-07L

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Run the cell below to view your default catalog and schema. Notice that your default catalog is **dbacademy** and your default schema is your unique **labuser** schema.
-- MAGIC
-- MAGIC **NOTE:** The default catalog and schema are pre-configured for you to avoid the need to specify the three-level name when writing your tables (i.e., catalog.schema.table).

-- COMMAND ----------

SELECT current_catalog(), current_schema()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## B. Lab - JSON Ingestion
-- MAGIC **Scenario:** You are working with your data team on ingesting a JSON file into Databricks. Your job is to ingest the JSON file as is into a bronze table, then create a second bronze table that flattens the JSON formatted string column in the raw bronze table for downstream processing.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### B1. Inspect the Dataset

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 1. In the cell below, view the value of the Python variable `DA.paths.working_dir`. This variable will reference your **dbacademy.ops.labuser** volume, as each user has a different source volume. This variable is created within the classroom setup script to dynamically reference your unique volume.
-- MAGIC
-- MAGIC    Run the cell and review the results. You’ll notice that the `DA.paths.working_dir` variable points to your `/Volumes/dbacademy/ops/labuser` volume.
-- MAGIC
-- MAGIC **Note:** Instead of using the `DA.paths.working_dir` variable, you could also specify the path name directly by right clicking on your volume and selecting **Copy volume path**.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC print(DA.paths.working_dir)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Run the cell to view the data in the `/Volumes/dbacademy/ops/your-labuser-name/json_demo_files/lab_kafka_events.json` file in the location from above.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC spark.sql(f'''
-- MAGIC           SELECT * 
-- MAGIC           FROM json.`{DA.paths.working_dir}/json_demo_files/lab_kafka_events.json`
-- MAGIC           ''').display()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### B2. Create the Raw Bronze Table
-- MAGIC
-- MAGIC Inspect and run the code below to ingest the raw JSON file `/Volumes/dbacademy/ops/your-labuser-name/json_demo_files/lab_kafka_events.json` and create the **lab7_lab_kafka_events_raw** table.
-- MAGIC
-- MAGIC Notice the following:
-- MAGIC - The **value** column is decoded.
-- MAGIC - The **decoded_value** column was created and returns the decoded column as a JSON-formatted string.
-- MAGIC

-- COMMAND ----------

CREATE OR REPLACE TABLE lab7_lab_kafka_events_raw
AS
SELECT 
  *,
  cast(unbase64(value) as STRING) as decoded_value
FROM read_files(
        DA.paths_working_dir || '/json_demo_files/lab_kafka_events.json',
        format => "json", 
        schema => '''
          key STRING, 
          timestamp DOUBLE, 
          value STRING
        ''',
        rescueddatacolumn => '_rescued_data'
      );

-- View the table
SELECT *
FROM lab7_lab_kafka_events_raw;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### B3. Create the Flattened Bronze Table

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 1. Your goal is to flatten the JSON formatted string column **decoded_value** from the table **lab7_lab_kafka_events_raw** to create a new table named **lab7_lab_kafka_events_flattened** for downstream processing. The table should contain the following columns:
-- MAGIC     - **key**
-- MAGIC     - **timestamp**
-- MAGIC     - **user_id**
-- MAGIC     - **event_type**
-- MAGIC     - **event_timestamp**
-- MAGIC     - **items**
-- MAGIC
-- MAGIC     You can use whichever technique you prefer:
-- MAGIC
-- MAGIC     - Parse the JSON formatted string (easiest) to flatten
-- MAGIC       - [Query JSON strings](https://docs.databricks.com/aws/en/semi-structured/json):
-- MAGIC
-- MAGIC     - Convert the JSON formatted string as a VARIANT and flatten
-- MAGIC       - [parse_json function](https://docs.databricks.com/gcp/en/sql/language-manual/functions/parse_json)
-- MAGIC
-- MAGIC     - Convert the JSON formatted string to a STRUCT and flatten
-- MAGIC       - [schema_of_json function](https://docs.databricks.com/aws/en/sql/language-manual/functions/schema_of_json)
-- MAGIC       - [from_json function](https://docs.databricks.com/gcp/en/sql/language-manual/functions/from_json)
-- MAGIC
-- MAGIC **NOTE:** View the lab solution notebook to view the solutions for each.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 2. To begin, run the code below to view the final solution table **lab7_lab_kafka_events_flattened_solution**. This will give you an idea of what your final table should look like.
-- MAGIC
-- MAGIC   **NOTE**: Depending on your solution, the data types of the columns may vary slightly.  
-- MAGIC
-- MAGIC
-- MAGIC ##### Optional Challenge
-- MAGIC
-- MAGIC   As a challenge, after flattening the table, try converting the data types accordingly. Depending on your skill set, you may not convert all columns to the correct data types within the allotted time.
-- MAGIC
-- MAGIC   - **key** STRING
-- MAGIC   - **timestamp** DOUBLE
-- MAGIC   - **user_id** STRING
-- MAGIC   - **event_type** STRING
-- MAGIC   - **event_timestamp** TIMESTAMP
-- MAGIC   - **items** (STRUCT or VARIANT) depending on the method you used.

-- COMMAND ----------

SELECT *
FROM lab7_lab_kafka_events_flattened_solution

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 3. Write the query in the cell below to read the **lab_kafka_events_raw** table and create the flattened table **lab7_lab_kafka_events_flattened** following the requirements from above.

-- COMMAND ----------

---- Parse the JSON formatted STRING
CREATE OR REPLACE TABLE lab7_lab_kafka_events_flattened_str
AS
SELECT 
  key,
  timestamp,
  decoded_value:user_id,
  decoded_value:event_type,
  cast(decoded_value:event_timestamp AS TIMESTAMP),
  from_json(decoded_value:items,'ARRAY<STRUCT<item_id: STRING, price_usd: DOUBLE, quantity: BIGINT>>') AS items
FROM lab7_lab_kafka_events_raw;


---- Display the table
SELECT *
FROM lab7_lab_kafka_events_flattened_str;

-- COMMAND ----------

---- Convert the JSON formatted string as a VARIANT
---- NOTE: The VARIANT decoded_value_variant column is included in this solution to display the column
---- NOTE: Variant data type will not work on Serverless Version 1.
CREATE OR REPLACE TABLE lab7_lab_kafka_events_flattened_variant
AS
SELECT
  key,
  timestamp,
  parse_json(decoded_value) AS decoded_value_variant,
  cast(decoded_value_variant:user_id AS STRING),
  decoded_value_variant:event_type :: STRING,
  decoded_value_variant:event_timestamp :: TIMESTAMP,
  decoded_value_variant:items
FROM lab7_lab_kafka_events_raw;


---- Display the table
SELECT *
FROM lab7_lab_kafka_events_flattened_variant;

-- COMMAND ----------

---- Convert the JSON formatted string as a STRUCT

---- Return the structure of the JSON formatted string
SELECT schema_of_json(decoded_value)
FROM lab7_lab_kafka_events_raw
LIMIT 1;


---- Use the JSON structure from above within the from_json function to convert the JSON formatted string to a STRUCT
---- NOTE: The STRUCT decoded_value_struct column is included in this solution to display the column
CREATE OR REPLACE TABLE lab7_lab_kafka_events_flattened_struct
AS
SELECT
  key,
  timestamp,
  from_json(decoded_value, 'STRUCT<event_timestamp: STRING, event_type: STRING, items: ARRAY<STRUCT<item_id: STRING, price_usd: DOUBLE, quantity: BIGINT>>, user_id: STRING>') AS decoded_value_struct,
  decoded_value_struct.user_id,
  decoded_value_struct.event_type,
  cast(decoded_value_struct.event_timestamp AS TIMESTAMP),
  decoded_value_struct.items
FROM lab7_lab_kafka_events_raw;


---- Display the table
SELECT *
FROM lab7_lab_kafka_events_flattened_struct;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC &copy; 2025 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="blank">Apache Software Foundation</a>.<br/>
-- MAGIC <br/><a href="https://databricks.com/privacy-policy" target="blank">Privacy Policy</a> | 
-- MAGIC <a href="https://databricks.com/terms-of-use" target="blank">Terms of Use</a> | 
-- MAGIC <a href="https://help.databricks.com/" target="blank">Support</a>