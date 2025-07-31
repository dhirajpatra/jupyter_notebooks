-- Databricks notebook source
-- MAGIC %md
-- MAGIC
-- MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
-- MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
-- MAGIC </div>
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # 4 - Handling CSV Ingestion with the Rescued Data Column
-- MAGIC
-- MAGIC In this demonstration, we will focus on ingesting CSV files into Delta Lake using the `CTAS` (`CREATE TABLE AS SELECT`) pattern with the `read_files()` method and exploring the rescued data column. 
-- MAGIC
-- MAGIC ### Learning Objectives
-- MAGIC
-- MAGIC By the end of this lesson, you will be able to:
-- MAGIC
-- MAGIC - Ingest CSV files as Delta tables using the `CREATE TABLE AS SELECT` (CTAS) statement with the `read_files()` function.
-- MAGIC - Define and apply an explicit schema with `read_files()` to ensure consistent and reliable data ingestion.
-- MAGIC - Handle and inspect rescued data that does not conform to the defined schema.

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

-- MAGIC %run ./Includes/Classroom-Setup-04

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Run the cell below to view your default catalog and schema. Notice that your default catalog is **dbacademy** and your default schema is your unique **labuser** schema.
-- MAGIC
-- MAGIC **NOTE:** The default catalog and schema are pre-configured for you to avoid the need to specify the three-level name for when writing your tables (i.e., catalog.schema.table).

-- COMMAND ----------

SELECT current_catalog(), current_schema()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## B. Overview of CTAS with `read_files()` for Ingestion of CSV Files
-- MAGIC
-- MAGIC CSV (Comma-Separated Values) files are a simple text-based format for storing data, where each line represents a row and values are separated by commas.
-- MAGIC
-- MAGIC In this demonstration, we will use CSV files imported from cloud storage. Let’s explore how to ingest these raw CSV files to Delta Lake.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### B1. Inspecting CSV files
-- MAGIC
-- MAGIC 1. List available CSV files from `dbacademy_ecommerce/v01/raw/sales-csv` directory. Confirm that 4 CSV files exist in the volume.

-- COMMAND ----------

LIST '/Volumes/dbacademy_ecommerce/v01/raw/sales-csv'

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 2. Query the CSV files by path in the `/Volumes/dbacademy_ecommerce/v01/raw/sales-csv`volume directly and view the results. Notice the following:
-- MAGIC
-- MAGIC    - The data files include a header row containing the column names.
-- MAGIC
-- MAGIC    - The columns are delimited by the pipe character (`|`). 
-- MAGIC
-- MAGIC      For example, the first row reads:  
-- MAGIC      ```order_id|email|transactions_timestamp|total_item_quantity|purchase_revenue_in_usd|unique_items|items```
-- MAGIC
-- MAGIC      The pipe (`|`) indicates column separation, meaning there are seven columns:  
-- MAGIC      **order_id**, **email**, **transactions_timestamp**, **total_item_quantity**, **purchase_revenue_in_usd**, **unique_items**, and **items**.

-- COMMAND ----------

-- DBTITLE 1,Query raw CSV files
SELECT * 
FROM csv.`/Volumes/dbacademy_ecommerce/v01/raw/sales-csv`
LIMIT 5;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 3. Run the cell below to query the CSV files using the default options in the `read_files` function.
-- MAGIC
-- MAGIC    Review the results. Notice that the CSV files were **not** queried correctly in the table output.
-- MAGIC
-- MAGIC    To fix this, we’ll need to provide additional options to the `read_files()` function for proper ingestion of CSV files.

-- COMMAND ----------

-- DBTITLE 1,Read CSV files with default options in read_files
SELECT * 
FROM read_files(
        "/Volumes/dbacademy_ecommerce/v01/raw/sales-csv",
        format => "csv"
      )
LIMIT 5;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### B2. Using CSV Options with `read_files()`
-- MAGIC
-- MAGIC 1. The code in the next cell ingests the CSV files using the `read_files()` function with some additional options.
-- MAGIC
-- MAGIC    In this example, we are using the following options with the `read_files()` function:    [CSV options](https://docs.databricks.com/aws/en/sql/language-manual/functions/read_files#csv-options)
-- MAGIC
-- MAGIC    - The first argument specifies the path to the CSV files.
-- MAGIC
-- MAGIC    - `format => "csv"` — Indicates that the files are in CSV format.
-- MAGIC
-- MAGIC    - `sep => "|"` — Specifies that columns are delimited by the pipe (`|`) character.
-- MAGIC
-- MAGIC    - `header => true` — Tells the reader to use the first row as column headers.
-- MAGIC    
-- MAGIC    - Although we're using CSV files in this demonstration, other file types (like JSON or Parquet) can also be used by specifying different options.
-- MAGIC
-- MAGIC    Run the cell and view the results. Notice the CSV files were read correctly, and a new column named **_rescued_data** appeared at the end of the result table.
-- MAGIC
-- MAGIC **NOTE:** A **_rescued_data** column is automatically included to capture any data that doesn't match the inferred or provided schema.

-- COMMAND ----------

-- DBTITLE 1,Specify CSV options in read_files
SELECT * 
FROM read_files(
        "/Volumes/dbacademy_ecommerce/v01/raw/sales-csv",
        format => "csv",
        sep => "|",
        header => true
      )
LIMIT 5;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 2. Now that we’ve successfully queried the CSV files using `read_files()`, let’s use a CTAS (`CREATE TABLE AS SELECT`) statement with the same query to complete the following:
-- MAGIC     - Create a Delta table named **sales_bronze**. 
-- MAGIC     - Add an ingestion timestamp and ingestion metadata columns to our **sales_bronze** table.
-- MAGIC
-- MAGIC         - **Ingestion Timestamp:** To record when the data was ingested, use the [`current_timestamp()`](https://docs.databricks.com/aws/en/sql/language-manual/functions/current_timestamp) function. It returns the current timestamp at the start of query execution and is useful for tracking ingestion time.
-- MAGIC
-- MAGIC         - **Metadata Columns:** To include file metadata, use the [`_metadata`](https://docs.databricks.com/en/ingestion/file-metadata-column.html) column, which is available for all input file formats. This hidden column allows access to various metadata attributes from the input files.
-- MAGIC             - Use `_metadata.file_modification_time` to capture the last modification time of the input file.
-- MAGIC             - Use `_metadata.file_name` to capture the name of the input file.
-- MAGIC             - [File metadata column](https://docs.databricks.com/gcp/en/ingestion/file-metadata-column)
-- MAGIC
-- MAGIC
-- MAGIC
-- MAGIC
-- MAGIC     Run the cell and review the results. You should see that the **sales_bronze** table was created successfully with the CSV data and additional metadata columns.

-- COMMAND ----------

-- DBTITLE 1,Create a table with read_files from CSV files
-- Drop the table if it exists for demonstration purposes
DROP TABLE IF EXISTS sales_bronze;


-- Create the Delta table
CREATE TABLE sales_bronze AS
SELECT 
  *,
  _metadata.file_modification_time AS file_modification_time,
  _metadata.file_name AS source_file, 
  current_timestamp() as ingestion_time 
FROM read_files(
        "/Volumes/dbacademy_ecommerce/v01/raw/sales-csv",
        format => "csv",
        sep => "|",
        header => true
      );


-- Display the table
SELECT *
FROM sales_bronze

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 3. View the column data types of the **sales_bronze** table. Notice that the `read_files()` function automatically infers the schema if one is not explicitly provided.
-- MAGIC
-- MAGIC       **NOTE:** When the schema is not provided, `read_files()` attempts to infer a unified schema across the discovered files, which requires reading all the files unless a LIMIT statement is used. Even when using a LIMIT query, a larger set of files than required might be read to return a more representative schema of the data.
-- MAGIC
-- MAGIC      - [Schema inference](https://docs.databricks.com/aws/en/sql/language-manual/functions/read_files#csv-options)

-- COMMAND ----------

-- DBTITLE 1,View the inferred schema
DESCRIBE TABLE EXTENDED sales_bronze;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### B3. (BONUS) Python Equivalent

-- COMMAND ----------

-- MAGIC %python
-- MAGIC
-- MAGIC df = (spark
-- MAGIC       .read 
-- MAGIC       .option("header", True) 
-- MAGIC       .option("sep","|") 
-- MAGIC       .option("rescuedDataColumn", "_rescued_data")       # <--------- Add the rescued data column
-- MAGIC       .csv("/Volumes/dbacademy_ecommerce/v01/raw/sales-csv")
-- MAGIC     )
-- MAGIC
-- MAGIC df.display()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## C. Troubleshooting Common CSV Issues
-- MAGIC
-- MAGIC
-- MAGIC 1. To begin, let’s quickly explore your data source raw files volume. Complete the following steps to view your course volume in **dbacademy.ops.labuser**:
-- MAGIC
-- MAGIC    a. In the left navigation bar, select the catalog icon ![Catalog Icon](./Includes/images/catalog_icon.png).
-- MAGIC
-- MAGIC    b. Expand the **dbacademy** catalog.
-- MAGIC
-- MAGIC    c. Expand the **ops** schema.
-- MAGIC
-- MAGIC    d. Expand **Volumes**. You should see a volume with your **labuser** name, which contains the source data to ingest.
-- MAGIC
-- MAGIC    e. Expand your **labuser** volume. Notice that this volume contains a series of subdirectories. We will be using the **csv_demo_files** directory in your volume.
-- MAGIC
-- MAGIC    f. Expand the **csv_demo_files** subdirectory. Notice that it contains the files:
-- MAGIC       - **malformed_example_1_data.csv**
-- MAGIC       - **malformed_example_2_data.csv**

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 2. In the cell below, view the value of the SQL variable `DA.paths_working_dir`. This variable will reference the path to your **labuser** volume from above, as each user has a different source volume. This variable is created within the classroom setup script to dynamically reference your unique volume.
-- MAGIC
-- MAGIC    Run the cell and review the results. You’ll notice that the `DA.paths_working_dir` variable points to your `/Volumes/dbacademy/ops/labuser` volume.
-- MAGIC
-- MAGIC **NOTE:** Instead of using the `DA.paths_working_dir` variable, you could also specify the path name directly by right clicking on your volume and selecting **Copy volume path**.

-- COMMAND ----------

values(DA.paths_working_dir)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 3. You can concatenate the `DA.paths_working_dir` SQL variable with a string to specify a specific subdirectory in your specific volume.
-- MAGIC
-- MAGIC    Run the cell below and review the results. You’ll notice that it returns the path to your **malformed_example_1_data.csv** file. This method will be used when referencing your volume within the `read_files` function.

-- COMMAND ----------

values(DA.paths_working_dir || '/csv_demo_files/malformed_example_1_data.csv')

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### C1. Defining a Schema During Ingestion
-- MAGIC
-- MAGIC We want to read the CSV file into the bronze table using a defined schema.
-- MAGIC
-- MAGIC **Explicit schemas benefits:**
-- MAGIC - Reduce the risk of inferred schema inconsistencies, especially with semi-structured data like JSON or CSV.
-- MAGIC - Enable faster parsing and loading of data, as Spark can immediately apply the correct types and structure without inferring the schema.
-- MAGIC - Improve performance with large datasets by significantly reducing compute overhead.
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 1. The query below will reference the **malformed_example_1_data.csv** file. This will allow you to view the CSV file as text for inspection.
-- MAGIC
-- MAGIC    Run the query and review the results. Notice the following:
-- MAGIC
-- MAGIC    - The CSV file is **|** delimited.
-- MAGIC
-- MAGIC    - The CSV file contains headers.
-- MAGIC    
-- MAGIC **NOTE:** The **transactions_timestamp** column contains the string *aaa* in the first row, which will cause issues during ingestion when attempting read the **transactions_timestamp** column as a BIGINT.

-- COMMAND ----------

-- DBTITLE 1,View the CSV file as text
-- MAGIC %python
-- MAGIC spark.sql(f'''
-- MAGIC     SELECT *
-- MAGIC     FROM text.`{DA.paths.working_dir}/csv_demo_files/malformed_example_1_data.csv`
-- MAGIC ''').display()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 2. Use the `read_files` function to see how this CSV file is read into the table. Run the cell and view the results. 
-- MAGIC
-- MAGIC     **IMPORTANT** Notice that the malformed value *aaa* in the **transactions_timestamp** column causes the column to be read as a STRING. However, we want the **transactions_timestamp** column to be read into the bronze table as a BIGINT.
-- MAGIC

-- COMMAND ----------

-- DBTITLE 1,Use read_files without a schema
SELECT *
FROM read_files(
        DA.paths_working_dir || '/csv_demo_files/malformed_example_1_data.csv',
        format => "csv",
        sep => "|",
        header => true
      );

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 3. You can define a schema for the `read_files()` function to read in the data with a specific structure.
-- MAGIC
-- MAGIC    a. Use the `schema` option to define the schema. In this case, we'll read in the following:
-- MAGIC    - **order_id** as INT  
-- MAGIC    - **email** as STRING  
-- MAGIC    - **transactions_timestamp** as BIGINT
-- MAGIC
-- MAGIC    b. Use the `rescuedDataColumn` option to collect all data that can’t be parsed due to data type mismatches or schema mismatches into a separate column for review.
-- MAGIC
-- MAGIC    Run the cell and review the results. Notice that row 1 (*aaa*) could not be read using the defined schema, so it was placed in the **_rescued_data** column. Keeping rows that don’t conform to the schema allows you to inspect and process them as needed.
-- MAGIC
-- MAGIC **NOTE:** Defining a schema when using `read_files` in Databricks improves performance by skipping the expensive schema inference step and ensures consistent, reliable data parsing. It's especially beneficial for large or semi-structured datasets.

-- COMMAND ----------

-- DBTITLE 1,Define a schema with read_files
SELECT *
FROM read_files(
        DA.paths_working_dir || '/csv_demo_files/malformed_example_1_data.csv',
        format => "csv",
        sep => "|",
        header => true,
        schema => '''
            order_id INT, 
            email STRING, 
            transactions_timestamp BIGINT''', 
        rescueddatacolumn => '_rescued_data'    -- Create the _rescued_data column
      );

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #### Summary: Rescued Data Column
-- MAGIC
-- MAGIC The rescued data column ensures that rows that don’t match with the schema are rescued instead of being dropped. The rescued data column contains any data that isn’t parsed for the following reasons:
-- MAGIC - The column is missing from the schema.
-- MAGIC - Type mismatches
-- MAGIC - Case mismatches

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### C2. Handling Missing Headers During Ingestion 
-- MAGIC In this example, the CSV file contains a missing header by mistake.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 1. Run the cell below to view the **malformed_example_2_data.csv** file. Notice that the first row contains headers, but the first column name is missing.

-- COMMAND ----------

-- DBTITLE 1,View the CSV file as text
-- MAGIC %python
-- MAGIC spark.sql(f'''
-- MAGIC   SELECT *
-- MAGIC   FROM text.`{DA.paths.working_dir}/csv_demo_files/malformed_example_2_data.csv`
-- MAGIC ''').display()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 2. Let's try to create a table using the **malformed_example_2_data.csv** file with the `read_files()` function. Run the cell and review the results.
-- MAGIC
-- MAGIC     Notice the following:
-- MAGIC     - The first column of the CSV file was not read into the table as a standard column and was instead placed in the **_rescued_data** column.
-- MAGIC
-- MAGIC     - The **_rescued_data** column stores the rescued data as a JSON-formatted string.
-- MAGIC     
-- MAGIC     - When inspecting the **_rescued_data** JSON-formatted string, you'll see that the unnamed column from the CSV file is represented with a key of **_c0**, which contains the column value as a string, along with a **_file_path** key.

-- COMMAND ----------

-- DBTITLE 1,Use read_files to read a malformed CSV file
-- Drop the table if it exists for demonstration purposes
DROP TABLE IF EXISTS demo_4_example_2_bronze;

-- Create Delta table by ingesting CSV file
CREATE OR REPLACE TABLE demo_4_example_2_bronze AS
SELECT *
FROM read_files(
        DA.paths_working_dir || '/csv_demo_files/malformed_example_2_data.csv',
        format => "csv",
        sep => "|",
        header => true
      );


-- Display the table
SELECT *
FROM demo_4_example_2_bronze;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 3. The **_rescued_data** column is a JSON-formatted string. We won’t go into detail on how to handle this type of data here, as it will be covered in a later demo and lab.
-- MAGIC
-- MAGIC     However, it's important to note that you can extract values from the **_rescued_data** column and add them to your bronze table. To obtain the value from the **_c0** field, you can use the `_rescued_data:_c0` syntax, as shown in the next cell.
-- MAGIC
-- MAGIC     **NOTE:** The output from running the next cell returns **order_id** as the rescued column.

-- COMMAND ----------

-- DBTITLE 1,Fix the rescued data column
SELECT
  cast(_rescued_data:_c0 AS BIGINT) AS order_id,
  *
FROM read_files(
        DA.paths_working_dir || '/csv_demo_files/malformed_example_2_data.csv',
        format => "csv",
        sep => "|",
        header => true
      )

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC &copy; 2025 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="blank">Apache Software Foundation</a>.<br/>
-- MAGIC <br/><a href="https://databricks.com/privacy-policy" target="blank">Privacy Policy</a> | 
-- MAGIC <a href="https://databricks.com/terms-of-use" target="blank">Terms of Use</a> | 
-- MAGIC <a href="https://help.databricks.com/" target="blank">Support</a>