-- Databricks notebook source
-- MAGIC %md
-- MAGIC
-- MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
-- MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
-- MAGIC </div>
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # 2B -  Create Streaming Tables with SQL using Auto Loader
-- MAGIC
-- MAGIC In this demonstration we will create a streaming table to incrementally ingest files from a volume using Auto Loader with SQL. 
-- MAGIC
-- MAGIC When you create a streaming table using the CREATE OR REFRESH STREAMING TABLE statement, the initial data refresh and population begin immediately. These operations do not consume DBSQL warehouse compute. Instead, streaming table rely on serverless DLT for both creation and refresh. A dedicated serverless DLT pipeline is automatically created and managed by the system for each streaming table.
-- MAGIC
-- MAGIC ### Learning Objectives
-- MAGIC
-- MAGIC By the end of this lesson, you should be able to:
-- MAGIC - Create streaming tables in Databricks SQL for incremental data ingestion.
-- MAGIC - Refresh streaming tables using the REFRESH statement.
-- MAGIC
-- MAGIC ### RECOMMENDATION
-- MAGIC
-- MAGIC The CREATE STREAMING TABLE SQL command is the recommended alternative to the legacy COPY INTO SQL command for incremental ingestion from cloud object storage. Databricks recommends using streaming tables to ingest data using Databricks SQL. 
-- MAGIC
-- MAGIC A streaming table is a table registered to Unity Catalog with extra support for streaming or incremental data processing. A DLT pipeline is automatically created for each streaming table. You can use streaming tables for incremental data loading from Kafka and cloud object storage.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## REQUIRED - SELECT CLASSIC COMPUTE
-- MAGIC
-- MAGIC **NOTE: We'll use a classic compute cluster to set up the demonstration files, as Python is required. After that, you'll need to switch to a SQL warehouse to create the streaming tables using SQL.**
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
-- MAGIC
-- MAGIC **TROUBLESHOOTING:** If you select a SQL Warehouse, an error will be returned since Python is used for the setup.

-- COMMAND ----------

-- MAGIC %run ./Includes/Classroom-Setup-Auto-Loader

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## !!!!! REQUIRED - SELECT YOUR SERVERLESS SQL WAREHOUSE !!!!!
-- MAGIC ## !!!!! REQUIRED - SELECT YOUR SERVERLESS SQL WAREHOUSE !!!!!
-- MAGIC
-- MAGIC **NOTE: Creating streaming tables with Databricks SQL requires a SQL warehouse.**.
-- MAGIC
-- MAGIC ![Select Cluster](./Includes/images/selecting_cluster_info.png)
-- MAGIC
-- MAGIC Before executing cells in this notebook, please select the **SHARED SQL WAREHOUSE** in the lab. Follow these steps:
-- MAGIC
-- MAGIC 1. Navigate to the top-right of this notebook and click the drop-down to select compute (it might say **Connect**). Complete one of the following below:
-- MAGIC
-- MAGIC    a. Under **Recent resources**, check to see if you have a **shared_warehouse SQL**. If you do, select it.
-- MAGIC
-- MAGIC    b. If you do not have a **shared_warehouse** under **Recent resources**, complete the following:
-- MAGIC
-- MAGIC     - In the same drop-down, select **More**.
-- MAGIC
-- MAGIC     - Then select the **SQL Warehouse** button.
-- MAGIC
-- MAGIC     - In the drop-down, make sure **shared_warehouse** is selected.
-- MAGIC
-- MAGIC     - Then, at the bottom of the pop-up, select **Start and attach**.
-- MAGIC
-- MAGIC <br></br>
-- MAGIC    <img src="./Includes/images/sql_warehouse.png" alt="SQL Warehouse" width="600">

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 1. Run the following cell to configure your working environment for this notebook.
-- MAGIC
-- MAGIC     **NOTE:** The `DA` object is only used in Databricks Academy courses and is not available outside of these courses. It will dynamically reference the information needed to run the course in the lab environment.

-- COMMAND ----------

-- MAGIC %run ./Includes/Classroom-Setup-SQL-Auto-Loader

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 2. View the default catalog and schema. Confirm the default catalog is **dbacademy** and the default schema is your **labuser** schema.

-- COMMAND ----------

-- DBTITLE 1,Check your default catalog and schema
SELECT current_catalog(), current_schema()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## C. Create Streaming Tables for Incremental Processing

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 1. Complete the following to explore the volume `/Volumes/dbacademy/your-lab-user-schema/csv_files_autoloader_source` and confirm it contains a single CSV file.
-- MAGIC
-- MAGIC    a. Select the catalog icon on the left ![Catalog Icon](./Includes/images/catalog_icon.png).
-- MAGIC
-- MAGIC    b. Expand the **dbacademy** catalog.
-- MAGIC
-- MAGIC    c. Expand your **labuser** schema.
-- MAGIC
-- MAGIC    d. Expand **Volumes**.
-- MAGIC
-- MAGIC    e. Expand the **csv_files_autoloader_source** volume.
-- MAGIC
-- MAGIC    f. Confirm it contains a single CSV file named **000.csv**.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 2. Run the query below to view the data in the CSV file(s) in your cloud storage location. Notice that it was returned in tabular format and contains 3,149 rows.

-- COMMAND ----------

-- DBTITLE 1,Preview the CSV file in your volume
SELECT *
FROM read_files(
  '/Volumes/dbacademy/' || DA.schema_name || '/csv_files_autoloader_source',
  format => 'CSV',
  sep => '|',
  header => true
);

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC #### Create a STREAMING TABLE using Databricks SQL
-- MAGIC 3. Your goal is to create an incremental pipeline that only ingests new files (instead of using traditional batch ingestion). You can achieve this by using [streaming tables in Databricks SQL](https://docs.databricks.com/aws/en/dlt/dbsql/streaming) (Auto Loader).
-- MAGIC
-- MAGIC    - The SQL code below creates a streaming table that will be scheduled to incrementally ingest only new data every week. 
-- MAGIC    
-- MAGIC    - A pipeline is automatically created for each streaming table. You can use streaming tables for incremental data loading from Kafka and cloud object storage.
-- MAGIC
-- MAGIC    **NOTE:** Incremental batch ingestion automatically detects new records in the data source and ignores records that have already been ingested. This reduces the amount of data processed, making ingestion jobs faster and more efficient in their use of compute resources.
-- MAGIC
-- MAGIC    **REQUIRED: Please insert the path of your csv_files_autoloader_source volume in the `read_files` function. This process will take about a minute to run and set up the incremental ingestion pipeline.**

-- COMMAND ----------

-- DBTITLE 1,Create a streaming table
-- YOU WILL HAVE TO REPLACE THE EXAMPLE PATH BELOW WITH THE PATH TO YOUR csv_file_autoloader_source VOLUME.
-- You can find the volume in your navigation bar on the right and insert the path
-- OR you can replace `your-labuser-name` with your specific labuser name (name of your schema)

CREATE OR REFRESH STREAMING TABLE sql_csv_autoloader
SCHEDULE EVERY 1 WEEK     -- Scheduling the refresh is optional
AS
SELECT *
FROM STREAM read_files(
  '/Volumes/dbacademy/labuser11045124_1753761040/csv_files_autoloader_source',  -- Insert the path to you csv_files_autoloader_source volume (example shown)
  format => 'CSV',
  sep => '|',
  header => true
);

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 4. Complete the following to view the streaming table in your catalog.
-- MAGIC
-- MAGIC    a. Select the catalog icon on the left ![Catalog Icon](./Includes/images/catalog_icon.png).
-- MAGIC
-- MAGIC    b. Expand the **dbacademy** catalog.
-- MAGIC
-- MAGIC    c. Expand your **labuser** schema.
-- MAGIC
-- MAGIC    d. Expand your **Tables**.
-- MAGIC
-- MAGIC    e. Find the the **sql_csv_autoloader** table. Notice that the Delta streaming table icon is slightly different from a traditional Delta table:
-- MAGIC     
-- MAGIC     ![Streaming table icon](./Includes/images/streaming_table_icon.png)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 5. Run the cell below to view the streaming table. Confirm that the results contain **3,149 rows**.

-- COMMAND ----------

-- DBTITLE 1,View the streaming table
SELECT *
FROM sql_csv_autoloader;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 6. Describe the STREAMING TABLE and view the results. Notice the following:
-- MAGIC
-- MAGIC - Under **Detailed Table Information**, notice the following rows:
-- MAGIC   - **View Text**: The query that created the table.
-- MAGIC   - **Type**: Specifies that it is a STREAMING TABLE.
-- MAGIC   - **Provider**: Indicates that it is a Delta table.
-- MAGIC
-- MAGIC - Under **Refresh Information**, you can see specific refresh details. Example shown below:
-- MAGIC
-- MAGIC ##### Refresh Information
-- MAGIC
-- MAGIC | Field                   | Value                                                                                                                                         |
-- MAGIC |-------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
-- MAGIC | Last Refreshed          | 2025-06-17T16:12:49.168Z                                                                                                                      |
-- MAGIC | Last Refresh Type       | INCREMENTAL                                                                                                                                   |
-- MAGIC | Latest Refresh Status   | Succeeded                                                                                                                                     |
-- MAGIC | Latest Refresh          | https://example.url.databricks.com/#joblist/pipelines/bed6c715-a7c1-4d45-b57c-4fdac9f956a7/updates/9455a2ef-648c-4339-b61e-d282fa76a92c (this is the path to the Declarative Pipeline that was created for you)|
-- MAGIC | Refresh Schedule        | EVERY 1 WEEKS                                                                                                                                 |

-- COMMAND ----------

-- DBTITLE 1,View the metadata of the streaming table
DESCRIBE TABLE EXTENDED sql_csv_autoloader;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 7. The `DESCRIBE HISTORY` statement displays a detailed list of all changes, versions, and metadata associated with a Delta streaming table, including information on updates, deletions, and schema changes.
-- MAGIC
-- MAGIC     Run the cell below and view the results. Notice the following:
-- MAGIC
-- MAGIC     - In the **operation** column, you can see that a streaming table performs three operations: **CREATE TABLE**, **DLT SETUP** and **STREAMING UPDATE**.
-- MAGIC     
-- MAGIC     - Scroll to the right and find the **operationMetrics** column. In row 1 (Version 2 of the table), the value shows that the **numOutputRows** is 3149, indicating that 3149 rows were added to the **sql_csv_autoloader** table.

-- COMMAND ----------

-- DBTITLE 1,View the history of the streaming table
DESCRIBE HISTORY sql_csv_autoloader;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 8. Complete the following steps to manually add another file to your cloud storage location:  
-- MAGIC    `/Volumes/dbacademy/your-lab-user-schema/csv_files_autoloader_source`.
-- MAGIC
-- MAGIC    a. Click the catalog icon on the left ![Catalog Icon](./Includes/images/catalog_icon.png).
-- MAGIC
-- MAGIC    b. Expand the **dbacademy** catalog.
-- MAGIC
-- MAGIC    c. Expand your **labuser** schema.
-- MAGIC
-- MAGIC    d. Expand **Volumes**.
-- MAGIC
-- MAGIC    e. Open the **auto_loader_staging_files** volume.
-- MAGIC
-- MAGIC    f. Right-click on the **001.csv** file and select **Download volume file** to download the file locally.
-- MAGIC
-- MAGIC    g. Upload the downloaded **001.csv** file to the **csv_files_autoloader_source** volume:
-- MAGIC
-- MAGIC       - Right-click on the **csv_files_autoloader_source** volume. 
-- MAGIC
-- MAGIC       - Select **Upload to volume**.  
-- MAGIC
-- MAGIC       - Choose and upload the **001.csv** file from your local machine.
-- MAGIC
-- MAGIC    h. Confirm your volume **csv_files_autoloader_source** contains two CSV files (**000.csv** and **001.csv**).
-- MAGIC
-- MAGIC
-- MAGIC     **NOTE:** Depending on your laptop’s security settings, you may not be able to download files locally.
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 9. Next, manually refresh the STREAMING TABLE using `REFRESH STREAMING TABLE table-name`. 
-- MAGIC
-- MAGIC - [Refresh a streaming table](https://docs.databricks.com/aws/en/dlt/dbsql/streaming#refresh-a-streaming-table) documentation
-- MAGIC
-- MAGIC     **NOTE:** You can also go back to **Create a STREAMING TABLE using Databricks SQL (direction number 3)** and rerun that cell to incrementally ingest only new files. Once complete come back to step 8.

-- COMMAND ----------

REFRESH STREAMING TABLE sql_csv_autoloader;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 10. Run the cell below to view the data in the **sql_csv_autoloader** table. Notice that the table now contains **6,081 rows**.
-- MAGIC

-- COMMAND ----------

-- DBTITLE 1,View the streaming table
SELECT *
FROM sql_csv_autoloader;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 11. Describe the history of the **sql_csv_autoloader** table. Observe the following:
-- MAGIC
-- MAGIC   - Version 3 of the streaming table includes another **STREAMING UPDATE**.
-- MAGIC
-- MAGIC   - Expand the **operationMetrics** column and note that only **2,932 rows** were incrementally ingested into the table from the new **001.csv** file.
-- MAGIC

-- COMMAND ----------

-- DBTITLE 1,View the history of the streaming table
DESCRIBE HISTORY sql_csv_autoloader;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 12. Drop the streaming table.

-- COMMAND ----------

-- DBTITLE 1,Drop the streaming table
DROP TABLE IF EXISTS sql_csv_autoloader;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Additional Resources
-- MAGIC
-- MAGIC - [Streaming Tables Documentation](https://docs.databricks.com/gcp/en/dlt/streaming-tables)
-- MAGIC
-- MAGIC - [CREATE STREAMING TABLE Syntax](https://docs.databricks.com/aws/en/sql/language-manual/sql-ref-syntax-ddl-create-streaming-table)
-- MAGIC
-- MAGIC - [Using Streaming Tables in Databricks SQL](https://docs.databricks.com/aws/en/dlt/dbsql/streaming)
-- MAGIC
-- MAGIC - [REFRESH (MATERIALIZED VIEW or STREAMING TABLE)](https://docs.databricks.com/aws/en/sql/language-manual/sql-ref-syntax-ddl-refresh-full)
-- MAGIC
-- MAGIC - [COPY INTO (legacy)](https://docs.databricks.com/aws/en/ingestion/#copy-into-legacy)
-- MAGIC
-- MAGIC - [Lakeflow Declarative Pipelines](https://docs.databricks.com/aws/en/dlt/)
-- MAGIC ---
-- MAGIC
-- MAGIC #### BONUS Material: Course Appendix
-- MAGIC
-- MAGIC In the course **Appendix** folder, you'll find a demonstration using Python Auto Loader in the **A2 - Python Auto Loader** notebook. This is extra material that you can explore outside of class.

-- COMMAND ----------



-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC &copy; 2025 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="blank">Apache Software Foundation</a>.<br/>
-- MAGIC <br/><a href="https://databricks.com/privacy-policy" target="blank">Privacy Policy</a> | 
-- MAGIC <a href="https://databricks.com/terms-of-use" target="blank">Terms of Use</a> | 
-- MAGIC <a href="https://help.databricks.com/" target="blank">Support</a>