-- Databricks notebook source
-- MAGIC %md
-- MAGIC
-- MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
-- MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
-- MAGIC </div>
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # 1 - Exploring the Lab Environment
-- MAGIC
-- MAGIC This demonstration is meant as a review for understanding data objects registered to Unity Catalog (UC) using Databricks. UC is a unified data governance solution designed to centralize and streamline the management of data, metadata, and access control across multiple Databricks workspaces. It provides interoperability across lakehouse formats like Delta lake and Apache Iceberg in addition to providing open APIs and built-in governance for data and AI applications. 
-- MAGIC
-- MAGIC ### Learning Objectives
-- MAGIC By the end of this lesson, you should be able to:
-- MAGIC - Identify and display available Unity Catalog objects, including catalogs, schemas, volumes, and tables within a Databricks.
-- MAGIC - Execute SQL queries to display data directly from files in cloud storage.
-- MAGIC
-- MAGIC **References** For more additional reading and learning, check out the [official UC GitHub repository](https://github.com/unitycatalog/unitycatalog) and [this video on UC on Databricks](https://www.databricks.com/resources/demos/videos/data-governance/unity-catalog-overview).

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## REQUIRED - SELECT CLASSIC COMPUTE
-- MAGIC
-- MAGIC Before executing cells in this notebook, please select your classic compute cluster in the lab. Be aware that **Serverless** is enabled by default and you have a Serverless SQL warehouse with a similar name.
-- MAGIC
-- MAGIC ![Select Cluster](./Includes/images/selecting_cluster_info.png)
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
-- MAGIC 1. Run the following cell to configure your working environment for this notebook.
-- MAGIC
-- MAGIC **NOTE:** The `DA` object is only used in Databricks Academy courses and is not available outside of these courses. It will dynamically reference the information needed to run the course in the lab environment.

-- COMMAND ----------

-- MAGIC %run ./Includes/Classroom-Setup-01

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 2. Complete the following to explore your **labuser** schema using the Catalog UI on the left.
-- MAGIC
-- MAGIC    a. In the left navigation bar, select the catalog icon:  ![Catalog Icon](./Includes/images/catalog_icon.png)
-- MAGIC
-- MAGIC    b. Locate the catalog called **dbacademy** and expand the catalog. 
-- MAGIC
-- MAGIC    c. Expand the **labuser** schema (database). This is your catalog for the course. It should be your lab username (for example, **labuser1234_5678**).

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 3. We want to modify our default catalog and default schema to use **dbacademy** and our **labuser** schema to avoid writing the three-level namespace every time we query and create tables in this course.
-- MAGIC
-- MAGIC     However, before we proceed, note that each of us has a different schema name. Your specific schema name has been stored dynamically in the SQL variable `DA.schema_name` during the classroom setup script.
-- MAGIC
-- MAGIC     Run the code below and confirm that the value of the `DA.schema_name` SQL variable matches your specific schema name (e.g., **labuser1234_678**).

-- COMMAND ----------

-- DBTITLE 1,View SQL variable with schema name
values(DA.schema_name)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 4. Let's modify our default catalog and schema using the `USE CATALOG` and `USE SCHEMA` statements. This eliminates the need to specify the three-level name for objects in your **labuser** schema (i.e., catalog.schema.object).
-- MAGIC
-- MAGIC     - `USE CATALOG` – Sets the current catalog.
-- MAGIC
-- MAGIC     - `USE SCHEMA` – Sets the current schema.
-- MAGIC
-- MAGIC     **NOTE:** Since our dynamic schema name is stored in the SQL variable `DA.schema_name` as a string, we will need to use the `IDENTIFIER` clause to interpret the constant string in our variable as a schema name. The `IDENTIFIER` clause can interpret a constant string as any of the following:
-- MAGIC     - Relation (table or view) name
-- MAGIC     - Function name
-- MAGIC     - Column name
-- MAGIC     - Field name
-- MAGIC     - Schema name
-- MAGIC     - Catalog name
-- MAGIC
-- MAGIC     [IDENTIFIER clause documentation](https://docs.databricks.com/aws/en/sql/language-manual/sql-ref-names-identifier-clause?language=SQL)
-- MAGIC
-- MAGIC     Run the following cell to set and view your default catalog and schema. Confirm that your default catalog is **dbacademy** and your schema is **labuser** (this uses the `DA.schema_name` variable created in the classroom setup script).
-- MAGIC
-- MAGIC **NOTE:** Alternatively, you can simply add your schema name without using the `IDENTIFIER` clause.
-- MAGIC

-- COMMAND ----------

-- DBTITLE 1,Set default catalog and schema to avoid three level namespace
-- Change the default catalog/schema
USE CATALOG dbacademy;
USE SCHEMA IDENTIFIER(DA.schema_name);


-- View current catalog and schema
SELECT 
  current_catalog(), 
  current_schema()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## B. Inspecting and Referencing Unity Catalog Objects
-- MAGIC
-- MAGIC ### Catalogs, Schemas, Volumes, and Tables
-- MAGIC In Unity Catalog, all metadata is registered in a metastore. The hierarchy of database objects in any Unity Catalog metastore is divided into three levels, represented as a three-level namespace (example, `<catalog>.<schema>.<object>`) when you reference tables, views, volumes, models, and functions.
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### B1. Catalogs
-- MAGIC
-- MAGIC Use the `SHOW SCHEMAS IN` statement to view available schemas in the **dbacademy** catalog. Run the cell and view the results. Notice that your **labuser** schema is within the **dbacademy** catalog.

-- COMMAND ----------

-- DBTITLE 1,View schemas in a catalog
SHOW SCHEMAS IN dbacademy;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### B2. Schemas
-- MAGIC Run the `DESCRIBE SCHEMA EXTENDED` statement to see information about your **labuser** schema (database) that was created for you within the **dbacademy** catalog. In the output below, your schema name is in the row called *Namespace Name*.  
-- MAGIC
-- MAGIC **NOTE:** Remember, we are using the `IDENTIFIER` clause to dynamically reference your specific schema name in the lab, since each user will have a different schema name. Alternatively, you can type in the schema name.

-- COMMAND ----------

-- DBTITLE 1,Describe your schema
DESCRIBE SCHEMA EXTENDED IDENTIFIER(DA.schema_name);

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### B3. Tables
-- MAGIC Use the `DESCRIBE TABLE EXTENDED` statement to describe the table `mydeltatable`.
-- MAGIC
-- MAGIC Run the cell and view the results. Notice the following:
-- MAGIC - In the first few cells, you can see column information.
-- MAGIC - Starting at cell 4, you can see additional **Delta Statistics Columns**.
-- MAGIC - Starting at cell 8, you can see additional **Detailed Table Information**.
-- MAGIC
-- MAGIC **NOTE:** Remember, we do not need to reference the three-level namespace (`catalog.schema.table`) because we set our default catalog and schema earlier.
-- MAGIC

-- COMMAND ----------

-- DBTITLE 1,View table metadata
DESCRIBE TABLE EXTENDED mydeltatable

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### B4. Volumes
-- MAGIC
-- MAGIC Volumes are Unity Catalog objects that enable governance over non-tabular datasets. Volumes represent a logical volume of storage in a cloud object storage location. Volumes provide capabilities for accessing, storing, governing, and organizing files.
-- MAGIC
-- MAGIC While tables provide governance over tabular datasets, volumes add governance over non-tabular datasets. You can use volumes to store and access files in **_any_** format, including structured, semi-structured, and unstructured data.
-- MAGIC
-- MAGIC Databricks recommends using volumes to govern access to all non-tabular data. Like tables, volumes can be managed or external.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #### B4.1 UI Exploration
-- MAGIC
-- MAGIC Complete the following to explore the **dbacademy_ecommerce** catalog:
-- MAGIC
-- MAGIC 1. In the left navigation bar, select the catalog icon:  ![Catalog Icon](./Includes/images/catalog_icon.png)
-- MAGIC
-- MAGIC 2. Locate the catalog called **dbacademy_ecommerce** and expand the catalog.
-- MAGIC
-- MAGIC 3. Expand the **v01** schema. Notice that this catalog contains two volumes, **delta** and **raw**.
-- MAGIC
-- MAGIC 4. Expand the **raw** volume. Notice that the volume contains a series of folders.
-- MAGIC
-- MAGIC 5. Expand the **users-historical** folder. Notice that the folder contains a series of files.
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #### B4.2 Volume Exploration with SQL
-- MAGIC
-- MAGIC Run the `DESCRIBE VOLUME` statement to return the metadata for the **dbacademy_ecommerce.v01.raw** volume. The metadata includes the volume name, schema, catalog, type, comment, owner, and more.
-- MAGIC
-- MAGIC Notice the following:
-- MAGIC - Under the **storage_location** column, you can see the cloud storage location for this volume.
-- MAGIC
-- MAGIC - Under the **volume_type** column, it indicates this is a *MANAGED* volume.
-- MAGIC

-- COMMAND ----------

-- DBTITLE 1,Describe a volume
DESCRIBE VOLUME dbacademy_ecommerce.v01.raw;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #### B4.3 List Files in a Volume
-- MAGIC
-- MAGIC
-- MAGIC Use the `LIST` statement to list the available files in the **raw** volume's **users-historical** directory (`/Volumes/dbacademy_ecommerce/v01/raw/users-historical`) and view the results.
-- MAGIC
-- MAGIC Notice the following:
-- MAGIC - Ignore any file names that begin with an underscore (_). These are temporary or intermediate files used when writing files to a location.
-- MAGIC - Scroll down in the results and expand one of the files where the **name** column begins with **part**. Confirm that this directory contains a series of Parquet files.
-- MAGIC
-- MAGIC
-- MAGIC **NOTE:**  When interacting with data in volumes, use the path provided by Unity Catalog, which always follows this format: */Volumes/catalog_name/schema_name/volume_name/*.
-- MAGIC
-- MAGIC For more information on exploring directories and data files managed with Unity Catalog volumes, check out the [Explore storage and find data files](https://docs.databricks.com/en/discover/files.html) documentation.
-- MAGIC

-- COMMAND ----------

LIST '/Volumes/dbacademy_ecommerce/v01/raw/users-historical'

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC &copy; 2025 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="blank">Apache Software Foundation</a>.<br/>
-- MAGIC <br/><a href="https://databricks.com/privacy-policy" target="blank">Privacy Policy</a> | 
-- MAGIC <a href="https://databricks.com/terms-of-use" target="blank">Terms of Use</a> | 
-- MAGIC <a href="https://help.databricks.com/" target="blank">Support</a>