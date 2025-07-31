# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy Workloads with Lakeflow Jobs
# MAGIC ---
# MAGIC
# MAGIC ### Course Agenda
# MAGIC | # | Notebook Name |
# MAGIC | --- | --- |
# MAGIC | 1 | [Creating a Job Using Lakeflow Jobs UI]($./1 - Creating a Job Using Lakeflow Jobs UI) |
# MAGIC | 2 Lab | [Create a Job with Multiple Tasks]($./2 Lab - Create a Job with Multiple Tasks) |
# MAGIC | 3 | [Explore Scheduling Options]($./3 - Explore Scheduling Options) |
# MAGIC | 4 | [Controlling the Flow of a Job]($./4 - Controlling the Flow of a Job) |
# MAGIC | 5 Lab | [Modular Orchestration]($./5 Lab - Modular Orchestration) |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run demo and lab notebooks, you need to use the following Databricks runtime: **`16.4.x-scala2.12`**
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="blank">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy" target="blank">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use" target="blank">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/" target="blank">Support</a>