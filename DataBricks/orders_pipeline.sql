----------------------------
-- ORDERS DLT PIPELINE
----------------------------

-- 1. In the left navigation pane, confirm you are in the 'Pipeline' tab. Here you should see your pipeline assets.


-- 2. In your pipeline project, you should see two folders: an 'exploration' folder and an 'orders' folder.

  -- a. The 'exploration' folder contains a sample exploration notebook that will not be included in the pipeline. 

  -- b. The 'orders' folder contains the orders pipeline code (orders_pipeline.sql, this file).

  -- NOTE: You can structure your pipeline project and files however you would like.


-- 3. Complete the following steps to see how to include or exclude folders from within your pipeline from the pipeline:

  -- a. Expand the 'exploration' folder and notice that the 'sample_exploration' notebook has an icon to the right of it. Hover over the icon to see that it is hiding this notebook from the pipeline. 

  -- b. Right-click on the 'exploration' folder. Notice the option 'Include folder as pipeline source code’. You can select this option and it will include this folder in the pipeline project. Keep the folder excluded. You can also include or exclude specific files within a folder.


-- 4. Complete the following steps to explore and modify the settings of the pipeline:

  -- a. In the left navigation pane within your pipeline project, below the 'Pipeline' and 'All files' tabs, select the 'settings' icon (looks like a gear). A pop-up should appear on the right with the settings for your pipeline.

  -- b. The first section provides information about your pipeline, such as the Pipeline ID, type, name, and creator. You can select the 'edit' icon to modify any of the settings. Leave as is.
  
  -- c. The 'Code assets' section automatically references all files in your pipeline project that are not hidden from the pipeline.

      -- The 'root folder' specifies this entire folder (this pipeline project).
      -- The 'source code' will display all files associated with this pipeline. In this example, it should show only orders folder.
      -- You can also reference a file or notebook outside of the project if necessary using the 'Configure paths' option.

  -- d. The 'Default location for data assets' section specifies the default catalog and schema to use in the pipeline.
  
    -- **TODO**: Select 'Edit the catalog and schema':
        -- Set the default catalog to your 'labuser' catalog
        -- Set the default schema to 'default'.
        -- Click 'Save'.
        -- NOTE: With Lakeflow Declarative Pipelines you can publish streaming tables and materialized views to any catalog and schema, you are not restricted to the defaults.

  -- e. The 'Compute' section enables you to select your specific compute. Complete the following steps to modify your compute cluster: 

      -- **TODO**: a. Select the 'edit' icon. You should see that Serverless is selected. 

      -- **TODO**: b. Deselect Serverless. Here, you can modify the Compute to your desired specifications and add any associated cluster policies or tags to this pipeline.

      -- **TODO**: c. Reselect Serverless Compute and navigate back to the pipeline settings.

  -- f. The 'Configuration' section enables you to add variables with default values to use in your pipeline code. Complete the following steps to add the 'source' variable pointing to your raw data volume:

      -- **TODO**: a. Select 'Add configuration'. 

      -- **TODO**: b. For 'Key', enter 'source'.

      -- **TODO**: c. For 'Value', paste the volume path of your `dbacademy.ops.labuser` volume. This is the volume with our JSON files to ingest that you copied earlier.
          -- EXAMPLE: `/Volumes/dbacademy/ops/labuser1234_5678`

      -- **TODO**: d. Select 'Save'

  -- g. The 'Budget' section enables you to add a budget policy for Serverless compute. Budget policies consist of tags that are applied to any serverless compute activity incurred by a user assigned to the policy. The tags are logged in your billing records, allowing you to attribute serverless usage to specific budgets.

  -- h. Expand the 'Advanced Settings' section and select 'Edit advanced settings'. In Advanced Settings, you can modify the following:

      -- a. 'Pipeline mode' - Sets the pipeline to run continuously, or triggered. Leave it as 'Triggered' for now.

      -- b. 'Pipeline user mode' - Sets the pipeline to 'Development' or 'Production'. We are developing, so leave it as 'Development'.

      -- c. 'Channel' - Controls the pipeline runtime version.
      
      -- d. 'Event Logs' - The pipeline event log contains all information related to a pipeline, including audit logs, data quality checks, pipeline progress, and data lineage. You can write this as a table to the schema. Leave this deselected for now.

  -- i. Select anywhere within the code here to close the settings.


-- 5. Explore the Declarative SQL code below (You can also use Python to create your pipeline. We will focus on SQL here).

-------------------------------------------------------
-- SQL Pipeline Code
-------------------------------------------------------
-- NOTE: The default catalog is set to your 'labuser' catalog, so specifying the catalog is not required for the code below since we are using the default catalog.
-------------------------------------------------------


-- A. Create the bronze streaming table in your labuser.1_bronze_db schema from a JSON files in your volume
  -- NOTE: read_files references the 'source' configuration key from your pipeline settings. 
  -- NOTE: 'source' = '/Volumes/dbacademy/ops/your-labuser-name'
CREATE OR REFRESH STREAMING TABLE 1_bronze_db.orders_bronze_demo2 
AS 
SELECT 
  *,
  current_timestamp() AS processing_time,
  _metadata.file_name AS source_file
FROM STREAM read_files(  -- Perform incremental reads with checkpoints
    "${source}/orders",  -- Uses the source configuration variable set in the pipeline settings
    format => 'JSON'
);


-- B. Create the silver streaming table in your labuser.2_silver_db schema (database)
CREATE OR REFRESH STREAMING TABLE 2_silver_db.orders_silver_demo2 
AS 
SELECT 
  order_id,
  timestamp(order_timestamp) AS order_timestamp, 
  customer_id,
  notifications
FROM STREAM 1_bronze_db.orders_bronze_demo2 ; -- References the streaming orders_bronze table for incrementally processing


-- C. Create the materialized view aggregation from the orders_silver table with the summarization
CREATE OR REFRESH MATERIALIZED VIEW 3_gold_db.gold_orders_by_date_demo2 
AS 
SELECT 
  date(order_timestamp) AS order_date, 
  count(*) AS total_daily_orders
FROM 2_silver_db.orders_silver_demo2  -- Aggregates the full orders_silver streaming table with optimizations where applicable
GROUP BY date(order_timestamp);
---------------------------------------------------------


-- 6. Complete the following steps to verify the correctness of the pipeline source code without updating any tables:

    -- **TODO**: a. Select 'Dry run' from the top navigation bar.

    -- b. Notice that the pipeline is in 'DRY RUN' mode in the left window and is processing each step (this takes about 1–2 minutes).

    -- c. Since we are creating two streaming tables and one materialized view, we expect to see three items in the 'Pipeline graph' after the dry run completes:
        -- [orders_bronze_demo2] -> [orders_silver_demo2] -> [gold_orders_by_date_demo2]

    -- d. After the 'Dry run' completes, explore the bottom window. Observe the following:
      -- In the 'Catalog' column, you’ll see the catalog the object will write to.
      -- In the 'Schema' column, you’ll see the schema the object will write to.
      -- In the 'Type' column, you’ll see the object type.

    -- TROUBLESHOOTING: If your pipeline returns an error, ensure the following are set correctly:
      -- The default catalog is set to your 'labuser' catalog.
      -- The default schema is 'default'.
      -- The 'source' configuration variable references your volume path: '/Volumes/dbacademy/ops/your-lab-user-name'.


-- 7. Once ready, run the pipeline by selecting the drop-down next to the 'Run pipeline' button in the top navigation bar. This provides two options:

    -- a. 'Run pipeline': This streams data through each table and computes results in the materialized view. Re-running it won’t reprocess files that were already ingested.

    -- b. WARNING!!!! 'Run pipeline with full table refresh': This reprocesses **ALL** data. Use with caution—if your source files are deleted periodically, a full refresh will remove all data and fail to re-read deleted files.
      -- TABLE PROPERTY TO AVOID FULL REFRESH: pipelines.reset.allowed = false

    -- **TODO**: c. Select 'Run pipeline'. Notice that the right window displays each stage of the pipeline.


-- 8. (1–3 minutes) While the pipeline is running, observe the 'Pipeline graph' on the right. As each object is created, a data flow appears. You’ll also see row counts update as processing completes.

    -- 174 rows are ingested from Raw JSON → Bronze → Silver, as expected.

    -- The materialized view aggregation contains 7 rows.

    -- NOTE: If you've already run the pipeline and run it again using 'Run pipeline', the result may be 0 rows since no new files were added.


-- 9. After the pipeline completes, explore it using the new editor:

  -- a. The 'Pipeline graph' in the editor visualizes the data flow.

  -- b. The bottom window provides details about the tables and materialized views created, including locations, durations, and row (record) counts.

  -- **TODO**: c. Select the 'Performance' tab to view performance metrics. Then return to 'Tables'.

  -- **TODO**: d. Select the 'orders_bronze_demo2' streaming table. You can view the data, 'Table metrics', and 'Performance' for that object.

  -- **TODO**: e. Select the arrow to the left of 'All tables' in the bottom window to return to the full pipeline objects.

  -- **TODO**: f. Select the 'gold_orders_by_date_demo2' materialized view. Confirm that it summarizes data by date as expected.

  -- **TODO**: g. Use the navigation bar in the bottom window to explore the 'Data', 'Table metrics', and 'Performance' for the materialized view.


-- 10. You can also select objects directly within the DAG to view related information on the right. Select the 'orders_silver' object to update the details in the bottom window.


-- 11. Run the pipeline again by selecting the 'Run pipeline' button. Once it completes, explore the updated results. While it's running, consider how many rows will be ingested?

  -- a. After the second run completes, note that no new rows were added—this is expected if no new files were added to the original data source.


-- 12. Keep this tab open and return to the notebook '2 - Developing a Simple Pipeline'. Follow the instructions in section 'C. Add a New File to Cloud Storage' to upload a new JSON file.


-- 13. After adding the new file, run the pipeline again. While it's running, monitor the 'Pipeline graph'. Notice that only 25 rows (from the new file `01.json`) were ingested into the streaming Bronze and Silver tables, and the materialized view was recomputed efficiently using the latest data.


-- 14. When the pipeline completes, you can close this tab and return to the notebook '2 - Developing a Simple Pipeline' to continue exploring the streaming tables and materialized view (section D. Exploring Your Streaming Tables).