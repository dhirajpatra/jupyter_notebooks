# Boston Housing Prediction MLOPS

This project demonstrates a Machine Learning Operations (MLOps) pipeline for predicting Boston Housing prices. It consists of a gRPC, REST, model saved in both pickle and joblib formats. It also has an inference server and a Flask web application.

## Project Structure

```
MLOPS/
├── model.proto          # gRPC service definition
├── requirements.txt     # Python dependencies
├── save_model.ipynb     # Notebook to train and save the model
├── server/
│   ├── server.py        # gRPC Inference Server
│   ├── client.py        # gRPC Client for testing
│   ├── boston_housing_prediction.pkl # Serialized Model (Pickle)
│   ├── model_pb2.py     # Generated gRPC code
│   └── model_pb2_grpc.py# Generated gRPC code
└── webapp/
    ├── app.py           # Flask Web Application
    ├── templates/       # HTML Templates
    └── ...
```

## Prerequisites

- Python 3.8+
- pip

## Installation

1.  Clone the repository or navigate to the project directory.
2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. gRPC Server

The gRPC server loads the trained model and listens for prediction requests.

1.  Navigate to the `server` directory:
    ```bash
    cd server
    ```
2.  Start the server:
    ```bash
    python server.py
    ```
    You should see: `gRPC Inference server listening on port: 50051`

### 2. gRPC Client (Optional)

You can test the gRPC server using the provided client script.

1.  Open a new terminal.
2.  Navigate to the `server` directory:
    ```bash
    cd server
    ```
3.  Run the client:
    ```bash
    python client.py
    ```
    It will send a sample request and print the prediction.

### 3. Flask Web Application

The web application provides a user interface for the model. It can load the model directly or potentially communicate with the backend (currently configured to load model locally).

1.  Navigate to the `webapp` directory:
    ```bash
    cd webapp
    ```
2.  Start the Flask app:
    ```bash
    python app.py
    ```
3.  Open your browser and go to `http://localhost:5000`.

## Tech Stack

-   **Model**: TensorFlow / Scikit-Learn
-   **Server**: gRPC, Python
-   **Web App**: Flask, HTML/CSS
-   **Serialization**: Pickle / Joblib

## Notes

-   The model is trained using `save_model.ipynb`.
-   Ensure the `server/` directory contains the `boston_housing_prediction.pkl` file before running the server or app.
