import os
# Set environment variables BEFORE importing tensorflow/grpc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# Removed CUDA_VISIBLE_DEVICES='-1' to allow GPU usage

import time
import grpc
import pickle
import numpy as np
import tensorflow as tf
from concurrent import futures

# Import generated gRPC classes
import model_pb2
import model_pb2_grpc 

# Load the model globally
model_path = os.path.join(os.path.dirname(__file__), 'boston_housing_prediction.pkl')
try:
    with open(model_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Extract model and scaler if saved as a dictionary
    if isinstance(checkpoint, dict):
        model = checkpoint.get('model')
        scaler = checkpoint.get('scaler')
    else:
        model = checkpoint
        scaler = None
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Could not find {model_path}")
    exit(1)

class InferenceService(model_pb2_grpc.InferenceServiceServicer):
    def Predict(self, request, context):
        feats = request.features
        
        # Standard Boston Housing feature order (13 features)
        order = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]

        try:
            # Build feature vector
            x_vec = np.array([[feats.get(name, 0.0) for name in order]], dtype=float)

            # Apply scaling if the scaler was found in the pickle
            if scaler is not None:
                x_vec = scaler.transform(x_vec)

            # Make prediction
            preds = model.predict(x_vec).ravel().tolist()

            return model_pb2.PredictResponse(
                prediction=preds,
                model_version="1.0.0"
            )
        except Exception as e:
            context.set_details(f"Inference error: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return model_pb2.PredictResponse()

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceService(), server)
    server.add_insecure_port("[::]:50051")
    
    # Print BEFORE blocking with wait_for_termination
    print("gRPC Inference server listening on port: 50051")
    server.start()
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.stop(0)

# FIX: Moved outside serve() and removed extra indentation
if __name__ == '__main__':
    serve()