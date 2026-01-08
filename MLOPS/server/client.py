import grpc
import model_pb2
import model_pb2_grpc

def run():
    # Connect to the server
    channel = grpc.insecure_channel('localhost:50051')
    stub = model_pb2_grpc.InferenceServiceStub(channel)

    # Create a test housing record (Boston dataset format)
    test_features = {
        "CRIM": 0.00632, "ZN": 18.0, "INDUS": 2.31, "CHAS": 0.0,
        "NOX": 0.538, "RM": 6.575, "AGE": 65.2, "DIS": 4.09,
        "RAD": 1.0, "TAX": 296.0, "PTRATIO": 15.3, "B": 396.9, "LSTAT": 4.98
    }

    # Make request
    response = stub.Predict(model_pb2.PredictRequest(features=test_features))
    print(f"Prediction received: {response.prediction}")

if __name__ == '__main__':
    run()