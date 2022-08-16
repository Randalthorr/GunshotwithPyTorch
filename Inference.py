import torch

# ----------------------------
# Inference
# ----------------------------
def inference(model, val_dl, device):
    correct_prediction = 0
    total_prediction = 0
    device
    counter = 0
    # Disable gradient updates
    with torch.no_grad():
        for data in val_dl:
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)
            print('Parameters for data' + str(counter) + ': ')
            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s
            print('labels: ', labels)
            print('inputs: ', inputs)

            # Get predictions
            outputs = model(inputs)
            print('outputs: ', outputs)
            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)
            print('prediction: ', prediction)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

    acc = correct_prediction / total_prediction
    print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')
