import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F          
from torchvision.utils import make_grid
from torch.utils.data import DataLoader  
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 4)

def show_mnist_images(images, labels):
    # Menampilkan citra-citra
    grid_image = make_grid(images[:12], nrow=12)
    grid_image = grid_image.permute(1, 2, 0)  # Ubah urutan dimensi menjadi (Width, Height, Color)

    # Menampilkan label
    st.write("Labels: ", labels[:12])

    # Menampilkan gambar menggunakan Streamlit
    st.image(np.array(grid_image))

def load_image(image_file):
	img = Image.open(image_file)
	return img

class MultilayerPerceptron(nn.Module):
    def __init__(self, in_sz=784, out_sz=10, layers=[256,128,64]):
        super().__init__()
        # linear combiner (sigma)
        self.fc1 = nn.Linear(in_sz,layers[0]) #input layer
        self.fc2 = nn.Linear(layers[0],layers[1]) #hidden layer 1
        self.fc3 = nn.Linear(layers[1],layers[2]) #hidden layer 2
        self.fc4 = nn.Linear(layers[2],out_sz) #hidden layer 2 to output
    
    def forward(self,X):
        X = F.relu(self.fc1(X)) #forward input layer to hidden layer 1
        X = F.relu(self.fc2(X)) #result from previous layer, pass to the hidden layer 1 - 2
        X = F.relu(self.fc3(X)) #result from previous layer, pass to the hidden layer 2 - 3
        X = self.fc4(X) #forward process from hidden layer 2 to output
        return F.log_softmax(X, dim=1) #dim=dimension

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    preprocessed_image = transform(image)
    preprocessed_image = preprocessed_image.unsqueeze(0)
    preprocessed_image = preprocessed_image.view(-1, 28*28)  # Mengubah dimensi menjadi (1, 784)
    return preprocessed_image

def main():
    # logo
    logo = Image.open("D:\KULIAH\TA\GUI\logo.png")
    st.image(logo, width = 100)
    st.title("Classification MNIST Dataset Using Artificial Neural Network")

    menu = ["Home", "Train and Test", "Predict"]
    choice = st.sidebar.selectbox("Menu", menu)

    # input dataset
    transform = transforms.ToTensor()
    cwd = os.getcwd()
    data_path = os.path.join(cwd, 'MNIST')

    train_data = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)

    model = MultilayerPerceptron()

    if choice == "Home":
        st.header('Tugas Multimodal Biomedical Image Analysis')
        st.subheader('Anastasia Berlianna Febiola - 07311940000004')

    if choice == "Train and Test":
         st.header("Train and Validation on Dataset")
         bs_tr = st.sidebar.slider("Batch Size Train", 50, 1000, step=50)
         bs_ts = st.sidebar.slider("Batch Size Test", 50, 1000, step=50)
         ep = st.sidebar.number_input("Epoch", min_value=5, max_value=100)
         LR = st.sidebar.selectbox("Learning Rate", [0.00001, 0.0001, 0.001, 0.01])

         st.write("Dataset Type : {}".format(type(train_data[0])))
         st.subheader("Dataset")
         train_loader = DataLoader(train_data, batch_size=bs_tr, shuffle=True)
         for images, labels in train_loader:
             show_mnist_images(images, labels)
             break
         
         st.info("Selected Hyperparameters:\n"
                "- Batch Size Train: {}\n"
                "- Batch Size Test: {}\n"
                "- Epoch: {}\n"
                "- Learning Rate: {}".format(bs_tr, bs_ts, ep, LR))

         if st.sidebar.button("Start Train", use_container_width=True):
             torch.manual_seed(101)
             train_loader = DataLoader(train_data, batch_size=bs_tr, shuffle=True)
             test_loader = DataLoader(test_data, batch_size=bs_ts, shuffle=False)
             criterion = nn.CrossEntropyLoss()
             optimizer = torch.optim.Adam(model.parameters(), lr=LR)
             start_time = time.time()
             epochs = ep

             train_losses = []
             test_losses = []
             train_correct = []
             test_correct = []

             for i in range(epochs):
                trn_corr = 0 #train correct currently
                tst_corr = 0 #test correst
    
                # Run the training batches
                # with enumerate, we're actually going to keep track of what batch number we're on with B.
                for b, (X_train, y_train) in enumerate(train_loader): #y_train=output = label, b = batches, train_loader = return back the image and its label
                    b+=1
        
                    # Apply the model
                    y_pred = model(X_train.view(bs_tr, -1))  # Here we flatten X_train
                    loss = criterion(y_pred, y_train) #calculating error difference
 
                    # calculate the number of correct predictions
                    predicted = torch.max(y_pred.data, 1)[1] #check print(y_pred.data) to know data of one epoch, 1 = actual predicted value
                    batch_corr = (predicted == y_train).sum()
                    trn_corr += batch_corr
        
                    # Update parameters
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Print results per epoch
                    if b == len(train_loader) - 1:
                        st.write(f"Epoch: {i+1}/{epochs} | Batch: {b+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Accuracy: {trn_corr.item()}/{len(train_data)} = {trn_corr.item()*100/len(train_data):.2f}%")
                
                # Update train loss & accuracy for the epoch
                train_losses.append(loss.item())
                train_correct.append(trn_corr.item())

                # Run the testing batches
                with torch.no_grad(): #don't update weight and bias in test data
                    for b, (X_test, y_test) in enumerate(test_loader):

                    # Apply the model
                     y_val = model(X_test.view(bs_ts, -1))  

                     # Calculating the number of correct predictions
                     predicted = torch.max(y_val.data, 1)[1] 
                     tst_corr += (predicted == y_test).sum()

                # Update test loss & accuracy for the epoch
                loss = criterion(y_val, y_test)
                test_losses.append(loss)
                test_correct.append(tst_corr)

             # Calculate training time
             end_time = time.time()
             training_time = end_time - start_time
             st.text(f"Training time: {training_time:.2f} seconds")
             # Save the trained model
             torch.save(model.state_dict(), 'model1.pth')

             # Plot train and test loss

             plt.plot(train_losses, label='Train Loss')
             plt.plot(test_losses, label='Test Loss')
             plt.xlabel('Epoch')
             plt.ylabel('Loss')
             plt.title('Train and Test Loss')
             plt.legend()
             plt.savefig('loss_plot.png')  # Save the plot as an image

             # Plot train and test accuracy
             train_accuracy = [100 * correct / len(train_data) for correct in train_correct]
             test_accuracy = [100 * correct / len(test_data) for correct in test_correct]

             plt.figure()
             plt.plot(train_accuracy, label='Train Accuracy')
             plt.plot(test_accuracy, label='Test Accuracy')
             plt.xlabel('Epoch')
             plt.ylabel('Accuracy (%)')
             plt.title('Train and Test Accuracy')
             plt.legend()
             plt.savefig('accuracy_plot.png')  # Save the plot as an image

             # Show the saved plots using Streamlit
             st.image('loss_plot.png', caption='Train and Test Loss')
             st.image('accuracy_plot.png', caption='Train and Test Accuracy')


         if st.sidebar.button("Start Test", use_container_width=True):
            # Load the saved model
            model.load_state_dict(torch.load('model1.pth'))

            # Extract the data all at once, not in batches
            test_load_all = DataLoader(test_data, batch_size=10000, shuffle=False)

            with torch.no_grad():
                correct = 0
                for X_test, y_test in test_load_all:
                    y_val = model(X_test.view(len(X_test), -1))  # pass in a flattened view of X_test
                    predicted = torch.max(y_val,1)[1]
                    correct += (predicted == y_test).sum()
                st.header('Test on Dataset Result')
                st.markdown(f'**Test accuracy: {correct.item()}/{len(test_data)} = :red[{correct.item()*100/(len(test_data)):7.3f}%]**')

    elif choice == "Predict":
        st.header('Prediction Result on Image')
        st.write(type(test_data))
        index = st.sidebar.slider("Select Image Index", 0, len(test_data) - 1)
        image, label = test_data[index]
        image_array = torch.transpose(image, 0, 2).numpy()
        st.image(image_array, width=150)

        # Function to perform the prediction
        def predict_class(image):
            # Load the trained model
            model = MultilayerPerceptron()  
            model.load_state_dict(torch.load('model1.pth'))
            model.eval()  # Set the model to evaluation mode

            # Perform the prediction
            with torch.no_grad():
                output = model(image)

            predicted_class = torch.argmax(output).item()
            return predicted_class

        # Perform the prediction when the "Predict Class" button is pressed
        if st.button("Predict Class"):
            preprocessed_image = preprocess_image(image)
            predicted_class = predict_class(preprocessed_image)
            #predicted_class = predicted_class.item()  # Convert tensor to Python scalar
            st.write(f"Predicted Class: {predicted_class}")

if __name__ == '__main__':
    main()
