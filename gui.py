import tkinter as tk
from tkinter import Radiobutton, filedialog
from tkinter.filedialog import askopenfile
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk,ImageFilter
from matplotlib import pyplot as plt
import cv2
import tkinter
import numpy as np


from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.image as mpimg
from sklearn.cluster import KMeans


my_w = tk.Tk()
my_w.geometry("1275x720")  # Size of the window 
my_w.title('Brain Tumor Detection')
my_font1=('times', 18, 'bold')
l1 = tk.Label(my_w,text='Input Image',width=30,font=my_font1)  
l1.grid(row=1,column=1)
l2 = tk.Label(my_w,text='Denoised Image',width=30,font=my_font1)  
l2.grid(row=1,column=2)
l2 = tk.Label(my_w,text='Segmented Image',width=30,font=my_font1)  
l2.grid(row=1,column=3)
b1 = tk.Button(my_w, text='Browse', 
   width=20,command = lambda:choose_file())
#b1.grid(row=10,column=6) 
b1.place(x=100, y=300)
b1 = tk.Button(my_w, text='Denoise', 
   width=20,command = lambda:noise())
#b1.grid(row=2,column=1) 
b1.place(x=500, y=300)

b1 = tk.Button(my_w, text='Image Segmentation', 
   width=20,command = lambda:segment())
#b1.grid(row=2,column=1) 
b1.place(x=900, y=300)

b1 = tk.Button(my_w, text='Detect Tumor', 
   width=20,command = lambda:detect_tumor())
#b1.grid(row=2,column=1) 
b1.place(x=450, y=500)


def choose_file():
    global img,filename
    f_types = [('Jpg Files', '*.jpg')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    img=Image.open(filename)
    img_resized=img.resize((250,250)) # new width & height
    img=ImageTk.PhotoImage(img_resized)
    b2 =tk.Button(my_w,image=img) # using Button 
    b2.grid(row=3,column=1)
def noise():
    import cv2 as cv
    imageread = cv.imread(filename)
    #using medinaBlur() function to remove the noise from the given image
    imagenormal = cv.medianBlur(imageread, 5)
    #displaying the noiseless image as the output on the screen
    cv.imwrite("medianfilter2.png",imagenormal)
    # gorntuyu kaydeder
  
    img = Image.open("/home/whsk3y/PycharmProjects/Python-Median-Filter-master/medianfilter2.png")
    img_resized=img.resize((250,250))
    img = ImageTk.PhotoImage(img_resized)
   

    label1 = tkinter.Label(image=img)
    label1.image = img

    # Position image
    #label1.place(x=<x_coordinate>, y=<y_coordinate>)
    label1.grid(row=3,column=2)
    

def segment():
    
    img=image1 = cv2.imread("medianfilter2.png")
    
    #binarythreshold
    # applying different thresholding
    # techniques on the input image
    # all pixels value above 120 will
    # be set to 255
    ret, thresh1 = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
    # the window showing output images
    # with the corresponding thresholding
    # techniques applied to the input images
    cv2.imwrite("binarythresh2.png",thresh1)
    # Create a photoimage object of the image in the path
    img = Image.open("/home/whsk3y/PycharmProjects/Python-Median-Filter-master/binarythresh2.png")
    img_resized=img.resize((250,250))
    img = ImageTk.PhotoImage(img_resized)
   

    label1 = tkinter.Label(image=img)
    label1.image = img

    # Position image
    #label1.place(x=<x_coordinate>, y=<y_coordinate>)
    label1.grid(row=3,column=3)


    

def detect_tumor():
    img = cv2.imread('binarythresh2.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # threshold input image using otsu thresholding as mask and refine with morphology
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
    kernel = np.ones((9,9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # put mask into alpha channel of result
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask
    # save resulting masked image
    cv2.imwrite('segimage1.png', result)


    
    # Path to the directory containing the images
    p = Path("images/")

    # Initialize the lists to store image data and labels
    image_data = []
    labels = []

    #    Define the labels dictionary
    labels_dict = {'yes': 0, 'no': 1}

    # Load and preprocess the images
    for category in p.glob("*"):
    # Extract the label from the folder name
        label = category.stem
    
        for img_file in category.glob("*"):
            img = Image.open(img_file).convert("L")  # Convert the image to grayscale
            img = img.resize((32, 32))  # Set the desired target size
            img_array = np.array(img)  # Convert PIL image to NumPy array
            image_data.append(img_array)
            labels.append(labels_dict[label])

    # Convert the image data and labels to NumPy arrays
    X = np.array(image_data)
    y = np.array(labels)

    # Split the data into training and testing sets
    train_size = 0.8
    test_size = 0.1
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=42)

    # Perform feature scaling (optional but recommended for SVM)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
    X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1))

    # Create an SVM classifier
    classifier = svm.SVC(kernel='linear')

    # Train the classifier
    classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Load and preprocess a new image for prediction
    img_path = "segimage1.png"
    img = Image.open(img_path).convert("L")  # Convert the image to grayscale
    img = img.resize((32, 32))  # Set the desired target size
    img_array = np.array(img)  # Convert PIL image to NumPy array
    preprocessed_img = scaler.transform(img_array.reshape(1, -1))

    # Predict the class of the new image
    prediction = classifier.predict(preprocessed_img)
    print("Prediction:",prediction)
    if(prediction==0):
        resLabel = tkinter.Label(text="Tumor Detected", height=1, width=20) 
        resLabel.configure(background="White", font=("Comic Sans MS", 16, "bold"), fg="red")
        resLabel.place(x=700, y=500)
        #b1 = tk.Button(my_w, text=' Tumor Type', 
        #width=20,command = lambda:detect_tumor())
        #b1.grid(row=2,column=1) 
        #b1.place(x=450, y=600)
        
        # Path to the directory containing the images
        p = Path("Images/")

# Initialize the lists to store image data and labels
        image_data = []
        labels = []

# Define the labels dictionary
        labels_dict = {'benign': 0, 'malignant': 1}

# Load and preprocess the images
        for category in p.glob("*"):
    # Extract the label from the folder name
            label = category.stem
    
            for img_file in category.glob("*"):
                img = Image.open(img_file).convert("L")  # Convert the image to grayscale
                img = img.resize((32, 32))  # Set the desired target size
                img_array = np.array(img)  # Convert PIL image to NumPy array
                image_data.append(img_array)
                labels.append(labels_dict[label])
        # Convert the image data and labels to NumPy arrays
        X = np.array(image_data)
        y = np.array(labels)

        # Split the data into training and testing sets
        train_size = 0.8
        test_size = 0.1
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=42)

# Perform feature scaling (optional but recommended for SVM)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
        X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1))


        # Create an SVM classifier
        classifier = svm.SVC(kernel='linear')

# Train the classifier
        classifier.fit(X_train, y_train)

# Make predictions on the test set
        y_pred = classifier.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy of tumor type:", accuracy)

# Load and preprocess a new image for prediction
        img_path = "segimage1.png"
        img = Image.open(img_path).convert("L")  # Convert the image to grayscale
        img = img.resize((32, 32))  # Set the desired target size
        img_array = np.array(img)  # Convert PIL image to NumPy array
        preprocessed_img = scaler.transform(img_array.reshape(1, -1))

# Predict the class of the new image
        prediction2 = classifier.predict(preprocessed_img)
        print("Prediction:", prediction2)

        if prediction2 == 0:
           resLabel = tkinter.Label(text="Benign", height=1, width=20)
           resLabel.configure(background="White", font=("Comic Sans MS", 16, "bold"), fg="black")
           resLabel = tkinter.Label(text="BENIGN!!\nThese tumors are non-cancerous and tend to stay in one place without spreading.")
           resLabel.configure(font=("Modern", 11, "bold"), fg="black")
           resLabel.place(x=650, y=600) 
        else:
            print( "Malignant Tumor Detected")
            resLabel = tkinter.Label(text="Malignant.", height=1, width=20)
            resLabel.configure(background="White", font=("Comic Sans MS", 16, "bold"), fg="black")
                                #l1 = tk.Label(my_w,text='',width=30,font=my_font1)  
                                #l1.grid(row=8,column=2) 
            resLabel = tkinter.Label(text="MALIGNANT!!\nThese tumors are cancerous and can spread to other parts of the brain.")
            resLabel.configure(font=("Modern", 11, "bold"), fg="black")
            resLabel.place(x=650, y=600) 
   











    else:
        resLabel = tkinter.Label(text="No Tumor", height=1, width=20)
        resLabel.configure(background="White", font=("Comic Sans MS", 16, "bold"), fg="green")
        resLabel.place(x=700, y=500)
    



            

            


my_w.mainloop()