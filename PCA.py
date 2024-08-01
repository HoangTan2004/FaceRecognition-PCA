import os
import numpy as np
import matplotlib.pyplot as plot

dataset_path = 'C:/Users/HP/Downloads/database/'

img_width = 92
img_height = 112

#Training tensor
training_tensor = np.ndarray(shape = (98, img_height*img_width), dtype = np.float64)
for i in range (98):
    img = plot.imread(dataset_path + 'training/' + str(i + 1) + '.pgm')
    training_tensor[i,:] = np.array(img, dtype = 'float64').flatten()
    plot.subplot(14, 7, i + 1)
    plot.imshow(img, cmap = 'gray')
    plot.tick_params(labelleft = False, labelbottom = False, bottom = False, top = False, left = False, right = False, which = 'both')
    plot.subplots_adjust(right = 0.6, top = 1)
    
plot.show()

#Testing tensor
testing_tensor = np.ndarray(shape = (42, img_height*img_width), dtype = np.float64)
for i in range (42):
    img = plot.imread(dataset_path + 'testing/' + str(i + 1) + '.pgm')
    testing_tensor[i, :] = np.array(img, dtype = 'float64').flatten()
    plot.subplot(14, 3, i + 1)
    plot.imshow(img, cmap = 'gray')
    plot.tick_params(labelleft = False, labelbottom = False, bottom =  False, top = False, left = False, right = False, which = 'both')
    plot.subplots_adjust(right = 0.6, top = 1)
    
plot.show()

#PCA Algorithm
mean_face = np.zeros((1, img_height*img_width))
for i in training_tensor:
    mean_face = np.add(mean_face, i)
    
mean_face = np.divide(mean_face, float(len(training_tensor))).flatten()
plot.imshow(mean_face.reshape(img_height, img_width), cmap = 'gray')
plot.tick_params(labelleft = False, labelbottom = False, bottom = False, top = False, left = False, right = False, which = 'both')
plot.title("Mean face of training set")
plot.show()

normalize_training_tensor = np.ndarray(shape = (len(training_tensor), img_height*img_width))
for i in range(len(training_tensor)):
    normalize_training_tensor[i] = np.subtract(training_tensor[i], mean_face)
    
for i in range(len(training_tensor)):
    img = normalize_training_tensor[i].reshape(img_height, img_width)
    plot.subplot(14, 7, i + 1)
    plot.imshow(img, cmap = 'gray')
    plot.tick_params(labelleft = False, labelbottom = False, bottom = False, top = False, left = False, right = False, which = 'both')
    plot.subplots_adjust(right = 0.6, top = 1)
    
plot.show()

cov_matrix = np.dot(normalize_training_tensor, normalize_training_tensor.T)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

eig_pairs = [(eigenvalues[idx], eigenvectors[:, idx]) for idx in range(len(eigenvalues))]
eig_pairs.sort(reverse = True)
eigvalues_sort = [eig_pairs[idx][0] for idx in range(len(eigenvalues))]
eigvectors_sort = [eig_pairs[idx][1] for idx in range(len(eigenvalues))]

var_comp_sum = np.cumsum(eigvalues_sort) / sum(eigvalues_sort)
num_comp = range(1, len(eigvalues_sort) + 1)

plot.title("Cumulative Proportion of Variance and Components Kept")
plot.xlabel("Principal Components")
plot.ylabel("Cumulative Proportion of Variance")
plot.scatter(num_comp, var_comp_sum)
plot.show()

print("Number of eigenvectors:", len(eigvectors_sort))
n_components = 80
print("The value of k:", n_components)
reduced_data = np.array(eigvectors_sort[:n_components]).transpose()

proj_data = np.dot(training_tensor.transpose(), reduced_data)
proj_data = proj_data.transpose()

for i in range(proj_data.shape[0]):
    img = proj_data[i].reshape(img_height, img_width)
    plot.subplot(10, 8, i + 1)
    plot.imshow(img, cmap = 'gray')
    plot.tick_params(labelleft = False, labelbottom = False, bottom = False, top = False, left = False, right = False, which = 'both')
    plot.subplots_adjust(right = 0.6, top = 1)

plot.show()

w = np.array([np.dot(proj_data, img) for img in normalize_training_tensor])
print(w.shape)

#Function to calculate the best accuracy and the best t0 value
def evaluate_performance(t0, testing_tensor, training_tensor, mean_face, proj_data, w):
    correct = 0
    total = len(testing_tensor)
    for imgNum in range(total):
        unknown_face_vector = testing_tensor[imgNum,:]
        normalize_uface_vector = np.subtract(unknown_face_vector, mean_face)
        w_unknown = np.dot(proj_data, normalize_uface_vector)
        diff = w - w_unknown
        norms = np.linalg.norm(diff, axis = 1)
        idx = np.argmin(norms)
        if norms[idx] < t0:
            setNum = int(imgNum / 3)
            if (idx >= (7*setNum) and idx < (7*(setNum + 1))):
                correct += 1
        else:
            if (imgNum >= 43):
                correct += 1
    accuracy = correct / total
    return accuracy

#Find the best t0 value for the best accuracy
t0_values = [1000000*i for i in range(20, 80)]
best_t0 = None
best_accuracy = 0

for t0 in t0_values:
    accuracy = evaluate_performance(t0, testing_tensor, training_tensor, mean_face, proj_data, w)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_t0 = t0

print(f"The most optimal t0: {best_t0} with the best accuracy: {round(best_accuracy, 4)}")

#Regconition the face with the best t0 and the training tensor
count = 0
num_img = 0
correct = 0
def recognition(imgNum, proj_data, w, best_t0):
    global count, highest_min, num_img, correct
    num_img += 1
    #Add the unknown image to the plot
    unknown_face_vector = testing_tensor[imgNum, :]
    normalize_uface_vector = np.subtract(unknown_face_vector, mean_face)
    plot.subplot(14, 6, 1 + count)
    plot.imshow(unknown_face_vector.reshape(img_height, img_width), cmap = 'gray')
    plot.title("Input: " + str(imgNum + 1), fontsize = 10)
    plot.tick_params(labelleft = False, labelbottom = False, bottom = False, top = False, left = False, right = False, which='both')
    count += 1
    
    w_unknown = np.dot(proj_data, normalize_uface_vector)
    diff = w - w_unknown
    norms = np.linalg.norm(diff, axis = 1)
    idx = np.argmin(norms)
    plot.subplot(14, 6, count + 1)
    setNum = int(imgNum/3)
    t0 = best_t0
    
    #Find the matched images
    if norms[idx] < t0:
        if (idx >= (7*setNum) and idx < (7*(setNum + 1))):
            plot.title("Matched with: " + str(idx + 1), color = 'g', fontsize = 10)
            plot.imshow(training_tensor[idx, :].reshape(img_height, img_width), cmap = 'gray')
            correct += 1
        else:
            plot.title("Matched with: " + str(idx + 1), color = 'r', fontsize = 10)
            plot.imshow(training_tensor[idx, :].reshape(img_height, img_width), cmap = 'gray')
    else:
        if (imgNum >= 43):
            plot.title("Unknown face!", color = 'g', fontsize = 10)
            correct += 1
        else:
            plot.title("Unknown face!", color = 'r', fontsize = 10)
    plot.tick_params(labelleft = False, labelbottom = False, bottom = False, top = False, left = False, right = False, which = 'both')
    plot.subplots_adjust(right = 0.6, top = 1)
    count += 1

fig = plot.figure(figsize = (40, 40))
for i in range(len(testing_tensor)):
    recognition(i, proj_data, w, best_t0)

plot.show()
print("Accuracy: {}/{} = {}".format(correct, num_img, round(correct/num_img, 4)))