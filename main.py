import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


################################################################################################################
# 1.1
################################################################################################################

mat = np.array([[180, 28, 8],
                [170, 24, 0]])  # Create a matrix. Each row is a person, each column a certain characteristic

# We set up our plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(mat[:, 0], mat[:, 1], mat[:, 2], marker='o') # We plot our points
ax.set(title='1.1 plot', xlabel='height', ylabel='Age', zlabel='Zodiac') # Label our plot
ax.set(xlim=[150,200],ylim=[10, 40], zlim=[0, 12])  # Set the axis

plt.show()

################################################################################################################
# 1.2 Calculate distances
################################################################################################################
euclidean_distance = np.linalg.norm(mat[0, :] - mat[1, :]) # The squareroot of the sum of the squared differences
manhattan_distance = np.sum(mat[0, :] - mat[1, :])  # The difference in each dimension summed
print(euclidean_distance)
print(manhattan_distance)

################################################################################################################
# 1.3 Find midpoint and plot
################################################################################################################
# The choosen technique is to start at one point and then travel in the direction towards the other.
# We only want to travel half the distance.

# We seek half the length of our distance between the two points vector.
vira_position = mat[1, :]   # We start with one vector
vector_from_vira_to_davide = mat[0, :] - mat[1, :]  # Calculate the vector from that vector
# Then start at our vector and "walk" half the distance to reach the middle between the two points
point_in_middle = vira_position + 1/2*vector_from_vira_to_davide
print(point_in_middle)
new_mat = np.append(mat.ravel(), point_in_middle).reshape([3, 3])   # We "append" this new point to our matrix


# Below we create our plot, same as before
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(new_mat[:, 0], new_mat[:, 1], new_mat[:, 2], marker='o')
ax.set(title='1.3 plot', xlabel='height', ylabel='Age', zlabel='Zodiac')
ax.set(xlim=[150,200],ylim=[20, 60], zlim=[0, 12])

plt.show()

# Comment: The zodiac of that person would be 4 so he/she is a Gemini


################################################################################################################
# 1.4 1000 random vectors
################################################################################################################
mu, sigma = 0, 1 # mean and standard deviation
fig2 = plt.figure() # We set up our plot
ax2 = fig2.add_subplot(111)
ax2.grid()

for i in range(1000):   # We create 1000 vectors and plot them
     vector = np.random.normal(mu, sigma, 2)
     ax2.scatter(vector[0], vector[1], marker='.')
plt.show()

################################################################################################################
# 1.5 multiply by matrix
################################################################################################################
mu, sigma = 0, 1 # mean and standard deviation
fig2 = plt.figure() # We set up our plot
ax2 = fig2.add_subplot(111)
ax2.grid()
matrix = np.array([[2,1],[1,1]])    # We create the matrix that we want to multiply our vectors by

for i in range(1000):
     vector = np.random.normal(mu, sigma, 2)    # We create our vectors
     vector = matrix @ vector   # We then multiply them by the matrix 2x2 @ 2 x 1 = 2 x 1 are the dimensions
     ax2.scatter(vector[0], vector[1], marker='.')  # And then plot them
plt.show()



################################################################################################################
# 1.6 Describe what happened
################################################################################################################
# Multiplying a vector by a matrix can be seen as rescaling a vector.
# In this case the x and y coordinate were "squished together" which now makes it appear as more of a "line"

################################################################################################################
# 1.7 multiply by inverse matrix
################################################################################################################
mu, sigma = 0, 1 # mean and standard deviation
fig2 = plt.figure() # Set up a plot
ax2 = fig2.add_subplot(111)
ax2.grid()
matrix = np.array([[2,1],[1,1]])
inverse_matrix = np.linalg.inv(matrix)  # Create an inverse matrix
for i in range(1000):
     vector = np.random.normal(mu, sigma, 2)    # Create the vectors and multiply them as before
     vector = vector @ matrix
     vector = vector @ inverse_matrix   # Multiply with inverse matrix, thereby rescaling the vector
     ax2.scatter(vector[0], vector[1], marker='.')
plt.show()

################################################################################################################
# 1.8 Uniform samples
################################################################################################################
dim = 2 # We set the dimension
fig2 = plt.figure() # And set up our plot
ax2 = fig2.add_subplot(111)
ax2.grid()

A_many_vectors = np.random.uniform(-1, 1, dim * 1000) # We start by creating a vector with uniformly distributed values
A_many_vectors = A_many_vectors.reshape(1000, dim) # and reshaping it into a 1000 * dim shape



E_vector_w_distances = np.array([]) # This keeps track of all our distances

for i in range(1000):  # We iterate through our vector with uniformly distributed values
    # The following vector will be used to calculate distances from a point to all other points
    B_vector_w_rows = np.ones((999, dim)) * A_many_vectors[0, :]

    A_many_vectors = A_many_vectors[1:, :] # We remove the first vector, so as not to calculate distance to itself

    C_difference_vector = B_vector_w_rows - A_many_vectors  # Contains distance vectors between the point and all others

    # We re-append the point to our vector that keeps track of all points, added at the end so we don't count it twice
    A_many_vectors = A_many_vectors.reshape(1, dim*999)
    A_many_vectors = np.append(A_many_vectors, B_vector_w_rows[0, :])
    A_many_vectors = A_many_vectors.reshape(1000, dim)

    # Below we create a vector with all 999 distances between this point and others and append it our vector
    # that keeps track of ALL distances

    # The euclidean distance is calculated by first adding the square of each axis (x, y etc for higher dimensions)
    F_vector_w_euclidiean_distances = np.square(C_difference_vector[:, 0])  # The x-axis
    for column in range(1, dim):
        F_vector_w_euclidiean_distances += np.square(C_difference_vector[:, column]) # The y axis etc added

    # We take the square root, thereby completing the calculation of the euclidean distances
    F_vector_w_euclidiean_distances = np.sqrt(F_vector_w_euclidiean_distances)

    # Lastly we append these values to a vector that keeps track of all distances
    E_vector_w_distances = np.append(E_vector_w_distances,F_vector_w_euclidiean_distances.reshape(1, 999))


# Below we plot a histogram with all distances
ax2.hist(E_vector_w_distances, bins=50, density=True)
ax2.set(xlim=[0, 10], ylabel = 'Probability density', xlabel='Euclidean Distance')

plt.show()


################################################################################################################
# 1.9 Uniform samples 10
################################################################################################################
# The exact same thing as above (1.8), only the dimension is changed.
dim = 10
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.grid()

A_many_vectors = np.random.uniform(-1, 1, dim * 1000) # We start by creating a vector and reshaping it
A_many_vectors = A_many_vectors.reshape(1000, dim)



E_vector_w_distances = np.array([])

for i in range(1000):
    B_vector_w_rows = np.ones((999, dim)) * A_many_vectors[0, :]  # This will be used to calculate distances

    A_many_vectors = A_many_vectors[1:, :] # We remove the first vector, so as not to calculate distance to itself

    C_difference_vector = B_vector_w_rows - A_many_vectors  # Contains all the vectors between the points. (999,2)

    A_many_vectors = A_many_vectors.reshape(1, dim*999) # We re-append the point to our vector that keeps track of all points
    A_many_vectors = np.append(A_many_vectors, B_vector_w_rows[0, :])
    A_many_vectors = A_many_vectors.reshape(1000, dim)

    F_vector_w_euclidiean_distances = np.square(C_difference_vector[:, 0])
    for column in range(1, dim):
        F_vector_w_euclidiean_distances += np.square(C_difference_vector[:, column])


    F_vector_w_euclidiean_distances = np.sqrt(F_vector_w_euclidiean_distances)


    E_vector_w_distances = np.append(E_vector_w_distances,F_vector_w_euclidiean_distances.reshape(1, 999))



ax2.hist(E_vector_w_distances, bins=50, density=True)
ax2.set(xlim=[0, 10], ylabel = 'Probability density', xlabel='Euclidean Distance')

plt.show()


################################################################################################################
# 1.10 Uniform samples 100
################################################################################################################
dim = 100
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.grid()

A_many_vectors = np.random.uniform(-1, 1, dim * 1000) # We start by creating a vector and reshaping it
A_many_vectors = A_many_vectors.reshape(1000, dim)



E_vector_w_distances = np.array([])

for i in range(1000):
    B_vector_w_rows = np.ones((999, dim)) * A_many_vectors[0, :]  # This will be used to calculate distances

    A_many_vectors = A_many_vectors[1:, :] # We remove the first vector, so as not to calculate distance to itself

    C_difference_vector = B_vector_w_rows - A_many_vectors  # Contains all the vectors between the points. (999,2)

    A_many_vectors = A_many_vectors.reshape(1, dim*999) # We re-append the point to our vector that keeps track of all points
    A_many_vectors = np.append(A_many_vectors, B_vector_w_rows[0, :])
    A_many_vectors = A_many_vectors.reshape(1000, dim)


    F_vector_w_euclidiean_distances = np.square(C_difference_vector[:, 0])
    for column in range(1, dim):
        F_vector_w_euclidiean_distances += np.square(C_difference_vector[:, column])


    F_vector_w_euclidiean_distances = np.sqrt(F_vector_w_euclidiean_distances)


    E_vector_w_distances = np.append(E_vector_w_distances,F_vector_w_euclidiean_distances.reshape(1, 999))



ax2.hist(E_vector_w_distances, bins=50, density=True)
ax2.set(xlim=[0, 10], ylabel = 'Probability density', xlabel='Euclidean Distance')

plt.show()


################################################################################################################
# 1.11 Answers
################################################################################################################
# The distance between points in the n-dimension equals

# spuare_root_of ( x1^2 + x2^2 + ... + xn^2)
# Since x is the distance between two points in one dimension, all x are similarly distributed.
# Thus we have square_root_of((n-1)*x^2 + x^2*2) = square_root_of(n-1) * square_root_of (x1^2 + x2^2)
# As we increase the dimension, as n increases, the euclidean distances increase by a factor X*(n-1)^(0.5) roughly
# This is why we see an increase in value as the number of dimensions increase