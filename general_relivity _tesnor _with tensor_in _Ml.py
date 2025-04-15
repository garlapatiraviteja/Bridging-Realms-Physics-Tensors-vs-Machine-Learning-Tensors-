# import numpy as np
# import matplotlib.pyplot as plt

# # 1. Physics Tensor - Stress-Energy Tensor (2D)
# def lorentz_transform(tensor, velocity, speed_of_light=1):
#     gamma = 1 / np.sqrt(1 - (velocity ** 2) / (speed_of_light ** 2))
#     lorentz_matrix = np.array([[gamma, -gamma * velocity],
#                                [-gamma * velocity, gamma]])
#     transformed_tensor = np.dot(lorentz_matrix, np.dot(tensor, lorentz_matrix.T))
#     return transformed_tensor

# stress_energy_tensor = np.array([[1, 0],
#                                  [0, 1]])
# velocity = 0.5
# transformed_tensor = lorentz_transform(stress_energy_tensor, velocity)

# print("Original Stress-Energy Tensor:\n", stress_energy_tensor)
# print("\nTransformed Stress-Energy Tensor:\n", transformed_tensor)

# # 2. Machine Learning Tensor - Rank-M Tensor Operations
# def ml_tensor_operations(tensor):
#     reshaped_tensor = np.reshape(tensor, (-1, tensor.shape[-1]))
#     # Safely transpose only last 3 dimensions (height, width, channels)
#     transposed_tensor = np.transpose(tensor, axes=(0, 3, 1, 2))  # (batch, channels, height, width)
#     return reshaped_tensor, transposed_tensor

# # Create a rank-4 tensor (e.g., for batch of RGB images)
# ml_tensor = np.random.rand(2, 3, 3, 3)  # Shape: (batch=2, height=3, width=3, channels=3)
# reshaped_tensor, transposed_tensor = ml_tensor_operations(ml_tensor)

# print("\nOriginal ML Tensor (Rank-4):", ml_tensor.shape)
# print("\nReshaped ML Tensor (flattened spatial dims):", reshaped_tensor.shape)
# print("\nTransposed ML Tensor (channels first):", transposed_tensor.shape)

# # 3. Visualization (Comparison of Tensor Transformations)
# def plot_tensor_comparison():
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#     axes[0].imshow(stress_energy_tensor, cmap='coolwarm', interpolation='none')
#     axes[0].set_title('Original Stress-Energy Tensor')
#     axes[1].imshow(transformed_tensor, cmap='coolwarm', interpolation='none')
#     axes[1].set_title(f'Transformed Stress-Energy Tensor (v={velocity})')
#     plt.tight_layout()
#     plt.show()

# plot_tensor_comparison()
#======================================================
# import numpy as np
# import matplotlib.pyplot as plt

# # 1. Physics Tensor - Stress-Energy Tensor (2D)
# def lorentz_transform(tensor, velocity, speed_of_light=1):
#     gamma = 1 / np.sqrt(1 - (velocity ** 2) / (speed_of_light ** 2))
#     lorentz_matrix = np.array([[gamma, -gamma * velocity],
#                                [-gamma * velocity, gamma]])
#     transformed_tensor = np.dot(lorentz_matrix, np.dot(tensor, lorentz_matrix.T))
#     return transformed_tensor

# stress_energy_tensor = np.array([[1, 0],
#                                  [0, 1]])
# velocity = 0.5
# transformed_tensor = lorentz_transform(stress_energy_tensor, velocity)

# print("Original Stress-Energy Tensor:\n", stress_energy_tensor)
# print("\nTransformed Stress-Energy Tensor:\n", transformed_tensor)

# # 2. Machine Learning Tensor - Rank-M Tensor Operations
# def ml_tensor_operations(tensor):
#     reshaped_tensor = np.reshape(tensor, (-1, tensor.shape[-1]))
#     transposed_tensor = np.transpose(tensor, axes=(0, 3, 1, 2))  # (batch, channels, height, width)
#     return reshaped_tensor, transposed_tensor

# # Create a rank-4 tensor (e.g., for batch of RGB images)
# ml_tensor = np.random.rand(2, 3, 3, 3)  # Shape: (batch=2, height=3, width=3, channels=3)
# reshaped_tensor, transposed_tensor = ml_tensor_operations(ml_tensor)

# print("\nOriginal ML Tensor (Rank-4):\n", ml_tensor.shape)
# print("\nReshaped ML Tensor (flattened spatial dims):\n", reshaped_tensor.shape)
# print("\nTransposed ML Tensor (channels first):\n", transposed_tensor.shape)

# # 3. Visualization (Comparison of Tensor Transformations)
# def plot_tensor_comparison():
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#     axes[0].imshow(stress_energy_tensor, cmap='coolwarm', interpolation='none')
#     axes[0].set_title('Original Stress-Energy Tensor')
#     axes[1].imshow(transformed_tensor, cmap='coolwarm', interpolation='none')
#     axes[1].set_title(f'Transformed Stress-Energy Tensor (v={velocity})')
#     plt.tight_layout()
#     plt.show()

# plot_tensor_comparison()

# # 4. Visualization of ML Tensor Slices
# def plot_ml_tensor_slices(tensor):
#     batch, channels, height, width = tensor.shape
#     fig, axes = plt.subplots(batch, channels, figsize=(channels * 3, batch * 3))
#     for b in range(batch):
#         for c in range(channels):
#             ax = axes[b, c] if batch > 1 else axes[c]
#             ax.imshow(tensor[b, c], cmap='viridis')
#             ax.set_title(f'Batch {b} - Channel {c}')
#             ax.axis('off')
#     plt.tight_layout()
#     plt.show()

# plot_ml_tensor_slices(transposed_tensor)

#=======================================


# import numpy as np
# import matplotlib.pyplot as plt

# # 1. Physics Tensor - Stress-Energy Tensor (2D)
# def lorentz_transform(tensor, velocity, speed_of_light=1):
#     print("\nApplying Lorentz transformation...")
#     gamma = 1 / np.sqrt(1 - (velocity ** 2) / (speed_of_light ** 2))
#     lorentz_matrix = np.array([[gamma, -gamma * velocity],
#                                [-gamma * velocity, gamma]])
#     print("Lorentz Matrix:\n", lorentz_matrix)
#     transformed_tensor = np.dot(lorentz_matrix, np.dot(tensor, lorentz_matrix.T))
#     return transformed_tensor

# stress_energy_tensor = np.array([[1, 0],
#                                  [0, 1]])
# velocity = 0.5
# transformed_tensor = lorentz_transform(stress_energy_tensor, velocity)

# print("Original Stress-Energy Tensor:\n", stress_energy_tensor)
# print("\nTransformed Stress-Energy Tensor:\n", transformed_tensor)

# # 2. Machine Learning Tensor - Rank-M Tensor Operations
# def ml_tensor_operations(tensor):
#     print("\nPerforming ML tensor operations...")
#     reshaped_tensor = np.reshape(tensor, (-1, tensor.shape[-1]))
#     print("Reshaped Tensor Shape:", reshaped_tensor.shape)
#     transposed_tensor = np.transpose(tensor, axes=(0, 3, 1, 2))  # (batch, channels, height, width)
#     print("Transposed Tensor Shape:", transposed_tensor.shape)
#     return reshaped_tensor, transposed_tensor

# # Create a rank-4 tensor (e.g., for batch of RGB images)
# ml_tensor = np.random.rand(2, 3, 3, 3)  # Shape: (batch=2, height=3, width=3, channels=3)
# print("\nOriginal ML Tensor Shape:", ml_tensor.shape)
# reshaped_tensor, transposed_tensor = ml_tensor_operations(ml_tensor)

# # 3. Visualization (Comparison of Tensor Transformations)
# def plot_tensor_comparison():
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#     axes[0].imshow(stress_energy_tensor, cmap='coolwarm', interpolation='none')
#     axes[0].set_title('Original Stress-Energy Tensor')
#     axes[0].set_xlabel('Axis 0')
#     axes[0].set_ylabel('Axis 1')
    
#     axes[1].imshow(transformed_tensor, cmap='coolwarm', interpolation='none')
#     axes[1].set_title(f'Transformed Stress-Energy Tensor (v={velocity})')
#     axes[1].set_xlabel('Axis 0')
#     axes[1].set_ylabel('Axis 1')

#     plt.tight_layout()
#     plt.show()

# plot_tensor_comparison()

# # 4. Visualization of ML Tensor Slices
# def plot_ml_tensor_slices(tensor):
#     print("\nVisualizing ML tensor slices...")
#     batch, channels, height, width = tensor.shape
#     fig, axes = plt.subplots(batch, channels, figsize=(channels * 3, batch * 3))
#     for b in range(batch):
#         for c in range(channels):
#             ax = axes[b, c] if batch > 1 else axes[c]
#             ax.imshow(tensor[b, c], cmap='viridis')
#             ax.set_title(f'Batch {b} - Channel {c}')
#             ax.set_xlabel('Width')
#             ax.set_ylabel('Height')
#             ax.axis('off')
#     plt.tight_layout()
#     plt.show()

# plot_ml_tensor_slices(transposed_tensor)

import numpy as np
import matplotlib.pyplot as plt

# 1. Physics Tensor - Stress-Energy Tensor (2D)
def lorentz_transform(tensor, velocity, speed_of_light=1):
    print("\nApplying Lorentz transformation...")
    gamma = 1 / np.sqrt(1 - (velocity ** 2) / (speed_of_light ** 2))
    lorentz_matrix = np.array([[gamma, -gamma * velocity],
                               [-gamma * velocity, gamma]])
    print("Lorentz Matrix:\n", lorentz_matrix)
    transformed_tensor = np.dot(lorentz_matrix, np.dot(tensor, lorentz_matrix.T))
    return transformed_tensor

stress_energy_tensor = np.array([[1, 0],
                                 [0, 1]])
velocity = 0.5
transformed_tensor = lorentz_transform(stress_energy_tensor, velocity)

print("Original Stress-Energy Tensor:\n", stress_energy_tensor)
print("\nTransformed Stress-Energy Tensor:\n", transformed_tensor)

# 2. Machine Learning Tensor - Rank-M Tensor Operations
def ml_tensor_operations(tensor):
    print("\nPerforming ML tensor operations...")
    reshaped_tensor = np.reshape(tensor, (-1, tensor.shape[-1]))
    print("Reshaped Tensor Shape:", reshaped_tensor.shape)
    transposed_tensor = np.transpose(tensor, axes=(0, 3, 1, 2))  # (batch, channels, height, width)
    print("Transposed Tensor Shape:", transposed_tensor.shape)
    return reshaped_tensor, transposed_tensor

# Create a rank-4 tensor (e.g., for batch of RGB images)
ml_tensor = np.random.rand(2, 3, 3, 3)  # Shape: (batch=2, height=3, width=3, channels=3)
print("\nOriginal ML Tensor Shape:", ml_tensor.shape)
reshaped_tensor, transposed_tensor = ml_tensor_operations(ml_tensor)

# 3. Visualization (Comparison of Tensor Transformations)
def plot_tensor_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(stress_energy_tensor, cmap='coolwarm', interpolation='none')
    axes[0].set_title('Original Stress-Energy Tensor')
    axes[0].set_xlabel('Axis 0')
    axes[0].set_ylabel('Axis 1')
    for (j,i),label in np.ndenumerate(stress_energy_tensor):
        axes[0].text(i, j, f"{label}", ha='center', va='center', color='black', fontsize=12)

    axes[1].imshow(transformed_tensor, cmap='coolwarm', interpolation='none')
    axes[1].set_title(f'Transformed Stress-Energy Tensor (v={velocity})')
    axes[1].set_xlabel('Axis 0')
    axes[1].set_ylabel('Axis 1')
    for (j,i),label in np.ndenumerate(transformed_tensor):
        axes[1].text(i, j, f"{label:.2f}", ha='center', va='center', color='black', fontsize=12)

    plt.tight_layout()
    plt.show()

plot_tensor_comparison()

# 4. Visualization of ML Tensor Slices
def plot_ml_tensor_slices(tensor):
    print("\nVisualizing ML tensor slices with annotations...")
    batch, channels, height, width = tensor.shape
    fig, axes = plt.subplots(batch, channels, figsize=(channels * 3, batch * 3))
    for b in range(batch):
        for c in range(channels):
            ax = axes[b, c] if batch > 1 else axes[c]
            ax.imshow(tensor[b, c], cmap='viridis')
            for (j,i), val in np.ndenumerate(tensor[b, c]):
                ax.text(i, j, f"{val:.2f}", ha='center', va='center', color='white', fontsize=8)
            ax.set_title(f'Batch {b} - Channel {c}')
            ax.set_xlabel('Width')
            ax.set_ylabel('Height')
            ax.axis('off')
    plt.tight_layout()
    plt.show()

plot_ml_tensor_slices(transposed_tensor)