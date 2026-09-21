# deeponet_comsol.py
"""
DeepONet for Fluid-Thermal Field Predictions with Visualizations
Output Fields: [U, V, P, Theta] (Velocity-X, Velocity-Y, Pressure, Temperature)
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. DeepONet Architecture Definition
# ==========================================
class DeepONet(tf.keras.Model):
    def __init__(self, num_sensors, p=64):
        """
        :param num_sensors: Number of sensor measurement points
        :param p: Number of latent basis features per output variable
        """
        super(DeepONet, self).__init__()
        self.p = p

        # Branch Network: maps boundary/inlet condition profiles
        self.branch = tf.keras.Sequential([
            tf.keras.Input(shape=(num_sensors,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(p * 4)  # Outputs p features for each [U, V, P, Theta]
        ])

        # Trunk Network: maps spatial domain coordinates (X, Y)
        self.trunk = tf.keras.Sequential([
            tf.keras.Input(shape=(2,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(p, activation='relu')
        ])

        # Trainable bias parameters for each target output
        self.b_u = self.add_weight(shape=(1,), initializer='zeros', trainable=True, name='bias_u')
        self.b_v = self.add_weight(shape=(1,), initializer='zeros', trainable=True, name='bias_v')
        self.b_p = self.add_weight(shape=(1,), initializer='zeros', trainable=True, name='bias_p')
        self.b_theta = self.add_weight(shape=(1,), initializer='zeros', trainable=True, name='bias_theta')

    def call(self, inputs):
        branch_input, trunk_input = inputs

        b_out = self.branch(branch_input)  # Shape: (N, 4*p)
        t_out = self.trunk(trunk_input)   # Shape: (N, p)

        # Separate branch representations for each target field
        b_u = b_out[:, 0:self.p]
        b_v = b_out[:, self.p:2*self.p]
        b_p = b_out[:, 2*self.p:3*self.p]
        b_theta = b_out[:, 3*self.p:4*self.p]

        # Inner product (dot product) between Branch and Trunk output
        pred_u = tf.reduce_sum(b_u * t_out, axis=1, keepdims=True) + self.b_u
        pred_v = tf.reduce_sum(b_v * t_out, axis=1, keepdims=True) + self.b_v
        pred_p = tf.reduce_sum(b_p * t_out, axis=1, keepdims=True) + self.b_p
        pred_theta = tf.reduce_sum(b_theta * t_out, axis=1, keepdims=True) + self.b_theta

        return tf.concat([pred_u, pred_v, pred_p, pred_theta], axis=1)


# ==========================================
# 2. Data Generation & Setup
# ==========================================
def generate_synthetic_data(nx=50, ny=30, num_sensors=50):
    """
    Generates a structured grid and physical field data.
    Replace this function with pd.read_csv('your_comsol_data.csv') for real data.
    """
    x = np.linspace(0, 3, nx)
    y = np.linspace(-0.5, 0.5, ny)
    X, Y = np.meshgrid(x, y)
    
    # Flatten spatial grid
    xy_coords = np.hstack([X.flatten()[:, None], Y.flatten()[:, None]]).astype(np.float32)
    n_points = xy_coords.shape[0]

    # Synthetic boundary profile (sensor data)
    sensor_input = np.sin(np.linspace(0, np.pi, num_sensors))[None, :].astype(np.float32)
    sensor_data = np.repeat(sensor_input, n_points, axis=0)

    # Synthetic solution fields: U, V, P, Theta
    U = (1.0 - 4.0 * (xy_coords[:, 1:2]**2)).astype(np.float32)  # Poiseuille flow
    V = (0.1 * np.sin(np.pi * xy_coords[:, 0:1])).astype(np.float32)
    P = (3.0 - xy_coords[:, 0:1]).astype(np.float32)
    Theta = (np.sin(np.pi * xy_coords[:, 0:1]) * np.cos(np.pi * xy_coords[:, 1:2])).astype(np.float32)

    targets = np.hstack([U, V, P, Theta])
    return sensor_data, xy_coords, targets, X, Y


# ==========================================
# 3. Visual Representation Functions
# ==========================================
def plot_field_comparisons(X, Y, targets_true, targets_pred):
    """
    Plots COMSOL Ground Truth vs DeepONet Predictions and Absolute Error Maps.
    """
    field_names = ['Horizontal Velocity (U)', 'Vertical Velocity (V)', 'Pressure (P)', 'Temperature (Theta)']
    nx, ny = X.shape[1], X.shape[0]

    for i in range(4):
        true_field = targets_true[:, i].reshape(ny, nx)
        pred_field = targets_pred[:, i].reshape(ny, nx)
        abs_error = np.abs(true_field - pred_field)

        fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))

        # Ground Truth
        c1 = axes[0].contourf(X, Y, true_field, levels=25, cmap='jet')
        axes[0].set_title(f'COMSOL Ground Truth - {field_names[i]}')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        fig.colorbar(c1, ax=axes[0])

        # DeepONet Prediction
        c2 = axes[1].contourf(X, Y, pred_field, levels=25, cmap='jet')
        axes[1].set_title(f'DeepONet Prediction - {field_names[i]}')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        fig.colorbar(c2, ax=axes[1])

        # Absolute Error
        c3 = axes[2].contourf(X, Y, abs_error, levels=25, cmap='magma')
        axes[2].set_title(f'Absolute Error Map - {field_names[i]}')
        axes[2].set_xlabel('X')
        axes[2].set_ylabel('Y')
        fig.colorbar(c3, ax=axes[2])

        plt.tight_layout()
        plt.show()


# ==========================================
# 4. Main Execution Pipeline
# ==========================================
def main():
    num_sensors = 50
    latent_dim_p = 32
    epochs = 30
    batch_size = 64

    # 1. Load Data
    print("--- 1. Generating/Loading Dataset ---")
    sensor_data, xy_coords, targets, X, Y = generate_synthetic_data(nx=60, ny=30, num_sensors=num_sensors)

    # 2. Compile Model
    print("--- 2. Building & Compiling DeepONet ---")
    model = DeepONet(num_sensors=num_sensors, p=latent_dim_p)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )

    # 3. Train Model
    print("--- 3. Training Model ---")
    history = model.fit(
        x=[sensor_data, xy_coords],
        y=targets,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )

    # 4. Evaluate & Predict
    print("\n--- 4. Running Model Predictions ---")
    predictions = model.predict([sensor_data, xy_coords])

    # 5. Generate Contour Plots
    print("--- 5. Generating Visual Representations ---")
    plot_field_comparisons(X, Y, targets, predictions)


if __name__ == "__main__":
    main()
