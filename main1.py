import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow.keras import layers, optimizers, regularizers
import scipy.stats as stats


class GasSystemAnalyzer:
    def __init__(self, pipe_diameter, gas_constant=287, gamma=1.4):
        self.pipe_diameter = pipe_diameter
        self.gas_constant = gas_constant
        self.gamma = gamma

    def calculate_thermodynamic_properties(self, df):
        """Calculate thermodynamic and hydrodynamic properties."""
        try:
            df['Reynolds'] = (df['Density'] * df['Flow_in']
                              * self.pipe_diameter) / df['Viscosity']
            df['Speed_of_sound'] = np.sqrt(
                self.gamma * self.gas_constant * df['T_in'])
            df['Mach'] = df['Flow_in'] / df['Speed_of_sound']
            df['Compressibility'] = (
                df['P_in'] * self.gas_constant * df['T_in']) / (df['Density'] + 1e-6)
        except KeyError as e:
            print(f"Missing key: {e}")
        return df


class NavierStokesSolver:
    def __init__(self, density, viscosity, dx=0.1, dt=0.01):
        self.density = density
        self.viscosity = viscosity
        self.dx = dx
        self.dt = dt

    def solve_mass_spring(self, force, mass, spring_constant, n_steps=1000):
        """Solve the mass-spring equation for vibration analysis."""
        positions = np.zeros(n_steps)
        velocity = 0
        position = 0

        for i in range(1, n_steps):
            acceleration = (force[i % len(force)] -
                            spring_constant * position) / mass
            velocity += acceleration * self.dt
            position += velocity * self.dt
            positions[i] = position

        return positions

    def solve_navier_stokes_1D(self, initial_velocity, pressure_gradient, length, n_steps=1000):
        """Solve a simplified 1D Navier-Stokes equation."""
        n_points = len(initial_velocity)
        velocity = np.array(initial_velocity)

        for _ in range(n_steps):
            dudt = np.zeros_like(velocity)
            dudt[1:-1] = (-velocity[1:-1] * (velocity[2:] - velocity[:-2]) / (2 * self.dx) +
                          self.viscosity * (velocity[2:] - 2 * velocity[1:-1] + velocity[:-2]) / self.dx**2 -
                          pressure_gradient / self.density)

            velocity += dudt * self.dt

        return velocity


class DeepPowerPredictor:
    def __init__(self, input_dim):
        self.model = self.build_model(input_dim)
        self.scaler = RobustScaler()

    def build_model(self, input_dim):
        """Construct a deep learning model for power prediction."""
        model = tf.keras.Sequential([
            layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(
                0.001), input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu',
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.Dense(64, activation='relu',
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.Dense(1)  # Output layer
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=512):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        history = self.model.fit(X_train_scaled, y_train,
                                 validation_data=(X_val_scaled, y_val),
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 verbose=1)
        return history

    def evaluate(self, X_test, y_test):
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)

        print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
        print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
        print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
        print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
        print(f"MAPE: {mean_absolute_percentage_error(
            y_test, y_pred) * 100:.2f}%")

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.3)
        plt.plot([y_test.min(), y_test.max()], [
                 y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual Power')
        plt.ylabel('Predicted Power')
        plt.title('Actual vs Predicted Power')
        plt.show()


class AnomalyDetector:
    def __init__(self, contamination=0.1):
        self.model = IsolationForest(contamination=contamination)
        self.scaler = RobustScaler()

    def detect_anomalies(self, df, features, fit=True):
        """Detect anomalies in operational data."""
        if fit:
            scaled_data = self.scaler.fit_transform(df[features])
            self.model.fit(scaled_data)
        else:
            scaled_data = self.scaler.transform(df[features])
            anomalies = self.model.predict(scaled_data)

        analyze_col = 'Actual_power'
        anomalies = self.model.predict(scaled_data)
        df['Anomaly'] = np.where(anomalies == -1, 1, 0)

        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df[analyze_col], color='blue', label='Normal')
        plt.scatter(df[df['Anomaly'] == 1].index,
                    df[df['Anomaly'] == 1][analyze_col],
                    color='red', label='Anomaly')
        plt.title('Anomaly Detection Results')
        plt.xlabel('Time')
        plt.ylabel('Power')
        plt.legend()
        plt.show()

        return df


class EnhancedDeepPowerPredictor(DeepPowerPredictor):
    def build_model(self, input_shape):
        """Construct an advanced model with residual connections and attention."""
        inputs = tf.keras.Input(shape=input_shape)

        x = layers.Dense(256, activation='swish',
                         kernel_regularizer=regularizers.l2(0.001))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

        # Residual Block 1
        residual = x
        x = layers.Dense(128, activation='swish',
                         kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='swish',
                         kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.add([x, residual])
        x = layers.Activation('swish')(x)

        # Attention Layer
        attention = layers.Dense(256, activation='sigmoid')(x)
        x = layers.multiply([x, attention])

        # Residual Block 2
        residual = x
        x = layers.Dense(128, activation='swish',
                         kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='swish',
                         kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.add([x, residual])
        x = layers.Activation('swish')(x)

        # Compression Layers
        x = layers.Dense(128, activation='swish',
                         kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.Dense(64, activation='swish',
                         kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.Dense(32, activation='swish',
                         kernel_regularizer=regularizers.l2(0.001))(x)

        outputs = layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        optimizer = optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-5)

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.Huber(),
            metrics=[
                tf.keras.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsolutePercentageError(),
                self.r_squared_metric
            ]
        )
        return model

    def r_squared_metric(self, y_true, y_pred):
        """Custom R-squared metric."""
        SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
        SS_tot = tf.reduce_sum(
            tf.square(y_true - tf.reduce_mean(y_true))) + tf.keras.backend.epsilon()
        return 1 - SS_res / SS_tot

    def train(self, X_train, y_train, X_val, y_val, epochs=200, batch_size=1024):
        """Train the model with adaptive learning rate and early stopping."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_r_squared_metric',
            patience=15,
            mode='max',
            restore_best_weights=True
        )

        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[lr_scheduler, early_stop],
        )
        return history

    def evaluate(self, X_test, y_test):
        """Advanced evaluation with residuals analysis."""
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)

        # Baseline metrics
        print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
        print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

        # Residual analysis
        residuals = y_test - y_pred.flatten()

        plt.figure(figsize=(15, 6))

        plt.subplot(1, 3, 1)
        sns.histplot(residuals, kde=True)
        plt.title('Distribution of Residuals')

        plt.subplot(1, 3, 2)
        plt.scatter(y_pred, residuals, alpha=0.3)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted')

        plt.subplot(1, 3, 3)
        stats.probplot(residuals, plot=plt)
        plt.title('Q-Q Plot of Residuals')

        plt.tight_layout()
        plt.show()


# Main Execution
if __name__ == "__main__":
    # Generate synthetic industrial data
    df = generate_industrial_data(n_records=1000000)

    # Calculate density (assumed)
    df['Density'] = df['P_in'] / (df['T_in'] * 287)

    # 1. Thermodynamic analysis
    thermo_analyzer = GasSystemAnalyzer(pipe_diameter=0.5)
    df = thermo_analyzer.calculate_thermodynamic_properties(df)

    # 2. Vibration analysis using mass-spring equation
    vibration_analyzer = NavierStokesSolver(density=1.225, viscosity=1.8e-5)
    mass = 500  # kg
    spring_constant = 1e4  # N/m
    force = df['Vibration'].values * 1000  # Convert to Newtons

    vibration_results = []
    for f in force[:1000]:  # Sample for quick computations
        vibration_results.extend(
            vibration_analyzer.solve_mass_spring(f, mass, spring_constant))

    plt.figure(figsize=(12, 6))
    plt.plot(vibration_results[:500])
    plt.title('Vibration Simulation Results')
    plt.xlabel('Time Steps')
    plt.ylabel('Displacement')
    plt.show()

    # 3. Anomaly detection
    anomaly_features = ['P_in', 'T_in', 'Flow_in',
                        'Vibration', 'Actual_power', 'Health_index']
    anomaly_detector = AnomalyDetector(contamination=0.05)
    df = anomaly_detector.detect_anomalies(df, anomaly_features)

    # Training the advanced deep learning model
    features = ['P_in', 'T_in', 'Flow_in', 'Viscosity', 'Reynolds', 'Mach', 'Health_index',
                'Pressure_flow_ratio', 'Energy_index', 'Vibration_ma_1h']
    target = 'Actual_power'

    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.2, random_state=42
    )

    enhanced_predictor = EnhancedDeepPowerPredictor(input_dim=(len(features),))
    history = enhanced_predictor.train(X_train, y_train, X_test, y_test)

    # Learning analysis
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Evolution')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['r_squared_metric'], label='Train R²')
    plt.plot(history.history['val_r_squared_metric'], label='Validation R²')
    plt.title('R² Score Evolution')
    plt.legend()

    plt.show()

    # Evaluate the model
    enhanced_predictor.evaluate(X_test, y_test)

    # Save the model
    enhanced_predictor.model.save('enhanced_power_model.keras')
