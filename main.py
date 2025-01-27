from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_realistic_industrial_data(n_records=10_000_000):
    np.random.seed(42)

    # Create a timestamp index for time series analysis
    timestamps = pd.date_range('2023-01-01', periods=n_records, freq='1min')

    # Create more realistic features
    data = {
        "P_in": np.clip(np.random.normal(1.2e6, 2e5, n_records), 1e6, 1.6e6).astype(np.float32),
        "T_in": np.clip(np.random.normal(310, 10, n_records), 290, 330).astype(np.float32),
        "Flow_in": np.abs(np.random.normal(2500, 200, n_records)).astype(np.float32),
        "Flow_out": np.abs(np.random.normal(2400, 200, n_records)).astype(np.float32),
        "Vibration": (np.random.weibull(1.75, n_records) * 2.5).astype(np.float32),
        "Torque": np.clip(np.random.normal(350, 30, n_records), 250, 400).astype(np.float32),
        "RPM": np.random.randint(1500, 8000, n_records).astype(np.int16),
        "Sound": np.clip(np.random.lognormal(4.1, 0.4, n_records), 60, 150).astype(np.float32),
        "Viscosity": np.abs(np.random.normal(0.06, 0.01, n_records)).astype(np.float32),
        "Mole_concentration": np.clip(np.random.normal(1.0, 0.1, n_records), 0.8, 1.2).astype(np.float32),
    }

    df = pd.DataFrame(data, index=timestamps)

    # Advanced feature engineering
    df['Pressure_flow_ratio'] = df['P_in'] / (df['Flow_in'] + 1e-6)
    df['Energy_index'] = (df['Torque'] * df['RPM']) / 9549
    df['Viscosity_temperature'] = df['Viscosity'] * df['T_in']

    # Time-based features
    df['Vibration_ma_1h'] = df['Vibration'].rolling(
        window='1H', min_periods=1).mean()
    df['Torque_ma_30min'] = df['Torque'].rolling(
        window='30T', min_periods=1).mean()
    df['Flow_diff'] = df['Flow_in'] - df['Flow_out']
    df['Pressure_derivative'] = df['P_in'].diff().fillna(0)
    df['Vibration_derivative'] = df['Vibration'].diff().fillna(0)
    df['Torque_diff_6h'] = df['Torque'].diff(periods=360).fillna(0)

    # Theoretical and actual power calculations
    R = 287  # specific gas constant for air (J/kg·K)
    gamma = 1.4  # specific heat ratio
    df['Theoretical_power'] = (df['Flow_in'] * R * df['T_in'] / (gamma - 1) *
                               ((df['P_in']*1.1/df['P_in'])**((gamma-1)/gamma) - 1))
    efficiency = 0.85 - 0.0001 * df['Vibration'] + 0.005 * np.sin(2 * np.pi * (
        df.index.minute / 60)) - 0.00002 * df['RPM'] + 0.01 * df['Viscosity']
    df['Actual_power'] = df['Theoretical_power'] / \
        np.clip(efficiency, 0.5, 0.9)

    # Health index calculation
    df['Health_index'] = np.clip(
        1.0 - (0.3 * (df['Vibration']/10) +
               0.2 * (df['Torque']/500) +
               0.5 * (df['Sound']/150)),
        0.0, 1.0
    )

    # Generate labels with more realistic conditions and some noise
    compressor_risk = (
        0.6 * (df['Vibration'] > 4) +
        0.4 * (df['Flow_in'] < 320) +
        0.5 * (df['T_in'] > 325) +
        0.3 * (df['Health_index'] < 0.75)
    )

    bearing_risk = (
        0.6 * (df['Torque'] < 260) +
        0.5 * (df['RPM'] > 7500) +
        0.7 * (df['Sound'] > 120) +
        0.4 * (df['Vibration_derivative'] > 1)
    )

    # Generating failure labels
    df["Failure_compressor"] = (compressor_risk > 0.5).astype(int)
    df["Failure_bearing"] = (bearing_risk > 0.5).astype(int)

    # Fill missing values
    return df.fillna(method='bfill').fillna(0)


# Example Usage
industrial_data = generate_realistic_industrial_data()
print(industrial_data.head())
# Then continue with your training code as previously defined.


# Data Preparation
# Using the improved generate_realistic_industrial_data function from previous response
df = generate_realistic_industrial_data(n_records=5_000_000)
features = ['P_in', 'T_in', 'Flow_in', 'Flow_out', 'Vibration', 'Torque',
            'RPM', 'Sound', 'Viscosity', 'Mole_concentration',
            'Pressure_flow_ratio', 'Energy_index', 'Vibration_ma_1h',
            'Torque_ma_30min', 'Flow_diff', 'Pressure_derivative',
            'Vibration_derivative', 'Torque_diff_6h', 'Health_index']

# Splitting Data
X = df[features].values
y_compressor = df['Failure_compressor'].values
y_bearing = df['Failure_bearing'].values

X_train, X_test, yc_train, yc_test, yb_train, yb_test = train_test_split(
    X, y_compressor, y_bearing, test_size=0.3, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compressor Failure Model


def build_compressor_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(512, activation='swish',
                     kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='swish',
                     kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='swish'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss='binary_crossentropy',
                  metrics=['accuracy',
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall')])

    return model


compressor_model = build_compressor_model(X_train_scaled.shape[1])
history_compressor = compressor_model.fit(X_train_scaled, yc_train,
                                          epochs=100,
                                          batch_size=512,
                                          validation_split=0.2,
                                          # Weighted class handling
                                          class_weight={0: 1, 1: 10},
                                          callbacks=[callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

# Bearing Failure Model


def build_bearing_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(512, activation='swish',
                     kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='swish',
                     kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='swish'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss='binary_crossentropy',
                  metrics=['accuracy',
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall')])

    return model


bearing_model = build_bearing_model(X_train_scaled.shape[1])
history_bearing = bearing_model.fit(X_train_scaled, yb_train,
                                    epochs=100,
                                    batch_size=512,
                                    validation_split=0.2,
                                    # Weighted class handling
                                    class_weight={0: 1, 1: 8},
                                    callbacks=[callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

# Evaluation and Visualization Functions


def evaluate_classification(model, X_test, y_test, model_name):
    y_pred = (model.predict(X_test) > 0.5).astype(int)

    # Classification Report
    print(f"\n{classification_report(y_test, y_pred,
          target_names=['Normal', 'Failure'])}")

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Failure'],
                yticklabels=['Normal', 'Failure'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, model.predict(X_test))
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.show()


# Evaluate All Models
print("\n" + "="*40)
print("Compressor Failure Model Evaluation")
print("="*40)
evaluate_classification(compressor_model, X_test_scaled,
                        yc_test, "Compressor Failure")

print("\n" + "="*40)
print("Bearing Failure Model Evaluation")
print("="*40)
evaluate_classification(bearing_model, X_test_scaled,
                        yb_test, "Bearing Failure")
