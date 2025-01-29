import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_industrial_data(n_records=10_000_000):
    # تولید ویژگی‌های اصلی با توزیع‌های واقع‌بینانه‌تر
    np.random.seed(42)

    # ایجاد ایندکس زمانی برای تحلیل سری زمانی
    timestamps = pd.date_range('2023-01-01', periods=n_records, freq='1min')

    data = {
        "P_in": np.clip(np.random.normal(1.2e6, 3e5, n_records), 8e5, 2e6).astype(np.float32),
        "T_in": np.clip(np.random.normal(315, 15, n_records), 280, 350).astype(np.float32),
        "Flow_in": np.abs(np.random.normal(2000, 500, n_records)).astype(np.float32),
        "Flow_out": np.abs(np.random.normal(1900, 475, n_records)).astype(np.float32),
        "Vibration": (np.random.weibull(1.5, n_records) * 2).astype(np.float32),
        "Torque": np.clip(np.random.normal(300, 40, n_records), 150, 450).astype(np.float32),
        "RPM": np.random.randint(1500, 9500, n_records).astype(np.int16),
        "Sound": np.clip(np.random.lognormal(4.0, 0.3, n_records), 50, 120).astype(np.float32),
        "Viscosity": np.abs(np.random.normal(0.05, 0.02, n_records)).astype(np.float32),
        "Mole_concentration": np.clip(np.random.normal(1.0, 0.15, n_records), 0.8, 1.2).astype(np.float32),
    }

    df = pd.DataFrame(data, index=timestamps)

    # مهندسی ویژگی‌های پیشرفته
    df['Pressure_flow_ratio'] = df['P_in'] / (df['Flow_in'] + 1e-6)
    df['Energy_index'] = (df['Torque'] * df['RPM']) / 9549
    df['Viscosity_temperature'] = df['Viscosity'] * df['T_in']

    # ویژگی‌های زمانی پیشرفته
    df['Vibration_ma_1h'] = df['Vibration'].rolling(
        window='1H', min_periods=1).mean()
    df['Torque_ma_30min'] = df['Torque'].rolling(
        window='30T', min_periods=1).mean()
    df['Flow_diff'] = df['Flow_in'] - df['Flow_out']
    df['Pressure_derivative'] = df['P_in'].diff().fillna(0)
    df['Vibration_derivative'] = df['Vibration'].diff().fillna(0)
    df['Torque_diff_6h'] = df['Torque'].diff(periods=360).fillna(0)

    # ویژگی جدید: محاسبه توان تئوری و واقعی بر اساس معادلات ترمودینامیکی
    R = 287  # ثابت گاز هوا (J/kg·K)
    gamma = 1.4  # نسبت گرمای ویژه
    df['Theoretical_power'] = (df['Flow_in'] * R * df['T_in'] / (gamma - 1) *
                               ((df['P_in']*1.2/df['P_in'])**((gamma-1)/gamma) - 1))

    # محاسبه راندمان واقعی با در نظر گرفتن تلفات
    efficiency = 0.85 - 0.0001*df['Vibration'] - \
        0.00002*df['RPM'] + 0.01*df['Viscosity']
    df['Actual_power'] = df['Theoretical_power'] / \
        np.clip(efficiency, 0.6, 0.95)

    # ویژگی جدید: شاخص سلامت تجهیزات (PHM)
    df['Health_index'] = np.clip(
        1.0 - (0.3*(df['Vibration']/10) +
               0.2*(df['Torque']/500) +
               0.5*(df['Sound']/120)),
        0.0, 1.0
    )

    # تولید برچسب‌های خرابی پیشرفته
    compressor_risk = (
        0.6 * (df['Vibration'] > 5) +
        0.4 * (df['Flow_in'] < 300) +
        0.5 * (df['T_in'] > 330) +
        0.3 * (df['Health_index'] < 0.7)
    ).clip(0, 1)

    bearing_risk = (
        0.7 * (df['Torque'] < 250) +
        0.4 * (df['RPM'] > 8000) +
        0.6 * (df['Sound'] > 100) +
        0.3 * (df['Vibration_derivative'] > 0.5)
    ).clip(0, 1)

    # تولید برچسب‌ها با توزیع پواسون برای شبیه‌سازی خرابی‌های دوره‌ای
    df["Failure_compressor"] = np.random.poisson(
        compressor_risk * 0.1).clip(0, 1).astype(np.int8)
    df["Failure_bearing"] = np.random.poisson(
        bearing_risk * 0.1).clip(0, 1).astype(np.int8)

    # پر کردن مقادیر NaN با روش backward fill و سپس صفر
    return df.fillna(method='bfill').fillna(0)


# نمونه استفاده:
industrial_data = generate_industrial_data()
print(industrial_data.head())


# class AnomalyDetector:
#     def __init__(self, data, contamination=0.05):
#         self.data = data
#         self.contamination = contamination
#         self.anomalies = None
#         self.model = IsolationForest(
#                 n_estimators=200,
#                 max_samples=256,
#                 contamination=0.01,  # if 1% anomalies are expected
#                 max_features=1.0,
#                 bootstrap=False,
#                 random_state=42
#             )

#     def preprocess_data(self):
#         # Drop non-numeric columns if they exist
#         cols_to_drop = ['Failure_compressor', 'Failure_bearing']
#         existing_cols = [
#             col for col in cols_to_drop if col in self.data.columns]
#         self.data = self.data.drop(existing_cols, axis=1)

#         # Fill or drop missing values
#         self.data = self.data.fillna(method='ffill')

#         # Scale the features
#         scaler = StandardScaler()
#         self.data_scaled = scaler.fit_transform(self.data)

#     def detect_anomalies(self):
#         self.model.fit(self.data_scaled)
#         self.anomalies = self.model.predict(self.data_scaled)
#         self.data['anomaly'] = self.anomalies

#     def get_anomalies(self):
#         return self.data[self.data['anomaly'] == -1]

#     def plot_anomalies(self, x_col='P_in', y_col='T_in'):
#         anomaly_data = self.get_anomalies()

#         plt.figure(figsize=(12, 6))

#         plt.scatter(self.data[self.data['anomaly'] == 1][x_col],
#                     self.data[self.data['anomaly'] == 1][y_col],
#                     color='blue',
#                     label='Normal Data', alpha=0.6)

#         plt.scatter(anomaly_data[x_col], anomaly_data[y_col],
#                     color='red',
#                     label='Anomalies', alpha=0.8)

#         plt.xlabel(x_col)
#         plt.ylabel(y_col)
#         plt.title('Anomaly Detection Visualization')
#         plt.legend()
#         plt.grid()
#         plt.tight_layout()
#         plt.show()


# # Usage
# # industrial_data should be a pandas DataFrame with your data
# anomaly_detector = AnomalyDetector(industrial_data)
# anomaly_detector.preprocess_data()
# anomaly_detector.detect_anomalies()
# anomaly_detector.plot_anomalies(x_col='P_in', y_col='T_in')


class AnomalyDetector:
    def __init__(self, data, contamination=0.05):
        self.data = data
        self.contamination = contamination
        self.anomalies = None
        self.model = IsolationForest(
            n_estimators=200,
            max_samples=256,
            contamination=contamination,  # Use the passed contamination value
            max_features=1.0,
            bootstrap=False,
            random_state=42
        )

    def preprocess_data(self):
        # Drop non-numeric columns if they exist
        cols_to_drop = ['Failure_compressor', 'Failure_bearing']
        existing_cols = [
            col for col in cols_to_drop if col in self.data.columns]
        self.data = self.data.drop(existing_cols, axis=1)

        # Fill or drop missing values
        self.data = self.data.fillna(method='ffill')

        # Scale the features
        self.scaler = StandardScaler()
        self.data_scaled = self.scaler.fit_transform(self.data)

    def detect_anomalies(self):
        self.model.fit(self.data_scaled)
        self.anomalies = self.model.predict(self.data_scaled)
        self.data['anomaly'] = self.anomalies

    def get_anomalies(self):
        return self.data[self.data['anomaly'] == -1]

    def plot_anomalies(self, x_col='P_in', y_col='T_in'):
        anomaly_data = self.get_anomalies()

        plt.figure(figsize=(12, 6))

        plt.scatter(self.data[self.data['anomaly'] == 1][x_col],
                    self.data[self.data['anomaly'] == 1][y_col],
                    color='blue',
                    label='Normal Data', alpha=0.6)

        plt.scatter(anomaly_data[x_col], anomaly_data[y_col],
                    color='red',
                    label='Anomalies', alpha=0.8)

        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title('Anomaly Detection Visualization')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    def save_model(self, model_filename='anomaly_detector_model.pkl'):
        joblib.dump(self.model, model_filename)
        joblib.dump(self.scaler, 'scaler.pkl')  # Save the scaler as well

    def load_model(self, model_filename='anomaly_detector_model.pkl', scaler_filename='scaler.pkl'):
        self.model = joblib.load(model_filename)
        self.scaler = joblib.load(scaler_filename)  # Load the scaler

        # Note: You might also want to handle resetting any existing data or anomalies.
        self.anomalies = None
        self.data_scaled = None  # Reset scaled data


# Usage
# industrial_data should be a pandas DataFrame with your data
anomaly_detector = AnomalyDetector(industrial_data)
anomaly_detector.preprocess_data()
anomaly_detector.detect_anomalies()
anomaly_detector.plot_anomalies(x_col='P_in', y_col='T_in')

# Save the model after training
anomaly_detector.save_model()

# Load the model (in a new instance or later)
# new_anomaly_detector = AnomalyDetector(data)  # You can handle data if needed
# new_anomaly_detector.load_model()
