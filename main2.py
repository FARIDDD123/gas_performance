import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def generate_realistic_industrial_data(n_records=1_000_000):
    np.random.seed(42)

    # Create a timestamp index for time series analysis
    timestamps = pd.date_range('2023-01-01', periods=n_records, freq='1min')

    # Generate base features
    P_in = np.clip(np.random.normal(1.2e6, 2e5, n_records),
                   1e6, 1.6e6).astype(np.float32)
    T_in = np.clip(np.random.normal(310, 10, n_records),
                   290, 330).astype(np.float32)

    # Constants for calculations
    gamma = 1.4  # specific heat ratio

    # Generate P_OUT and T_OUT with realistic relationships
    P_out = np.clip(P_in * 1.1 + np.random.normal(0, 5e4,
                    n_records), 1.1e6, 1.7e6).astype(np.float32)

    pressure_ratio = P_out / P_in
    temperature_ratio = pressure_ratio ** ((gamma - 1) / gamma)
    T_out_adiabatic = T_in * temperature_ratio
    T_OUT = np.clip(T_out_adiabatic + np.random.normal(0, 5,
                    n_records), 300, 350).astype(np.float32)

    # Create more realistic features
    data = {
        "P_in": P_in,
        "T_in": T_in,
        "P_OUT": P_out,
        "T_OUT": T_OUT,
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

    # Theoretical and actual power calculations (updated with P_OUT)
    R = 287  # specific gas constant for air (J/kg·K)
    df['Theoretical_power'] = (df['Flow_in'] * R * df['T_in'] / (gamma - 1) *
                               ((df['P_OUT']/df['P_in'])**((gamma-1)/gamma) - 1))
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

    # Generate labels with realistic conditions and noise
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

    # Generate failure labels
    df["Failure_compressor"] = (compressor_risk > 0.5).astype(int)
    df["Failure_bearing"] = (bearing_risk > 0.5).astype(int)

    # Fill missing values
    return df.fillna(method='bfill').fillna(0)


# Example Usage
industrial_data = generate_realistic_industrial_data()
print(industrial_data.head())


class GasSystemAnalyzer:
    def __init__(self, pipe_diameter, gas_constant=287, gamma=1.4):
        self.pipe_diameter = pipe_diameter
        self.gas_constant = gas_constant
        self.gamma = gamma

    def calculate_thermodynamic_properties(self, df):
        """محاسبه خواص ترمودینامیکی و هیدرودینامیکی"""
        cross_section_area = np.pi * (self.pipe_diameter / 2) ** 2
        df['Velocity'] = df['Flow_in'] / cross_section_area
        # محاسبه عدد رینولدز
        df['Reynolds'] = (df['Density'] * df['Flow_in'] *
                          self.pipe_diameter) / df['Viscosity']

        # محاسبه عدد ماخ
        df['Speed_of_sound'] = np.sqrt(
            self.gamma * self.gas_constant * df['T_in'])
        df['Mach'] = df['Flow_in'] / df['Speed_of_sound']

        # محاسبه ضریب تراکم‌پذیری
        df['Compressibility'] = (
            df['P_in'] * self.gas_constant * df['T_in']) / (df['Density'] + 1e-6)

        return df


class NavierStokesSolver:
    def __init__(self, density, viscosity, dx=0.1, dt=0.01):
        self.density = density
        self.viscosity = viscosity
        self.dx = dx
        self.dt = dt

    def solve_mass_spring(self, force, mass, spring_constant, n_steps=1000):
        """حل معادله جرم-فنر برای تحلیل ارتعاشات"""
        positions = np.zeros(n_steps)
        velocity = 0
        position = 0

        for i in range(1, n_steps):
            acceleration = (force - spring_constant * position) / mass
            velocity += acceleration * self.dt
            position += velocity * self.dt
            positions[i] = position

        return positions

    def solve_navier_stokes_1D(self, initial_velocity, pressure_gradient, length, n_steps=1000):
        """حل ساده شده معادله نویر-استوکس در یک بعد"""
        n_points = len(initial_velocity)
        velocity = np.array(initial_velocity)

        for _ in range(n_steps):
            dudt = np.zeros_like(velocity)
            dudt[1:-1] = (-velocity[1:-1] * (velocity[2:] - velocity[:-2]) / (2 * self.dx) +
                          self.viscosity * (velocity[2:] - 2*velocity[1:-1] + velocity[:-2]) / self.dx**2 -
                          pressure_gradient / self.density)

            velocity += dudt * self.dt

        return velocity


gas = GasSystemAnalyzer(pipe_diameter=0.5)
industrial_data['Density'] = industrial_data['P_in'] / \
    (industrial_data['T_in'] * 287)
df = gas.calculate_thermodynamic_properties(industrial_data)


# 2. تحلیل ارتعاشات با معادله جرم-فنر
vibration_analyzer = NavierStokesSolver(density=1.225, viscosity=1.8e-5)
mass = 500  # kg
spring_constant = 1e4  # N/m
force = df['Vibration'].values * 1000  # تبدیل به نیوتن

vibration_results = []
for f in force[:1000]:  # نمونه‌گیری برای محاسبات سریع
    vibration_results.extend(
        vibration_analyzer.solve_mass_spring(f, mass, spring_constant))

plt.figure(figsize=(12, 6))
plt.plot(vibration_results[:500])
plt.title('Vibration Simulation Results')
plt.xlabel('Time Steps')
plt.ylabel('Displacement')
plt.show()

x = df.iloc[:, :4].values
y = df.iloc[:, 22].values


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)


sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


model = LinearRegression()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))


plt.figure(figsize=(12, 7))
plt.scatter(x_test[:, :1], y_test[:])
plt.scatter(x_test[:, :1], model.predict(x_test))
plt.show()
