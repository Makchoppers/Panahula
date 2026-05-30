import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from scipy import linalg
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Weather Analyzer & Plotter",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


# ==================== OOP - Abstract Base Classes ====================

class DataProcessor(ABC):
    """Abstract base class for data processing"""
    
    @abstractmethod
    def process(self, data: np.ndarray) -> dict:
        """Process weather data using NumPy"""
        pass


class Plotter(ABC):
    """Abstract base class for plotting"""
    
    @abstractmethod
    def plot(self, data: pd.DataFrame) -> Figure:
        """Create a plot"""
        pass


class Predictor(ABC):
    """Abstract base class for predictions"""
    
    @abstractmethod
    def predict(self, data: np.ndarray) -> dict:
        """Make predictions"""
        pass


# ==================== Concrete Implementations ====================

class NumPyDataProcessor(DataProcessor):
    """Processes weather data using NumPy - organizes the chaos"""
    
    def process(self, data: np.ndarray) -> dict:
        """NumPy quickly computes averages, finds min/max, creates matrices"""
        if data is None or len(data) == 0:
            return {}
        
        return {
            'mean': np.mean(data),
            'min': np.min(data),
            'max': np.max(data),
            'std': np.std(data),
            'median': np.median(data),
            'variance': np.var(data),
            'sum': np.sum(data)
        }


class TemperatureBarPlotter(Plotter):
    """Creates bar chart for daily temperatures - one bar per day"""
    
    def plot(self, data: pd.DataFrame) -> Figure:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        day_numbers = np.arange(len(data))
        dates_str = [d.strftime('%m/%d') if isinstance(d, datetime) else str(d) for d in data['date']]
        
        # Color bars based on temperature
        colors = np.where(data['temperature'].values < 15, '#4A90E2',
                         np.where(data['temperature'].values > 25, '#FF6B6B', '#90EE90'))
        
        bars = ax.bar(day_numbers, data['temperature'].values, 
                     color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        
        ax.set_title('Daily Temperature Bar Chart - Weather Mood Swings', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Day', fontsize=12)
        ax.set_ylabel('Temperature (°C)', fontsize=12)
        ax.set_xticks(day_numbers)
        ax.set_xticklabels(dates_str, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, temp in zip(bars, data['temperature'].values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{temp:.1f}°', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        return fig


class RainfallBarPlotter(Plotter):
    """Creates bar chart for weekly rainfall totals"""
    
    def plot(self, data: pd.DataFrame) -> Figure:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        if len(data) >= 7:
            # Weekly totals using NumPy
            n_weeks = len(data) // 7
            weekly_rainfall = []
            week_labels = []
            
            for week in range(n_weeks):
                start_idx = week * 7
                end_idx = min((week + 1) * 7, len(data))
                week_total = np.sum(data['rainfall'].iloc[start_idx:end_idx].values)
                weekly_rainfall.append(week_total)
                week_labels.append(f'Week {week + 1}')
            
            if len(data) % 7 > 0:
                start_idx = n_weeks * 7
                week_total = np.sum(data['rainfall'].iloc[start_idx:].values)
                weekly_rainfall.append(week_total)
                week_labels.append(f'Week {n_weeks + 1}')
            
            x_pos = np.arange(len(weekly_rainfall))
            bars = ax.bar(x_pos, weekly_rainfall, color='#4A90E2', alpha=0.7, 
                         edgecolor='black', linewidth=1)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(week_labels)
        else:
            # Daily rainfall
            day_numbers = np.arange(len(data))
            dates_str = [d.strftime('%m/%d') if isinstance(d, datetime) else str(d) for d in data['date']]
            bars = ax.bar(day_numbers, data['rainfall'].values, color='#4A90E2', 
                         alpha=0.7, edgecolor='black', linewidth=1)
            ax.set_xticks(day_numbers)
            ax.set_xticklabels(dates_str, rotation=45, ha='right')
        
        ax.set_title('Weekly Rainfall Totals', fontsize=14, fontweight='bold')
        ax.set_xlabel('Period', fontsize=12)
        ax.set_ylabel('Rainfall (mm)', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}mm', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        return fig


class HumidityBarPlotter(Plotter):
    """Creates bar chart for humidity - rising and falling like a drumbeat"""
    
    def plot(self, data: pd.DataFrame) -> Figure:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        day_numbers = np.arange(len(data))
        dates_str = [d.strftime('%m/%d') if isinstance(d, datetime) else str(d) for d in data['date']]
        
        humidity_values = data['humidity'].values
        colors = plt.cm.viridis(humidity_values / 100)
        
        bars = ax.bar(day_numbers, humidity_values, color=colors, alpha=0.8, 
                     edgecolor='black', linewidth=1)
        
        ax.set_title('Humidity Levels - Rising and Falling Like a Slow Drumbeat', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Day', fontsize=12)
        ax.set_ylabel('Humidity (%)', fontsize=12)
        ax.set_xticks(day_numbers)
        ax.set_xticklabels(dates_str, rotation=45, ha='right')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, hum in zip(bars, humidity_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{hum:.0f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        return fig


class LinearRegressionPlotter(Plotter):
    """Plots linear regression line with data points and prediction"""
    
    def plot_temperature_regression(self, data: pd.DataFrame, predictions: dict) -> Figure:
        """Plot temperature data with linear regression line and prediction"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        temp_array = data['temperature'].values
        dates = data['date'].values
        
        # Plot actual data points
        ax.scatter(range(len(temp_array)), temp_array, color='#ff6b6b', 
                  s=100, alpha=0.7, label='Actual Temperature', zorder=3)
        
        # If we have trend prediction, plot regression line
        if 'trend' in predictions and len(temp_array) >= 5:
            # Use last 5 days for regression
            x_reg = np.arange(len(temp_array[-5:]))
            y_reg = temp_array[-5:]
            
            # Fit linear regression
            X = np.column_stack([np.ones(len(x_reg)), x_reg])
            result = linalg.lstsq(X, y_reg)
            slope = result[0][1]
            intercept = result[0][0]
            
            # Extend regression line to include tomorrow
            x_line = np.arange(len(temp_array[-5:]) + 1)
            y_line = intercept + slope * x_line
            
            # Plot regression line
            x_plot = np.arange(len(temp_array) - 5, len(temp_array) + 1)
            ax.plot(x_plot, y_line, '--', color='#4ecdc4', linewidth=2, 
                   label='Linear Regression Trend', zorder=2)
            
            # Plot tomorrow's prediction
            tomorrow_pred = intercept + slope * len(temp_array[-5:])
            ax.scatter(len(temp_array), tomorrow_pred, color='#45b7d1', 
                      s=150, marker='*', label='Tomorrow Prediction', zorder=4, edgecolors='black', linewidths=1)
        
        # Plot 3-day average prediction if available
        if 'tomorrow_temp' in predictions:
            pred = predictions['tomorrow_temp']
            ax.axhline(y=pred['prediction'], color='#90EE90', linestyle=':', 
                      linewidth=2, label=f"3-day Avg ({pred['prediction']}°C)", alpha=0.7)
        
        ax.set_title('Temperature Linear Regression & Prediction', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Day Index', fontsize=12)
        ax.set_ylabel('Temperature (°C)', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    def plot_rainfall_regression(self, data: pd.DataFrame, predictions: dict) -> Figure:
        """Plot rainfall data with linear regression line and prediction"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        rainfall_array = data['rainfall'].values
        dates = data['date'].values
        
        # Plot actual data points
        ax.bar(range(len(rainfall_array)), rainfall_array, color='#4A90E2', 
              alpha=0.7, label='Actual Rainfall', edgecolor='black', linewidth=1)
        
        # If we have rainfall prediction, plot regression line
        if 'tomorrow_rainfall' in predictions and len(rainfall_array) >= 5:
            # Use last 5 days for regression
            x_reg = np.arange(len(rainfall_array[-5:]))
            y_reg = rainfall_array[-5:]
            
            # Fit linear regression
            X = np.column_stack([np.ones(len(x_reg)), x_reg])
            result = linalg.lstsq(X, y_reg)
            slope = result[0][1]
            intercept = result[0][0]
            
            # Extend regression line to include tomorrow
            x_line = np.arange(len(rainfall_array[-5:]) + 1)
            y_line = intercept + slope * x_line
            y_line = np.maximum(y_line, 0)  # Can't be negative
            
            # Plot regression line
            x_plot = np.arange(len(rainfall_array) - 5, len(rainfall_array) + 1)
            ax.plot(x_plot, y_line, '--', color='#FF6B6B', linewidth=2, 
                   marker='o', markersize=6, label='Linear Regression Trend', zorder=3)
            
            # Plot tomorrow's prediction
            tomorrow_pred = max(0, intercept + slope * len(rainfall_array[-5:]))
            rain_pred = predictions['tomorrow_rainfall']
            color = '#FF6B6B' if rain_pred['will_be_rainy'] else '#90EE90'
            ax.bar(len(rainfall_array), tomorrow_pred, color=color, alpha=0.8, 
                  label='Tomorrow Prediction', edgecolor='black', linewidth=2, width=0.6)
            
            # Add threshold line
            ax.axhline(y=2.0, color='orange', linestyle=':', linewidth=1.5, 
                      label='Rainy Threshold (2mm)', alpha=0.7)
        
        ax.set_title('Rainfall Linear Regression & Prediction', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Day Index', fontsize=12)
        ax.set_ylabel('Rainfall (mm)', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        return fig
    
    def plot(self, data: pd.DataFrame) -> Figure:
        """Default plot method"""
        return self.plot_temperature_regression(data, {})


class SimplePredictor(Predictor):
    """Simple ML model using SciPy linear regression - makes simple predictions"""
    
    def predict_temperature(self, temp_data: np.ndarray) -> dict:
        """Predicts tomorrow's temperature using 3-day average or trend detection"""
        if temp_data is None or len(temp_data) < 3:
            return {}
        
        predictions = {}
        
        # 1. Tomorrow's temperature: average of last 3 days
        if len(temp_data) >= 3:
            last_3_avg = np.mean(temp_data[-3:])
            predictions['tomorrow_temp'] = {
                'prediction': round(last_3_avg, 1),
                'method': '3-day average',
                'last_3_days': temp_data[-3:].tolist()
            }
        
        # 2. Trend detection using SciPy linear regression
        if len(temp_data) >= 5:
            x = np.arange(len(temp_data[-5:]))
            y = temp_data[-5:]
            
            # Add intercept term
            X = np.column_stack([np.ones(len(x)), x])
            
            # Use SciPy's least squares for linear regression
            result = linalg.lstsq(X, y)
            slope = result[0][1]
            intercept = result[0][0]
            
            if slope > 0.5:
                trend = 'increasing'
                trend_strength = 'strong' if slope > 1 else 'moderate'
            elif slope < -0.5:
                trend = 'decreasing'
                trend_strength = 'strong' if slope < -1 else 'moderate'
            else:
                trend = 'stable'
                trend_strength = 'weak'
            
            predictions['trend'] = {
                'trend': trend,
                'strength': trend_strength,
                'slope': round(slope, 2),
                'method': 'SciPy linear regression'
            }
        
        return predictions
    
    def predict_rainfall(self, rainfall_data: np.ndarray) -> dict:
        """Predicts if tomorrow will be rainy using SciPy linear regression"""
        if rainfall_data is None or len(rainfall_data) < 3:
            return {}
        
        predictions = {}
        
        # 1. Average rainfall of last 3 days
        last_3_avg = np.mean(rainfall_data[-3:])
        
        # 2. Trend detection using SciPy linear regression
        if len(rainfall_data) >= 5:
            x = np.arange(len(rainfall_data[-5:]))
            y = rainfall_data[-5:]
            
            # Add intercept term
            X = np.column_stack([np.ones(len(x)), x])
            
            # Use SciPy's least squares for linear regression
            result = linalg.lstsq(X, y)
            slope = result[0][1]
            intercept = result[0][0]
            
            # Predict tomorrow's rainfall (next point in the trend)
            tomorrow_prediction = intercept + slope * len(rainfall_data[-5:])
            tomorrow_prediction = max(0, tomorrow_prediction)  # Can't be negative
            
            # Determine if it will be rainy (threshold: > 2mm)
            will_be_rainy = tomorrow_prediction > 2.0 or last_3_avg > 2.0
            
            predictions['tomorrow_rainfall'] = {
                'will_be_rainy': will_be_rainy,
                'predicted_amount': round(tomorrow_prediction, 1),
                'average_last_3_days': round(last_3_avg, 1),
                'method': 'SciPy linear regression + 3-day average',
                'confidence': 'high' if (tomorrow_prediction > 5 or last_3_avg > 5) else 'medium' if (tomorrow_prediction > 2 or last_3_avg > 2) else 'low'
            }
        else:
            # Simple threshold-based prediction if not enough data
            will_be_rainy = last_3_avg > 2.0
            predictions['tomorrow_rainfall'] = {
                'will_be_rainy': will_be_rainy,
                'predicted_amount': round(last_3_avg, 1),
                'average_last_3_days': round(last_3_avg, 1),
                'method': '3-day average threshold',
                'confidence': 'medium'
            }
        
        return predictions
    
    def predict(self, data: np.ndarray) -> dict:
        """Predicts tomorrow's temperature using 3-day average or trend detection"""
        return self.predict_temperature(data)


# ==================== Weather Analyzer Class ====================

class WeatherAnalyzer:
    """Main analyzer class using OOP composition"""
    
    def __init__(self):
        self.processor = NumPyDataProcessor()
        self.predictor = SimplePredictor()
        self._data = None
    
    def load_data(self, temperatures: list, humidities: list, rainfall: list = None, days: int = None):
        """Load weather data into NumPy arrays - tidy drawers for numbers"""
        if days is None:
            days = len(temperatures)
        
        base_date = datetime.now()
        dates = [base_date - timedelta(days=days-i-1) for i in range(days)]
        
        # Convert to NumPy arrays
        temp_array = np.array(temperatures[:days])
        hum_array = np.array(humidities[:days])
        
        if rainfall is None:
            rain_array = np.zeros(days)
        else:
            rain_array = np.array(rainfall[:days])
        
        # Create DataFrame
        self._data = pd.DataFrame({
            'date': dates,
            'temperature': temp_array,
            'humidity': hum_array,
            'rainfall': rain_array
        })
        
        return self._data
    
    def parse_text_input(self, text: str, days: int = 7):
        """Parse simple text like 'hot/cold/rainy' into weather data"""
        keywords = text.lower()
        
        # Generate data based on keywords
        np.random.seed(42)
        base_date = datetime.now()
        dates = [base_date - timedelta(days=days-i-1) for i in range(days)]
        
        # Simple mapping
        if 'hot' in keywords:
            temps = np.random.uniform(25, 35, days)
            hums = np.random.uniform(30, 60, days)
        elif 'cold' in keywords:
            temps = np.random.uniform(0, 15, days)
            hums = np.random.uniform(40, 70, days)
        elif 'rainy' in keywords:
            temps = np.random.uniform(10, 20, days)
            hums = np.random.uniform(70, 95, days)
            rains = np.random.exponential(5, days)
            rains = np.where(rains > 20, 0, rains)
        else:
            temps = np.random.uniform(15, 25, days)
            hums = np.random.uniform(40, 80, days)
            rains = np.zeros(days)
        
        if 'rainy' not in keywords:
            rains = np.zeros(days)
        
        self._data = pd.DataFrame({
            'date': dates,
            'temperature': np.round(temps, 1),
            'humidity': np.round(hums, 1),
            'rainfall': np.round(rains, 1)
        })
        
        return self._data
    
    def get_statistics(self):
        """Get statistics using NumPy - computes averages, finds min/max"""
        if self._data is None:
            return {}
        
        temp_stats = self.processor.process(self._data['temperature'].values)
        hum_stats = self.processor.process(self._data['humidity'].values)
        rain_stats = self.processor.process(self._data['rainfall'].values)
        
        return {
            'temperature': temp_stats,
            'humidity': hum_stats,
            'rainfall': rain_stats
        }
    
    def get_predictions(self):
        """Get simple predictions using SciPy linear regression"""
        if self._data is None:
            return {}
        
        temp_array = self._data['temperature'].values
        rainfall_array = self._data['rainfall'].values
        
        # Get temperature predictions
        temp_predictions = self.predictor.predict_temperature(temp_array)
        
        # Get rainfall predictions
        rain_predictions = self.predictor.predict_rainfall(rainfall_array)
        
        # Combine predictions
        return {**temp_predictions, **rain_predictions}
    
    def create_feature_matrix(self):
        """Convert inputs into clean matrix for ML"""
        if self._data is None:
            return None
        
        return np.column_stack([
            self._data['temperature'].values,
            self._data['humidity'].values,
            self._data['rainfall'].values
        ])


# ==================== Main Application ====================

def main():
    st.markdown('<h1 class="main-header">Weather Analyzer & Plotter</h1>', unsafe_allow_html=True)
    
    analyzer = WeatherAnalyzer()
    
    # Sidebar
    with st.sidebar:
        st.header("Input Weather Data")
        
        input_type = st.radio("Input Type", ["Manual Entry", "Simple Text"])
        
        if input_type == "Manual Entry":
            st.subheader("Enter Daily Data")
            days = st.slider("Number of Days", min_value=1, max_value=30, value=7)
            
            st.write("**Temperatures (°C)**")
            temp_input = st.text_area("Enter temperatures (comma-separated)", 
                                     value="20, 22, 18, 25, 23, 21, 19",
                                     help="Example: 20, 22, 18, 25")
            
            st.write("**Humidity (%)**")
            hum_input = st.text_area("Enter humidity values (comma-separated)", 
                                    value="60, 65, 70, 55, 58, 62, 68",
                                    help="Example: 60, 65, 70, 55")
            
            st.write("**Rainfall (mm)** - Optional")
            rain_input = st.text_area("Enter rainfall values (comma-separated)", 
                                     value="0, 0, 5, 0, 0, 2, 0",
                                     help="Example: 0, 0, 5, 0")
            
            if st.button("Load Data", type="primary", use_container_width=True):
                try:
                    temps = [float(x.strip()) for x in temp_input.split(',')]
                    hums = [float(x.strip()) for x in hum_input.split(',')]
                    rains = [float(x.strip()) for x in rain_input.split(',')] if rain_input.strip() else None
                    
                    df = analyzer.load_data(temps, hums, rains, days)
                    st.session_state['weather_data'] = df
                    st.session_state['analyzer'] = analyzer
                    st.success(f"Loaded {len(df)} days of data!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        else:  # Simple Text
            st.subheader("Simple Text Input")
            days = st.slider("Number of Days", min_value=1, max_value=30, value=7)
            
            text_input = st.text_area("Enter weather description", 
                                     value="today was hot, yesterday was rainy",
                                     help="Examples: 'hot', 'cold', 'rainy', 'today was hot'")
            
            if st.button("Load Data", type="primary", use_container_width=True):
                try:
                    df = analyzer.parse_text_input(text_input, days)
                    st.session_state['weather_data'] = df
                    st.session_state['analyzer'] = analyzer
                    st.success(f"Loaded {len(df)} days of data!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Main content
    if 'weather_data' in st.session_state:
        df = st.session_state['weather_data']
        analyzer = st.session_state['analyzer']
        
        st.divider()
        
        # 1. Statistics - NumPy organizes the chaos
        st.header("Statistics (NumPy)")
        stats = analyzer.get_statistics()
        
        if stats:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Temperature")
                st.metric("Average", f"{stats['temperature']['mean']:.1f}°C")
                st.caption(f"Min: {stats['temperature']['min']:.1f}°C | Max: {stats['temperature']['max']:.1f}°C")
                st.caption(f"Std: {stats['temperature']['std']:.2f}")
            
            with col2:
                st.subheader("Humidity")
                st.metric("Average", f"{stats['humidity']['mean']:.1f}%")
                st.caption(f"Min: {stats['humidity']['min']:.1f}% | Max: {stats['humidity']['max']:.1f}%")
            
            with col3:
                st.subheader("Rainfall")
                st.metric("Total", f"{stats['rainfall']['sum']:.1f} mm")
                st.caption(f"Average: {stats['rainfall']['mean']:.1f} mm/day")
        
        st.divider()
        
        # 2. Bar Charts - Weather mood swings
        st.header("Bar Charts (Matplotlib)")
        
        chart_type = st.selectbox("Select Chart", 
                                  ["Temperature Bars", "Rainfall Bars", "Humidity Bars",
                                   "Temperature Linear Regression", "Rainfall Linear Regression"])
        
        if chart_type == "Temperature Bars":
            plotter = TemperatureBarPlotter()
            fig = plotter.plot(df)
            st.pyplot(fig)
        elif chart_type == "Rainfall Bars":
            plotter = RainfallBarPlotter()
            fig = plotter.plot(df)
            st.pyplot(fig)
        elif chart_type == "Humidity Bars":
            plotter = HumidityBarPlotter()
            fig = plotter.plot(df)
            st.pyplot(fig)
        elif chart_type == "Temperature Linear Regression":
            reg_plotter = LinearRegressionPlotter()
            predictions = analyzer.get_predictions()
            fig = reg_plotter.plot_temperature_regression(df, predictions)
            st.pyplot(fig)
            st.caption("Shows actual data points, linear regression trend line, and tomorrow's prediction")
        elif chart_type == "Rainfall Linear Regression":
            reg_plotter = LinearRegressionPlotter()
            predictions = analyzer.get_predictions()
            fig = reg_plotter.plot_rainfall_regression(df, predictions)
            st.pyplot(fig)
            st.caption("Shows actual rainfall bars, linear regression trend line, and tomorrow's prediction (with 2mm threshold)")
        
        st.divider()
        
        # 3. Simple Predictions - Tiny ML model
        st.header("Simple Predictions (SciPy Linear Regression)")
        
        predictions = analyzer.get_predictions()
        
        if predictions:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'tomorrow_temp' in predictions:
                    pred = predictions['tomorrow_temp']
                    st.metric("Tomorrow's Temperature", f"{pred['prediction']}°C")
                    st.caption(f"Method: {pred['method']}")
                    st.caption(f"Based on: {pred['last_3_days']}")
            
            with col2:
                if 'trend' in predictions:
                    trend = predictions['trend']
                    st.metric("Temperature Trend", trend['trend'].title())
                    st.caption(f"Strength: {trend['strength']}")
                    st.caption(f"Method: {trend['method']}")
            
            with col3:
                if 'tomorrow_rainfall' in predictions:
                    rain_pred = predictions['tomorrow_rainfall']
                    rain_text = "Yes, Rainy" if rain_pred['will_be_rainy'] else "No, Dry"
                    st.metric("Tomorrow Will Be Rainy?", rain_text)
                    st.caption(f"Predicted: {rain_pred['predicted_amount']}mm")
                    st.caption(f"Avg last 3 days: {rain_pred['average_last_3_days']}mm")
                    st.caption(f"Confidence: {rain_pred['confidence']}")
                    st.caption(f"Method: {rain_pred['method']}")
        
        st.divider()
        
        # 4. Feature Matrix
        st.header("Feature Matrix (For ML)")
        feature_matrix = analyzer.create_feature_matrix()
        if feature_matrix is not None:
            st.dataframe(pd.DataFrame(feature_matrix, 
                                    columns=['Temperature', 'Humidity', 'Rainfall']),
                        use_container_width=True)
            st.caption("This clean matrix is used as a compass for ML predictions")
        
        st.divider()
        
        # Raw Data
        st.header("Raw Data")
        st.dataframe(df, use_container_width=True)
        
        # Download
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"weather_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    else:
        st.info("Enter weather data in the sidebar to get started!")
        
        with st.expander("How It Works"):
            st.markdown("""
            ### How It Works
            
            1. **You feed it the weather**: Give simple inputs (temperatures, rainfall, humidity, or "hot/cold/rainy")
               - These slip into NumPy arrays, which act like tidy drawers for numbers
            
            2. **NumPy organizes the chaos**: NumPy quickly:
               - Computes averages
               - Finds hottest/coldest days
               - Converts your inputs into a clean matrix for ML
               - Handles weeks or months of data
               - The program doesn't panic no matter how many days you throw at it
            
            3. **It draws bar charts**: Using matplotlib, it creates bar charts:
               - A bar for each day's temperature
               - Weekly rainfall totals
               - Humidity levels rising and falling like a slow drumbeat
            
            4. **A tiny ML model makes simple predictions**: Using SciPy linear regression:
               - "Tomorrow will be close to the average of the last 3 days"
               - "Temperature is trending upward"
               - It uses your weather matrix as its compass
            
            5. **OOP keeps it clean and expandable**: Clean code structure with polymorphism and inheritance
            """)

if __name__ == "__main__":
    main()
