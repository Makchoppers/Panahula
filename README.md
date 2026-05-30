# Weather Analyzer & Plotter

A comprehensive Python application for analyzing and visualizing weather data with an intuitive web-based user interface.

## Features

- **Statistical Analysis**: Calculate mean, min, max, standard deviation, and trends for weather metrics
- **Interactive Visualizations**: Multiple plot types including:
  - Temperature over time
  - Multi-metric comparisons
  - Humidity vs Pressure scatter plots
  - Correlation heatmaps
  - Distribution histograms
  - Weather conditions pie charts
- **Real Weather Data**: Optional integration with OpenWeatherMap API for live data
- **Data Export**: Download weather data as CSV files
- **Modern UI**: Beautiful, responsive web interface built with Streamlit

## Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage (Sample Data)

1. **Run the application**:
   ```bash
   streamlit run weather_analyzer.py
   ```

2. The application will open in your default web browser

3. Enter a city name and click "Fetch Weather Data"

4. Explore the statistics, visualizations, and analysis!

### Using Real Weather Data (Optional)

1. **Get a free API key**:
   - Sign up at [OpenWeatherMap](https://openweathermap.org/api)
   - Get your free API key from the dashboard

2. **Enable API in the app**:
   - Check "Use OpenWeatherMap API" in the sidebar
   - Enter your API key
   - Fetch weather data as usual

## Requirements

- Python 3.8 or higher
- See `requirements.txt` for package dependencies

## Application Structure

- `weather_analyzer.py`: Main application file with UI and analysis logic
- `requirements.txt`: Python package dependencies
- `README.md`: This file

## Features in Detail

### Statistical Analysis
- Temperature statistics (mean, min, max, trend)
- Humidity analysis
- Wind speed metrics
- Atmospheric pressure data

### Visualizations
- **Temperature Over Time**: Line chart showing temperature trends
- **Multiple Metrics Comparison**: Normalized comparison of all metrics
- **Humidity vs Pressure**: Scatter plot with temperature coloring
- **Correlation Heatmap**: Shows relationships between different weather metrics
- **Distribution Charts**: Histograms for temperature and wind speed
- **Weather Conditions**: Pie chart of weather descriptions

### Data Management
- View raw data in an interactive table
- Download data as CSV
- Support for 1-30 days of data analysis

## Screenshots

The application features:
- Clean, modern interface
- Interactive charts powered by Plotly
- Real-time data fetching
- Comprehensive statistical analysis

## Troubleshooting

### API Issues
- If API calls fail, the app will use sample data
- Make sure your API key is valid and has sufficient quota
- Check your internet connection

### Display Issues
- Make sure all dependencies are installed correctly
- Try refreshing the browser if charts don't appear

## License

This project is open source and available for personal and educational use.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## Future Enhancements

Potential features for future versions:
- Historical weather data analysis
- Weather forecasting integration
- Multiple city comparison
- Custom date range selection
- Export plots as images
- Email reports

---

**Enjoy analyzing weather data!**


# Panahula
