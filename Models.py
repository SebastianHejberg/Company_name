import sys
import pycountry
import numpy as np
import wbgapi as wb
import pandas as pd
import seaborn as sns
from io import StringIO
import statsmodels.api as sm
import ipywidgets as widgets
from openpyxl import Workbook
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.optimize import curve_fit
from ipywidgets import interactive, Layout


# Define a class for forecasting time series data
    
class ForecastClass:
    


    
    def __init__(self, time_series_list):
        """
        Initializes the ForecastClass with a list of time series data.

        Parameters:
        - time_series_list (list): A list of pandas Series representing time series data.

        This constructor initializes the class instance with the provided time series data, which will be used for forecasting.
        """
        self.time_series_list = time_series_list
    


    
    def read_excel_and_process(file_path, sheet_name):
        """
        Read data from an Excel file, process it, and return a list of time series.
        
        Parameters:
        - file_path (str): Path to the Excel file.
        - sheet_name (str): Name of the sheet in the Excel file.
        
        Returns:
        - time_series_list (list): List of pandas Series representing time series data.
        
        This function reads data from an Excel file, processes it, and returns a list of time series represented as pandas Series.
        
        Steps:
        1. Read data from the specified Excel file and sheet.
        2. Set a multi-index for the DataFrame using the columns ["Variable", "Country", "Region"].
        3. Convert the column names to datetime format using the specified format.
        4. Sort columns in ascending order based on the datetime values.
        5. Iterate through each row in the DataFrame's index and create a pandas Series using the row data.
        6. Append each created Series to the time_series_list.
        7. Return the list of time series data.
        
        This function provides a streamlined approach to reading and processing time series data from an Excel file, making it easier to work with and analyze the data.
        """

        # Read data from Excel file
        Marcodata = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # Set multi-index for the DataFrame
        Marcodata.set_index(["Variable", "Country", "Region"], inplace=True)
        
        # Convert columns to datetime format
        Marcodata.columns = pd.to_datetime(Marcodata.columns, format='%Y')
        
        # Sort columns by date
        Marcodata.sort_index(axis=1, inplace=True)
        
        # Create a list of time series using rows from DataFrame
        time_series_list = [pd.Series(data=Marcodata.loc[row], name=row) for row in Marcodata.index]
        
        # Return the list of time series data
        return time_series_list
    


    

    def ets_damped_trend_forecast(self, row, n_preds=30, damping_factor=1, trend=True, plot=False, latex=False, DATAFRAME=False, output=False):  
        """
        Forecast future values using the ETS (Exponential Smoothing) model with damped trend.

        Parameters:
        - row (int): Index of the time series in the time_series_list to be forecasted.
        - n_preds (int): Number of future periods to forecast.
        - damping_factor (float): Damping factor for the trend. If trend is False, this parameter is ignored.
        - trend (bool): Whether to include a damped trend in the model.
        - plot (bool): Whether to plot the forecasted values.
        - latex (bool): Whether to display the ETS model parameters in LaTeX format.
        - DATAFRAME (bool): Whether to return the forecasted data as a DataFrame.
        - output (bool): Whether to print output information.

        Returns:
        - If output is True, prints output information.
        - If DATAFRAME is True, returns the forecasted data as a DataFrame.

        This method forecasts future values using the Exponential Smoothing (ETS) model with an optional damped trend component.
        
        Steps:
        1. Check if the input time series contains NaN values. If yes, interpolate the missing values.
        2. Determine whether to use the original time series or the interpolated series.
        3. Extract relevant information (Variable, Country, Region) from the time series index.
        4. Define forecast dates for the prediction period.
        5. Create an ETS model based on the specified parameters.
        6. Fit the model to the data and forecast future values.
        7. If latex is True, display the ETS model parameters in LaTeX format.
        8. If plot is True, create a plot showing the original data and the forecasted values.
        9. If output is True, print the variable value for the region in the country.
        10. If DATAFRAME is True, create a DataFrame containing the forecasted data.
        11. Return the forecasted data or print output information based on the specified flags.

        This function facilitates the forecasting of future values using the ETS model with optional visualization and output customization.
        """

        # Check for NaN values and interpolate if necessary
        contains_nan = np.isnan(self.time_series_list[row].values).any()

        if contains_nan:
            # Interpolate missing values
            original_series = self.time_series_list[row].values
            nan_indexes = np.isnan(original_series)
            not_nan_indexes = np.arange(len(original_series))[~nan_indexes]

            interpolated_values = np.interp(
                nan_indexes.nonzero()[0], not_nan_indexes, original_series[not_nan_indexes]
            )
            interpolated_series = original_series.copy()  # Make a copy of the original series
            interpolated_series[nan_indexes] = interpolated_values

            # Create a new pandas Series with interpolated values and assign the name
            interpolated_series = pd.Series(interpolated_series.tolist(), name=(
                self.time_series_list[row].name[0], 
                self.time_series_list[row].name[1], 
                self.time_series_list[row].name[2]
            ))

            interpolated_series.index = self.time_series_list[row].index
            series = interpolated_series
        else:
            series = self.time_series_list[row].values  # Use the original series

        # Extract information from the time series index
        last_date = self.time_series_list[row].index[-1]
        X = self.time_series_list[row].name
        
        # Extract the names from the time series list
        Variable = X[0]
        Country = X[1]
        Region = X[2]
        
        # Define forecast dates and create the ETS model
        forecast_dates = pd.date_range(start=last_date, periods=n_preds, freq='AS')
        if contains_nan:
            model = sm.tsa.ExponentialSmoothing(series, trend='add', seasonal=None, damped_trend=trend,freq='AS')
        else:
            model = sm.tsa.ExponentialSmoothing(series, trend='add', seasonal=None, damped_trend=trend)

        # Fit the model, forecast, and determine damping factor label
        fitted_model = model.fit(damping_slope=damping_factor)
        forecast = fitted_model.forecast(steps=n_preds)

        Y = "with" if trend else "without"
        damping_factor=damping_factor if trend else 1

        # Display LaTeX parameter table if requested
        if latex:
            params_table = fitted_model.params_formatted.to_latex()
            print(f'ETS Model Parameters for {Region} in {Country} ({Y} Damped Trend):')
            print(params_table)
        
        # Create a plot if requested
        if plot:
            # Create a date range for the forecast period
            last_date = self.time_series_list[row].index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.DateOffset(years=1), periods=n_preds, freq='AS')

            # Example hightech color codes
            original_data_color = "#00b3b3"  # Teal
            forecast_color = "#ff6600"      # Orange

            # Plot the original data and the forecast
            plt.figure(figsize=(18, 7))
            plt.plot(self.time_series_list[row].index, series, label='Original Data', color=original_data_color)
            plt.plot(forecast_dates, forecast, label='Forecast', color=forecast_color)
            plt.axvline(x=last_date, color="#FF1493", linestyle='--', label='Forecast Start')  # Adding the vertical line
            plt.ylabel(f'{Variable} ', fontsize=14)

            # Set the y-axis ticks with thousand separators and more decimal places
            plt.gca().yaxis.set_major_formatter("{:,.3f}".format)
            plt.rcParams["font.family"] = "Latin Modern Roman"
            plt.rcParams["font.size"] = 12
            plt.title(f'{Variable} Forecast for {Region} in {Country} (ETS Model {Y} Damped Trend)', fontsize=13, loc='left')
            plt.legend(fontsize=12, loc='lower right', title='Damping factor, $\\phi=${:.2f}'.format(damping_factor))
            plt.grid(True)
            plt.show()

        if output: 
            return print(f'{Variable} for {Region} in {Country}')
        else:
            contains_nan = np.isnan(self.time_series_list[row].values).any()
            if contains_nan:
                print(f"Data serie: {Variable} for {Region} in {Country} (time_series_list[{row}]) contains NaN values and was interpolated")
        
        if DATAFRAME:
            last_date = self.time_series_list[row].index[-1]
            start_date = last_date + pd.DateOffset(years=1)
            contains_nan = np.isnan(self.time_series_list[row].values).any()

            if contains_nan:
                data = interpolated_series

            else:
                data = self.time_series_list[row]

            forecast_series = pd.Series(forecast, index=pd.date_range(start=start_date, periods=len(forecast), freq="AS"), name=self.time_series_list[row].name)
            data = data.append(forecast_series)
            return data
            


    
    def value_list(self):
        """
        Extract and format output values from ETS forecast for each time series.
        
        Returns:
        - value_list (list): List of formatted output values for each time series.
        
        This method is defined within a class and is used to extract and format output values from Exponential Smoothing (ETS) forecasts for each time series in the provided list.

        Steps:
        1. Redirect the standard output (stdout) to capture the output of the forecasts.
        2. Initialize an empty list to store the formatted output values.
        3. Create an output buffer using StringIO to capture the forecast output.
        4. Store the original stdout to restore it later.
        5. Iterate through each time series and call the ETS forecast method with specified parameters.
        6. Capture the output of the forecast in the output buffer and append it to the value_list.
        7. Clear the output buffer for the next iteration.
        8. Restore the original stdout to its initial state.
        9. Format specific lines in the captured output values. For instance, replace 'Population' with 'Urban' in relevant lines.
        10. Return the list of formatted output values.
        
        This function streamlines the process of extracting and formatting forecast output values for multiple time series, enhancing readability and making it easier to analyze and present the results.
        """
        # Redirect stdout to capture the output of the forecasts
        value_list = [] 
        from io import StringIO
        output_buffer = StringIO()
        original_stdout = sys.stdout
        sys.stdout = output_buffer

        # Iterate through each time series and capture output
        for row in range(len(self.time_series_list)):
            self.ets_damped_trend_forecast(row=row, n_preds=28, damping_factor=0.97, trend=True, plot=False, latex=False, DATAFRAME=False, output=True)
            output_value = output_buffer.getvalue().strip()
            value_list.append(output_value)
            output_buffer.truncate(0)
            output_buffer.seek(0)
        
        # Restore original stdout
        sys.stdout = original_stdout
        
        # Format specific lines in the output values
        for output_value in value_list:
            lines = output_value.split('\n')
            formatted_lines = []
            for line in lines:
                if 'Population for Kenya' in line:
                    formatted_line = line.replace('Population', 'Urban')
                    formatted_lines.append(formatted_line)
                else:
                    formatted_lines.append(line)
        
        return value_list
        


    
    def generate_forecast_data(self, phi_values):
        """
        Generate forecast data for multiple time series using different damping factors.
        
        Parameters:
        - phi_values (list): List of damping factors to be used for forecasting.
        
        Returns:
        - forecast_data (DataFrame): Combined forecast data for all time series and damping factors.
        """
        forecast_combined = []
        dataframes = []

        # Iterate through each time series and corresponding damping factor
        for row, phi in zip(range(len(self.time_series_list)), phi_values):

            # Generate forecast for the current time series with the given damping factor
            result_df = self.ets_damped_trend_forecast(row=row, n_preds=28, damping_factor=phi, trend=True, plot=False, latex=False, DATAFRAME=True)
            forecast_combined.append(result_df)

            # Extract name data and create a DataFrame with forecast results
            name_data = result_df.name
            df_temp = pd.DataFrame(result_df.items(), columns=['Date', 'Value'])
            df_temp['Variable'] = name_data[0]
            df_temp['Country'] = name_data[1]
            df_temp['Region'] = name_data[2]
            dataframes.append(df_temp)

        # Combine individual forecast DataFrames into a single DataFrame
        forecast_data = pd.concat(dataframes, ignore_index=True)
        forecast_data['Date'] = pd.to_datetime(forecast_data['Date'])
        
        return forecast_data
    



    def get_unique_values(dataframe):
        """
        Extracts unique values for specified columns in a DataFrame and prints regions for each country.
        
        Parameters:
        - dataframe (DataFrame): The input DataFrame containing data.
        
        This method takes a DataFrame as input and performs the following tasks:
        
        1. Initialize dictionaries to store unique values for specified columns and country-region associations.
        2. Iterate through each row in the DataFrame.
        3. For each row, examine specified columns ('Variable', 'Country', 'Region'):
            - If the column is 'Country', add the country value to the list of unique countries if it's not already present.
            - If the column is 'Region', extract the first part of the region value (e.g., 'East Asia' from 'East Asia & Pacific') and associate it with the respective country in the country_regions dictionary.
            - For all other columns, add the value to the list of unique values if it's not already present.
        4. After processing all rows, print the regions associated with each country.
        
        This method helps in understanding the unique values present in specific columns of the DataFrame,
        and it also provides a way to gather and display regional information for each country in the dataset.
        """
        column_names = ['Variable', 'Country', 'Region']
        unique_values_dict = {col: [] for col in column_names}
        country_regions = {}

        for index, row in dataframe.iterrows():
            for column_name in column_names:
                value = row[column_name]
                if column_name == 'Country':
                    country = value
                    if country not in unique_values_dict['Country']:
                        unique_values_dict['Country'].append(country)
                elif column_name == 'Region':
                    region = value
                    if country not in country_regions:
                        country_regions[country] = []
                    region_name = region.split()[0]
                    if region_name not in country_regions[country]:
                        country_regions[country].append(region_name)
                elif value not in unique_values_dict[column_name]:
                    unique_values_dict[column_name].append(value)

        for country, regions in country_regions.items():
            print(f"Region in {country} =", regions)
    


    def calculate_custom_metric(df, country, gamma_values):
        """
        Calculate a custom metric for specified country and regions using data from a DataFrame.
        
        Parameters:
        - df (DataFrame): The input DataFrame containing data.
        - country (str): The country for which the custom metric is calculated.
        - gamma_values (list): List of gamma values corresponding to unique regions (excluding the country).
        
        Returns:
        - metric_df (DataFrame): DataFrame containing calculated custom metric values for each region and year.
        
        This function calculates a custom metric using the following steps:
        
        1. Extract unique regions associated with the specified country from the DataFrame.
        2. Filter out the country itself from the list of unique regions.
        3. Initialize an empty list to store the calculated metric values.
        4. Iterate through each unique year in the DataFrame.
        5. For each year, iterate through each unique region (excluding the country):
            - Extract population, household, and urban values for the current year and region.
            - Retrieve the corresponding gamma value for the region from the gamma_values list.
            - Calculate the custom metric value using the formula: (population / household) * urban * gamma.
            - Store the calculated metric value along with the year and region in the result list.
        6. Create a DataFrame from the result list to store the custom metric values.
        
        This function provides a way to compute a custom metric that considers population, household, urbanization,
        and gamma values for each region, providing insights into the specified country's data in the DataFrame.
        """
        # Extract unique regions associated with the specified country
        unique_regions = df[df['Country'] == country]['Region'].unique()
        unique_regions = list(filter(lambda x: x != country, unique_regions))

        result = []

        for year in df['Date'].unique():
            for region in unique_regions:
                population = df[(df['Date'] == year) & (df['Region'] == region)]['Value'].values[0]
                household = df[(df['Date'] == year) & (df['Region'] == region) & (df['Variable'] == 'Household')]['Value'].values[0]
                urban = df[(df['Date'] == year) & (df['Region'] == region) & (df['Variable'] == 'Urban')]['Value'].values[0]
                gamma = gamma_values[unique_regions.index(region)]

                metric_value = (population / household ) * urban * gamma
                result.append({
                    'Date': year,
                    'Region': region,
                    'MetricValue': metric_value
                })

        return pd.DataFrame(result)

    


    
    def regression(country, time_range=range(2005, 2023), Model_summary=True, LaTex=True, Scatterplot=True, Parameters_print=True, MaxGDP=15000, X=False):
        """
        Perform linear regression analysis on GDP per capita and access to clean fuels data.

        Parameters:
        - country (str): The country for which the regression analysis is performed.
        - time_range (range): The range of years for data analysis.
        - Model_summary (bool): Whether to display the model summary.
        - LaTex (bool): Whether to display the model summary in LaTeX tabular format.
        - Scatterplot (bool): Whether to display the scatter plot.
        - Parameters_print (bool): Whether to print the estimated slope, intercept, and equation.
        - MaxGDP (int): Maximum GDP per capita value to include in the analysis.
        - X (bool): Whether to include ISO3 country codes (X=True) or not (X=False).

        Returns:
        - estimated_slope (float): Estimated slope coefficient from the regression analysis.
        - estimated_constant (float): Estimated intercept (constant) from the regression analysis.

        This function performs linear regression analysis on GDP per capita and access to clean fuels data for the specified country. It includes the following steps:

        1. Fetch GDP per capita and access to clean fuels data for the specified country and time range.
        2. Calculate the percentage of the population without access to clean fuels.
        3. Merge the GDP per capita and clean fuels data into a single DataFrame.
        4. Remove rows with missing values and filter out data points where GDP per capita exceeds MaxGDP.
        5. Perform linear regression using the Ordinary Least Squares (OLS) method.
        6. Display the model summary, if Model_summary is True.
        7. Display the model summary in LaTeX tabular format, if LaTex is True.
        8. Display a scatter plot of GDP per capita vs. access to clean fuels, if Scatterplot is True.
        9. Print the estimated slope, intercept, and equation of the regression line, if Parameters_print is True.
        10. Return the estimated slope and constant (intercept) of the regression.

        This function provides insights into the relationship between GDP per capita and access to clean fuels, including visualization and statistical analysis.
        """

        # Fetch GDP per capita and access to clean fuels data for the specified country and time range
        if X:
            # Convert country names to ISO3 codes
            def convertSO3(countries):
                iso3_codes = []
                for country_name in countries:
                    try:
                        if country_name == 'Tanzania':
                            iso3_codes.append('TZA')
                        else:
                            country = pycountry.countries.get(name=country_name)
                            iso3_codes.append(country.alpha_3 if country else None)
                    except AttributeError:
                        iso3_codes.append(None)  # Or an appropriate value if the country code is not found
                return iso3_codes
            
            liste = convertSO3(country)

            def get_indicator_data(indicator, time_range, skip_blanks=True):
                data = wb.data.DataFrame(indicator, liste, time=time_range, skipBlanks=skip_blanks, columns='series')
                return data
        else:
            def get_indicator_data(indicator, time_range, skip_blanks=True):
                data = wb.data.DataFrame(indicator, time=time_range, skipBlanks=skip_blanks, columns='series')
                return data

        # Define indicator codes for GDP per capita and access to clean fuels
        gdp_indicator = 'NY.GDP.PCAP.CD'
        fuel_indicator = 'EG.CFT.ACCS.ZS'

        gdp_data = get_indicator_data(gdp_indicator, time_range)
        fuel_data = get_indicator_data(fuel_indicator, time_range)
        
        # Calculate the percentage of the population without access to clean fuels
        fuel_data = (100 - fuel_data)

        # Merge GDP per capita and access to clean fuels data into a single DataFrame
        merged_data = pd.concat([gdp_data, fuel_data], axis=1)
        merged_data.columns = ['GDP per capita', 'Without Access to Clean Fuels']

        # Remove rows with missing values and filter out data points where GDP per capita exceeds MaxGDP
        merged_data.dropna(inplace=True)
        merged_data = merged_data[merged_data['GDP per capita'] <= MaxGDP]

        # Prepare data for regression analysis
        y = merged_data['Without Access to Clean Fuels']
        X = merged_data['GDP per capita']

        X_with_constant = sm.add_constant(X)  

        # Perform linear regression using OLS
        model = sm.OLS(y, X_with_constant).fit()
        regression_table = model.summary().tables[1]
        regression_table = regression_table.as_latex_tabular(column_format='lcc')

        # Display model summary 
        if Model_summary:
            print(model.summary())

        # Display model summary as LaTeX tabular format
        if LaTex:
            print(regression_table)

        # Display scatter plot
        if Scatterplot:
            plt.figure(figsize=(18, 7))
            plt.rcParams["font.family"] = "Latin Modern Roman"
            plt.rcParams["font.size"] = 12
            

            # Generate a colormap based on the number of unique years
            unique_years = merged_data.index.get_level_values('time').unique()
            num_years = len(unique_years)
            color_map = plt.get_cmap('viridis', num_years)

            for i, year in enumerate(unique_years):
                year_data = merged_data.xs(year, level='time')
                plt.scatter(year_data['GDP per capita'], year_data['Without Access to Clean Fuels'], color=color_map(i), s=20)  # Adjust 's' to change point size

            plt.xlabel('GDP per capita in current US dollars')
            plt.ylabel('Without access to Clean Fuels (as % of the population)')
            plt.title('Scatter Plot of GDP per Capita vs. Without access to Clean Fuels')
            #plt.grid(True)

            annotation_text = f"Note: Excluded data points where GDP\n per capita is higher than {MaxGDP}"
            plt.annotate(annotation_text, xy=(0.90, 0.95), xycoords='axes fraction', ha='right', va='top', fontsize=11, color='red')
            plt.xlim(0, MaxGDP)
            plt.ylim(0, 100)

            r_squared = model.rsquared
            r_squared_text = f"R-squared = {r_squared:.2f}"
            plt.annotate(r_squared_text, xy=(0.90, 0.80), xycoords='axes fraction', ha='right', fontsize=11, color='black')
            # Plot the regression line
            plt.plot(merged_data['GDP per capita'], model.predict(X_with_constant), color=(238/255, 130/255, 238/255),linewidth=2.5, label='Regression Model')

            legend_entries = [str(year)[2:] for year in unique_years]
            plt.legend(legend_entries, title='Year', handlelength=0, fontsize=10)
            plt.show()

        # Print estimated parameters and equation
        if Parameters_print:
            estimated_slope = model.params['GDP per capita']
            estimated_constant = model.params['const']  # Extract the constant (intercept) parameter
            print("Estimated slope coefficient:", estimated_slope/100)
            print("")
            print("Estimated constant (intercept):", estimated_constant/100)

            X_1 = estimated_slope / 100
            X_2 = estimated_constant / 100
            print("")
            print(f'Equation: f(GDP_t)={X_2:.10f}{X_1:.10f} * GDP_t')


        # Return estimated slope and constant
        estimated_slope = model.params['GDP per capita']
        estimated_constant = model.params['const']  # Extract the constant (intercept) parameter

        estimated_slope = estimated_slope/100
        estimated_constant = 1
        
        return estimated_slope, estimated_constant
    


    
    def calculate_and_adjust_metric(df, country, gamma_values, parameter):
        """
        Calculate and adjust a custom metric based on GDP values for specified country and regions.

        Parameters:
        - df (DataFrame): The input DataFrame containing data.
        - country (str): The country for which the analysis is performed.
        - gamma_values (list): List of gamma values corresponding to unique regions (excluding the country).
        - parameter (float): The parameter used for adjusting the metric.

        Returns:
        - result_df (DataFrame): DataFrame containing adjusted metric values for each region and year.

        This function calculates a custom metric for specified regions and years using population, household, urbanization,
        and gamma values. It then applies an adjustment based on GDP values to generate potential customer estimates.

        Steps:
        1. Extract unique regions associated with the specified country, excluding "National".
        2. Iterate through each unique year and region to calculate the base metric value.
        3. Fetch relevant data for population, household, urbanization, and gamma.
        4. Calculate the base metric value using the specified formula.
        5. Extract national GDP values for Tanzania from the DataFrame.
        6. Create a result DataFrame to store the calculated metric values.
        7. Define an adjustment function that applies the parameter-based adjustment to the metric value.
        8. Apply the adjustment function to each row in the result DataFrame using GDP values.
        9. Return the adjusted metric values along with the date and region in the result DataFrame.

        This function provides insights into potential customer estimates based on a custom metric and GDP-based adjustments,
        allowing for a deeper understanding of the relationship between different factors in the data.
        """

        # Define an adjustment function that applies the parameter-based adjustment
        def apply_adjustment(row, gdp_values):
            date = row['Date']
            metric_value = row['MetricValue']
            if date in gdp_values.index:
                gdp = gdp_values.loc[date, 'GDP']
                adjustment = (1 + (parameter * gdp))  # Using plus ( + ) since parameter is negative such that (1 - | parameter | * gdp)
                adjusted_value = metric_value * adjustment
                return adjusted_value
            else:
                return metric_value
        
        # Extract unique regions associated with the specified country, excluding "National"
        unique_regions = df[df['Country'] == country]['Region'].unique()
        unique_regions = [region for region in unique_regions if region not in (country, "National")]
        
        result = []

        # Iterate through each unique year and region
        for year in df['Date'].unique():
            for region in unique_regions:
                data_filter = (df['Date'] == year) & (df['Region'] == region)
                population = df[data_filter]['Value'].values[0]
                household = df[data_filter & (df['Variable'] == 'Household')]['Value'].values[0]
                urban = df[data_filter & (df['Variable'] == 'Urban')]['Value'].values[0]
                gamma = gamma_values[unique_regions.index(region)]

                # Calculate the base metric value
                metric_value = (population / household) * urban * gamma
                result.append({
                    'Date': year,
                    'Region': region,
                    'MetricValue': metric_value
                })

        # Extract national GDP values for Tanzania
        national_data = df[(df['Region'] == 'National') & (df['Country'] == "Tanzania")].iloc[:, :-3].rename(columns={'Value': 'GDP'}).reset_index(drop=True)
        gdp_values = national_data.set_index('Date')

        # Create a result DataFrame to store the calculated metric values
        result_df = pd.DataFrame(result)

        # Apply the adjustment function to each row in the result DataFrame using GDP values
        result_df['Potential customers'] = result_df.apply(apply_adjustment, args=(gdp_values,), axis=1)
        
        # Return the adjusted metric values along with the date and region
        return result_df[['Date', 'Region', 'Potential customers']]
    


    

    def plot_potential_customers_by_region(data_frame):
        """
        Plot potential customers by region over time.

        Parameters:
        - data_frame (DataFrame): The input DataFrame containing adjusted potential customer data.

        This function generates a stacked bar plot to visualize the potential customers by region over time. It performs the following steps:

        1. Create a copy of the input DataFrame to avoid modifying the original data.
        2. Convert the 'Date' column to datetime format and extract the 'Year' component.
        3. Group the data by 'Year' and 'Region', summing the 'Potential customers' values.
        4. Pivot the grouped DataFrame to create a pivot table with 'Year' as index and 'Region' as columns.
        5. Generate a stacked bar plot using the pivot table, setting the width and figure size.
        6. Format the y-axis labels with comma-separated thousands.
        7. Set font preferences for the plot title and labels.
        8. Add labels to the x and y axes, and set the plot title and legend.
        9. Display the stacked bar plot.

        This function provides a visual representation of potential customers by region over time, aiding in the interpretation of trends and variations.
        """

        # Create a copy of the input DataFrame
        df = data_frame.copy()
        
        # Convert 'Date' column to datetime format and extract the 'Year' component
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year

        # Group data by 'Year' and 'Region', summing the 'Potential customers' values
        df = df.groupby(['Year', 'Region'])['Potential customers'].sum().reset_index()

        # Pivot the grouped DataFrame to create a pivot table
        pivot_df = df.pivot_table(index='Year', columns='Region', values='Potential customers')
        
        # Generate a stacked bar plot with specified width and figure size
        ax = pivot_df.plot(kind='bar', stacked=True, width=0.5, figsize=(18, 7))  # Store the axes object

        # Format y-axis labels with comma-separated thousands
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: format(x, ',.0f')))
        
        # Set font preferences for plot title and labels
        plt.rcParams["font.family"] = "Latin Modern Roman"
        plt.rcParams["font.size"] = 12

        # Add labels to x and y axes, set plot title and legend
        plt.xlabel('Year')
        plt.ylabel('Potential customers')
        plt.title('Potential Customers by Region Over Time')
        plt.legend(title='Region', fontsize='small')
        #ax.ticklabel_format(style='plain', axis='y')

        # Display the stacked bar plot
        plt.show()
    


        


    
class EstimationClass:
    def __init__(self, estimation_list):
        self.estimation_list = estimation_list

    def process_excel_data(file_path, sheet_name):
        Marcodata = pd.read_excel(file_path, sheet_name=sheet_name)
        Marcodata.set_index(["Variable", "Country", "Region"], inplace=True)
        Marcodata.columns = pd.to_datetime(Marcodata.columns, format='%d.%m.%Y')
        Marcodata.sort_index(axis=1, inplace=True)
        Marcodata_monthly = Marcodata.resample('MS', axis=1).mean()
        estimation_list = [pd.Series(data=Marcodata_monthly.loc[row], name=row) for row in Marcodata_monthly.index]

        data = []

        for i in range(len(estimation_list)):
            country_name = estimation_list[i].name[1]
            region_name = estimation_list[i].name[2]
            data.extend([
                {'Date': date, 'Country': country_name, 'Region': region_name, 'Accumulated customers': value}
                for date, value in zip(estimation_list[i].index, estimation_list[i].values)
            ])

        big_df = pd.DataFrame(data)
        return big_df
    
    def interpolate_monthly(data, columns=["Date", "Region", "Potential customers", "Country"]):
        df = pd.DataFrame(data, columns=columns)
        df["Date"] = pd.to_datetime(df["Date"])

        monthly_data = []

        for region in df["Region"].unique():
            region_data = df[df["Region"] == region]

            for i in range(len(region_data) - 1):
                current_row = region_data.iloc[i]
                next_row = region_data.iloc[i + 1]

                months_diff = (next_row["Date"].year - current_row["Date"].year) * 12 + next_row["Date"].month - current_row["Date"].month

                for j in range(1, months_diff + 1):
                    date = current_row["Date"] + pd.DateOffset(months=j)
                    potential_customers = current_row["Potential customers"] + (next_row["Potential customers"] - current_row["Potential customers"]) * j / months_diff
                    country = current_row["Country"] 
                    monthly_data.append([date, region, potential_customers, country])

        monthly_df = pd.DataFrame(monthly_data, columns=columns)

        return monthly_df
    

    def perform_logistic_regression_estimation(X, Y, plot=True, estimates=True):    

        def logistic_curve(x, K, C, P):
            return 1 / (1 + C * np.exp(-K * x + P))

        initial_guess = [0.003, 1.0, 0.0]

        params, covariance = curve_fit(logistic_curve, X, Y, p0=initial_guess)

        estimated_K, estimated_C, estimated_P = params
        
        result_list = [estimated_K, estimated_C, estimated_P]

        if estimates:
            print("Estimated K:", estimated_K)
            print("Estimated C:", estimated_C)
            print("Estimated P:", estimated_P)

        if plot:
            plt.figure(figsize=(18, 7))
            plt.scatter(X, Y, label='Actual Data')
            plt.plot(X, logistic_curve(X, estimated_K, estimated_C, estimated_P), label='Estimated Curve', color='red')
            plt.xlabel('Months')
            plt.ylabel('Data')
            plt.legend()
            plt.title('Logistic Regression Estimation')
            
            plt.show()

        return result_list
    
    def replace_negative_with_mean(df):
        for column in df.columns[2:]:
            non_negative_values = df[df[column] >= 0][column]
            mean_value = non_negative_values.mean()
            df[column] = np.where(df[column] < 0, mean_value, df[column])
        
        return df

    def estimate_parameters_for_regions(df, plot=False, estimates=False):
        unique_regions = df['Region'].unique()
        results_list = []

        for region in unique_regions:
            region_data = df[df['Region'] == region]
            data_years = region_data['Date'].tolist()
            data_year = np.arange(1, len(data_years) + 1)
            accumulated_customers_ratio = region_data['Ratio'].tolist()
            print("")
            print(region)
            regression = EstimationClass.perform_logistic_regression_estimation(X=data_year, Y=accumulated_customers_ratio, plot=plot, estimates=estimates)

            results_dict = {
                'Country': region_data['Country'].iloc[0],
                'Region': region_data['Region'].iloc[0],
                'Estimated_K': regression[0],
                'Estimated_C': regression[1],
                'Estimated_P': regression[2]
            }
            results_list.append(results_dict)
            

        results_df = pd.DataFrame(results_list)
        return results_df

    def merge_results_with_customers(data, customers_data):
        # Calculate the average row
        average_row = data[['Estimated_K', 'Estimated_C', 'Estimated_P']].mean()
        average_row['Country'] = 'Average'
        average_row['Region'] = 'Average'
        data = data.append(average_row, ignore_index=True)

        # Merge customers_data with data
        merged_df = customers_data.merge(data, on='Region', how='left')

        missing_columns = ['Estimated_K', 'Estimated_C', 'Estimated_P']
        for column in missing_columns:
            avg_value = data[column].loc[data['Region'] == 'Average'].values[0]
            merged_df[column] = merged_df.apply(
                lambda row: avg_value if pd.isnull(row[column]) else row[column],
                axis=1
            )

        # Drop the unwanted 'Country_y' column and rename 'Country_x' to 'Country'
        merged_df.drop(columns=['Country_y'], inplace=True)
        merged_df.rename(columns={'Country_x': 'Country'}, inplace=True)

        return merged_df

    def generate_potential_customers_excel(region_start_dates, final_merged_df):
        unique_regions = final_merged_df['Region'].unique()
        data = {}

        for region in unique_regions:
            region_data = final_merged_df[final_merged_df['Region'] == region]
            start_date = region_start_dates.get(region)

            if start_date is None:
                x_values = np.arange(len(region_data))
                potential_customers = 0.0000000000000001 * x_values
                
            else:
                region_data = region_data[region_data['Date'] >= start_date]
                x_values = np.arange(len(region_data))

                Estimated_K = region_data['Estimated_K'].iloc[0]
                Estimated_C = region_data['Estimated_C'].iloc[0]
                Estimated_P = region_data['Estimated_P'].iloc[0]

                potential_customers = (
                    region_data['Potential customers'] *
                    1 / (1 + Estimated_C * np.exp(-Estimated_K * x_values + Estimated_P))
                )

            data[region] = {'Date': region_data['Date'].tolist(), 'Potential_Customers': potential_customers.tolist()}

        # Create a new Excel workbook
        wb = Workbook()

        # Create a new worksheet
        ws = wb.active
        ws.title = 'Potential Customers Data'

        # Add headers for dates
        dates = sorted(set(date for region_data in data.values() for date in region_data['Date']))
        ws.append(['Date'] + dates)

        # Add data rows
        for region in unique_regions:
            row = [data[region]['Potential_Customers'][data[region]['Date'].index(date)] if date in data[region]['Date'] else "" for date in dates]
            ws.append([region] + row)

        # Save the Excel workbook
        wb.save('Potential_customers_data.xlsx')
        print('Data is successfully saved as: "Potential_customers_data.xlsx"')
        print("")

    def generate_potential_customers_dataframe(region_start_dates, final_merged_df, region_sigma ,Churn_Rate=False):
        unique_regions = final_merged_df['Region'].unique()
        data = []

        for region in unique_regions:
            region_data = final_merged_df[final_merged_df['Region'] == region]
            start_date = region_start_dates.get(region)

            if start_date is None:
                x_values = np.arange(len(region_data))
                potential_customers = 0 * x_values
                
            else:
                region_data = region_data[region_data['Date'] >= start_date]
                x_values = np.arange(len(region_data))

                Estimated_K = region_data['Estimated_K'].iloc[0]
                Estimated_C = region_data['Estimated_C'].iloc[0]
                Estimated_P = region_data['Estimated_P'].iloc[0]

                if Churn_Rate:
                    potential_customers_t = (
                        region_data['Potential customers'] *
                        1 / (1 + Estimated_C * np.exp(-Estimated_K * x_values + Estimated_P))
                    )
                    
                    potential_customers_t_1 = (
                        region_data['Potential customers'] *
                        1 / (1 + Estimated_C * np.exp(-Estimated_K * (x_values - 1) + Estimated_P))
                    )
                    if (potential_customers_t - potential_customers_t_1) > 0:

                        potential_customers = potential_customers_t_1 + (potential_customers_t - potential_customers_t_1) * (1-region_sigma[region])

                    else:

                        potential_customers = potential_customers_t_1 + (potential_customers_t - potential_customers_t_1) * (1)
                    
                else:
                    potential_customers = (
                        region_data['Potential customers'] *
                        1 / (1 + Estimated_C * np.exp(-Estimated_K * x_values + Estimated_P))
                    )

            region_dates = region_data['Date'].tolist()
            region_countries = region_data['Country'].tolist()
            region_potential_customers = potential_customers.tolist()

            data.extend(
                [{'Region': region, 'Country': country, 'Date': date, 'Potential_Customers': pc}
                for country, date, pc in zip(region_countries, region_dates, region_potential_customers)])

        df = pd.DataFrame(data)
        print("Data is successfully saved in the format: DATAFRAME")
        return df


    def plot_potential_customers(data, region_start_dates):
        unique_regions = data['Region'].unique()

        # Create a dropdown-widget with options for regions
        region_dropdown = widgets.Dropdown(
            options=list(unique_regions) + ['All Regions'],
            description='Region:',
            disabled=False,
        )

        # Function to update the plot based on the dropdown value
        def update_combined_plot(region):
            plt.figure(figsize=(18, 7))

            for r in unique_regions if region == 'All Regions' else [region]:
                region_data = data[data['Region'] == r]
                start_date = region_start_dates.get(r)

                if start_date is None:
                    continue

                region_data = region_data[region_data['Date'] >= start_date]
                x_values = np.arange(len(region_data))

                Estimated_K = region_data['Estimated_K'].iloc[0]
                Estimated_C = region_data['Estimated_C'].iloc[0]
                Estimated_P = region_data['Estimated_P'].iloc[0]

                potential_customers = (
                    region_data['Potential customers'] *
                    1 / (1 + Estimated_C * np.exp(-Estimated_K * x_values + Estimated_P))
                )

                plt.plot(region_data['Date'], potential_customers, marker='o', label=r)

            plt.xlabel('Date')
            plt.ylabel('Potential Customers')
            plt.title('Potential Customer Projections for Different Regions')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        # Interactively update the combined plot based on the dropdown value
        interactive_combined_plot = widgets.interactive(update_combined_plot, region=region_dropdown)
        display(interactive_combined_plot)

class ConsumptionClass:


    def process_consumption_data(file_path, sheet_name, forecast_data):
        Marcodata = pd.read_excel(file_path, sheet_name=sheet_name)
        Marcodata.set_index(["Variable", "Country", "Region"], inplace=True)
        Marcodata.columns = pd.to_datetime(Marcodata.columns, format='%d.%m.%Y')
        Marcodata.sort_index(axis=1, inplace=True)
        consumption_list = [pd.Series(data=Marcodata.loc[row], name=row) for row in Marcodata.index]

        # Omdan data til en stor DataFrame
        data = []
        for i in range(len(consumption_list)):
            country_name = consumption_list[i].name[1]
            region_name = consumption_list[i].name[2]
            data.extend([
                {'Date': date, 'Country': country_name, 'Region': region_name, 'Consumption': value}
                for date, value in zip(consumption_list[i].index, consumption_list[i].values)
            ])
        big_df = pd.DataFrame(data)

        result = []
        for year in big_df['Date'].dt.year.unique():
            year_df = big_df[big_df['Date'].dt.year == year]
            
            for region in year_df['Region'].unique():
                region_df = year_df[year_df['Region'] == region]
                
                avg_consumption = region_df['Consumption'].mean()
                p95_consumption = np.percentile(region_df['Consumption'], 95)
                p5_consumption = np.percentile(region_df['Consumption'], 5)
                
                country = region_df['Country'].iloc[0] 
                
                result.append({
                    'Year': year,
                    'Country': country,
                    'Region': region,
                    'Avg_Consumption': avg_consumption,
                    'P95_Consumption': p95_consumption,
                    'P5_Consumption': p5_consumption
                })

        summary_df = pd.DataFrame(result)

        forecast_data["Year"] = pd.to_datetime(forecast_data["Date"]).dt.year
        new_dataset = forecast_data[forecast_data["Variable"] == "Household"].copy()
        new_dataset.drop(columns=['Date', 'Variable'], inplace=True)
        unique_combinations = new_dataset.drop_duplicates(subset=['Year', 'Country', 'Region'])
        merged_dataset = pd.merge(summary_df, unique_combinations, on=['Year', 'Country', 'Region'], how='left')
        merged_dataset.dropna(subset=['Value'], inplace=True)
        merged_dataset.rename(columns={'Value': 'Household'}, inplace=True)
        
        return big_df, merged_dataset



    def create_interactive_plot(data):
        sns.set_palette("Blues")
        
        region_dropdown = widgets.Dropdown(options=data['Region'].unique(), description='Region:')
        
        def update_plot(region):
            region_df = data[data['Region'] == region]

            plt.figure(figsize=(18, 7))
            ax = sns.boxplot(data=region_df, y='Consumption', x=region_df['Date'].dt.year)

            plt.title(f'Consumption Box Plot for {region}')
            plt.xlabel('Year')
            plt.ylabel('Consumption')
            plt.xticks(rotation=45) 

            plt.tight_layout()
            plt.show()

        return widgets.interactive(update_plot, region=region_dropdown)

    def perform_linear_regression(df, LaTex=False, Plot=False, Summary=False):
        X = df[['Household']]
        y = df['Avg_Consumption']

        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()
        n, c = model.params['Household'], model.params['const']

        if LaTex:
            latex_table = model.summary().as_latex()
            print(latex_table)

        if Summary:
            print(model.summary())
        
        if Plot:
            plt.figure(figsize=(18, 7))
            y_pred = model.predict(X)
            plt.scatter(X['Household'], y, color='darkblue', label='Actual Data')
            plt.plot(X['Household'], y_pred, color=(238/255, 130/255, 238/255), label='Regression Line')
            plt.xlabel('Average household size')
            plt.ylabel('LPG consumption in grams')
            plt.title('Linear Regression: Average LPG consumption vs Average household size')
            plt.legend()
            plt.show()
        
        print(f'The constant: {c}')
        print("")
        print(f'The slope coefficient: {n}')
        
        return [n, c]
    

    def calculate_regression_coefficients(Regress_data_df, LaTex=False, Plot=False, Summary=False):
        def convert_country_names_to_ISO3(countries):
            iso3_codes = []
            for country_name in countries:
                try:
                    if country_name == 'Tanzania':
                        iso3_codes.append('TZA')
                    else:
                        country = pycountry.countries.get(name=country_name)
                        iso3_codes.append(country.alpha_3 if country else None)
                except AttributeError:
                    iso3_codes.append(None)  # Or an appropriate value if the country code is not found
            return iso3_codes

        unique_regions = Regress_data_df['Country'].unique()
        years = Regress_data_df['Year'].unique()

        countries = convert_country_names_to_ISO3(unique_regions)

        gdp_data_reset = wb.data.DataFrame('NY.GDP.PCAP.CD', countries, time=range(years[0], years[-1]), skipBlanks=True, columns='series')
        gdp_data_reset.index = gdp_data_reset.index.set_levels(gdp_data_reset.index.levels[1].str.replace('YR', ''), level=1)
        gdp_data_reset = gdp_data_reset.rename_axis(index={'time': 'Year'}).reset_index()
        gdp_data_reset['Country'] = gdp_data_reset['economy'].map(dict(zip(countries, unique_regions)))
        gdp_data_reset = gdp_data_reset.rename(columns={'NY.GDP.PCAP.CD': 'GDP'})[['Year', 'Country', 'GDP']]

        agg_df = Regress_data_df.groupby(["Year", "Country"]).agg({
            "Avg_Consumption": "mean",
            "P95_Consumption": "mean",
            "P5_Consumption": "mean",
            "Household": "mean"  
        }).reset_index()

        gdp_data_reset['Year'] = gdp_data_reset['Year'].astype(int)
        merged_data = pd.merge(agg_df, gdp_data_reset, on=['Country', 'Year'], how='left')
        merged_data.drop('Household', axis=1, inplace=True)
        merged_data.dropna(inplace=True)

        X = sm.add_constant(merged_data["GDP"])  
        Y = merged_data["Avg_Consumption"]

        model = sm.OLS(Y, X).fit()

        if LaTex:
            latex_table = model.summary().as_latex()
            print(latex_table)


        if Summary:
            print(model.summary())
        
        
        if Plot:
            plt.figure(figsize=(18, 7))
            plt.scatter(merged_data["GDP"], merged_data["Avg_Consumption"], label="Data", color="darkblue")
            plt.xlabel("GDP per capita (current US$)")
            plt.ylabel("Average LPG consumption in grams")
            plt.title("Scatterplot of Linear Regression")
            plt.legend()

            # Regressionslinje
            plt.plot(merged_data["GDP"], model.predict(), color=(238/255, 130/255, 238/255), label="Regression Line")
            plt.legend()

            plt.show()
        
        print(f'The constant: {model.params["const"]}')
        print("")
        print(f'The slope coefficient: {model.params["GDP"]}')

        return [model.params['const'], model.params['GDP']]
    

    def interpolate_data(forecast_data):
        filtered_df = forecast_data[forecast_data["Variable"] == "Household"].copy()
        filtered_df.rename(columns={"Value": "Household"}, inplace=True)
        filtered_df.drop('Variable', axis=1, inplace=True)

        Data_df = forecast_data[forecast_data["Variable"] == "GDP"].copy()
        Data_df.rename(columns={"Value": "GDP"}, inplace=True)
        Data_df.drop('Variable', axis=1, inplace=True)
        Data_df.drop('Year', axis=1, inplace=True)

        filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])
        Data_df['Date'] = pd.to_datetime(Data_df['Date'])

        merged_df = filtered_df.merge(Data_df[['Country', 'GDP', 'Date']], on=['Country', 'Date'])

        merged_df['Date'] = pd.to_datetime(merged_df['Date'], format='%Y-%m-%d')

        interpolated_rows = []

        for _, row in merged_df.iterrows():
            
            dates = pd.date_range(start=row['Date'], periods=12, freq='MS')

            
            for date in dates:
                interpolated_rows.append({
                    'Date': date,
                    'Household': row['Household'],
                    'Country': row['Country'],
                    'Region': row['Region'],
                    'GDP': row['GDP']
                })

        
        interpolated_df = pd.DataFrame(interpolated_rows)
        
        return interpolated_df


    def calculate_consumption_for_dataframe(data, coefficients, params, alpha=400):
        def f(Household, params):
            return params[1] + params[0] * Household

        def g(GDP, coefficients):
            return coefficients[1] * GDP

        def calculate_consumption(row, params, coefficients, alpha):
            f_result = f(row['Household'], params)
            g_result = g(row['GDP'], coefficients)
            consumption = f_result + g_result if f_result + g_result <= alpha else alpha
            return consumption
        
        result_df = data.copy()

        result_df['Consumption'] = result_df.apply(lambda row: calculate_consumption(row, params, coefficients, alpha), axis=1)
        
        Consumption_df = result_df.drop(columns=['Household', 'GDP'])
        
        return Consumption_df
    
    def plot_lpg_consumption(data, columns='Region'):
        data['Date'] = pd.to_datetime(data['Date'])

        pivot_df = data.pivot(index='Date', columns=columns, values='Total_LPG_monthly')
        ax = pivot_df.plot(kind='bar', stacked=True, width=1.2, figsize=(18, 7))
        years = pivot_df.index.year
        plt.xticks(range(len(years)), years, rotation=45)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=int(round((len(data['Date'].unique()) / 12), 0))))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: format(x, ',.0f')))

        plt.rcParams["font.family"] = "Latin Modern Roman"
        plt.rcParams["font.size"] = 12
        plt.ylabel('Average monthly LPG consumption in kilograms')
        plt.title('Predicted: LPG consumption for Circle Gas')
        plt.xticks(rotation=45)
        plt.legend(title=columns, fontsize='small')

        plt.show()