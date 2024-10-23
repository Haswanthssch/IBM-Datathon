# IBM-Datathon
# Food Wastage Prediction with PMML

This project predicts **food wastage figures** for different cities using **Random Forest Regressor** models and exports them in **PMML (Predictive Model Markup Language)** format for easy integration.

## 📂 Project Structure

```plaintext
├── pmml_code_format.py           # Main script to train models and export to PMML
├── fs1.xlsx                      # Dataset containing food wastage data
├── README.md                     # Documentation (this file)
├── {city_name}_food_wastage_model.pmml  # PMML files generated per city

# City-wise Food Availability Prediction using Random Forest

This project uses **Random Forest Regression** to predict the food availability across different hotels in various cities. The predictions are based on historical food wastage estimates and are grouped by cities. The code handles missing data, creates lagged features, and generates forecasts with slight variations for the next ten days. It also assigns hotels to accommodate user-provided food requirements.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Output](#output)
6. [Assumptions](#assumptions)
7. [Error Handling](#error-handling)
8. [License](#license)

---

## Project Overview
The code:
- Loads and processes data from an **Excel file** (`fs1.xlsx`).
- Uses **RandomForestRegressor** to predict food availability.
- Generates a **10-day forecast** for hotels in each city with some random variation.
- Supports **user input** to calculate whether there is enough food for a given number of people in a particular city.
- If sufficient food is available, assigns hotels to meet the demand.

---

## Dataset Description
- **Input**: `fs1.xlsx` (an Excel file containing food service data per city and hotel).
- **Required Columns**:
  - `Food service estimate (kg/capita/year)`
  - `combined figures (kg/capita/year)`
  - Optionally, `Hotel Name` for hotel-specific predictions.
- **Generated Column**: 
  - `Date` column for daily entries starting from `2023-01-01`.

---

## Installation

1. Clone this repository:
    ```bash
    git clone <your-github-repo-url>
    cd <your-repo-folder>
    ```

2. Install dependencies:
    ```bash
    pip install pandas numpy scikit-learn openpyxl
    ```

3. Place your Excel data file (`fs1.xlsx`) in the root directory.

---

## Usage

1. **Run the script**:
    ```bash
    python your_script_name.py
    ```

2. **User Inputs**:
    - Number of people to be accommodated.
    - Name of the city for which predictions are required.

---

## Output
- **City-wise predictions**: 
  - Lists the top 3 hotels with the highest predicted food availability.
  - Provides information on whether the selected hotels can meet the required food demand.
  
Example Output:
