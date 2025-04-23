import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Adding TransOrg logo to the sidebar
st.set_page_config(page_title="Trade Promotion Optimization App")
st.sidebar.image("https://transorg.com/wp-content/uploads/2022/04/transorg-logo.png")

# Adding radio buttons for user selection
option = st.sidebar.radio(
    "Choose an Option:",
    ("Category1", "Category2", "Category3", "Category4", "Category5")
)

# Sidebar inputs for user-defined parameters
max_investment = st.sidebar.number_input("Maximum Investment", min_value=0, value=100000)
lower_discount = st.sidebar.number_input("Lower Discount Range (%)", min_value=0, value=0)
upper_discount = st.sidebar.number_input("Upper Discount Range (%)", min_value=0, value=10)

data = {
    'Retailer': ['Toss']*14,
    'PL4 Sub-Category': ['Liquid Detergent 50ml', 'Liquid Detergent 250ml', 'Front load Washing 250ml', 
                         'Powder Detergent 1', 'Powder Cloth Disinfectant 500ml', 'Powder Cloth Disinfectant 600ml',
                         'Washing Capsule 1', 'Washing Capsule 2', 'Washing Capsule 3', 
                         'Washing Capsule 4', 'Washing Capsule 5', 'Cloth Freshener 1',
                         'Cloth Freshener 2', 'Cloth Freshener 3'],
    'Forecast Avg. Base Volume': [600, 471, 440, 620, 203, 842, 979, 425, 17, 134, 740, 24, 306, 313],
    'Units': [600, 471, 440, 620, 203, 842, 979, 425, 17, 134, 740, 24, 306, 313],
    'Per Unit COGS': [50, 60, 170, 180, 200, 200, 170, 90, 70, 150, 225, 150, 280, 290],
    'List/ Base Price': [120, 220, 250, 320, 330, 340, 200, 210, 100, 300, 450, 250, 450, 500],
    'Per Unit Selling Price': [120, 220, 250, 320, 330, 340, 200, 210, 100, 300, 450, 250, 450, 500],
    'Discount': [0.0]*14,
    'Revenue': [72000, 103620, 110000, 198400, 66990, 286280, 195800, 89250, 1700, 40200, 333000, 6000, 137700, 156500],
    'Per Unit Margin': [70, 160, 80, 140, 130, 140, 30, 120, 30, 150, 225, 100, 170, 210],
    'Margin': [42000, 75360, 35200, 86800, 26390, 117880, 29370, 51000, 510, 20100, 166500, 2400, 52020, 65730],
    'Per Unit Rebate': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Dollor Investment': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Lower Discount': [lower_discount]*14,
    'Upper Discount': [upper_discount]*14,
    'Discount Uplift': [2.174962489, 1.174962489, 3.00034343, 1.174962489, 1.099384733, 2.174962489, 2.174962489, 2.974962489, 2.56904535, 1.625284225, 1.889298444, 3.038315124, 2.700573718, 5.485022623],
    'Tactic Uplift': [1]*14
}

# Create DataFrames
Category1 = pd.DataFrame(data)
Category2 = pd.DataFrame(data)
Category3 = pd.DataFrame(data)
Category4 = pd.DataFrame(data)
Category5 = pd.DataFrame(data)

total = {
    'Data-Totals': ['Units', 'Revenue', 'Margins', 'Investment'],
    'Planned': [6114, 1797440, 771260, 0],
    'Optimized': [6114, 1797440, 771260, 0],
    '% Change': [0.0, 0.0, 0.0, 0.0]
}

df = pd.DataFrame(data)
tf = pd.DataFrame(total)

st.title("KPI Optimization", anchor='Trade Optimization')

category_dict = {
    "Category1": Category1,
    "Category2": Category2,
    "Category3": Category3,
    "Category4": Category4,
    "Category5": Category5
}

# Fetch and display the data based on the selected option
df = category_dict.get(option, pd.DataFrame())

if st.button("Get Data"):
    st.write("Planned Data:")
    edited_df = st.data_editor(df, key='planned_data_editor')
    st.session_state['edited_df'] = edited_df

# Optimize function
def optimize_discounts(df, max_investment, lower_discount, upper_discount):
    def objective(discounts):
        df['Optimized Discount'] = discounts
        optimized_units = df['Forecast Avg. Base Volume'] * np.exp(df['Discount Uplift'] * df['Optimized Discount'] / 100) * df['Tactic Uplift']
        revenue = optimized_units * (df['List/ Base Price'] - discounts)
        return -revenue.sum()

    x0 = np.full(len(df), (lower_discount + upper_discount) / 2)
    bounds = [(lower_discount, upper_discount)] * len(df)

    def investment_constraint(discounts):
        df['Optimized Discount'] = discounts
        optimized_units = df['Forecast Avg. Base Volume'] * np.exp(df['Discount Uplift'] * df['Optimized Discount'] / 100) * df['Tactic Uplift']
        investment = (df['Optimized Discount'] / 100) * df['List/ Base Price'] * optimized_units
        return max_investment - investment.sum()

    constraints = [{'type': 'ineq', 'fun': investment_constraint}]

    result = minimize(objective, x0, bounds=bounds, constraints=constraints, method='trust-constr', options={'disp': True})

    if result.success:
        df['Optimized Discount'] = np.round(result.x, 2)
        df['Optimized Units'] = np.int64(df['Forecast Avg. Base Volume'] * np.exp(df['Discount Uplift'] * df['Optimized Discount'] / 100) * df['Tactic Uplift'])
        df['Optimized Per Unit Selling Price'] = df['List/ Base Price']-(df['List/ Base Price']*(df['Optimized Discount']/100))
        df['Optimized Revenue'] = np.int64(df['Optimized Units'] * df['Optimized Per Unit Selling Price'])
        df['Optimized Per Unit Margin'] = df['Optimized Per Unit Selling Price']-df['Per Unit COGS']
        df['Optimized Margin'] = np.int64(df['Optimized Units'] * df['Optimized Per Unit Margin'])
        df['Optimized Per Unit Rebate'] = df['List/ Base Price'] - df['Optimized Per Unit Selling Price']
        df['Optimized investment'] = df['Optimized Per Unit Rebate'] * df['Optimized Units']
    else:
        st.error("Optimization failed. Please check constraints and data.")
        df = pd.DataFrame()  # Return an empty DataFrame on failure
    
    return df

if st.button("Optimize"):
    optimized_df = optimize_discounts(df, max_investment, lower_discount, upper_discount)

    if not optimized_df.empty:
        st.session_state['optimized_df'] = optimized_df

        total_units = optimized_df['Optimized Units'].sum()
        total_revenue = optimized_df['Optimized Revenue'].sum()
        total_margin = optimized_df['Optimized Margin'].sum()
        total_investment = optimized_df['Optimized investment'].sum()

        tf.loc[tf['Data-Totals'] == 'Units', 'Optimized'] = total_units
        tf.loc[tf['Data-Totals'] == 'Revenue', 'Optimized'] = total_revenue
        tf.loc[tf['Data-Totals'] == 'Margins', 'Optimized'] = total_margin
        tf.loc[tf['Data-Totals'] == 'Optimized Investment', 'Optimized'] = total_investment

        st.write("Total Metrics:")
        st.dataframe(tf)

        st.write("Optimized Data:")
        st.data_editor(optimized_df)
        st.session_state['editable_optimized_df'] = optimized_df

        # Plot comparison chart
        st.write("Comparison of Planned and Optimized Values:")
        comparison_data = tf.set_index('Data-Totals')[['Planned', 'Optimized']]
        st.bar_chart(comparison_data)
    else:
        st.write("No optimized data available.")

if st.button("Process"):
    if 'editable_optimized_df' in st.session_state:
        edf = st.session_state['editable_optimized_df']

        edf['Units'] = edf['Forecast Avg. Base Volume'] * np.exp(edf['Discount Uplift'] * edf['Discount'] / 100) * edf['Tactic Uplift']
        edf['Revenue'] = edf['Units'] * edf['Per Unit Selling Price']
        edf['Margin'] = edf['Revenue'] - (edf['Units'] * edf['Per Unit COGS'])
        edf['Dollor Investment'] = edf['Units'] * (edf['Discount']/100) * edf['List/ Base Price']

        st.write("Updated Planned Data:")
        st.dataframe(edf)

        tf.at[0, 'Optimized'] = edf['Optimized Units'].sum()
        tf.at[1, 'Optimized'] = edf['Optimized Revenue'].sum()
        tf.at[2, 'Optimized'] = edf['Optimized Margin'].sum()
        tf.at[3, 'Optimized'] = edf['Optimized investment'].sum()

        tf.at[0, 'Planned'] = df['Units'].sum()
        tf.at[1, 'Planned'] = df['Revenue'].sum()
        tf.at[2, 'Planned'] = df['Margin'].sum()
        tf.at[3, 'Planned'] = df['Dollor Investment'].sum()

        tf['% Change'] = ((tf['Optimized'] - tf['Planned']) / tf['Planned']) * 100

        st.write("Updated Total Comparison Data")
        st.dataframe(tf)

        # Plot updated comparison chart
        st.write("Updated Comparison of Planned and Optimized Values:")
        comparison_data_updated = tf.set_index('Data-Totals')[['Planned', 'Optimized']]
        st.bar_chart(comparison_data_updated)

    else:
        st.write("No editable optimized data available.")
