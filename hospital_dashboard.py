import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar 
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="hospital", page_icon=":hospital:", layout="wide")

# Display hospital title and icon
st.title(" :hospital: ABC Medical Center")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)

# Load your dataset
@st.cache_data
def load_data():
    import pandas as pd
    url = 'https://raw.githubusercontent.com/velascogringo/Hospital_Operations_Interactive_Dashboard_Deployed_via_Streamlit-App/main/hosp_data.csv'
    return pd.read_csv(url)

df = load_data()

# Convert 'ADMISSION DATE' and 'DISCHARGE DATE' columns to datetime format
df['ADMISSION DATE'] = pd.to_datetime(df['ADMISSION DATE'])
df['DISCHARGE DATE'] = pd.to_datetime(df['DISCHARGE DATE']) 


###############################################################################################
# Create sidebar date filters
st.sidebar.header('Date Filter')

# Set default date range from 01/01/2017 to 12/31/2023
default_start_date = pd.to_datetime('2017-01-01')
default_end_date = pd.to_datetime('2020-12-31')

start_date = st.sidebar.date_input("Start Date", value=default_start_date)
end_date = st.sidebar.date_input("End Date", value=default_end_date)

# Convert date inputs to Timestamp objects
start_date = pd.Timestamp(start_date)
end_date = pd.Timestamp(end_date)

# Check if the start date is before or equal to the end date
if start_date > end_date:
    st.error("Date filters invalid, please select a valid date.")
else:
    # Check if the selected start and end dates are found in the dataset
    if (start_date not in df['ADMISSION DATE'].values) or (end_date not in df['ADMISSION DATE'].values):
        st.error("Date filters invalid, please select a valid date.")
    else:
        # Sidebar buttons
        st.sidebar.header('Dashboard Sections')
        try:
            selected_section = st.sidebar.radio(' ', ['Patients Trend', 'Operating Performance', 'Patient Satisfaction', 'Financial Performance'])
        except NameError:
            st.error("Date filters invalid, please select a valid date.")
            selected_section = None


########################################  PATIENT TREND SECTION  ######################################## 
def display_patients_trend():
    st.header('Patients Trend Section')  
    # Function to display Patients Trend section

    st.markdown("---")
    # Filter data based on date range
    filtered_data = df[(df['ADMISSION DATE'] >= start_date) & (df['ADMISSION DATE'] <= end_date)]

################# KPIs #################

   # Calculate KPIs
    total_patients = filtered_data.shape[0]
    total_patients_formatted = "{:,}".format(total_patients)
    total_days = (end_date - start_date).days + 1
    average_patients_per_day = total_patients / total_days
    average_age = filtered_data['AGE'].mean()

    # Display KPIs in 3 columns
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patients", total_patients_formatted)
    col2.metric("Average Patients per Day", round(average_patients_per_day, 1))
    col3.metric("Average Age of Patients", round(average_age, 1))
    st.markdown("---")

################# PATIENT PER MONTH / PATIENT PER DEPARTMENT #################

 # Create layout columns
    col1, col2 = st.columns(2)

    with col1:
        # Convert month numbers to month names
        month_names = [calendar.month_abbr[i] for i in range(1, 13)]
        # Calculate number of patients per month
        filtered_data.loc[:, 'Year'] = filtered_data['ADMISSION DATE'].dt.year
        filtered_data.loc[:, 'Month'] = filtered_data['ADMISSION DATE'].dt.month
        patients_per_month = filtered_data.groupby(['Year', 'Month']).size().unstack()
        # Create traces for each year
        traces = []
        for year in patients_per_month.index:
            trace = go.Scatter(
                x=month_names,
                y=patients_per_month.loc[year],
                mode='lines+markers',
                name=str(year)
            )
            traces.append(trace)

        # Create layout
        layout = go.Layout(
            title='Patients Per Month',
            xaxis=dict(title='Month'),
            yaxis=dict(title='Patients'),
            legend=dict(title='Year')
        )

        # Create figure
        fig = go.Figure(data=traces, layout=layout)

        # Show the plot
        st.plotly_chart(fig)

        # Create an expander for displaying the data
        with st.expander("View Patients per Month"):
            # Calculate number of patients per month
            filtered_data.loc[:, 'Year'] = filtered_data['ADMISSION DATE'].dt.year
            filtered_data.loc[:, 'Month'] = filtered_data['ADMISSION DATE'].dt.month
            patients_per_month = filtered_data.groupby(['Year', 'Month']).size().reset_index(name='Number of Patients')
        
            # Display the data
            st.write(patients_per_month)
        
            # Add a download button
            st.download_button(
                label="Download Data as CSV",
                data=patients_per_month.to_csv(index=False),
                file_name='patients_data.csv',
                mime='text/csv'
            )

    # Calculate total number of patients per department/specialty
    patients_per_department = filtered_data.groupby('DEPARTMENT/SPECIALTY').size().reset_index(name='Total Patients')

    # Sort the data by 'Total Patients' in descending order
    patients_per_department_sorted = patients_per_department.sort_values(by='Total Patients', ascending=True)

    with col2:
        # Get unique department/specialty values
        department_specialties = patients_per_department_sorted['DEPARTMENT/SPECIALTY'].tolist()

        # Create a bar chart using Plotly Graph Objects
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=patients_per_department_sorted['Total Patients'],
            y=patients_per_department_sorted['DEPARTMENT/SPECIALTY'],
            orientation='h',
            marker=dict(color='skyblue')
        ))

        fig.update_layout(
            title='Patients per Department/Specialty',
            xaxis=dict(title='Total Patients'),
            yaxis=dict(title='Department/Specialty', tickvals=department_specialties)
        )

        # Show the plot
        st.plotly_chart(fig)

        # Add an expander for displaying the data
        with st.expander("View Patients per Department/Specialty"):
            # Sort the data by 'Total Patients' in descending order
            patients_per_department = patients_per_department.sort_values(by='Total Patients', ascending=False)

            # Reset the index for displaying the data
            patients_per_department_display = patients_per_department.reset_index(drop=True)

            # Display the data with reset index
            st.write(patients_per_department_display)

            # Add a download button
            st.download_button(
                label="Download Data as CSV",
                data=patients_per_department.to_csv(index=False),
                file_name='patients_per_department_display.csv',
                mime='text/csv'
            )
   
################# INPATIENT VS OUTPATIENT / PAYER MIX #################

    # Create layout columns
    col1, col2 = st.columns(2)

    with col1:
        # Extract year and month from 'ADMISSION DATE'
        filtered_data.loc[:, 'Year'] = filtered_data['ADMISSION DATE'].dt.year
        filtered_data.loc[:, 'Month'] = filtered_data['ADMISSION DATE'].dt.month

        # Group by 'Year', 'Month', and 'ADMISSION TYPE' and count the number of occurrences
        patients_per_month = filtered_data.groupby(['Year', 'Month', 'ADMISSION TYPE']).size().unstack(fill_value=0)

        # Combine 'Year' and 'Month' into a single column for x-axis
        patients_per_month['Date'] = patients_per_month.apply(lambda x: pd.Timestamp(year=x.name[0], month=x.name[1], day=1), axis=1)

        # Create traces for inpatient and outpatient trends
        trace_inpatient = go.Scatter(
            x=patients_per_month['Date'],
            y=patients_per_month['INPATIENT'],
            mode='lines+markers',
            name='Inpatient',
            marker=dict(size=8),
            line=dict(color='blue')
        )

        trace_outpatient = go.Scatter(
            x=patients_per_month['Date'],
            y=patients_per_month['OUTPATIENT'],
            mode='lines+markers',
            name='Outpatient',
            marker=dict(size=8),
            line=dict(color='green')
        )

        
        # Create layout
        layout = go.Layout(
            title='Inpatient vs Outpatient Trend',
            xaxis=dict(
                title='Month/Year',
                tickmode='array',
                tickvals=patients_per_month['Date'],
                ticktext=patients_per_month['Date'].dt.strftime('%b %Y')  # Format tick labels as Month Year
            ),
            yaxis=dict(title='Number of Patients'),
        )

        # Create figure
        fig = go.Figure(data=[trace_inpatient, trace_outpatient], layout=layout)

        # Show the plot
        st.plotly_chart(fig)

        # Add an expander for displaying data and download button
        with st.expander("View Inpatient & Outpatient Data"):
            st.write(patients_per_month)

            # Download button for the data
            csv = patients_per_month.to_csv().encode('utf-8')
            st.download_button(label="Download Data", data=csv, file_name='patients_per_month.csv', mime='text/csv')

  
    # Calculate payer mix
    with col2:   
        payer_mix = filtered_data['PAYER'].value_counts()

        # Create a pie chart to visualize payer mix
        fig_payer_mix = px.pie(payer_mix, values=payer_mix.values, names=payer_mix.index, 
                            title='Payer Mix', hole=0.2)
        # Add labels to the pie chart
        fig_payer_mix.update_traces(textposition='inside', textinfo='percent')

        # Add the pie chart to the expander
        st.plotly_chart(fig_payer_mix)

        # Create an expander for displaying payer mix data and chart
        with st.expander("View Payer Mix Data"):
            # Create a pie chart to visualize payer mix
            fig_payer_mix = px.pie(payer_mix, values=payer_mix.values, names=payer_mix.index, 
                                    title='Payer Mix Distribution', hole=0.3)

            # Display the payer mix data
            st.write(payer_mix)

            # Add a download button for the data
            st.download_button(
                label="Download Payer Mix Data as CSV",
                data=payer_mix.to_csv(),
                file_name='payer_mix.csv',
                mime='text/csv'
            )

################# PATIENT VOLUME HEATMAP #################

    # Filter data based on date range
    filtered_data = df[(df['ADMISSION DATE'] >= start_date) & (df['ADMISSION DATE'] <= end_date)]

    # Extract month and year from 'ADMISSION DATE'
    df['Month'] = df['ADMISSION DATE'].dt.month
    df['Year'] = df['ADMISSION DATE'].dt.year

    # Extract year and month from 'ADMISSION DATE'
    filtered_data.loc[:, 'Year'] = filtered_data['ADMISSION DATE'].dt.year
    filtered_data.loc[:, 'Month'] = filtered_data['ADMISSION DATE'].dt.month

    # Filter data based on date range selected by the user

    filtered_data = filtered_data[(filtered_data['ADMISSION DATE'] >= start_date) & (filtered_data['ADMISSION DATE'] <= end_date)]

    # Recalculate patient volume per department and month
    patient_volume = filtered_data.groupby(['DEPARTMENT/SPECIALTY', 'Month']).size().unstack(fill_value=0)

    # Recalculate total patients per month (sum across departments)
    total_patients_per_month = patient_volume.sum(axis=0)

    # Create heatmap using Plotly graph objects
    fig = go.Figure(data=go.Heatmap(
            z=patient_volume.values,
            x=[calendar.month_abbr[i] for i in range(1, 13)],  # Abbreviated month names
            y=patient_volume.index,
            colorscale='Viridis',
            hoverongaps=False))

    # Customize layout
    fig.update_layout(
        title='Patient Volume Heatmap',
        yaxis=dict(tickmode='array', tickvals=list(range(len(patient_volume.index))), ticktext=patient_volume.index.tolist()),
        width=1000,  # Adjust the width
        height=700, 
    )

    # Add annotations for each box
    for y_index, department in enumerate(patient_volume.index):
        for x_index, patient_count in enumerate(patient_volume.iloc[y_index]):
            # Access patient count for current department and month
            text = str(patient_count)  # Convert to string for display
            fig.add_annotation(
                x=calendar.month_abbr[x_index + 1],  # Adjust index to start from 1
                y=department,
                text=text,
                showarrow=False,
                font=dict(color='black'),
                xref='x',
                yref='y'
            )

    # Display the heatmap
    st.plotly_chart(fig, use_container_width=True) 
    # Replace numeric month values with month names
    month_names = [calendar.month_abbr[i] for i in range(1, 13)]
    patient_volume.columns = [month_names[month - 1] for month in patient_volume.columns]

    # Create an expander for displaying data and download button
    with st.expander("View Pt Volume Data"):
        # Display the patient volume data with month names
        st.write(patient_volume)

        # Download button for the patient volume data with month names
        csv = patient_volume.to_csv().encode('utf-8')
        st.download_button(label="Download Data", data=csv, file_name='patient_volume_with_month_names.csv', mime='text/csv')



########################################  OPERATING PERFORMANCE SECTION  ########################################

# Function to display Operating Performance section
def display_operating_performance():
    st.header('Operating Performance Section')

    st.markdown("---")
    # Filter data based on date range
    filtered_data = df[(df['ADMISSION DATE'] >= start_date) & (df['ADMISSION DATE'] <= end_date)]

################# KPIs #################

# Function to map string values to numeric values
    def map_readmission(value):
        if value.isdigit():
            return int(value)
        elif value.strip().lower() == 'yes':
            return 1
        elif value.strip().lower() == 'no':
            return 0
        else:
            return np.nan  # Handle other cases gracefully
        
    def convert_hospital_charges(value):
        try:
            return float(value)
        except ValueError:
            return np.nan  # Handle non-numeric values gracefully

    # Convert 'HOSPITAL CHARGES' column to numeric
    filtered_data['HOSPITAL CHARGES'] = filtered_data['HOSPITAL CHARGES'].map(convert_hospital_charges)

    # Convert 'READMISSION' column to numeric
    filtered_data['READMISSION'] = filtered_data['READMISSION'].map(map_readmission)
    filtered_data['HOSPITAL CHARGES'] = filtered_data['HOSPITAL CHARGES'].map(convert_hospital_charges).astype('float')

     # Calculate KPIs
    average_length_of_stay = filtered_data['LENGTH OF STAY'].mean()
    readmission_rate = filtered_data['READMISSION'].mean()
    average_treatment_cost = filtered_data['HOSPITAL CHARGES'].mean(skipna=True)

    # Filter data for inpatient admissions within the date range
    filtered_data = df[(df['ADMISSION DATE'] >= start_date) & (df['ADMISSION DATE'] <= end_date)]
    inpatient_data = filtered_data[filtered_data['ADMISSION TYPE'] == 'INPATIENT']

    # Calculate total inpatient bed days
    total_inpatient_bed_days = inpatient_data['LENGTH OF STAY'].sum()

    # Define hospital bed capacity
    bed_capacity = 150  # Set the bed capacity as needed

    # Calculate total days in the selected period
    num_days_in_period = (end_date - start_date).days + 1

    # Calculate total bed days available
    total_bed_days_available = bed_capacity * num_days_in_period

    # Calculate bed occupancy rate
    bed_occupancy_rate = (total_inpatient_bed_days / total_bed_days_available) * 100

    # Create a gauge chart to represent bed occupancy rate
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bed_occupancy_rate,
        title={'text': "Bed Occupancy Rate (%)",
        'font': {'size': 15}  # Set font size to 14 (adjust as needed)
        },
        gauge={
            'shape': 'angular',
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 33], 'color': "lightgray"},
                {'range': [33, 66], 'color': "gray"},
                {'range': [66, 100], 'color': "red"},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 1,
                'value': bed_occupancy_rate
            },
            'bar': {'color': "blue"},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "gray"
        }
    ))
    
    
   # Set the layout properties for the gauge chart
    fig.update_layout(
        width=200,  # Set the width of the gauge chart
        height=200,  # Set the height of the gauge chart
        margin={'t': 5, 'b': 40, 'l': 20, 'r': 30}  # Set the margins
    )

    # Display KPIs as 4 columns
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Average Length of Stay", f"{round(average_length_of_stay, 1)} days")
    col2.metric("Readmission Rate", f"{round(readmission_rate * 100, 2)}%")
    col3.metric("Average Treatment Cost", f"${average_treatment_cost:,.2f}")
    col4.plotly_chart(fig)  # Display the gauge chart in column 4
 
################# INPATIENT ADMISSION BY DEPARTMENT #################
    
    # Filtered data for inpatient admissions
    filtered_data = df[(df['ADMISSION DATE'] >= start_date) & (df['ADMISSION DATE'] <= end_date)]

    inpatient_data = filtered_data[filtered_data['ADMISSION TYPE'] == 'INPATIENT']

    # Group data by 'DEPARTMENT/SPECIALTY'
    grouped_data = inpatient_data.groupby('DEPARTMENT/SPECIALTY').agg(
    total_admissions=('PATIENT ID', 'count'),
    average_length_of_stay=('LENGTH OF STAY', 'mean')
    ).reset_index()

    # Create a bubble chart 
    fig = px.scatter(
        grouped_data,
        x='DEPARTMENT/SPECIALTY',
        y='total_admissions',
        size='total_admissions',
        color='average_length_of_stay',
        hover_name='DEPARTMENT/SPECIALTY',
        labels={
            'DEPARTMENT/SPECIALTY': 'Department/Specialty',
            'total_admissions': 'Total Admissions',
            'average_length_of_stay': 'Average Length of Stay (days)'
        },
        title='Inpatient Admissions by Department',
        color_continuous_scale='Viridis'
    )

    # Display the chart 
    st.plotly_chart(fig, use_container_width=True)

    # Create an expander for displaying the data
    with st.expander("View Inpatient Admissions by Department Data"):
        # Display the grouped data
        st.write(grouped_data)
        
        # Add a download button for the data
        st.download_button(
            label="Download Data as CSV",
            data=grouped_data.to_csv(index=False),
            file_name='inpatient_admissions_by_department.csv',
            mime='text/csv'
        )


    ################# AVE LOS / INPATIENT STAYS BY PAYER #################

    # Create layout columns
    col1, col2 = st.columns(2)

    with col1:
        # Convert month numbers to month names
        month_names = [calendar.month_abbr[i] for i in range(1, 13)]

        # Calculate average length of stay per year and month
        filtered_data['Year'] = filtered_data['ADMISSION DATE'].dt.year
        filtered_data['Month'] = filtered_data['ADMISSION DATE'].dt.month
        avg_length_of_stay_per_year_month = filtered_data.groupby(['Year', 'Month'])['LENGTH OF STAY'].mean()

    # Create traces for each year with different colors
        color_palette = px.colors.qualitative.Plotly  # Use Plotly's color palette
        traces = []
        for idx, (year, group) in enumerate(avg_length_of_stay_per_year_month.groupby(level=0)):
            trace = go.Scatter(
                x=month_names,
                y=group.values,
                mode='lines+markers',
                name=str(year),
                marker=dict(color=color_palette[idx % len(color_palette)])  # Assign a color from the palette
            )
            traces.append(trace)

        # Create layout
        layout = go.Layout(
            title='Average Length of Stay per Month',
            xaxis=dict(title='Month'),
            yaxis=dict(title='Average Length of Stay'),
            legend=dict(title='Year')
        )

        # Create figure
        fig = go.Figure(data=traces, layout=layout)

        # Show the plot
        st.plotly_chart(fig)

    # Create an expander for displaying the data
        with st.expander("View Average Length of Stay per Month"):
            # Calculate average length of stay per month and year
            avg_length_of_stay_per_month = filtered_data.groupby(['Year', 'Month'])['LENGTH OF STAY'].mean().reset_index()
            
            # Convert the 'Year' column data to integers
            avg_length_of_stay_per_month['Year'] = avg_length_of_stay_per_month['Year'].astype(int)

            # Display the data
            st.write(avg_length_of_stay_per_month)
            
            # Add a download button
            st.download_button(
                label="Download Data as CSV",
                data=avg_length_of_stay_per_month.to_csv(index=False),
                file_name='average_length_of_stay_per_month.csv',
                mime='text/csv'
            )

        # Filter the data to include only inpatient stays
        filtered_data = df[(df['ADMISSION DATE'] >= start_date) & (df['ADMISSION DATE'] <= end_date)]

        inpatient_data = filtered_data[filtered_data['ADMISSION TYPE'] == 'INPATIENT']

        # Group the filtered data by 'PAYER' and calculate the total length of stay for each payer
        stays_by_payer = inpatient_data.groupby('PAYER')['LENGTH OF STAY'].sum().reset_index()

        # Create a doughnut chart using Plotly Express
        # The 'hole' parameter defines the size of the doughnut hole (0 for pie chart, a value > 0 for doughnut chart)
        fig_stays_by_payer = px.pie(
        stays_by_payer, values='LENGTH OF STAY', names='PAYER', title='Inpatient Stays by Payer', hole=0.5
        )

    
    with col2:
    # Plot the doughnut chart using Streamlit
        st.plotly_chart(fig_stays_by_payer)

    # Create an expander for displaying the data
        with st.expander("View Inpatient Stays by Payer Data"):
        # Display the grouped data
            st.write(stays_by_payer)

        # Add a download button for the data
            st.download_button(
                label="Download Data as CSV",
                data=stays_by_payer.to_csv(index=False),
                file_name='inpatient_stays_by_payer.csv',
                mime='text/csv'
            )
################# AVE TREATMENT COSTS BY AGE GROUP / TOP 10 DEPARTMENT #################

# Create layout columns
    col1, col2 = st.columns(2)

    with col1:

        # Filter data based on the selected date range
        filtered_data = df[(df['ADMISSION DATE'] >= start_date) & (df['ADMISSION DATE'] <= end_date)]

    # Create a function to categorize patients into age groups
        def categorize_age_group(age):
            if 1 <= age <= 17:
                return '1-17'
            elif 18 <= age <= 64:
                return '18-64'
            elif 65 <= age <= 90:
                return '65-90'
            else:
                return 'Other'

        # Create a new column in the data frame that categorizes patients into age groups
        filtered_data['Age Group'] = filtered_data['AGE'].apply(categorize_age_group)

        # Filter data to include only specified age groups
        filtered_data = filtered_data[filtered_data['Age Group'].isin(['1-17', '18-64', '65-90'])]

        # Group data by age group and calculate average treatment costs
        avg_treatment_cost_by_age_group = filtered_data.groupby('Age Group')['HOSPITAL CHARGES'].mean().reset_index()

        # Create a bar chart showing the average treatment costs for each age group
        fig = px.bar(
            avg_treatment_cost_by_age_group,
            x='Age Group',
            y='HOSPITAL CHARGES',
            labels={'HOSPITAL CHARGES': 'Average Treatment Cost'},
            title='Average Treatment Costs by Age Group',
            text_auto=True,  # Automatically display data labels on bars
            color_discrete_sequence=['#1f77b4']  # Set the bar color to blue
        )

        # Customize the y-axis to display currency values
        fig.update_yaxes(tickprefix='$', tickformat=',.2f')

        # Update layout to adjust bar gap for thinner bars
        fig.update_layout(bargap=0.5)  # Adjust bargap value as needed for thinner bars

        # Display the bar chart in Streamlit
        st.plotly_chart(fig)

        # Create an expander for additional data and download button
        with st.expander("View Average Treatment Costs by Age Group Data"):
            # Display the grouped data
            st.write(avg_treatment_cost_by_age_group)
            
            # Add a download button for the data
            st.download_button(
                label="Download Data as CSV",
                data=avg_treatment_cost_by_age_group.to_csv(index=False),
                file_name='average_treatment_costs_by_age_group.csv',
                mime='text/csv'
            )

    with col2:

            # Filter data based on the selected date range
        filtered_data = df[(df['ADMISSION DATE'] >= start_date) & (df['ADMISSION DATE'] <= end_date)]

        # Group data by 'DEPARTMENT/SPECIALTY' and sum the hospital charges for each group
        department_income = filtered_data.groupby('DEPARTMENT/SPECIALTY')['HOSPITAL CHARGES'].sum().reset_index()

        # Sort departments by total hospital charges generated in descending order
        department_income = department_income.sort_values(by='HOSPITAL CHARGES', ascending=False)

        # Filter to show only the top 10 departments
        top_10_departments = department_income.head(10)

        # Create a horizontal bar chart using Plotly Express to visualize the top 10 departments generating the most income
        fig = px.bar(
            top_10_departments,
            x='HOSPITAL CHARGES',
            y='DEPARTMENT/SPECIALTY',
            orientation='h',  # Horizontal bar chart
            labels={'HOSPITAL CHARGES': 'Total Hospital Charges ($)', 'DEPARTMENT/SPECIALTY': 'Department/Specialty'},
            title='Top 10 Departments Generating Top Income',
            text_auto=True,  # Automatically display data labels on bars
            
        )

                # Update the x-axis
        fig.update_xaxes(
            tickprefix='$',  # Prefix with a dollar sign
            tickformat=',.2s',  # Format values as SI units with two decimal places
        )

        # Update the hover template to display the actual values
        fig.update_traces(
            hovertemplate='%{y}: $%{x:,.2f}'  # Display the department/specialty as y and actual x value formatted with two decimal places
        )
        
         # Reverse the order of the bars so the highest income department is at the top
        fig.update_yaxes(autorange='reversed')
        # Display the bar chart in Streamlit
        st.plotly_chart(fig)

        # Create an expander for additional data and a download button
        with st.expander("View Top 10 Departments Generating Top Income Data"):
            # Display the grouped data
            st.write(top_10_departments)
            
            # Add a download button for the data
            st.download_button(
                label="Download Data as CSV",
                data=top_10_departments.to_csv(index=False),
                file_name='top_10_departments_generating_top_income.csv',
                mime='text/csv'
            )



########################################  PATIENT SATISFACTION SECTION  ########################################

# Function to display Patient Satisfaction section
def display_patient_satisfaction():
    st.header('Patient Satisfaction Section')

    st.markdown("---")

################# KPIs #################

    # Filter data based on date range
    filtered_data = df[(df['ADMISSION DATE'] >= start_date) & (df['ADMISSION DATE'] <= end_date)]

    # Calculate overall satisfaction rate

    # Count the number of 'Yes' responses in the 'CLIENT SATISFACTION' column
    positive_responses = filtered_data['CLIENT SATISFACTION'].value_counts().get('Yes', 0)
    
    # Count the total number of responses in the 'CLIENT SATISFACTION' column
    total_responses = filtered_data['CLIENT SATISFACTION'].count()

    # Calculate the overall satisfaction rate as a percentage
    overall_satisfaction_rate = (positive_responses / total_responses) * 100

    # Format the overall satisfaction rate to two decimal places
    overall_satisfaction_rate_formatted = f"{overall_satisfaction_rate:.2f}%"
    
    # Calculate average response time based on "WAITING TIME" column (in minutes)
    average_response_time = filtered_data['WAITING TIME'].mean()

    # Format the average response time to one decimal place (or as needed)
    average_response_time_formatted = f"{average_response_time:.1f} min"
    # Calculate the top performing department
    # Group data by department
    grouped_data = filtered_data.groupby('DEPARTMENT/SPECIALTY')

    # Calculate satisfaction rate for each department
    department_satisfaction_rate = grouped_data.apply(lambda group: 
                                                    (group['CLIENT SATISFACTION'].value_counts().get('Yes', 0) / 
                                                    group['CLIENT SATISFACTION'].count()) * 100)

    # Identify the department with the highest satisfaction rate
    top_performing_department = department_satisfaction_rate.idxmax()
    highest_satisfaction_rate = department_satisfaction_rate.max()

    # Format the highest satisfaction rate to two decimal places
    highest_satisfaction_rate_formatted = f"{highest_satisfaction_rate:.2f}%"

    # Display KPIs as 3 columns
    col1, col2, col3 = st.columns(3)

    # Display the overall satisfaction rate in column 1
    col1.metric("Overall Satisfaction Rate", overall_satisfaction_rate_formatted)

    # Display the average response time in column 2
    col2.metric("Average Response Time", average_response_time_formatted)

    # Display the top performing department in column 3
    col3.metric("Top Performing Department", top_performing_department, highest_satisfaction_rate_formatted)

    st.markdown("---")

################# PT SATISFACTION & UNSATISFACTION RATE / PATIENT SATISFACTION #################
  # Display the charts in two columns
    col1, col2 = st.columns(2)

    with col1:

    # Filter data based on date range and "CLIENT SATISFACTION" column being "YES"
        filtered_data = df[
            (df['ADMISSION DATE'] >= start_date) &
            (df['ADMISSION DATE'] <= end_date) 
            ]

    # Convert 'ADMISSION DATE' to datetime and extract the year
        filtered_data['ADMISSION DATE'] = pd.to_datetime(filtered_data['ADMISSION DATE'])
        filtered_data['YEAR'] = filtered_data['ADMISSION DATE'].dt.year

        # Group data by year
        grouped_data = filtered_data.groupby('YEAR')

        # Calculate total patients per year
        total_patients_per_year = grouped_data.size()

        # Calculate satisfied and unsatisfied counts per year based on "CLIENT SATISFACTION" column
        satisfied_count_per_year = grouped_data['CLIENT SATISFACTION'].apply(
            lambda x: (x.str.strip().str.lower() == 'yes').sum()
        )
        unsatisfied_count_per_year = grouped_data['CLIENT SATISFACTION'].apply(
            lambda x: (x.str.strip().str.lower() == 'no').sum()
        )

        # Calculate satisfaction and unsatisfaction rates per year
        satisfaction_rate_per_year = (satisfied_count_per_year / total_patients_per_year) * 100
        unsatisfaction_rate_per_year = (unsatisfied_count_per_year / total_patients_per_year) * 100

        # Create a grouped bar chart showing satisfied and unsatisfied rates per year
        fig = go.Figure()

        # Add bar for satisfaction rate
        fig.add_trace(go.Bar(
            x=satisfaction_rate_per_year.index,
            y=satisfaction_rate_per_year,
            name='Satisfaction Rate',
            marker_color='blue',
            hovertemplate='Satisfaction Rate: %{y:.2f}%<extra></extra>'
        ))

        # Add bar for unsatisfaction rate
        fig.add_trace(go.Bar(
            x=unsatisfaction_rate_per_year.index,
            y=unsatisfaction_rate_per_year,
            name='Unsatisfaction Rate',
            marker_color='red',
            hovertemplate='Unsatisfaction Rate: %{y:.2f}%<extra></extra>'
        ))

        # Add a horizontal line for the target satisfaction rating (85%)
        fig.add_trace(go.Scatter(
            x=satisfaction_rate_per_year.index,  # Use the same x-values (years) as the bars
            y=[85] * len(satisfaction_rate_per_year.index),  # Constant y-value for target rating
            mode='lines',
            name='Target Rating (85%)',
            line=dict(color='green', dash='dash')
        ))

        # Customize chart layout
        fig.update_layout(
            title='Patient Satisfaction and Unsatisfaction Rates per Year',
            yaxis_title='Rate (%)',
            xaxis_title='Year',
            barmode='group',  # Display bars side by side
            showlegend=True,
            yaxis=dict(
                range=[0, 100],  # Set y-axis range from 0 to 100
                tickvals=list(range(0, 101, 20)),  # Display ticks at 0, 20, 40, 60, 80, 100
                ticktext=[f"{i}%" for i in range(0, 101, 20)]  # Format ticks as percentages
            ),
            xaxis=dict(
                tickmode='array',
                tickvals=satisfaction_rate_per_year.index,
                ticktext=satisfaction_rate_per_year.index # Display years as x-axis labels
            )
        )

        # Display the chart 
        st.plotly_chart(fig)

# Wrap the data values in an expander
        with st.expander("Satisfaction and Unsatisfaction Rates Data"):
            # Prepare data for display and download
            display_data = pd.DataFrame({
                'Year': satisfaction_rate_per_year.index,
                'Satisfaction Rate (%)': satisfaction_rate_per_year.apply(lambda x: f"{x:.2f}").values,
                'Unsatisfaction Rate (%)': unsatisfaction_rate_per_year.apply(lambda x: f"{x:.2f}").values
            })
               
            st.write(display_data)

            # Add a download button
            csv = display_data.to_csv(index=False)
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name='satisfaction_unsatisfaction_rates.csv',
                mime='text/csv'
            )


    with col2:

    # Convert 'ADMISSION DATE' to datetime and extract the year
        filtered_data['ADMISSION DATE'] = pd.to_datetime(filtered_data['ADMISSION DATE'])
        filtered_data['YEAR'] = filtered_data['ADMISSION DATE'].dt.year

        # Group data by year
        grouped_data = filtered_data.groupby('YEAR')

        # Calculate total patients per year
        total_patients_per_year = grouped_data.size()

        # Calculate satisfied count per year based on "CLIENT SATISFACTION" column
        satisfied_count_per_year = grouped_data['CLIENT SATISFACTION'].apply(
            lambda x: (x.str.strip().str.lower() == 'yes').sum()
        )

        # Calculate satisfaction rate per year
        satisfaction_rate_per_year = (satisfied_count_per_year / total_patients_per_year) * 100

        # Create a line chart showing satisfaction rate per year
        fig = go.Figure()

        # Add line for satisfaction rate
        fig.add_trace(go.Scatter(
            x=satisfaction_rate_per_year.index,
            y=satisfaction_rate_per_year,
            mode='lines+markers',
            name='Satisfaction Rate',
            line=dict(color='blue', width=2),
            hovertemplate='Satisfaction Rate: %{y:.2f}%<extra></extra>'
        ))

        # Add a horizontal line for the target satisfaction rating (85%)
        fig.add_trace(go.Scatter(
            x=satisfaction_rate_per_year.index,  # Use the same x-values (years) as the line
            y=[85] * len(satisfaction_rate_per_year.index),  # Constant y-value for target rating
            mode='lines',
            name='Target Rating (85%)',
            line=dict(color='green', dash='dash')
        ))

        # Customize chart layout
        fig.update_layout(
            title='Patient Satisfaction Rate per Year',
            yaxis_title='Rate (%)',
            xaxis_title='Year',
            showlegend=True,
            yaxis=dict(
                range=[0, 100],  # Set y-axis range from 0 to 100
                tickvals=list(range(0, 101, 20)),  # Display ticks at 0, 20, 40, 60, 80, 100
                ticktext=[f"{i}%" for i in range(0, 101, 20)]  # Format ticks as percentages
            ),
            xaxis=dict(
                tickmode='array',
                tickvals=satisfaction_rate_per_year.index,
                ticktext=satisfaction_rate_per_year.index, # Display years as x-axis labels
                title='Year'  # Label the x-axis as 'Year'
            )
        )

        # Display the chart  
        st.plotly_chart(fig)

        with st.expander("Satisfaction Rates Data"):
            # Prepare data for display and download
            display_data = pd.DataFrame({
                'Year': satisfaction_rate_per_year.index,
                'Satisfaction Rate (%)': satisfaction_rate_per_year.apply(lambda x: f"{x:.2f}").values,
            })
                
            st.write(display_data)

            # Prepare data for download
            csv = display_data.to_csv(index=False)
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name='satisfaction_rates.csv',
                mime='text/csv'
            )


################# LIKERT SCALE / SATISFACTION & UNSATISFACTION COUNTS #################
 
    # Display the charts in two columns
    col1, col2 = st.columns(2)

    with col1:
    # Filter data based on the selected date range
        filtered_data = df[
        (df['ADMISSION DATE'] >= start_date) &
        (df['ADMISSION DATE'] <= end_date)
        ]

        # Calculate the frequencies of each Likert scale response in the "SCALE" column
        scale_counts = filtered_data['SCALE'].value_counts(normalize=True) * 100

        # Calculate the raw counts of each Likert scale response
        scale_counts_value = filtered_data['SCALE'].value_counts()

        # Define the order of Likert scale responses including "VERY UNSATISFIED"
        likert_order = ['VERY UNSATISFIED', 'UNSATISFIED', 'AVERAGE', 'SATISFIED', 'HAPPY']

        # Reindex both Series to ensure they have the same order of Likert scale responses
        scale_counts = scale_counts.reindex(likert_order)
        scale_counts_value = scale_counts_value.reindex(likert_order)

        # Create custom data list with both raw counts and percentages for each Likert scale response
        custom_data = [(scale_counts_value[response], scale_counts[response]) for response in likert_order]

        # Define custom colors for each Likert scale response
        color_map = {
            'VERY UNSATISFIED': '#FF4500',  # Orange-red for Very Unsatisfied
            'UNSATISFIED': '#FF6347',       # Tomato red for Unsatisfied
            'AVERAGE': '#FFD700',           # Gold for Average
            'SATISFIED': '#90EE90',         # Light green for Satisfied
            'HAPPY': '#32CD32'              # Lime green for Happy
        }

        # Create a horizontal bar chart to visualize the Likert scale responses
        fig = px.bar(
            scale_counts,
            y=scale_counts.index,
            x=scale_counts.values,
            labels={'y': 'Likert Scale Response', 'x': 'Percentage (%)'},
            title='Distribution of Likert Scale Responses',
            text_auto=True,              # Display data values on the bars
            orientation='h',            # Use horizontal orientation
            color=scale_counts.index,   # Different colors for each scale value
            color_discrete_map=color_map  # Use the defined color map
        )

        # Update hover text to include both raw value count and percentage
        fig.update_traces(
            text=scale_counts.values,  # Display percentage on bars
            texttemplate='%{x:.2f}%',
            textposition='inside'
        )

        # Update chart layout for better visualization
        fig.update_layout(
            xaxis=dict(
                range=[0, 100],              # Set x-axis range from 0 to 100
                tickvals=list(range(0, 101, 20)),  # Display ticks at 0, 20, 40, 60, 80, 100
                ticktext=[f"{i}%" for i in range(0, 101, 20)]  # Format ticks as percentages
            ),
            yaxis_title='Likert Scale Response',
            xaxis_title='Percentage (%)'
        )

        # Display the chart in Streamlit
        st.plotly_chart(fig)

        # Wrap the data values in an expander
        with st.expander("Likert Scale Response Distribution"):
             # Prepare data for display and download
            display_data = pd.DataFrame({
                'Likert Scale Response': likert_order,
                'Raw Count': scale_counts_value.values,
                'Percentage (%)': scale_counts.apply(lambda x: f"{x:.2f}").values
            })
            st.write(display_data)

            # Prepare data for download
            csv = display_data.to_csv(index=False)
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name='likert_scale_distribution.csv',
                mime='text/csv'
            )

    with col2:

        # Calculate total satisfied and unsatisfied counts based on "CLIENT SATISFACTION" column
        satisfied_count = filtered_data['CLIENT SATISFACTION'].apply(lambda x: x.strip().lower() == 'yes').sum()
        unsatisfied_count = filtered_data['CLIENT SATISFACTION'].apply(lambda x: x.strip().lower() == 'no').sum()

        # Create a doughnut chart with satisfied and unsatisfied counts
        labels = ['Satisfied', 'Unsatisfied']
        values = [satisfied_count, unsatisfied_count]

        
    # Define custom colors for the satisfied and unsatisfied slices
        colors = ['#FFFF00', '#F44336']  # Customize these color codes as desired

        # Create a doughnut chart with satisfied and unsatisfied counts
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,  # Size of the hole in the middle to create a doughnut chart
            marker=dict(colors=colors),  # Set custom colors for each slice
            textinfo='percent',  # Display label and percentage in each slice
            texttemplate='%{percent:.2%}',  # Format the percentage with 2 decimal places
            hovertemplate='%{label}: %{percent}<extra> (%{value})</extra>'  # Include raw count in hover

        )])

        # Add a title to the chart
        fig.update_layout(
            title='Satisfaction and Unsatisfaction Counts',
            showlegend=True
        )

        # Display the chart 
        st.plotly_chart(fig)

            # Display the values and percentages in an expander with a download button
        with st.expander("View Satisfaction and Unsatisfaction Counts"):
            # Display the values and percentages
            st.write("Satisfied:", satisfied_count, f"({(satisfied_count / (satisfied_count + unsatisfied_count)):.2%})")
            st.write("Unsatisfied:", unsatisfied_count, f"({(unsatisfied_count / (satisfied_count + unsatisfied_count)):.2%})")
            
        # Add a download button for the satisfaction and unsatisfaction counts data
            st.download_button(
                label="Download Data as CSV",
                data=pd.DataFrame({'Category': ['Satisfied', 'Unsatisfied'], 'Count': [satisfied_count, unsatisfied_count]}).to_csv(index=False),
                file_name='satisfaction_unsatisfaction_counts.csv',
                mime='text/csv'
            )



################# SATISFACTION RATE BY DEPARTMENT #################
     # Filter data based on date range
    filtered_data = df[(df['ADMISSION DATE'] >= start_date) & (df['ADMISSION DATE'] <= end_date)]

    # Group data by department and calculate satisfaction rate
    grouped_data = filtered_data.groupby('DEPARTMENT/SPECIALTY')

    # Calculate total cases and satisfied cases per department
    total_cases_per_department = grouped_data.size()
    satisfied_cases_per_department = grouped_data['CLIENT SATISFACTION'].apply(
    lambda x: (x.str.strip().str.lower() == 'yes').sum()
    )

    # Calculate satisfaction rate per department
    satisfaction_rate_per_department = (satisfied_cases_per_department / total_cases_per_department) * 100

    # Create a horizontal bar chart with satisfaction rates by department
    fig = go.Figure(go.Bar(
    y=satisfaction_rate_per_department.index,  # Departments on y-axis
    x=satisfaction_rate_per_department,  # Satisfaction rates on x-axis
    orientation='h',  # Set orientation to horizontal
    name='Satisfaction Rate',
    marker_color='green',
    hovertemplate='Satisfaction Rate: %{x:.2f}%<extra></extra>'
    ))

    # Add a horizontal line for the target satisfaction rating (85%)
    fig.add_trace(go.Scatter(
    x=[85] * len(satisfaction_rate_per_department.index),  # Constant target rating
    y=satisfaction_rate_per_department.index,  # Same departments as the bar chart
    mode='lines',
    name='Target Rating (85%)',
    line=dict(color='red', dash='dash')
    ))

    # Customize chart layout
    fig.update_layout(
        title='Satisfaction Rate by Department',
        xaxis_title='Rate (%)',
        yaxis_title='Department',
        height=150 + len(satisfaction_rate_per_department) * 20,  # Adjust height based on number of departments
        width=1600,  # Adjust the width of the chart
        xaxis=dict(
            range=[0, 100],  # Set x-axis range from 0 to 100
            tickvals=list(range(0, 101, 20)),  # Tick values every 20%
            ticktext=[f"{i}%" for i in range(0, 101, 20)]  # Format ticks as percentages
        )
    )

    # Display the chart
    st.plotly_chart(fig)

    # Use an expander to encapsulate the chart and download button
    with st.expander("View Satisfaction Rates by Department"):
    # Display the chart in Streamlit within the expander
        st.write(satisfaction_rate_per_department)
    
    
    # Add a download button for the satisfaction rate data
        st.download_button(
            label="Download Data as CSV",
            data=satisfaction_rate_per_department.to_csv(index=True),
            file_name='satisfaction_rate_per_department.csv',
            mime='text/csv'
        )

########################################  FINANCIAL PERFORMANCE SECTION  ########################################

    # Function to display Financial Performance section
def display_financial_performance():
    st.header('Financial Performance Section')
    # Filter data based on the selected date range
    filtered_data = df[
        (df['ADMISSION DATE'] >= start_date) &
        (df['ADMISSION DATE'] <= end_date)
    ]

################# KPIs #################

    # Calculate Total Patient Revenue in $M
    total_patient_revenue = filtered_data['HOSPITAL CHARGES'].sum() / 1_000_000_000

    # Calculate Average Revenue per Inpatient Day
    inpatient_revenue = filtered_data[filtered_data['ADMISSION TYPE'] == 'INPATIENT']['HOSPITAL CHARGES'].sum()
    total_inpatient_days = filtered_data[filtered_data['ADMISSION TYPE'] == 'INPATIENT']['LENGTH OF STAY'].sum()
    avg_revenue_per_inpatient_day = inpatient_revenue / total_inpatient_days

    # Calculate Average Revenue per Outpatient Visit
    outpatient_revenue = filtered_data[filtered_data['ADMISSION TYPE'] == 'OUTPATIENT']['HOSPITAL CHARGES'].sum()
    total_outpatient_visits = filtered_data[filtered_data['ADMISSION TYPE'] == 'OUTPATIENT'].shape[0]
    avg_revenue_per_outpatient_visit = outpatient_revenue / total_outpatient_visits

    # Display KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patient Revenue", f"${total_patient_revenue:,.2f}B")
    col2.metric("Avg Revenue per Inpatient Day", f"${avg_revenue_per_inpatient_day:,.2f}")
    col3.metric("Avg Revenue per Outpatient Visit", f"${avg_revenue_per_outpatient_visit:,.2f}")

    st.markdown("---")

   ################# DEPARTMENT REVENUE / ADMISSION TYPE REVENUE #################

    # Display the charts in two columns
    col1, col2 = st.columns(2)
    
    # Revenue by Department/Specialty
    with col1:
        # Filter data based on the selected date range
        filtered_data = df[(df['ADMISSION DATE'] >= start_date) & (df['ADMISSION DATE'] <= end_date)]

        # Group data by 'DEPARTMENT/SPECIALTY' and sum the hospital charges for each group
        department_income = filtered_data.groupby('DEPARTMENT/SPECIALTY')['HOSPITAL CHARGES'].sum().reset_index()

        # Sort departments by total hospital charges generated in descending order
        department_income = department_income.sort_values(by='HOSPITAL CHARGES', ascending=False)

        # Filter to show only the top 10 departments
        top_10_departments = department_income.head(10)
        # Assuming department_income is a DataFrame with 'DEPARTMENT/SPECIALTY' and 'HOSPITAL CHARGES' columns

        # Sort the DataFrame by revenue (HOSPITAL CHARGES) in descending order
        department_income_sorted = department_income.sort_values(by='HOSPITAL CHARGES', ascending=False)

        # Define the number of top departments to highlight
        top_n = 10

        # Create a color list for bars (default gray, then a distinct color for top 10)
        colors = ['lightgray'] * len(department_income_sorted)
        colors[:top_n] = ['royalblue'] * top_n  # Set 'royalblue' for top 10

        # Create a vertical bar chart using Plotly Express
        fig = px.bar(
            department_income_sorted,
            x='DEPARTMENT/SPECIALTY',
            y='HOSPITAL CHARGES',
            labels={'DEPARTMENT/SPECIALTY': 'Department/Specialty', 'HOSPITAL CHARGES': 'Total Hospital Revenue ($)'},
            title='Departmental Revenue',
            text_auto=True
            )

        # Update the y-axis (revenue)
        fig.update_yaxes(
            tickprefix='$',  # Prefix with a dollar sign
            tickformat=',.2s',  # Format values as SI units with two decimal places
        )

        # Update the hover template to display department and actual revenue
        fig.update_traces(
            hovertemplate='%{x}: $%{y:,.2f}',
            marker_color=colors  # Display department and actual revenue with two decimal places
        )

        # Rotate text labels
        fig.update_xaxes(tickangle=45)  # Rotate x-axis labels by 45 degrees

        # Increase chart height
        fig.update_layout(height=600,width=800, showlegend=False) 

        # Display the bar chart in Streamlit
        st.plotly_chart(fig)

        #Use an expander to encapsulate the chart and download button
        with st.expander("View Departmental Revenue"):
            # Reset index for display only
            department_income_display = department_income_sorted.reset_index(drop=True)
            # Rename the column "HOSPITAL CHARGES" to "REVENUE"
            department_income_display = department_income_display.rename(columns={"HOSPITAL CHARGES": "REVENUE"})

            st.write(department_income_display)

            # Add a download button for the satisfaction rate data
            st.download_button(
                label="Download Data as CSV",
                data=department_income_display.to_csv(index=True),
                file_name='departmental_revenue.csv',
                mime='text/csv'
            )


    # Revenue by Admission Type
    with col2:

        filtered_data = df[(df['ADMISSION DATE'] >= start_date) & (df['ADMISSION DATE'] <= end_date)]
              # Assuming filtered_data is a DataFrame with 'ADMISSION TYPE' and 'HOSPITAL CHARGES' columns

        revenue_by_admission_type = filtered_data.groupby('ADMISSION TYPE')['HOSPITAL CHARGES'].sum().reset_index()

        # Extract labels and values
        labels = revenue_by_admission_type['ADMISSION TYPE'].tolist()
        values = revenue_by_admission_type['HOSPITAL CHARGES'].tolist()

        # Define custom colors if needed
        colors = ['#00FF00', '#FF0000']  # Example colors for different admission types

        # Create the doughnut chart
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,  # Size of the hole in the middle to create a doughnut chart
            marker=dict(colors=colors),  # Set custom colors for each slice
            textinfo='percent',  # Display label and percentage in each slice
            texttemplate='%{percent:.2%}',  # Format the percentage with 2 decimal places
            hovertemplate='%{label}: %{percent}<extra> ($%{value})</extra>'  # Include raw count in hover
        )])

        # Add a title to the chart
        fig.update_layout(
            title_text='Admission Type Revenue')

        # Display the chart using Streamlit
        st.plotly_chart(fig)

        #Use an expander to encapsulate the chart and download button
        with st.expander("View Admission Type Revenue"):
            # Rename the column "HOSPITAL CHARGES" to "REVENUE"
            revenue_by_admission_type = revenue_by_admission_type.rename(columns={"HOSPITAL CHARGES": "REVENUE"})
            st.write(revenue_by_admission_type)

                # Add a download button for the satisfaction rate data
            st.download_button(
                label="Download Data as CSV",
                data=revenue_by_admission_type.to_csv(index=True),
                file_name='admission_type_revenue.csv',
                mime='text/csv'
            )

################# REVENUE TRENDS #################

    # Revenue Trends Over Time
    filtered_data = df[(df['ADMISSION DATE'] >= start_date) & (df['ADMISSION DATE'] <= end_date)]
    filtered_data['YearMonth'] = filtered_data['ADMISSION DATE'].dt.to_period('M').astype(str)
    revenue_trends = filtered_data.groupby('YearMonth')['HOSPITAL CHARGES'].sum().reset_index()
    fig2 = px.line(revenue_trends, x='YearMonth', y='HOSPITAL CHARGES', title='Revenue Trends Over Time', markers=True,
                   hover_data={'HOSPITAL CHARGES': ':$,.0f'}) 
    fig2.update_traces(line=dict(color='#FF0000'), hovertemplate='<b>YearMonth</b>: %{x}<br><b>Revenue</b>: $%{y:,.0f}<extra></extra>') 
    
    # Update y-axis to include dollar sign
    fig2.update_layout(yaxis_tickprefix='$', xaxis_title='', yaxis_title='')
    st.plotly_chart(fig2, use_container_width=True)

    #Use an expander to encapsulate the chart and download button
    with st.expander("View Revenue Trends Over Time"):
        revenue_trends = revenue_trends.rename(columns={"HOSPITAL CHARGES": "REVENUE"})
        st.write(revenue_trends)

    # Add a download button for the satisfaction rate data
        st.download_button(
            label="Download Data as CSV",
            data=revenue_trends.to_csv(index=True),
            file_name='revenue_trends.csv',
            mime='text/csv'
        )

################# REVENUE BY PAYER / REVENUE BY AGE GROUP #################
    # Display the charts in two columns
    col1, col2 = st.columns(2)

    #Revenue by payer
    with col1:
        filtered_data = df[(df['ADMISSION DATE'] >= start_date) & (df['ADMISSION DATE'] <= end_date)]
        
        # Payer Mix Analysis
        revenue_by_payer = filtered_data.groupby('PAYER')['HOSPITAL CHARGES'].sum().reset_index()

        # Define custom colors excluding blue, red, light blue, and pink
        custom_colors = ['#FFD700', '#32CD32', '#800080', '#FFA500', '#8A2BE2', '#FF1493']  

        # Create a pie chart for Revenue by Payer with custom colors
        fig4 = px.pie(revenue_by_payer, values='HOSPITAL CHARGES', names='PAYER', title='Revenue by Payer', 
              color_discrete_sequence=custom_colors,
              hover_data={'HOSPITAL CHARGES': ':$, .2f'})

        # Add a prefix to the hover text for Hospital Charges
        fig4.update_traces(textinfo='percent+label')

        # Add a prefix to the hover text for Hospital Charges
        fig4.update_traces(textinfo='percent+label', hovertemplate='<b>PAYER</b>: %{label}<br><b>REVENUE</b>: $%{value:,.2f}<extra></extra>')

        st.plotly_chart(fig4)

        #Use an expander to encapsulate the chart and download button
        with st.expander("View Revenue by Payer"):
            revenue_by_payer = revenue_by_payer.rename(columns={"HOSPITAL CHARGES": "REVENUE"})
            st.write(revenue_by_payer)

            # Add a download button for the satisfaction rate data
            st.download_button(
                label="Download Data as CSV",
                data=revenue_by_payer.to_csv(index=True),
                file_name='revenue_payer.csv',
                mime='text/csv'
            )
 
        
    #Revenue by Age group
    with col2:
    
        # Filter the data based on the date range
        filtered_data = df[(df['ADMISSION DATE'] >= start_date) & (df['ADMISSION DATE'] <= end_date)]

        age_bins = [0, 18, 35, 50, 65, 80, 100]
        age_labels = ['0-18', '19-35', '36-50', '51-65', '66-80', '81-100']
        
        # Create 'Age Group' column
        filtered_data['Age Group'] = pd.cut(filtered_data['AGE'], bins=age_bins, labels=age_labels)

        # Group by 'Age Group' and sum the hospital charges
        revenue_by_age_group = filtered_data.groupby('Age Group')['HOSPITAL CHARGES'].sum().reset_index()

        # Create a bar chart for Revenue per Patient by Age Group
        fig5 = px.bar(revenue_by_age_group, x='Age Group', y='HOSPITAL CHARGES', title=' Revenue By Age Group', color_discrete_sequence=['#FFA07A'])

        fig5.update_layout(yaxis_tickprefix='$', xaxis_title='Age Group', yaxis_title='')
        fig5.update_traces(hovertemplate='<b>Age Group</b>: %{x}<br><b>REVENUE</b>: $%{y:,.2f}<extra></extra>')
        st.plotly_chart(fig5)

        #Use an expander to encapsulate the chart and download button
        with st.expander("View Revenue by Age Group"):
            revenue_by_age_group = revenue_by_age_group.rename(columns={"HOSPITAL CHARGES": "REVENUE"})
            st.write(revenue_by_age_group)

        # Add a download button for the satisfaction rate data
            st.download_button(
                label="Download Data as CSV",
                data=revenue_by_age_group.to_csv(index=True),
                file_name='revenue_by_age_group.csv',
                mime='text/csv'
            )

# Main section selection
if selected_section == 'Patients Trend':
    display_patients_trend()
elif selected_section == 'Operating Performance':
    display_operating_performance()
elif selected_section == 'Patient Satisfaction':
    display_patient_satisfaction()
elif selected_section == 'Financial Performance':
    display_financial_performance()


