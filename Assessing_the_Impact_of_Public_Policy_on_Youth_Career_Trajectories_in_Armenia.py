""" The Impact of Public Policy on Youth Career Trajetories in Armenia """

# Hypothesis: The number of applicants in STEM faculties in Armenia will increase 
# after the Government's new policy adopted in 2023.

# Import the required dependecies
import pandas as pd
import numpy as np
import re
import datetime
from dateutil import parser
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLSResults
from sys import displayhook
import plotly.express as px

# File paths containing the raw data
ysu_data_file = "./data/ysu_annual_applications_processed_data - English.csv"

def prep_var(file_path):
    var = pd.read_csv(file_path, na_values=np.nan) # 47 cross-sections
    var.columns = list(var.columns[:1]) + [pd.to_datetime(int(year), format="%Y") for year in var.columns[1:]]
    var = var.set_index("Group").T
    var = var.reset_index().melt(id_vars=["index"])
    return(var)

"""___ Difference-in-Differences Estimation for 2023's Government policy ___"""

ysu_data = pd.read_csv(ysu_data_file)
entire_fac_list = ysu_data["faculty"].to_list()
gov_prom_fac = ["Physics, Applied Mathematics and Physics", "Physics of Nuclear Reactors", "Radiophysics and Electronics", "Chemistry"]

qs_subject_classification = {
    "Arts & Humanities": [
        "History",
        "Cultural Studies, Archeology and Ethnography",
        "Art Science",
        "Philosophy",
        "Armenian Language and Literature",
        "Russian Language and Literature",
        "Linguistic and Intercultural Communication - Russian Language",
        "Linguistic and Intercultural Communication - English Language",
        "Translation Work: English and Armenian Languages",
        "Translation Work: French and Armenian Languages",
        "Translation Work: German and Armenian Languages",
        "Spanish Language and Literature",
        "Italian Language and Literature",
        "Arabic Studies",
        "Iranology",
        "Turkic Studies",
        "Theology",
        "Caucasian Studies, Azerbaijani Studies"
    ],
    "Engineering & Technology": [
        "Mechanics",
        "Informatics and Applied Mathematics, Applied Statistics and Data Science",
        "Mathematics and Computer Sciences, Information Security",
        "Semiconductor Physics and Microelectronics",
        "Telecommunications and Signal Processing",
        "Cartography and Cadastre Work"
    ],
    "Life Sciences & Medicine": [
        "Ecological Chemistry, Biochemistry and Food Safety",
        "Pharmaceutical Chemistry, Pharmacy",
        "Biology",
        "Biophysics, Bioinformatics"
    ],
    "Natural Sciences": [
        "Mathematics",
        "Actuarial and Financial Mathematics",
        "Geology",
        "Geography"
    ],
    "Social Sciences & Management": [
        "Service",
        "Political Science",
        "International Relations",
        "Public Administration",
        "Economics",
        "Management, Finance",
        "Psychology",
        "Social Work",
        "Sociology",
        "Social Pedagogy",
        "Journalism",
        "Law"
    ],
    
    "Government Promoted Faculties" : [
        "Physics, Applied Mathematics and Physics", 
        "Physics of Nuclear Reactors", 
        "Radiophysics and Electronics", 
        "Chemistry"
    ]
}

# Function to aggregate data for a group
def aggregate_group(group, data):
    return data[data["faculty"].isin(group)].sum().drop("faculty")

# Create a list to store aggregated data for each group
aggregated_data = []

# Aggregate data for each group
for group_name, faculties in qs_subject_classification.items():
    group_data = aggregate_group(faculties, ysu_data)
    group_data["Group"] = group_name
    aggregated_data.append(group_data)

# Concatenate all aggregated data into a single DataFrame
aggregated_df = pd.concat(aggregated_data, axis=1).T

# Set "Group" as the index
aggregated_df.set_index("Group", inplace=True)
aggregated_df.reset_index(drop = False, inplace = True)

# Display the aggregated data
print(aggregated_df)

# Save the aggregated data frame to a CSV file
# aggregated_df.to_csv("./data/applications_to_qs_aggregated_faculties.csv", index = False)

aggregated_df = prep_var("./data/aggregated_qs_faculty_applications_government_promoted_faculties.csv")

aggregated_df = aggregated_df.rename(columns={"index":"year", "value":"fac_app"})

aggregated_df = aggregated_df.dropna(subset=["fac_app"])
aggregated_df = aggregated_df.assign(fac_app=aggregated_df["fac_app"].astype(int))

aggregated_df = aggregated_df.assign(fac_app_div_mean=aggregated_df.groupby("Group")["fac_app"].transform(lambda x: x / x.mean()))
px.line(aggregated_df, y="fac_app_div_mean", x="year", line_group="Group", color="Group")

# aggregated_df.rename(columns={'fac_app_div_mean':'Scaled applications', 'year':'Date', 'Group':'faculty group'}, inplace=True)
# sns.set(style='whitegrid', rc={"grid.linewidth": 0.1}, font_scale=1.1)
# sns.lineplot(data=aggregated_df, x='Date', y='Scaled applications', hue='faculty group')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.show()

post_2023_law = ["2023", "2024"] # at least two data points for years are requiired.
aggregated_df["post_2023_law"] = np.where(aggregated_df["year"].isin(post_2023_law), 1, 0)
cond = aggregated_df["Group"].isin(["Government Promoted Faculties"])
aggregated_df["stem"] = np.where(cond, 1, 0)
aggregated_df["post_2023_law_stem"] = aggregated_df["post_2023_law"] * aggregated_df["stem"]

aggregated_df["year"] = pd.factorize(aggregated_df["year"], sort = True) [0] + 1
aggregated_df["year"].unique()

aggregated_df["Group"] = pd.factorize(aggregated_df["Group"], sort = True) [0] + 1

endg = pd.DataFrame(aggregated_df["fac_app_div_mean"])
exog = aggregated_df[["post_2023_law_stem"]]
aggregated_df.info()
aggregated_df.describe()

clustered_2ways_ols_2022 = sm.OLS(endg, sm.add_constant(exog))
fitted_mdl_2022 = clustered_2ways_ols_2022.fit(cov_type = "cluster", cov_kwds={"groups" : np.array(aggregated_df[["Group", "year"]])})
fitted_mdl_2022.summary()
