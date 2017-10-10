from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt

summary_df = pd.read_csv("exdata_data_NEI_data/summarySCC_PM25.csv")
scc_df = pd.read_csv("exdata_data_NEI_data/Source_Classification_Code.csv")
summary_df.fips = summary_df.fips.astype(str)
summary_df.SCC = summary_df.SCC.astype(str)
scc_df.SCC = scc_df.SCC.astype(str)

ylabel = "Tons of PM2.5"

#1
total_emissions = summary_df.pivot_table('Emissions', index='year', aggfunc='sum')
total_plot = total_emissions.plot(title="Total PM2.5 emissions for USA 1999-2008")
total_plot.set_ylabel(ylabel)
total_plot.legend_.remove()
#2
baltimore_emissions = summary_df[summary_df.fips == '24510']
baltimore_total = baltimore_emissions.pivot_table('Emissions', index='year', aggfunc='sum')
baltimore_total_plot = baltimore_total.plot(title="Total PM2.5 emissions for Baltimore City 1999-2008")
baltimore_total_plot.set_ylabel(ylabel)
baltimore_total_plot.legend_.remove()
#3
baltimore_total_by_type = baltimore_emissions.pivot_table('Emissions', index='year', columns='type', aggfunc='sum')
baltimore_types_plot = baltimore_total_by_type.plot(title="Total PM2.5 emissions by type for Baltimore City 1999-2008")
baltimore_types_plot.set_ylabel(ylabel)
#4
coal_combustion = scc_df[(scc_df.Short_Name.str.contains('Coal') & scc_df.Short_Name.str.contains('Comb'))]
coal_emissions = summary_df[summary_df.SCC.isin(coal_combustion.SCC)]
coal_total = coal_emissions.pivot_table('Emissions', index='year', aggfunc='sum')
coal_total_plot = coal_total.plot(title='PM2.5 emissions from coal for USA 1999-2008')
coal_total_plot.set_ylabel(ylabel)
coal_total_plot.legend_.remove()
#5
vehicles = scc_df[(scc_df.Data_Category.str.contains('road')) & (scc_df.Short_Name.str.contains('Vehicle'))]
baltimore_vehicle_emissions = baltimore_emissions[baltimore_emissions.SCC.isin(vehicles.SCC)]
baltimore_vehicle_total = baltimore_vehicle_emissions.pivot_table('Emissions', index='year', aggfunc='sum')
baltimore_vehicle_total_plot = baltimore_vehicle_total.plot(title='PM2.5 emissions from vehicles for Baltimore City 1999-2008')
baltimore_vehicle_total_plot.set_ylabel(ylabel)
baltimore_vehicle_total_plot.legend_.remove()
#6
la_emissions = summary_df[summary_df.fips == '6037']
la_vehicle_emissions = la_emissions[la_emissions.SCC.isin(vehicles.SCC)]
la_vehicle_total = la_vehicle_emissions.pivot_table('Emissions', index='year', aggfunc='sum')
baltimore_vs_la = DataFrame({'bc_emissions': baltimore_vehicle_total.Emissions, 'la_emissions': la_vehicle_total.Emissions}, index=la_vehicle_total.index, columns = ['bc_emissions', 'la_emissions'])
fig, axes = plt.subplots(1,2)
bvt = baltimore_vehicle_total.plot(title="Baltimore vehicle emissions 1999-2008", ax=axes[0])
bvt.legend_.remove()
lvt = la_vehicle_total.plot(title="LA vehicles emissions 1999-2008", ax=axes[1])
lvt.legend_.remove()
bvt.set_ylabel(ylabel)
