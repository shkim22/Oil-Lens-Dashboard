# Oil-Lens-Dashboard
The Oil Lens Dashboard is a dashboard with comprehensive oil data from the EnergyAspects API. The idea behind this dashboard is to allow users to access Oil data and perform data analytics in an industry that does not have convenient ways to access the data.

Oil Lens Dashboard
1. Launching the Application
a. For Windows users, please install the latest version of Power BI. You can download it from the Power BI website(https://powerbi.microsoft.com/). Unfortunately, Power BI Desktop is not available for macOS users. You would need to use a Virtual Machine, Apple’s Boot Camp, or Remote Desktop.
b. Once installed, open the "Oil Lens Dashboard_final.pbix" file.
c. Upon loading the dashboard, all market-related data is separated into tabs such as "Global Overview" and "ARIMA Forecasting". 

2. Using the Dashboard
a. Navigating tabs
Navigate through the different named tabs by clicking on them at the bottom of the Power BI window.
b. Interactive Features
The dashboard pages are interactive. Here's how you can make the most of them:
-	Date/time: to adjust the date views for the different tabs, see the Date sliders at the top of the screen. By clicking the current dates, manual entry can be done. The calendar icon can also be selected to manually click the specified dates. Lastly, the sliders can be adjusted to change the time horizon and can be moved around by clicking and dragging. 
-	Multiple Selection: For tabs that have multiple selection cards, such as the "Refining Production" tab, click the dropdown menu and hold Ctrl + click to select multiple time series sets to view. For example, Ctrl + click multiple countries in the "Refining Production" tab to have their refining runs and maintenance schedules plotted on both charts.
-	Series Focus: by clicking on an individual time series on a chart, it will change the dashboard page to show that series specifically. This can also be done by clicking the colored circles in the legend at the top of respective charts.
c. Exporting: In the Power BI application under "File" and "Export" in the top left menu, the dashboard can be exported as a merged PDF file. Note that the current views set by the user for all the dashboard pages are what will be exported and shown in the merged PDF file. The dashboard can also be published on PowerBI service and viewed through the web. 

