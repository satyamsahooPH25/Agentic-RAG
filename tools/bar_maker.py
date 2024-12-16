import matplotlib.pyplot as plt

def generate_bar_chart(chart_data, save_path):
    # Extract the labels and values from the "data" list in the JSON
    labels = [item["label"] for item in chart_data["data"]]
    values = [item["value"] for item in chart_data["data"]]
    
    # Set chart title and axis labels
    title = chart_data.get("title", "Bar Chart")
    x_axis_label = chart_data.get("xAxisLabel", "Categories")
    y_axis_label = chart_data.get("yAxisLabel", "Values")
    
    # Create the bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color="skyblue")
    plt.title(title)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    
    # Save the chart to the specified path
    plt.savefig(save_path)
    plt.close()
    
    return save_path