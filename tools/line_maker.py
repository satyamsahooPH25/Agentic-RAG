import matplotlib.pyplot as plt

def generate_line_chart(chart_data, save_path):
    try:
        # Extract data for the primary chart
        data = chart_data.get("data", [])
        if not data:
            raise ValueError("The 'data' field is missing or empty.")

        labels = [entry["label"] for entry in data]
        values = [entry["value"] for entry in data]
        
        # Extract metadata for the chart
        title = chart_data.get("title", "Line Chart")
        x_axis_label = chart_data.get("xAxisLabel", "X-Axis")
        y_axis_label = chart_data.get("yAxisLabel", "Y-Axis")
        
        # Create the line chart
        plt.figure(figsize=(8, 5))
        plt.plot(labels, values, marker="o", color="blue")
        plt.title(title)
        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)
        plt.grid()
        
        # Save the chart to the specified path
        plt.savefig(save_path)
        plt.close()
        return save_path
    except Exception as e:
        print(f"Error generating chart: {e}")
        return None