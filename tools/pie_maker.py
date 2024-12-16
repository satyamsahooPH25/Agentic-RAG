import matplotlib.pyplot as plt

def generate_pie_chart(chart_data, save_path):
    labels = chart_data["labels"]
    values = chart_data["values"]
    colors = chart_data.get("colors", None)
    title = chart_data.get("title", "Pie Chart")
    legend = chart_data.get("legend", {})

    plt.figure(figsize=(8, 5))
    plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
    plt.title(title)
    
    if legend.get("display", False):
        plt.legend(labels=labels, loc=legend.get("position", "best"))

    plt.savefig(save_path)
    plt.close()
    return save_path