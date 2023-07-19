import pandas as pd
import plotly.subplots as sp
import plotly.graph_objects as go

# Read the CSV file
data = pd.read_csv('../gensim_results.csv')

# Extract the values from the DataFrame
topics_number = data['Number of Topics']
perplexity = data['Perplexity']
umass = data['Umass']
cv = data['C_v']
cuci = data['C_uci']
cnpmi = data['C_npmi']
topic_diversity = data['Topic Diversity']

# Create subplots
fig = sp.make_subplots(rows=2, cols=3, subplot_titles=("Perplexity", "UMass", "Cv", "Cuci", "Cnpmi", "Topic Diversity"))

# Add traces to subplots
fig.add_trace(go.Scatter(x=topics_number, y=perplexity, mode='lines', name='Perplexity'), row=1, col=1)
fig.add_trace(go.Scatter(x=topics_number, y=umass, mode='lines', name='UMass'), row=1, col=2)
fig.add_trace(go.Scatter(x=topics_number, y=cv, mode='lines', name='Cv'), row=1, col=3)
fig.add_trace(go.Scatter(x=topics_number, y=cuci, mode='lines', name='Cuci'), row=2, col=1)
fig.add_trace(go.Scatter(x=topics_number, y=cnpmi, mode='lines', name='Cnpmi'), row=2, col=2)
fig.add_trace(go.Scatter(x=topics_number, y=topic_diversity, mode='lines', name='Topic Diversity'), row=2, col=3)

# Update layout
fig.update_layout(title='',
                  showlegend=True)

# Show the figure
fig.show()

