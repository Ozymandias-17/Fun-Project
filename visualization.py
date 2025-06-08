import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# Define and set the custom Plotly template
pio.templates["plotly_dark_biedge"] = go.layout.Template(
    layout=dict(
        title=dict(font=dict(color="white")),
        xaxis=dict(color="#D9CAB3", gridcolor="#736E66", titlefont=dict(color="#D9CAB3")), 
        yaxis=dict(color="#D9CAB3", gridcolor="#736E66", titlefont=dict(color="#D9CAB3")), 
        plot_bgcolor="#2D2A26",
        paper_bgcolor="#2D2A26",
        font=dict(color="#D9CAB3"), 
        legend=dict(font=dict(color="#D9CAB3"))))

pio.templates.default = "plotly_dark_biedge"


# Distribution of views
def views_distribution(df):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.Date, y=df["Views_Count"], marker_color="#99A3B8", name="Views"))
    fig.update_layout(
        title_text="Distribution of views",
        title_font_size=16)
    fig.update_layout(width=1200, height=600) 
    fig.update_xaxes(nticks=15)
    fig.write_image("Views_distribution.png")
    

# Toxicity and emotions
def toxic_and_emotion_stat(df, object_name):
    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=(f"Toxicity of {object_name}",
                                        f"Emotions of {object_name}",
                                        f"Tonality of {object_name}"),
                        vertical_spacing=0.1) 

    toxicity_counts = df["Toxicity"].value_counts()

    toxicity_colors = {"non toxic": "#6B8E7A",
                       "toxic": "#2A6478"}

    fig.add_trace(go.Bar(x=toxicity_counts.index, y=toxicity_counts.values,
                         marker_color=[toxicity_colors[col] for col in toxicity_counts.index], width=0.3, name="Toxicity"),
                  row=1, col=1)

    emotion_counts = df["Prior Emotion"].value_counts()
    fig.add_trace(go.Bar(x=emotion_counts.index, y=emotion_counts.values,
                         marker_color="palevioletred", name="Emotions"),
                  row=2, col=1)

    tonality_means = df[["Neutral", "Negative", "Positive"]].mean()

    tonality_colors = {"Neutral": "slategrey",
                       "Negative": "#C47451",
                       "Positive": "teal"}

    fig.add_trace(go.Bar(x=tonality_means.index, y=tonality_means.values,
                         marker_color=[tonality_colors[col] for col in tonality_means.index], width=0.4, name="Tonality"), row=3, col=1)

    fig.update_layout(height=700, width=600, showlegend=False) 
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_yaxes(title_text="Mean Coefficient", row=3, col=1)

    for annotation in fig.layout.annotations:
        annotation.font.size = 13

    fig.write_image("Result.png")


# Most active commentators
def top_commentators(data, how_many=15):
    top_comm = data["Username"].value_counts().nlargest(how_many)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=top_comm.index, y=top_comm.values, marker_color="goldenrod", name="Comments"))
    fig.update_layout(
        title_text="Most active commentators",
        title_font_size=16,
        xaxis_tickangle=-55,
        yaxis_title="Number of Comments",
        xaxis_tickfont_size=11, 
        yaxis_tickfont_size=11)
    
    fig.update_layout(width=1000, height=550) 
    fig.write_image("Top_commentators.png")