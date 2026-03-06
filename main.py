# =================== IMPORTS ===================
import pandas as pd
import base64
from io import BytesIO
import re

import plotly.express as px
import plotly.graph_objs as go

from dash import Dash, html, dcc, Input, Output

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns



import nltk
nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer
# NLTK imports with proper setup
import nltk
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from nltk.corpus import stopwords
except LookupError:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from nltk.corpus import stopwords

from wordcloud import WordCloud

# =================== CONFIGURATION ===================
DATA_PATH = "Final_Master_Data.csv"
REVIEWS_PATH = "74responses.txt" 

# =================== DATA LOADING ===================
try:
    df = pd.read_csv(DATA_PATH)
    print(f"✓ Data loaded successfully: {len(df)} rows")
except FileNotFoundError:
    print(f"ERROR: Could not find {DATA_PATH}")
    exit(1)

# Calculate KPIs
kpi_1 = df["ChannelPartnerID"].nunique()
kpi_2 = df["total_sales_2021"].sum()
kpi_3 = df["total_sales_2022"].sum()

# =================== MODEL DATA PREPARATION ===================
# Define features as per your notebook
feature_cols_raw = [
    'loyalty', 'nps', 'n_yrs', 'email', 'sms', 'call', 
    'brand_B1_contribution_2022', 'n_comp', 'portal', 'Region',
    'total_sales_2021', 'total_sales_2022', 'brand_B1_sales_2022',
    'buying_frequency_2022', 'brand_engagement_2022', 
    'buying_frequency_B1_2022', 'active_last_quarter', 'active_last_quarter_B1'
]
target = 'response'

# Split data first
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Process train data
processed_train_df = train_df[feature_cols_raw].copy()
processed_test_df = test_df[feature_cols_raw].copy()

# Convert 'Yes'/'No' columns to 0/1
for col in ['active_last_quarter', 'active_last_quarter_B1']:
    if col in processed_train_df.columns:
        processed_train_df[col] = processed_train_df[col].apply(lambda x: 1 if x == 'Yes' else 0)
    if col in processed_test_df.columns:
        processed_test_df[col] = processed_test_df[col].apply(lambda x: 1 if x == 'Yes' else 0)

# One-hot encode 'Region'
if 'Region' in processed_train_df.columns:
    processed_train_df = pd.get_dummies(processed_train_df, columns=['Region'], prefix='Region')
if 'Region' in processed_test_df.columns:
    processed_test_df = pd.get_dummies(processed_test_df, columns=['Region'], prefix='Region')

# Align columns
train_cols = set(processed_train_df.columns)
test_cols = set(processed_test_df.columns)

# Add missing columns
for col in train_cols - test_cols:
    processed_test_df[col] = 0
for col in test_cols - train_cols:
    processed_train_df[col] = 0

# Ensure same column order
processed_test_df = processed_test_df[processed_train_df.columns]

X_train = processed_train_df
y_train = train_df[target]
X_test = processed_test_df
y_test = test_df[target]

# =================== TRAIN MODELS ===================
print("Training models...")

# Logistic Regression (with scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

blr_model = LogisticRegression(max_iter=1000, random_state=42)
blr_model.fit(X_train_scaled, y_train)

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_leaf=20)
dt_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=150, 
    random_state=42, 
    max_features='sqrt', 
    max_depth=8, 
    min_samples_split=15,
    oob_score=True
)
rf_model.fit(X_train, y_train)

models = {
    "blr": blr_model,
    "nb": nb_model,
    "dt": dt_model,
    "rf": rf_model
}

print("✓ All models trained successfully")

# =================== RF FEATURE IMPORTANCE ===================
rf_feature_importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=False).head(15)

# =================== SENTIMENT ANALYSIS ===================
print("\nProcessing sentiment analysis...")

# Read the file content
with open(REVIEWS_PATH, 'r', encoding='utf-8') as f:
    raw_data = f.read()

print(f"✓ Loaded reviews from {REVIEWS_PATH}")

sia = SentimentIntensityAnalyzer()

# Split the raw data into individual reviews (each line is a review)
reviews_list = [entry.strip() for entry in raw_data.split('\n') if entry.strip()]

print(f"✓ Found {len(reviews_list)} reviews in file")

# Initialize the list to store data for the DataFrame
data_for_df = []

for entry in reviews_list:
    review_content = entry.strip()
    
    if not review_content:  # Skip empty lines
        continue
    
    # Calculate Sentiment
    scores = sia.polarity_scores(review_content)
    compound = scores['compound']
    
    # Categorize sentiment (matching your exact logic)
    if compound >= 0.05:
        sentiment = 'Positive'
    elif compound <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    data_for_df.append({
        'Text': review_content,
        'Score': compound,
        'Sentiment': sentiment
    })

# Create sentiment DataFrame
sentiment_df = pd.DataFrame(data_for_df)

print(f"✓ Processed {len(sentiment_df)} reviews")
print(f"  - Positive: {(sentiment_df['Sentiment'] == 'Positive').sum()} ({(sentiment_df['Sentiment'] == 'Positive').sum()/len(sentiment_df)*100:.1f}%)")
print(f"  - Negative: {(sentiment_df['Sentiment'] == 'Negative').sum()} ({(sentiment_df['Sentiment'] == 'Negative').sum()/len(sentiment_df)*100:.1f}%)")
print(f"  - Neutral: {(sentiment_df['Sentiment'] == 'Neutral').sum()} ({(sentiment_df['Sentiment'] == 'Neutral').sum()/len(sentiment_df)*100:.1f}%)")

# Prepare text for word cloud
raw_data_clean = " ".join(sentiment_df["Text"])
clean_text = re.sub(r'\[.*?\]\s*', '', raw_data_clean)

# Setup Stopwords
stop_words = set(stopwords.words('english'))
# Adding more noise words for a better cloud
stop_words.update(["coffee", "product", "tastes", "one", "try", "make", "taste"])

# Generate Word Cloud with better styling
wc = WordCloud(
    width=1000, 
    height=500,
    background_color='white',
    stopwords=stop_words,
    colormap='viridis',
    max_words=100,
    relative_scaling=0.5,
    min_font_size=10
).generate(clean_text)

# Create Word Cloud Image
plt.figure(figsize=(12, 6))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title("Customer Feedback Word Cloud", fontsize=18, fontweight='bold', pad=20, color='#08306B')
plt.tight_layout(pad=2)

# Save to buffer
wc_buf = BytesIO()
plt.savefig(wc_buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
wc_buf.seek(0)
encoded_wc = base64.b64encode(wc_buf.read()).decode()

# Calculate the sentiment distribution
sentiment_counts = sentiment_df['Sentiment'].value_counts()

# Create Pie Chart with improved styling
plt.figure(figsize=(9, 9))
colors = ['#4ECDC4', '#FF6B6B', '#FFE66D']  # Custom colors: teal, coral, yellow
explode = (0.05, 0.05, 0.05)  # Slightly separate slices

# Sort to ensure consistent order: Positive, Negative, Neutral
ordered_sentiments = ['Positive', 'Negative', 'Neutral']
ordered_counts = [sentiment_counts.get(s, 0) for s in ordered_sentiments]
ordered_colors = [colors[i] for i, s in enumerate(ordered_sentiments) if sentiment_counts.get(s, 0) > 0]
ordered_explode = [explode[i] for i, s in enumerate(ordered_sentiments) if sentiment_counts.get(s, 0) > 0]
ordered_labels = [s for s in ordered_sentiments if sentiment_counts.get(s, 0) > 0]

plt.pie(
    [sentiment_counts.get(s, 0) for s in ordered_labels],
    labels=ordered_labels, 
    autopct='%1.1f%%', 
    startangle=90, 
    colors=ordered_colors,
    explode=ordered_explode,
    shadow=True,
    textprops={'fontsize': 14, 'fontweight': 'bold'}
)
plt.title('Sentiment Distribution of Customer Reviews', fontsize=18, fontweight='bold', pad=20, color='#08306B')
plt.axis('equal')
plt.tight_layout(pad=2)

# Save to buffer
pie_buf = BytesIO()
plt.savefig(pie_buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
pie_buf.seek(0)
encoded_pie = base64.b64encode(pie_buf.read()).decode()

print("✓ Word cloud and pie chart generated")

# =================== DASH APP ===================
app = Dash(__name__, suppress_callback_exceptions=True)

kpi_style = {
    "width": "32%",
    "padding": "15px",
    "border": "2px solid #08306B",
    "borderRadius": "12px",
    "textAlign": "center",
    "backgroundColor": "white",
    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"
}

# =================== LAYOUT ===================
app.layout = html.Div([

    html.H1("FMCG Marketing Analysis", style={
        "textAlign": "center",
        "backgroundColor": "#08306B",
        "color": "white",
        "padding": "20px",
        "margin": 0,
        "fontSize": "32px",
        "fontWeight": "bold"
    }),

    html.Div([

        html.Div([
            dcc.Tabs(id="tabs", value="eda", vertical=True, children=[
                dcc.Tab(label="EDA", value="eda", style={"padding": "50px", "fontSize": "20px"}),
                dcc.Tab(label=" Model", value="model", style={"padding": "50px", "fontSize": "20px"}),
                dcc.Tab(label=" Sentiment", value="sentiment", style={"padding": "50px", "fontSize": "20px"})
            ])
        ], style={"width": "10%", "backgroundColor": "#E6F0FA"}),

        html.Div([

            html.Div(id="content-area")

        ], style={"width": "85%", "padding": "20px", "backgroundColor": "#f8f9fa"})

    ], style={"display": "flex", "height": "calc(100vh - 50px)"})
])

# =================== TAB CONTENT ===================
@app.callback(Output("content-area", "children"), Input("tabs", "value"))
def render_tab(tab):

    if tab == "eda":
        return html.Div([
            
            # Top 3 KPIs for EDA
            html.Div([
                html.Div([html.H2("Channel Partners"), html.H2(kpi_1)], style=kpi_style),
                html.Div([html.H2("Total Sales 2021"), html.H2(f"₹{kpi_2:,.0f}")], style=kpi_style),
                html.Div([html.H2("Total Sales 2022"), html.H2(f"₹{kpi_3:,.0f}")], style=kpi_style)
            ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "20px"}),

            # Year Slicer
            html.Div([
                html.Label("Select Year:", 
                          style={"fontWeight": "bold", "fontSize": "16px", "marginRight": "10px"}),
                dcc.Dropdown(
                    id="year_dd",
                    options=[
                        {"label": "2021", "value": "2021"},
                        {"label": "2022", "value": "2022"}
                    ],
                    value="2022",
                    style={"width": "200px"}
                )
            ], style={"marginBottom": "20px", "padding": "15px", "backgroundColor": "#E6F0FA", "borderRadius": "8px"}),

            # ROW 1: Total Sales, Sales Distribution, Regional Sales
            html.Div([
                dcc.Graph(id="total_sales_box", style={"width": "33.33%"}),
                dcc.Graph(id="sales_distribution", style={"width": "33.33%"}),
                dcc.Graph(id="regional_sales", style={"width": "33.33%"})
            ], style={"display": "flex"}),

            # ROW 2: Buying Frequency, NPS, Complaints
            html.Div([
                dcc.Graph(id="buying_frequency_bar", style={"width": "33.33%"}),
                dcc.Graph(id="nps_box", style={"width": "33.33%"}),
                dcc.Graph(id="complaints_box", style={"width": "33.33%"})
            ], style={"display": "flex"})
        ])

    if tab == "model":
        return html.Div([
            
            # Top 3 KPIs for Model
            html.Div([
                html.Div([html.H2("Channel Partners"), html.H2(kpi_1)], style=kpi_style),
                html.Div([html.H2("Total Sales 2021"), html.H2(f"₹{kpi_2:,.0f}")], style=kpi_style),
                html.Div([html.H2("Total Sales 2022"), html.H2(f"₹{kpi_3:,.0f}")], style=kpi_style)
            ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "20px"}),
            
            # Model Selector
            html.Div([
                html.Label("Select Model:", style={"fontWeight": "bold", "marginRight": "10px"}),
                dcc.Dropdown(
                    id="model_dd",
                    options=[
                        {"label": "Logistic Regression", "value": "blr"},
                        {"label": "Naive Bayes", "value": "nb"},
                        {"label": "Decision Tree", "value": "dt"},
                        {"label": "Random Forest", "value": "rf"}
                    ],
                    placeholder="Select Model",
                    value=None,
                    style={"width": "300px"}
                )
            ], style={"marginBottom": "20px", "padding": "15px", "backgroundColor": "#E6F0FA", "borderRadius": "8px"}),
            
            html.H4(id="model_header", style={"color": "#08306B"}),
            
            # 3 Charts in a row
            html.Div([
                dcc.Graph(id="roc", style={"width": "33%"}),
                dcc.Graph(id="cm", style={"width": "33%"}),
                html.Div(id="third_chart", style={"width": "33%"})
            ], style={"display": "flex"}),
            
            # Decision Tree Visualization (will be empty for other models)
            html.Div(id="dt_viz_container", style={"marginTop": "30px"})
        ])

    # Sentiment tab - WITHOUT top 3 KPIs
    return html.Div([
        html.H3("Customer Sentiment Analysis", 
                style={"textAlign": "center", "color": "#08306B", "marginBottom": "40px", 
                       "fontSize": "28px", "fontWeight": "bold"}),
        
        # Only Sentiment Summary Stats Row (no top KPIs)
        html.Div([
            html.Div([
                html.H5("Total Reviews", style={"color": "#666", "marginBottom": "10px"}),
                html.H2(f"{len(sentiment_df)}", style={"color": "#08306B", "margin": "5px 0", "fontSize": "36px"})
            ], style={**kpi_style, "backgroundColor": "#f0f8ff"}),
            html.Div([
                html.H5("Positive Reviews", style={"color": "#666", "marginBottom": "10px"}),
                html.H2(f"{sentiment_counts.get('Positive', 0)}", 
                       style={"color": "#4ECDC4", "margin": "5px 0", "fontSize": "36px"})
            ], style={**kpi_style, "backgroundColor": "#e0f7f4"}),
            html.Div([
                html.H5("Negative Reviews", style={"color": "#666", "marginBottom": "10px"}),
                html.H2(f"{sentiment_counts.get('Negative', 0)}", 
                       style={"color": "#FF6B6B", "margin": "5px 0", "fontSize": "36px"})
            ], style={**kpi_style, "backgroundColor": "#ffe6e6"})
        ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "40px"}),
        
        # Both charts in a single row
        html.Div([
            # Word Cloud
            html.Div([
                html.H4("Customer Feedback Word Cloud", 
                       style={"textAlign": "center", "color": "#08306B", "marginBottom": "20px",
                              "fontSize": "20px", "fontWeight": "bold"}),
                html.Div([
                    html.Img(src="data:image/png;base64," + encoded_wc, 
                            style={"width": "100%", "maxWidth": "100%", "display": "block",
                                   "borderRadius": "8px"})
                ], style={"border": "3px solid #08306B", "borderRadius": "12px", "padding": "15px",
                         "backgroundColor": "white", "boxShadow": "0 4px 6px rgba(0,0,0,0.1)"})
            ], style={"width": "50%", "padding": "0 15px"}),
            
            # Sentiment Pie Chart
            html.Div([
                html.H4("Sentiment Distribution", 
                       style={"textAlign": "center", "color": "#08306B", "marginBottom": "20px",
                              "fontSize": "20px", "fontWeight": "bold"}),
                html.Div([
                    html.Img(src="data:image/png;base64," + encoded_pie, 
                            style={"width": "100%", "maxWidth": "100%", "display": "block",
                                   "borderRadius": "8px"})
                ], style={"border": "3px solid #08306B", "borderRadius": "12px", "padding": "15px",
                         "backgroundColor": "white", "boxShadow": "0 4px 6px rgba(0,0,0,0.1)"})
            ], style={"width": "50%", "padding": "0 15px"})
            
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start"})
    ])

# =================== EDA CALLBACK ===================
@app.callback(
    [
        Output("total_sales_box", "figure"),
        Output("sales_distribution", "figure"),
        Output("regional_sales", "figure"),
        Output("buying_frequency_bar", "figure"),
        Output("nps_box", "figure"),
        Output("complaints_box", "figure")
    ],
    Input("year_dd", "value")
)
def update_eda(year):

    # ROW 1 - Chart 1: Total Sales by Response
    sales_col = f"total_sales_{year}"
    fig1 = px.box(
        df, 
        x="response", 
        y=sales_col,
        title=f"Total Sales {year} by Response",
        color="response",
        color_discrete_map={0: "#FF6B6B", 1: "#4ECDC4"},
        labels={"response": "Response", sales_col: f"Total Sales {year}"}
    )
    fig1.update_layout(showlegend=False)

    # ROW 1 - Chart 2: Sales Distribution
    df_hist = df.copy()
    df_hist["response"] = df_hist["response"].astype(str)
    fig2 = px.histogram(
        df_hist, 
        x=sales_col, 
        color="response",
        title=f"Sales Distribution {year}",
        nbins=30,
        barmode="overlay",
        opacity=0.7,
        color_discrete_map={"0": "#FF6B6B", "1": "#4ECDC4"},
        category_orders={"response": ["1", "0"]},
        labels={sales_col: f"Total Sales {year}", "response": "response", "count": "Count"}
    )
    fig2.update_layout(yaxis_title="Count", xaxis_title=f"Total Sales {year}")

    # ROW 1 - Chart 3: Total Sales by Region
    region_sales = df.groupby("Region")[sales_col].sum().reset_index()
    fig3 = px.bar(
        region_sales, 
        x="Region", 
        y=sales_col,
        title=f"Total Sales by Region ({year})",
        color="Region",
        labels={"Region": "Region", sales_col: f"Total Sales {year}"}
    )
    fig3.update_layout(showlegend=False)

    # ROW 2 - Chart 4: Mean Buying Frequency 2022
    freq_data = df.groupby("response")["buying_frequency_2022"].mean().reset_index()
    fig4 = px.bar(
        freq_data, 
        x="response", 
        y="buying_frequency_2022",
        title="Mean Buying Frequency 2022 by Response",
        color="response",
        color_discrete_map={0: "#FF6B6B", 1: "#4ECDC4"},
        labels={"response": "Response", "buying_frequency_2022": "Mean Buying Frequency"}
    )
    fig4.update_layout(showlegend=False, yaxis_title="Mean Buying Frequency")

    # ROW 2 - Chart 5: NPS by Response
    fig5 = px.box(
        df, 
        x="response", 
        y="nps",
        title="NPS by Response",
        color="response",
        color_discrete_map={0: "#FF6B6B", 1: "#4ECDC4"},
        labels={"response": "Response", "nps": "NPS"}
    )
    fig5.update_layout(showlegend=False)

    # ROW 2 - Chart 6: Number of Complaints by Response
    fig6 = px.box(
        df, 
        x="response", 
        y="n_comp",
        title="Number of Complaints by Response",
        color="response",
        color_discrete_map={0: "#FF6B6B", 1: "#4ECDC4"},
        labels={"response": "Response", "n_comp": "Number of Complaints"}
    )
    fig6.update_layout(showlegend=False)

    return fig1, fig2, fig3, fig4, fig5, fig6

# =================== MODEL CALLBACK ===================
@app.callback(
    [
        Output("roc", "figure"),
        Output("model_header", "children"),
        Output("cm", "figure"),
        Output("third_chart", "children"),
        Output("dt_viz_container", "children")
    ],
    Input("model_dd", "value")
)
def update_model(model_key):

    if not model_key:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Select a model to view results",
            xaxis={"visible": False},
            yaxis={"visible": False}
        )
        return empty_fig, "Select a model", empty_fig, html.Div(), html.Div()

    model = models[model_key]
    
    model_names = {
        "blr": "Logistic Regression",
        "nb": "Naive Bayes",
        "dt": "Decision Tree",
        "rf": "Random Forest"
    }

    # Get predictions
    if model_key == "blr":
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_prob = model.predict_proba(X_test)[:, 1]

    y_pred = (y_prob >= 0.5).astype(int)

    # Calculate metrics
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_test, y_pred)

    # ROC Curve
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(
        x=fpr, y=tpr, 
        mode='lines',
        name=f"AUC={roc_auc:.3f}",
        line=dict(color='#08306B', width=2)
    ))
    roc_fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], 
        mode='lines',
        line=dict(dash="dash", color='gray'),
        name="Random"
    ))
    roc_fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        showlegend=True
    )

    # Confusion Matrix
    cm_fig = px.imshow(
        cm,
        text_auto=True,
        title="Confusion Matrix",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['No Response', 'Response'],
        y=['No Response', 'Response'],
        color_continuous_scale='Blues'
    )

    # Model Header
    header = f"Model: {model_names[model_key].upper()} | AUC Score: {roc_auc:.4f}"

    # Third Chart - Probability Predictions Table (First 5 samples)
    pred_df = pd.DataFrame({
        'Sample': range(1, 6),
        'Actual Response': y_test.iloc[:5].values,
        'Predicted Probability': y_prob[:5].round(4),
        'Predicted Class': y_pred[:5]
    })
    
    prob_table_fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Sample</b>', '<b>Actual Response</b>', '<b>Predicted Probability</b>', '<b>Predicted Class</b>'],
            fill_color='#08306B',
            font=dict(color='white', size=14),
            align='center',
            height=40
        ),
        cells=dict(
            values=[pred_df['Sample'], pred_df['Actual Response'], 
                   pred_df['Predicted Probability'], pred_df['Predicted Class']],
            fill_color='#E6F0FA',
            align='center',
            font=dict(size=13),
            height=35
        )
    )])
    
    prob_table_fig.update_layout(
        title="First 5 Test Samples with Predictions",
        height=400,
        margin=dict(t=50, b=20, l=20, r=20)
    )
    
    third_chart = dcc.Graph(figure=prob_table_fig)

    # Additional Visualization Container
    additional_viz = html.Div()
    
    if model_key == "dt":
        # Decision Tree Visualization
        plt.figure(figsize=(24, 12))
        plot_tree(
            dt_model,
            feature_names=X_train.columns,
            class_names=['No Response', 'Response'],
            filled=True,
            rounded=True,
            fontsize=9
        )
        plt.title('Decision Tree Visualization (max_depth = 5)', fontsize=16, pad=20)
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        dt_img_base64 = base64.b64encode(buf.read()).decode()
        
        additional_viz = html.Div([
            html.H3("Decision Tree Visualization", style={"textAlign": "center", "color": "#08306B"}),
            html.Img(
                src=f"data:image/png;base64,{dt_img_base64}",
                style={"width": "100%", "maxWidth": "1400px", "display": "block", "margin": "0 auto"}
            )
        ])
    
    elif model_key == "rf":
        # Random Forest Feature Importance
        rf_feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=rf_feature_importance.head(15),
            x='Importance',
            y='Feature',
            palette='viridis'
        )
        plt.title('Top 15 Feature Importances - Random Forest', fontsize=16, pad=20)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        rf_img_base64 = base64.b64encode(buf.read()).decode()
        
        additional_viz = html.Div([
            html.H3("Random Forest Feature Importance", style={"textAlign": "center", "color": "#08306B"}),
            html.Img(
                src=f"data:image/png;base64,{rf_img_base64}",
                style={"width": "100%", "maxWidth": "1000px", "display": "block", "margin": "0 auto"}
            )
        ])

    return roc_fig, header, cm_fig, third_chart, additional_viz

# =================== RUN ===================
# =================== SERVER FOR RENDER ===================
server = app.server

# =================== RUN ===================
if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 8050))

    app.run(host="0.0.0.0", port=port)


