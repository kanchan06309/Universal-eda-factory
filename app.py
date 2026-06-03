import os
import io
#from flask import Flask, render_with_template, render_template, request, response, send_file
from flask import Flask, render_template, request, send_file, session
import pandas as pd
from src.analytic import StatisticalAnalyzer
from src.cleaner import ProductionPipeline

app = Flask(__name__)
app.secret_key = "google_research_shortlist_secret_key"

# Global data buffer to simulate persistent memory across route steps
SESSION_DATA = {}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("dataset")
        if file and file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            
            # Store data frame inside memory cache
            SESSION_DATA["raw_df"] = df.copy()
            
            # Run Statistical profiling
            analyzer = StatisticalAnalyzer(df)
            desc_stats = analyzer.generate_descriptive_stats()
            
            # Structural variables mapping
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
            null_counts = df.isnull().sum().to_dict()
            
            return render_template(
                "preprocess.html",
                rows=df.shape[0],
                cols=df.shape[1],
                duplicates=df.duplicated().sum(),
                preview=df.head(5).to_html(classes="table table-dark table-striped", index=False),
                stats=desc_stats,
                num_cols=num_cols,
                cat_cols=cat_cols,
                null_counts=null_counts,
                all_cols=df.columns.tolist()
            )
            
    return render_template("index.html")

@app.route("/compile", methods=["POST"])
def compile_pipeline():
    raw_df = SESSION_DATA.get("raw_df")
    if raw_df is None:
        return "Session expired. Please upload your dataset again.", 400

    num_cols = raw_df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = raw_df.select_dtypes(exclude=['number']).columns.tolist()

    pipeline = ProductionPipeline(raw_df)
    processed_df = pipeline.run_transformations(request.form, num_cols, cat_cols)

    # Save output into a temporary in-memory IO buffer for download transmission
    buffer = io.BytesIO()
    processed_df.to_csv(buffer, index=False)
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype="text/csv",
        as_attachment=True,
        download_name="pipeline_output_training_ready.csv"
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)