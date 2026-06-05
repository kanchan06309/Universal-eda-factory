import os
import io
import json
from flask import Flask, render_template, request, send_file, session
import pandas as pd
from src.analytic import StatisticalAnalyzer
from src.cleaner import ProductionPipeline

app = Flask(__name__)
app.secret_key = "google_research_shortlist_secret_key"

SESSION_DATA = {}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("dataset")
        if file and file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            SESSION_DATA["raw_df"] = df.copy()
            
            analyzer = StatisticalAnalyzer(df)
            desc_stats = analyzer.generate_descriptive_stats()
            chart_payload = analyzer.get_chart_data()
            
            schema_info = []
            total_null_values = 0
            for col in df.columns:
                col_nulls = int(df[col].isnull().sum())
                total_null_values += col_nulls
                schema_info.append({
                    "column_name": col,
                    "data_type": str(df[col].dtype),
                    "null_count": col_nulls
                })
            
            duplicate_mask = df.duplicated(keep=False)
            dup_df = df[duplicate_mask].sort_values(by=df.columns.tolist())
            dup_preview_html = dup_df.head(10).to_html(classes="preview-table", index=False) if not dup_df.empty else ""

            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
            null_counts = df.isnull().sum().to_dict()
            
            return render_template(
                "preprocess.html",
                rows=f"{df.shape[0]:,}",
                cols=df.shape[1],
                duplicates=df.duplicated().sum(),
                total_nulls=f"{total_null_values:,}",
                preview=df.head(5).to_html(classes="preview-table", index=False),
                dup_preview=dup_preview_html,
                stats=desc_stats,
                schema=schema_info,
                num_cols=num_cols,
                cat_cols=cat_cols,
                null_counts=null_counts,
                all_cols=df.columns.tolist(),
                chart_json=json.dumps(chart_payload)
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
    # Receive both the final training vector and our clean pre-encoded reference snapshot
    processed_df, pre_encoded_snapshot = pipeline.run_transformations(request.form, num_cols, cat_cols)

    # Store final compiled data output in global memory for subsequent single-click extraction downloads
    SESSION_DATA["compiled_output"] = processed_df.copy()

    # Pass rows, features metrics, and unencoded previews to user verification layout views
    return render_template(
        "preprocess.html",
        rows=f"{processed_df.shape[0]:,}",
        cols=processed_df.shape[1],
        duplicates=0, 
        total_nulls="0",
        preview=raw_df.head(5).to_html(classes="preview-table", index=False), # Reference fallback
        pre_encoded_preview=pre_encoded_snapshot.head(5).to_html(classes="preview-table", index=False), # 🌟 Pre-encoded sample view
        stats=[],
        schema=[],
        num_cols=[],
        cat_cols=[],
        null_counts={},
        all_cols=[],
        chart_json=json.dumps({"has_numeric": False}),
        show_download_modal=True # Activates dynamic tab and overlay panels focus automatically
    )

@app.route("/download")
def download_asset():
    compiled_df = SESSION_DATA.get("compiled_output")
    if compiled_df is None:
        return "Data asset matrix not found. Please compile again.", 404
        
    buffer = io.BytesIO()
    compiled_df.to_csv(buffer, index=False)
    buffer.seek(0)
    return send_file(buffer, mimetype="text/csv", as_attachment=True, download_name="engineered_pipeline_output.csv")

if __name__ == "__main__":
    app.run(debug=True, port=5000)