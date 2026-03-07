from flask import Flask, request, jsonify
import os
import re
from dotenv import load_dotenv
from supabase import create_client, Client
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "https://buildxdesigner.site", "https://www.buildxdesigner.site"])
url: str | None = os.getenv("SUPABASE_URL")
key: str | None = os.getenv("SUPABASE_KEY")

if not url or not key:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in .env")

supabase: Client = create_client(url, key)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def normalize_text(value: object) -> str:
    """Normalize text for vectorization."""
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_onboarding_profile_text(onboarding_row: dict) -> str:
    fields = [
        onboarding_row.get("primary_role", ""),
        onboarding_row.get("workplace_type", ""),
        onboarding_row.get("experience", ""),
        onboarding_row.get("main_goal", ""),
        onboarding_row.get("team_size", ""),
    ]
    return normalize_text(" ".join(str(v) for v in fields if v is not None))


def fetch_like_counts(project_ids: list[str]) -> dict[str, int]:
    """Fetch like counts keyed by project_id from template_interactions."""
    if not project_ids:
        return {}

    unique_project_ids = list(dict.fromkeys(project_ids))
    counts: dict[str, int] = {project_id: 0 for project_id in unique_project_ids}

    for project_id in unique_project_ids:
        try:
            # JS equivalent:
            # .from('template_interactions').select('*', { count: 'exact', head: true })
            resp = (
                supabase
                .table("template_interactions")
                .select("*", count="exact", head=True)
                .eq("project_id", project_id)
                .execute()
            )
            counts[project_id] = int(resp.count or 0)
        except Exception:
            counts[project_id] = 0

    return counts


@app.route("/", methods=["GET"])
@app.route("/recommendations", methods=["GET"])
def recommend_projects():
    user_id = request.args.get("user_id", type=str)
    if not user_id:
        return jsonify({"error": "Query param 'user_id' is required."}), 400

    limit = request.args.get("limit", default=10, type=int)
    limit = max(1, min(limit, 50))

    onboarding_resp = (
        supabase
        .table("onboarding_data")
        .select("user_id, primary_role, workplace_type, experience, main_goal, team_size")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )

    onboarding_rows = onboarding_resp.data or []
    if not onboarding_rows:
        return jsonify({"error": f"No onboarding_data found for user_id '{user_id}'."}), 404

    onboarding_profile = build_onboarding_profile_text(onboarding_rows[0])
    if not onboarding_profile:
        return jsonify({"error": "Onboarding profile is empty for this user."}), 400

    templates_resp = (
        supabase
        .table("published_templates")
        .select("project_id, user_id, profiles(user_id, avatar_url, full_name), projects(projects_id, description, category, user_id, project_name, thumbnail)")
        .execute()
    )

    templates = templates_resp.data or []
    if not templates:
        return jsonify({"error": "No published templates found."}), 404

    corpus = []
    project_rows = []
    for template in templates:
        projects_data = template.get("projects")
        profiles_data = template.get("profiles")
        
        if not projects_data or not isinstance(projects_data, dict):
            continue
        
        project_description = normalize_text(projects_data.get("description", ""))
        project_category = normalize_text(projects_data.get("category", ""))
        project_text = normalize_text(f"{project_category} {project_description}")

        if not project_text:
            continue

        enriched_project = {
            **template,
            "projects": projects_data,
            "profiles": profiles_data,
            "project_description": project_description,
            "project_category": project_category,
        }
        corpus.append(project_text)
        project_rows.append(enriched_project)

    if not corpus:
        return jsonify({"error": "No usable project documents after preprocessing."}), 404

    like_counts = fetch_like_counts(
        [str(row.get("project_id")) for row in project_rows if row.get("project_id")]
    )

    user_embedding = embedding_model.encode([onboarding_profile])
    project_embeddings = embedding_model.encode(corpus)
    scores = cosine_similarity(user_embedding, project_embeddings).flatten()

    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    ranked = [(idx, score) for idx, score in ranked if score > 0][:limit]

    recommendations = []
    for idx, score in ranked:
        template = project_rows[idx]
        projects_data = template.get("projects", {})
        profiles_data = template.get("profiles", {})
        
        recommendations.append(
            {
                "project_id": template.get("project_id"),
                "like_count": like_counts.get(str(template.get("project_id")), 0),
                "template_user_id": template.get("user_id"),
                "author": {
                    "user_id": profiles_data.get("user_id"),
                    "full_name": profiles_data.get("full_name"),
                    "avatar_url": profiles_data.get("avatar_url"),
                },
                "project": {
                    "projects_id": projects_data.get("projects_id"),
                    "project_name": projects_data.get("project_name"),
                    "description": projects_data.get("description"),
                    "category": projects_data.get("category"),
                    "thumbnail": projects_data.get("thumbnail"),
                },
                "similarity_score": float(score),
            }
        )

    if not recommendations:
        return jsonify(
            {
                "user_id": user_id,
                "onboarding_profile": onboarding_profile,
                "message": "No recommendations found with positive similarity.",
                "recommendations": [],
            }
        ), 200

    return jsonify(
        {
            "user_id": user_id,
            "onboarding_profile": onboarding_profile,
            "recommendation_count": len(recommendations),
            "recommendations": recommendations,
        }
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
