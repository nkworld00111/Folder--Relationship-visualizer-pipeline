
import os
import json
import html
import math
import pathlib
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def _safe(s: Any) -> str:
    if s is None:
        return ""
    return html.escape(str(s))


def _file_times(path: str):
    try:
        mtime = os.path.getmtime(path)
        ctime = os.path.getctime(path)
        return (
            datetime.fromtimestamp(ctime).strftime("%Y-%m-%d %H:%M:%S"),
            datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S"),
        )
    except Exception:
        return ("N/A", "N/A")


def _word_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def _top_keywords_per_doc(corpus: List[str], top_k: int = 8):
    if len(corpus) == 0:
        return [[] for _ in range(0)], None, None
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    X = tfidf.fit_transform(corpus)
    kws = []
    vocab = np.array(tfidf.get_feature_names_out())
    for i in range(X.shape[0]):
        row = X.getrow(i)
        if row.nnz == 0:
            kws.append([])
            continue
        top_idx = np.argsort(row.toarray()[0])[::-1][:top_k]
        kws.append([vocab[j] for j in top_idx if row[0, j] > 0])
    return kws, tfidf, X


def _keyword_frequency(text: str, keywords: List[str]) -> int:
    if not text or not keywords:
        return 0
    low = text.lower()
    total = 0
    for k in keywords:
        if not k:
            continue
        total += low.count(k.lower())
    return total


def _color_by_type(file_type: str) -> str:
    if not file_type:
        return "#6b7280"
    ft = str(file_type).lower()
    if "pdf" in ft:
        return "#e11d48"
    if "csv" in ft:
        return "#2563eb"
    if "word" in ft or "doc" in ft:
        return "#16a34a"
    if "text" in ft or "txt" in ft:
        return "#f59e0b"
    if "json" in ft:
        return "#8b5cf6"
    if "png" in ft or "jpg" in ft or "jpeg" in ft:
        return "#06b6d4"
    return "#64748b"


def _edge_style(relation: str, score: float, shared_kw: int, same_type: bool):
    colors = {"similarity": "#38bdf8", "keywords": "#f59e0b", "meta": "#a78bfa"}
    color = colors.get(relation, "#94a3b8")
    base = 1.0
    try:
        if relation == "similarity":
            width = base + (max(0.0, min(1.0, float(score))) * 8.0)
        elif relation == "keywords":
            width = base + min(8.0, float(shared_kw or 0))
        else:
            width = base + (3.0 if same_type else 1.0)
    except Exception:
        width = base + (3.0 if same_type else 1.0)
    return color, max(1.0, width)


def _to_plain(o):
    if o is None:
        return None
    if isinstance(o, (np.ndarray, list, tuple)):
        try:
            return np.array(o).astype(float).tolist()
        except Exception:
            return [str(x) for x in o]
    if isinstance(o, (np.floating, np.float32, np.float64)):
        return float(o)
    if isinstance(o, (np.integer, )):
        return int(o)
    return o


def create_visualizations(docs: List[Dict], out_path: str = os.path.join(OUT_DIR, "folder_file_dashboard_v3.html"),
                          sim_threshold: float = 0.35, min_shared_keywords: int = 2):
   
    entries = []
    corpus = []
    for idx, d in enumerate(docs):
        text = d.get("text") or d.get("content") or ""
        meta = d.get("metadata", {}) or {}
        file_path = meta.get("filepath") or meta.get("file_path") or d.get("filepath") or d.get("path") or ""
        folder = meta.get("folder") or meta.get("dir") or (str(pathlib.Path(file_path).parent) if file_path else "Unknown")
        filename = meta.get("filename") or meta.get("file_name") or d.get("title") or os.path.basename(file_path) or f"Document_{idx+1}"
        ext = meta.get("ext") or meta.get("file_type") or (pathlib.Path(filename).suffix if filename else "")
        size_kb = meta.get("size_kb")
        if size_kb is None:
            try:
                if meta.get("size_bytes"):
                    size_kb = round(meta.get("size_bytes") / 1024.0, 2)
                elif file_path and os.path.exists(file_path):
                    size_kb = round(os.path.getsize(file_path) / 1024.0, 2)
                else:
                    size_kb = "N/A"
            except Exception:
                size_kb = "N/A"
        created_at = meta.get("created_at") or ( _file_times(file_path)[0] if file_path and os.path.exists(file_path) else "N/A")
        modified_at = meta.get("modified_at") or ( _file_times(file_path)[1] if file_path and os.path.exists(file_path) else "N/A")
        words = _word_count(text)
        emb = d.get("embedding")
        emb_arr = None
        if isinstance(emb, list):
            try:
                emb_arr = np.array(emb, dtype=float)
            except Exception:
                emb_arr = None
        elif isinstance(emb, np.ndarray):
            try:
                emb_arr = emb.astype(float)
            except Exception:
                emb_arr = None

        entry = {
            "id": str(idx + 1),
            "title": str(filename),
            "folder": str(folder),
            "ext": str(ext),
            "size_kb": size_kb,
            "size_bytes": meta.get("size_bytes", None),
            "created_at": created_at,
            "modified_at": modified_at,
            "path": str(file_path),
            "text": text,
            "words": words,
            "metadata": meta,
            "embedding": emb_arr
        }
        entries.append(entry)
        corpus.append(text)

    if not entries:
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("<html><body><h3>No documents to visualize.</h3></body></html>")
        return {"dashboard": out_path}

    kw_per_doc, tfidf_model, tfidf_matrix = _top_keywords_per_doc(corpus, top_k=10)
    for i, e in enumerate(entries):
        e["keywords"] = kw_per_doc[i] if i < len(kw_per_doc) else []

    folders = {}
    for e in entries:
        fpath = e["folder"] or "Unknown"
        if fpath not in folders:
            folders[fpath] = {"docs": [], "words": 0, "keywords": [], "embs": [], "size_bytes": 0}
        folders[fpath]["docs"].append(e)
        folders[fpath]["words"] += e["words"]
        folders[fpath]["keywords"].extend(e.get("keywords", []))
        if e.get("embedding") is not None:
            folders[fpath]["embs"].append(e.get("embedding"))
        if isinstance(e.get("size_bytes"), (int, float)):
            folders[fpath]["size_bytes"] += e["size_bytes"]

    folder_list = []
    for i, (fpath, info) in enumerate(folders.items()):
        emb = None
        if info["embs"]:
            try:
                M = np.vstack(info["embs"])
                norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
                Mn = M / norms
                mean = Mn.mean(axis=0)
                emb = mean / (np.linalg.norm(mean) + 1e-12)
            except Exception:
                emb = None
        folder_list.append({
            "id": f"F{i+1}",
            "path": fpath,
            "title": os.path.basename(fpath) or fpath,
            "words": info["words"],
            "keywords": list(dict.fromkeys(info["keywords"]))[:12],
            "embedding": emb,
            "doc_ids": [d["id"] for d in info["docs"]],
            "file_count": len(info["docs"]),
            "size_bytes": info.get("size_bytes", None)
        })

    use_folder_embed = all(f["embedding"] is not None for f in folder_list) and len(folder_list) > 1
    if use_folder_embed:
        M = np.vstack([f["embedding"] for f in folder_list])
        norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
        Mn = M / norms
        folder_sim = Mn @ Mn.T
    else:
        folder_corpus = ["\n".join([d["text"] for d in folders[f["path"]]["docs"]]) if folders.get(f["path"]) else "" for f in folder_list]
        if any(folder_corpus):
            tf = TfidfVectorizer(stop_words="english", max_features=5000)
            Xf = tf.fit_transform(folder_corpus)
            folder_sim = cosine_similarity(Xf)
        else:
            folder_sim = np.eye(len(folder_list))

    nodes = []
    for f in folder_list:
        size_value = max(18, math.sqrt(max(1, f["words"])) * 0.12)  
        nodes.append({
            "id": f["id"],
            "label": f["title"],
            "title": f"<b>{_safe(f['title'])}</b><br><b>Files:</b> {f['file_count']}<br><b>Words:</b> {f['words']}",
            "color": {"background": "#0ea5a4", "border": "#056d67"},
            "group": "folder",
            "value": size_value
        })
    for e in entries:
        color = _color_by_type(e["ext"])
        size_value = max(6, math.sqrt(max(1, e["words"])) * 0.08)
        nodes.append({
            "id": e["id"],
            "label": e["title"],
            "title": f"<b>{_safe(e['title'])}</b><br><b>Folder:</b> {_safe(e['folder'])}<br><b>Words:</b> {e['words']}",
            "color": {"background": color, "border": "#111827"},
            "group": "file",
            "value": size_value,
            "folder": e["folder"]
        })

    edges = []
    for i in range(len(folder_list)):
        for j in range(i + 1, len(folder_list)):
            try:
                s = float(folder_sim[i, j])
            except Exception:
                s = 0.0
            if s >= sim_threshold:
                color, width = _edge_style("similarity", s, 0, True)
                title = f"<b>Relation:</b> folder-similarity<br><b>Score:</b> {s:.2f}"
                edges.append({
                    "from": folder_list[i]["id"],
                    "to": folder_list[j]["id"],
                    "color": {"color": color},
                    "width": width,
                    "title": title
                })

    for f in folder_list:
        for doc_id in f.get("doc_ids", []):
            edges.append({
                "from": f["id"],
                "to": doc_id,
                "color": {"color": "#94a3b8"},
                "width": 1.0,
                "title": f"contains {doc_id}"
            })

    doc_use_embed = all(e["embedding"] is not None for e in entries) and len(entries) > 1
    doc_sim = None
    if doc_use_embed:
        Mdoc = np.vstack([e["embedding"] for e in entries])
        norms = np.linalg.norm(Mdoc, axis=1, keepdims=True) + 1e-12
        Mdn = Mdoc / norms
        doc_sim = Mdn @ Mdn.T
    else:
        if tfidf_matrix is not None and len(entries) > 1:
            doc_sim = cosine_similarity(tfidf_matrix)
        else:
            doc_sim = None

    if doc_sim is not None:
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                try:
                    s = float(doc_sim[i, j])
                except Exception:
                    s = 0.0
                shared_kw = len(set(entries[i].get("keywords", [])) & set(entries[j].get("keywords", [])))
                if (s >= 0.85) or (shared_kw >= max(3, min_shared_keywords) and s >= 0.5):
                    relation = "similarity" if s >= 0.85 else "keywords"
                    color, width = _edge_style(relation, s, shared_kw, entries[i]["ext"] == entries[j]["ext"])
                    reasons = []
                    if s >= sim_threshold:
                        reasons.append(f"sim={s:.2f}")
                    if shared_kw:
                        reasons.append(f"shared_kw={shared_kw}")
                    if entries[i]["ext"] == entries[j]["ext"]:
                        reasons.append("same_type")
                    title = f"<b>Relation:</b> doc-{relation}<br><b>Reasons:</b> {', '.join(reasons)}"
                    edges.append({
                        "from": entries[i]["id"],
                        "to": entries[j]["id"],
                        "color": {"color": color},
                        "width": width,
                        "title": title
                    })

    degree = {n["id"]: 0 for n in nodes}
    for ed in edges:
        if ed["from"] in degree:
            degree[ed["from"]] += 1
        if ed["to"] in degree:
            degree[ed["to"]] += 1

    x_labels = [f["title"] for f in folder_list]
    bar_values = [f["words"] for f in folder_list]
    top_folder_keywords = []
    for f in folder_list:
        top_folder_keywords.extend(f["keywords"])
    top_folder_keywords = list(dict.fromkeys(top_folder_keywords))[:6]
    line_values = [_keyword_frequency("\n".join([d["text"] for d in folders[f["path"]]["docs"]]) if folders.get(f["path"]) else "", top_folder_keywords) for f in folder_list]

    details = []
    for f in folder_list:
        details.append({
            "id": f["id"],
            "title": f["title"],
            "type": "folder",
            "path": f["path"],
            "words": f["words"],
            "file_count": f["file_count"],
            "size_bytes": f.get("size_bytes"),
            "size_kb": (round(f.get("size_bytes", 0) / 1024.0, 2) if f.get("size_bytes") else "N/A"),
            "keywords": f["keywords"],
            "metadata": {
                "folder_path": f["path"],
                "file_count": f["file_count"],
                "top_keywords": f["keywords"]
            },
            "degree": degree.get(f["id"], 0),
            "preview": (f.get("title") or "")
        })
    for e in entries:
        file_meta = {
            "filename": e["title"],
            "filepath": e["path"],
            "size_bytes": e.get("size_bytes"),
            "size_kb": e.get("size_kb"),
            "modified_at": e.get("modified_at"),
            "created_at": e.get("created_at"),
            "ext": e.get("ext"),
            "word_count": e.get("words"),
            "top_keywords": e.get("keywords", [])[:8]
        }
        details.append({
            "id": e["id"],
            "title": e["title"],
            "type": "file",
            "path": e["path"],
            "words": e["words"],
            "size_kb": e["size_kb"],
            "keywords": e.get("keywords", []),
            "metadata": file_meta,
            "degree": degree.get(e["id"], 0),
            "preview": (e["text"][:800].replace("\n", " ").replace("\r", " ")) if e["text"] else ""
        })

    def safe_json(obj):
        return json.dumps(obj, ensure_ascii=False, default=lambda o: _to_plain(o))

    html_path = out_path
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Folder—File Relationship Dashboard</title>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root{{--bg:#071019;--card:#0c1116;--muted:#9fb6cc;--accent:#8bd8ff;--panelbg:#071219}}
  body{{margin:0;font-family:Inter,Arial,Helvetica,sans-serif;background:var(--bg);color:#e6eef8}}
  header{{padding:18px 22px;background:linear-gradient(90deg,#081226,#071726);box-shadow:0 2px 10px rgba(0,0,0,0.5)}}
  header h1{{margin:0;font-weight:600;font-size:18px;color:var(--accent)}}
  .wrap{{padding:16px}}
  .grid{{display:grid;grid-template-columns:1.3fr 0.9fr;grid-gap:14px;align-items:start}}
  .card{{background:var(--card);border:1px solid #1a2432;border-radius:12px;padding:12px;box-shadow:0 8px 26px rgba(0,0,0,.45)}}
  .card h2{{margin:6px 0 10px 0;font-size:15px;color:var(--accent)}}
  .netwrap{{height:640px;border-radius:10px;border:2px solid #142030;overflow:hidden;background:#061018;padding:8px}}
  #mynetwork{{width:100%;height:100%;background:transparent}}
  .portrait{{background:#07151b;border-radius:10px;padding:14px;height:640px;overflow:auto;border:1px solid #131b25}}
  .kv{{
    display:flex;gap:8px;margin:8px 0;font-size:14px;align-items:center;
  }}
  .kv b{{color:var(--accent);min-width:140px}}
  .chips{{display:flex;gap:8px;flex-wrap:wrap}}
  .chip{{background:#10262f;border:1px solid #183443;color:#cfe9ff;padding:6px 9px;border-radius:999px;font-size:13px}}
  .muted{{color:var(--muted);font-size:13px}}
  .divider{{height:1px;background:#142030;margin:12px 0}}
  #chart{{height:360px}}
  .meta-block{{background:#071219;border:1px solid #142030;padding:10px;border-radius:8px;white-space:pre-wrap;font-family:monospace;font-size:13px;color:#cfe9ff}}
  footer{{text-align:center;color:#8aa7c2;font-size:12px;padding:10px;margin-top:12px}}
  /* legend */
  .legend{{display:flex;gap:8px;align-items:center;margin-bottom:8px}}
  .legend .item{{display:flex;gap:6px;align-items:center;font-size:13px}}
  .legend .dot{{width:14px;height:14px;border-radius:50%;display:inline-block}}
</style>
</head>
<body>
  <header><h1>📁 Folder—File Relationship Dashboard</h1></header>
  <div class="wrap">
    <div class="grid">
      <div class="card">
        <h2>Network — folders (large teal) & files (smaller colored by type)</h2>
        <div class="muted" style="margin-bottom:8px">Click nodes to inspect; folder nodes show aggregated stats. Cross-folder edges show why (similarity / keywords / same type).</div>
        <div class="netwrap"><div id="mynetwork"></div></div>
      </div>
      <div class="card portrait" id="portrait">
        <div class="muted">Click a folder or file node, or click a folder bar in the chart to inspect details here.</div>
      </div>
    </div>

    <div class="card" style="margin-top:14px;">
      <h2>Folder Distribution — Words & Top-keyword frequency</h2>
      <div id="chart"></div>
      <div class="muted" style="margin-top:8px">Click a bar to jump to folder details.</div>
    </div>
  </div>

  <footer>Generated by visualizer_v3 — files in <code>{_safe(str(pathlib.Path(OUT_DIR).resolve()))}</code></footer>

<script>
  // ------------------ Data injected from Python ------------------
  const NODES = {safe_json(nodes)};
  const EDGES = {safe_json(edges)};
  const DETAILS = {safe_json(details)};
  const FOLDER_LABELS = {safe_json([f["title"] for f in folder_list])};
  const FOLDER_IDS = {safe_json([f["id"] for f in folder_list])};
  const BAR_VALUES = {safe_json(bar_values)};
  const LINE_VALUES = {safe_json(line_values)};
  const TOP_KEYWORDS = {safe_json(top_folder_keywords)};

  // ------------------ Build vis-network ------------------
  const container = document.getElementById('mynetwork');
  const data = {{
    nodes: new vis.DataSet(NODES),
    edges: new vis.DataSet(EDGES)
  }};
  const options = {{
    physics: {{
      solver: 'barnesHut',
      barnesHut: {{
        gravitationalConstant: -3000,
        centralGravity: 0.2,
        springLength: 160,
        springConstant: 0.04,
        damping: 0.2
      }},
      stabilization: {{ iterations: 250 }}
    }},
    nodes: {{
      shape: 'dot',
      scaling: {{ min: 6, max: 60 }},
      borderWidthSelected: 3,
      font: {{ color: '#eaf6ff', size: 12 }},
    }},
    edges: {{ smooth: {{ type:'dynamic' }}, hoverWidth: 1.5 }},
    interaction: {{ hover: true, tooltipDelay: 80, navigationButtons: true, keyboard: true }}
  }};
  const network = new vis.Network(container, data, options);

  // utility: pretty-print JSON metadata block
  function prettyJSON(obj) {{
    try {{
      return JSON.stringify(obj, null, 2)
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    }} catch(e) {{
      return String(obj);
    }}
  }}

  // show portrait for id (folder or file)
  function showPortrait(id) {{
    const it = DETAILS.find(x => x.id === id);
    if (!it) return;
    let html = `<div>`;
    if (it.type === "folder") {{
      html += `<div class="kv"><b>Folder</b><div style="font-weight:700">${{_escape(it.title)}}</div></div>`;
      html += `<div class="kv"><b>Path</b><div>${{_escape(it.path)}}</div></div>`;
      html += `<div class="kv"><b>Files</b><div>${{it.file_count}}</div></div>`;
      html += `<div class="kv"><b>Words</b><div>${{it.words}}</div></div>`;
      html += `<div class="kv"><b>Top keywords</b><div class="chips">${{(it.keywords||[]).map(k=>'<span class="chip">'+_escape(k)+'</span>').join(' ')}}</div></div>`;
      html += `<div class="divider"></div>`;
      html += `<div style="font-weight:600;color:#8bd8ff;margin-bottom:6px">Metadata</div>`;
      html += `<div class="meta-block">${{prettyJSON(it.metadata)}}</div>`;
      html += `<div class="divider"></div>`;
      html += `<div style="font-weight:600;color:#8bd8ff;margin-bottom:6px">Contains (sample)</div>`;
      // show up to 25 contained docs
      const children = data.edges.get().filter(e => e.from === id).map(e => data.nodes.get(e.to)).filter(Boolean);
      html += `<div class="muted">${{children.slice(0,25).map(c=>'<div style="margin:6px 0">• <b>'+_escape(c.label)+'</b> <span style="color:#9fb6cc">('+(c.group)+')</span></div>').join('')}}</div>`;
    }} else {{
      html += `<div class="kv"><b>File</b><div style="font-weight:700">${{_escape(it.title)}}</div></div>`;
      html += `<div class="kv"><b>Folder</b><div>${{_escape(it.metadata && it.metadata.filepath ? it.metadata.filepath.replace(it.title,'') : '—')}}</div></div>`;
      html += `<div class="kv"><b>Type</b><div>${{_escape(it.metadata && it.metadata.ext ? it.metadata.ext : '—')}}</div></div>`;
      html += `<div class="kv"><b>Size (KB)</b><div>${{_escape(it.size_kb)}}</div></div>`;
      html += `<div class="kv"><b>Words</b><div>${{it.words}}</div></div>`;
      html += `<div class="kv"><b>Degree</b><div>${{it.degree}}</div></div>`;
      html += `<div class="kv"><b>Keywords</b><div class="chips">${{(it.keywords||[]).map(k=>'<span class="chip">'+_escape(k)+'</span>').join(' ')}}</div></div>`;
      html += `<div class="divider"></div>`;
      html += `<div style="font-weight:600;color:#8bd8ff;margin-bottom:6px">Metadata</div>`;
      html += `<div class="meta-block">${{prettyJSON(it.metadata)}}</div>`;
      html += `<div class="divider"></div>`;
      html += `<div style="font-weight:600;color:#8bd8ff;margin-bottom:6px">Preview</div>`;
      html += `<div style="line-height:1.45;color:#dfeaf8">${{_escape(it.preview || '—')}}</div>`;
    }}
    html += `</div>`;
    document.getElementById('portrait').innerHTML = html;
  }}

  // small helper for safe text injection
  function _escape(s) {{
    if (s === null || s === undefined) return '';
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }}

  // select & highlight neighbors on node select
  network.on("selectNode", function(params) {{
    if (!params.nodes || params.nodes.length === 0) return;
    const nid = params.nodes[0];
    const connEdges = data.edges.get({{ filter: e => e.from === nid || e.to === nid }});
    const neighborIds = new Set([nid]);
    connEdges.forEach(e => {{ neighborIds.add(e.from); neighborIds.add(e.to); }});
    data.nodes.forEach(n => {{
      data.nodes.update({{ id: n.id, opacity: neighborIds.has(n.id) ? 1.0 : 0.12 }});
    }});
    showPortrait(nid);
  }});

  network.on("deselectNode", function() {{
    data.nodes.forEach(n => data.nodes.update({{ id: n.id, opacity: 1.0 }}));
    document.getElementById('portrait').innerHTML = '<div class="muted">Click a folder or file to view details here.</div>';
  }});

  // ------------------ Plotly chart (folder-level) ------------------
  const traceBar = {{
    x: FOLDER_LABELS,
    y: BAR_VALUES,
    type: 'bar',
    name: 'Words',
    marker: {{ line: {{ width: 0 }} }}
  }};
  const traceLine = {{
    x: FOLDER_LABELS,
    y: LINE_VALUES,
    type: 'scatter',
    mode: 'lines+markers',
    name: 'Keyword Freq',
    yaxis: 'y2'
  }};
  const layout = {{
    paper_bgcolor: '#071019',
    plot_bgcolor: '#071019',
    font: {{ color: '#e6eef8' }},
    margin: {{ l: 60, r: 60, t: 12, b: 200 }},
    xaxis: {{ tickangle: -35, automargin: true }},
    yaxis: {{ title: 'Words', gridcolor: '#0e2a35' }},
    yaxis2: {{ title: 'Keyword Freq', overlaying: 'y', side: 'right', gridcolor: 'rgba(0,0,0,0)' }},
    hovermode: 'x unified'
  }};
  Plotly.newPlot('chart', [traceBar, traceLine], layout, {{displayModeBar: true, responsive: true}});

  // clicking a bar navigates to that folder node
  const plotDiv = document.getElementById('chart');
  plotDiv.on('plotly_click', function(eventData) {{
    try {{
      const point = eventData.points[0];
      const idx = point.pointIndex;
      const fid = FOLDER_IDS[idx];
      // focus the folder node in the network and show portrait
      network.focus(fid, {{scale:1.3, animation:true}});
      showPortrait(fid);
      // visually highlight (temporarily increase size)
      const node = data.nodes.get(fid);
      if (node) {{
        data.nodes.update({{ id: fid, value: node.value * 1.2 }});
        setTimeout(()=>data.nodes.update({{ id: fid, value: node.value }}), 900);
      }}
    }} catch(e) {{
      console.warn("plot click error", e);
    }}
  }});

  // edge tooltips handled by vis via title property

  // final: fit network nicely
  network.once("stabilizationIterationsDone", function() {{
    network.fit({{ animation: true }});
  }});
</script>
</body>
</html>
""")
    print(f"✅ Dashboard saved at: {html_path}")
    return {"dashboard": html_path}
