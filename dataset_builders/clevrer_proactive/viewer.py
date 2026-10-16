"""Review page for generated proactive questions (generate.py) on the video: any scene of train and validation.

A small http server: the page lists every scene that has questions, per split; a scene's data (events, per-frame
visibility / motion under BASE, questions) is built on request from its annotation file (else from the annotation
cache of the split), and its video is served
from the extracted CLEVRER zips <RAW>/video_<split>/video_XXXXX-YYYYY/, else from <RAW>/videos/ (download.py), else
fetched once from the remote CLEVRER zip and kept in <RAW>/videos/.
Per scene: the events (enter / exit / collisions / start / stop, BASE definitions), and the questions: group 1 =
answer from f* on, group 2 = an answer per interval; the overlay shows each question's current answer ('?' = not
answerable yet). Every question can be marked ok / wrong with a note (kept in localStorage, per question file);
"export" downloads the marks as JSON. The page is at http://localhost:8000/ (or /clevrer_proactive_viewer.html, the
old address); the video is kept in the URL as its number in the split: http://localhost:8000/#validation/17

    python -m dataset_builders.clevrer_proactive.viewer [--train train_balanced.json] [--validation validation_balanced.json] [--port 8000]
"""
import argparse
import json
import os
import re
import threading
import urllib.parse
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import fsspec

from .executor import annotation_path, build_cache, load_annotation
from .generate import BASE, OUT, State
from .paths import CACHE, RAW, video_path

BASE_URL = "https://data.csail.mit.edu/clevrer/"


class App:
    def __init__(self, files):
        self.files, self.qs, self.caches, self.zips = files, {}, {}, {}
        self.lock = threading.Lock()
        for split, f in files.items():
            by_scene = {}
            for q in json.loads((OUT / f).read_text(encoding="utf-8")):
                by_scene.setdefault(q["scene_index"], []).append(q)
            self.qs[split] = by_scene
            print(f"{split}: {f}, {len(by_scene)} scenes, {sum(map(len, by_scene.values()))} questions", flush=True)
        group = {q["family"]: q["group"] for by_scene in self.qs.values() for qs in by_scene.values() for q in qs}
        fams = sorted(group, key=lambda f: (group[f], f))
        fi = {f: i for i, f in enumerate(fams)}
        self.index = json.dumps({
            "families": [{"name": f, "group": group[f]} for f in fams],
            "splits": {split: {"file": files[split],
                               "scenes": [[sid, len(qs), sorted({fi[q["family"]] for q in qs})] for sid, qs in sorted(by_scene.items())]}
                       for split, by_scene in self.qs.items()}}).encode()

    def cache(self, split):
        with self.lock:
            if split not in self.caches:
                if not (CACHE / f"cache_{split}.pkl").exists() and not (RAW / "annotations" / split).exists():
                    raise RuntimeError(f"no {CACHE / f'cache_{split}.pkl'} and no {RAW / 'annotations' / split}: run download.py first")
                print(f"loading the annotation cache of {split} ...", flush=True)
                self.caches[split] = build_cache(split)
            return self.caches[split]

    def scene(self, split, sid):
        qs, p = self.qs[split][sid], annotation_path(split, sid)
        s = load_annotation(p)[1] if p.exists() else self.cache(split)[sid]  # one file: no wait for the whole split
        st = State(s, BASE)
        ev = [{"kind": k, "frame": f, "objects": o} for k, lst in (("enter", st.ins), ("exit", st.outs), ("collision", st.cols),
                                                                    ("start", st.starts), ("stop", st.stops)) for f, o in lst]
        return {"scene_index": sid, "split": split, "video": f"video_{sid:05d}.mp4", "objects": s["objs"],
                "visible": st.vis.tolist(), "moving": st.mov.tolist(), "events": sorted(ev, key=lambda e: e["frame"]),
                "questions": qs}

    def video(self, split, name):
        p = video_path(split, name)
        if not p.exists():
            with self.lock:
                if not p.exists():
                    if split not in self.zips:
                        zf = zipfile.ZipFile(fsspec.open(BASE_URL + f"videos/{split}/video_{split}.zip", "rb", block_size=2**20).open())
                        self.zips[split] = zf, {Path(i.filename).name: i for i in zf.infolist()}
                    zf, members = self.zips[split]
                    data = zf.read(members[name])
                    p.parent.mkdir(parents=True, exist_ok=True)
                    tmp = p.with_suffix(".part")
                    tmp.write_bytes(data)
                    tmp.replace(p)
                    print(f"[ok  ] {p}", flush=True)
        return p.read_bytes()


def handler(app):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urllib.parse.urlparse(self.path).path
            try:
                if path in ("/", "/index.html", "/clevrer_proactive_viewer.html"):
                    self.send(TEMPLATE.encode(), "text/html; charset=utf-8")
                elif path == "/api/index":
                    self.send(app.index, "application/json")
                elif m := re.fullmatch(r"/api/scene/(\w+)/(\d+)", path):
                    self.send(json.dumps(app.scene(m[1], int(m[2])), ensure_ascii=False).encode(), "application/json")
                elif m := re.fullmatch(r"/video/(\w+)/(video_\d{5}\.mp4)", path):
                    self.send(app.video(m[1], m[2]), "video/mp4")
                else:
                    self.send_error(404)
            except KeyError:
                self.send_error(404)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                self.send_error(500, str(e))
                raise

        def send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train_balanced.json")
    ap.add_argument("--validation", default="validation_balanced.json")
    ap.add_argument("--host", default="", help="'' = all interfaces, as python -m http.server")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    app = App({split: f for split, f in (("train", args.train), ("validation", args.validation)) if f})
    # on Windows SO_REUSEADDR lets a second server bind a busy port and share it silently (e.g. with an old
    # `python -m http.server 8000`): fail with "address in use" instead
    ThreadingHTTPServer.allow_reuse_address = os.name != "nt"
    srv = ThreadingHTTPServer((args.host, args.port), handler(app))
    print(f"serving on http://localhost:{args.port}/", flush=True)
    srv.serve_forever()


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>CLEVRER Proactive QA</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#111418;--panel:#1a1f26;--line:#2b323c;--text:#e6e9ee;--muted:#8b95a3;--accent:#4ea1ff;
--yes:#2ecc71;--no:#ff6b6b;--gt:#c678dd;--ask:#ffb020}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 system-ui,sans-serif;display:flex;height:100vh}
#left{flex:1;display:flex;flex-direction:column;min-width:0;padding:12px;gap:8px}
#right{width:620px;border-left:1px solid var(--line);display:flex;flex-direction:column;background:var(--panel)}
select,button,input{background:#222934;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:4px 8px;font:inherit}
button{cursor:pointer}button:hover{border-color:var(--accent)}
.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
#stage{flex:1;min-height:0;display:flex;align-items:center;justify-content:center;background:#000;border-radius:8px;position:relative}
#vid{width:100%;height:100%;object-fit:contain}
#overlay{position:absolute;left:10px;top:10px;font:600 13px ui-monospace,monospace;background:#000c;padding:4px 8px;border-radius:4px;max-width:calc(100% - 20px);white-space:pre-wrap}
#tlwrap{max-height:42vh;overflow:auto;background:var(--panel);border-radius:6px}
#timeline{position:relative;cursor:pointer;user-select:none}
#timeline .mk{position:absolute;font:600 9px ui-monospace,monospace;color:#000;text-align:center;white-space:nowrap;overflow:hidden}
#timeline .lbl{position:absolute;left:4px;font:10px ui-monospace,monospace;color:var(--muted);pointer-events:none;z-index:2;white-space:nowrap}
#cursor{position:absolute;top:0;bottom:0;background:#ffffff25;border-left:1px solid #fff;pointer-events:none;z-index:3}
#head{padding:8px 10px;border-bottom:1px solid var(--line);font-size:13px}
#list{overflow:auto;flex:1;font:12px/1.45 ui-monospace,monospace}
.q{padding:6px 10px;border-bottom:1px solid var(--line);cursor:pointer}
.q:hover{background:#222934}.q.sel{background:#203048;border-left:3px solid var(--accent)}
.q.ok{box-shadow:inset 4px 0 var(--yes)}.q.bad{box-shadow:inset 4px 0 var(--no)}
.muted{color:var(--muted)}.yes{color:var(--yes)}.gt{color:var(--gt)}.ask{color:var(--ask)}.no{color:var(--no)}
.chip{display:inline-block;border:1px solid var(--line);border-radius:3px;padding:0 4px;margin:1px 2px 0 0}
.chip.c{border-color:var(--yes);color:var(--yes)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:3px;vertical-align:middle}
.mark button{padding:0 6px;font-size:11px}
kbd{border:1px solid var(--line);border-radius:3px;padding:0 4px;font-size:11px}
.seg{display:inline-flex}.seg button{border-radius:0}.seg button:first-child{border-radius:6px 0 0 6px}.seg button:last-child{border-radius:0 6px 6px 0}
.seg button.on{background:var(--accent);border-color:var(--accent);color:#000;font-weight:600}
</style></head><body>
<div id="left">
  <div class="bar">
    <span id="splits" class="seg"></span>
    <label>video # <input id="num" type="number" min="0" style="width:80px" title="number of the video in the split, from 0; Enter to open"></label>
    <span id="nmax" class=muted></span>
    <button id="prev" title="previous scene  [">◀</button>
    <button id="next" title="next scene  ]">▶</button>
    <button id="rand" title="random scene  R">random</button>
    <span id="sinfo" class=muted style="font-size:12px"></span>
  </div>
  <div class="bar">
    <label>FPS <select id="fps"><option value="0">native</option><option>1</option><option>2</option><option selected>4</option><option>8</option></select></label>
    <button id="play">▶ Play</button>
    <label>speed <select id="speed"><option>0.1</option><option selected>0.25</option><option>0.5</option><option>1</option></select>×</label>
    <span id="time" style="font-family:ui-monospace,monospace"></span>
    <span style="color:var(--muted);font-size:12px"><kbd>Space</kbd> play · <kbd>←/→</kbd> ±1 frame · <kbd>J/K</kbd> question · <kbd>1</kbd>/<kbd>2</kbd> mark ok/wrong · <kbd>[</kbd>/<kbd>]</kbd>/<kbd>R</kbd> scene</span>
  </div>
  <div id="stage"><video id="vid" muted playsinline></video><div id="overlay"></div></div>
  <div id="now" style="font:13px ui-monospace,monospace;min-height:20px"></div>
  <div id="tlwrap"><div id="timeline"><div id="cursor"></div></div></div>
  <div style="font-size:12px;color:var(--muted)">top rows: objects while in view (<b>bold</b> = moving) · events: <span class=yes>▮</span> enter <span class=no>▮</span> exit <span style="color:#fff">▮</span> collision ·
    question rows: <span class=ask>▮</span> group 1 answer from f* (hatched before = not answerable yet) · group 2 intervals with their answers</div>
</div>
<div id="right">
  <div id="head"></div>
  <div class="bar" style="padding:6px 10px;border-bottom:1px solid var(--line)">
    <select id="fgroup"><option value="">all groups</option><option value="1">group 1</option><option value="2">group 2</option></select>
    <select id="ffam"><option value="">all families</option></select>
    <button id="export">export marks</button><span id="stat" class=muted></span>
  </div>
  <div id="list"></div>
</div>
<script>
const $=id=>document.getElementById(id),vid=$("vid");
const FPS_V=25,NF=128;
let IDX=null,SPL=null,pos=-1,FILE="",cur=null,t=0,sel=null,playing=false,timer=null,blobUrl=null,token=0,shown=[];
const esc=s=>String(s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const COL={gray:"#9aa0a6",red:"#e05252",blue:"#4f7de0",green:"#4caf50",brown:"#9c6b3c",purple:"#a05fd0",cyan:"#3fc6d0",yellow:"#e8c63a"};
const D=()=>NF/FPS_V,X=f=>f/NF*100,fr=()=>Math.min(NF-1,Math.floor(t*FPS_V+1e-6));
const FPS=()=>+$("fps").value,step=()=>FPS()?1/FPS():1/FPS_V;
const snap=s=>FPS()?Math.floor(s*FPS()+1e-6)/FPS():Math.floor(s*FPS_V+1e-6)/FPS_V;
const oname=i=>{const o=cur.objects[i];return `${o.color} ${o.material} ${o.shape}`};
// marks: localStorage, keyed by question id, one store per question file
let MARKS={};
const loadMarks=()=>{MARKS={};try{MARKS=JSON.parse(localStorage.getItem("clevrer_marks_"+FILE)||"{}")}catch(e){}};
const saveMarks=()=>{try{localStorage.setItem("clevrer_marks_"+FILE,JSON.stringify(MARKS))}catch(e){};stat()};
function stat(){const v=Object.values(MARKS);$("stat").textContent=`${FILE}: marked ${v.length} · ok ${v.filter(m=>m.ok).length} · wrong ${v.filter(m=>!m.ok).length}`}
const answerAt=(q,f)=>{for(const a of q.answers)if(a.start_frame<=f&&f<=a.end_frame)return a.answer;return null};

// scene navigation: prev / next / random go over the scenes that have a question passing the group / family filters
let SPLIT="";
const split=()=>SPLIT;
const matches=i=>{const g=$("fgroup").value,f=$("ffam").value;
  return SPL.scenes[i][2].some(k=>{const F=IDX.families[k];return (!g||F.group==g)&&(!f||F.name===f)})};
function info(){let n=0;for(let i=0;i<SPL.scenes.length;i++)if(matches(i))n++;
  $("sinfo").textContent=`scene ${SPL.scenes[pos][0]} · video_${String(SPL.scenes[pos][0]).padStart(5,"0")}.mp4`+(n<SPL.scenes.length?` · ${n} videos match the filters`:"")}
function nav(d){for(let i=pos+d;i>=0&&i<SPL.scenes.length;i+=d)if(matches(i))return go(i)}
function rand(){const ok=SPL.scenes.map((_,i)=>i).filter(matches);if(ok.length)go(ok[Math.floor(Math.random()*ok.length)])}
// the URL keeps the video number in the split: #validation/17
function go(i){pos=i;$("num").value=i;history.replaceState(null,"",`#${split()}/${i}`);info();load()}
function openNum(){const i=Math.floor(+$("num").value);
  if(!(i>=0&&i<SPL.scenes.length)){$("sinfo").innerHTML=`<span class=no>${split()} has videos 0–${SPL.scenes.length-1}</span>`;return}go(i)}
function setSplit(s,i){SPLIT=s;SPL=IDX.splits[s];FILE=SPL.file;loadMarks();
  document.querySelectorAll("#splits button").forEach(b=>b.classList.toggle("on",b.dataset.s===s));
  $("num").max=SPL.scenes.length-1;$("nmax").textContent=`of 0–${SPL.scenes.length-1}`;
  go(i>=0&&i<SPL.scenes.length?i:0)}

async function init(){IDX=await (await fetch("api/index")).json();
  for(const [s,v] of Object.entries(IDX.splits)){const b=document.createElement("button");b.dataset.s=s;
    b.textContent=`${s} · ${v.scenes.length}`;b.onclick=()=>s!==SPLIT&&setSplit(s,0);$("splits").append(b)}
  IDX.families.forEach(f=>$("ffam").add(new Option(`g${f.group} · ${f.name}`,f.name)));
  const [s,i]=decodeURIComponent(location.hash.slice(1)).split("/");
  setSplit(IDX.splits[s]?s:Object.keys(IDX.splits)[0],i?+i:0)}

async function load(){const my=++token,sp=split(),sid=SPL.scenes[pos][0];setPlay(false);
  $("head").innerHTML=`<span class=muted>loading scene ${sid} …</span>`;
  const r=await fetch(`api/scene/${sp}/${sid}`);if(my!==token)return;
  if(!r.ok){$("head").innerHTML=`<span class=no>scene ${sid}: ${r.status} ${esc(r.statusText)}</span>`;return}
  const sc=await r.json();if(my!==token)return;
  cur=sc;t=0;sel=null;
  $("head").innerHTML=`<b>${cur.split} · scene ${cur.scene_index}</b> · ${cur.video} · ${cur.questions.length} questions<div style="margin-top:4px">${cur.objects.map((o,i)=>`<span class=dot style="background:${COL[o.color]}"></span>${i}: ${oname(i)}`).join(" &nbsp; ")}</div>`+
    `<div class=muted style="margin-top:2px">${cur.events.map(e=>`${e.frame} ${e.kind} ${e.objects.join("↔")}`).join(" · ")}</div>`;
  fill();
  vid.removeAttribute("src");vid.load();
  const v=await fetch(`video/${sp}/${cur.video}`);if(my!==token)return;
  if(!v.ok){$("overlay").textContent=`video unavailable: ${v.status} ${v.statusText}`;return}
  const b=await v.blob();
  if(my!==token)return;if(blobUrl)URL.revokeObjectURL(blobUrl);blobUrl=URL.createObjectURL(b);vid.src=blobUrl;vid.onloadeddata=()=>seek(0)}

function fill(){if(!cur)return;const g=$("fgroup").value,f=$("ffam").value;
  shown=cur.questions.filter(q=>(!g||q.group==g)&&(!f||q.family===f));if(sel!=null&&sel>=shown.length)sel=null;
  $("list").innerHTML="";
  shown.forEach((q,k)=>{const d=document.createElement("div");const m=MARKS[q.id];d.className="q"+(m?(m.ok?" ok":" bad"):"");d.dataset.k=k;
    const ans=new Set(q.answers.map(a=>a.answer));
    d.innerHTML=`<span class=muted>g${q.group} · ${q.family} · ${q.source}${q.resolved_at_end?" · <span class=ask>no@end</span>":""}</span><br><b>${esc(q.question)}</b><br>`+
      q.choices.map(c=>`<span class="chip ${ans.has(c)?"c":""}">${esc(c)}</span>`).join("")+
      `<br>`+q.answers.map(a=>`<span class=gt>[${a.start_frame}–${a.end_frame}]</span> ${esc(a.answer)}`).join(" → ")+
      `<div class=mark style="margin-top:3px"><button data-ok=1>✓ ok</button> <button data-ok=0>✗ wrong</button> <input placeholder="note" style="width:260px;padding:1px 6px;font-size:11px" value="${esc(m?.note||"")}"></div>`;
    d.onclick=ev=>{if(ev.target.tagName==="INPUT")return;
      if(ev.target.dataset.ok!=null){MARKS[q.id]={ok:ev.target.dataset.ok==="1",note:d.querySelector("input").value,family:q.family,question:q.question};saveMarks();fill();return}
      sel=k;seek((q.answers[0].start_frame)/FPS_V);draw()};
    d.querySelector("input").onchange=e=>{if(MARKS[q.id]){MARKS[q.id].note=e.target.value;saveMarks()}};
    $("list").append(d)});
  draw();stat()}

function draw(){const tl=$("timeline");tl.querySelectorAll(".mk,.lbl").forEach(e=>e.remove());
  const add=(a,b,top,h,css,txt)=>{const d=document.createElement("div");d.className="mk";Object.assign(d.style,{left:X(a)+"%",width:`max(3px,${X(b)-X(a)}%)`,top:top+"px",height:h+"px"},css);if(txt)d.textContent=txt;tl.append(d);return d};
  const lbl=(y,s)=>{const d=document.createElement("div");d.className="lbl";d.style.top=y+"px";d.textContent=s;tl.append(d)};
  let y=2;const R=11;
  cur.objects.forEach((o,i)=>{const v=cur.visible[i],m=cur.moving[i];let a=null;
    for(let f=0;f<=NF;f++){const on=f<NF&&v[f];if(on&&a==null)a=f;if(!on&&a!=null){add(a,f,y,R-3,{background:COL[o.color],opacity:.35});a=null}}
    a=null;for(let f=0;f<=NF;f++){const on=f<NF&&v[f]&&m[f];if(on&&a==null)a=f;if(!on&&a!=null){add(a,f,y,R-3,{background:COL[o.color]});a=null}}
    lbl(y-1,String(i));y+=R});
  lbl(y,"events");cur.events.forEach(e=>{const c={enter:"var(--yes)",exit:"var(--no)",collision:"#fff",start:"var(--accent)",stop:"var(--muted)"}[e.kind];
    if(e.kind==="enter"||e.kind==="exit"||e.kind==="collision")add(e.frame,e.frame+1,y,R,{background:c})});y+=R+6;
  const Q=16;
  shown.forEach((q,k)=>{const yy=y+k*Q,hi=sel===k?{outline:"2px solid #fff",zIndex:2}:{};
    if(q.group===1){const a=q.answers[0];add(0,a.start_frame,yy,Q-4,{background:"repeating-linear-gradient(45deg,#ffffff18 0 3px,transparent 3px 6px)",...hi});
      add(a.start_frame,NF,yy,Q-4,{background:q.resolved_at_end?"var(--muted)":"var(--ask)",...hi},a.answer)}
    else q.answers.forEach((a,j)=>add(a.start_frame,a.end_frame+1,yy,Q-4,{background:`hsl(${(j*67+200)%360} 55% 62%)`,...hi},a.answer));
    lbl(yy,`${k+1}`)});
  tl.style.height=(y+shown.length*Q+4)+"px";
  document.querySelectorAll(".q").forEach(d=>d.classList.toggle("sel",+d.dataset.k===sel));
  const s=document.querySelector(".q.sel");if(s)s.scrollIntoView({block:"nearest"});
  update()}

function seek(s){t=snap(Math.min(Math.max(0,s),D()-1e-3));if(vid.readyState)vid.currentTime=t+1e-3;update()}
function update(){if(!cur)return;const f=fr();
  Object.assign($("cursor").style,{left:X(f)+"%",width:`max(2px,${X(f+Math.max(1,step()*FPS_V))-X(f)}%)`});
  $("time").textContent=`${t.toFixed(2)} s · frame ${f}/${NF-1}`;
  const inView=cur.objects.map((_,i)=>i).filter(i=>cur.visible[i][f]),mv=inView.filter(i=>cur.moving[i][f]);
  let o=`frame ${f} · in view ${inView.length} · moving ${mv.length}`;
  cur.events.forEach(e=>{if(Math.abs(e.frame-f)<=2&&e.kind!=="start"&&e.kind!=="stop")o+=`\n${e.kind}: ${e.objects.map(oname).join(" ↔ ")}`});
  $("overlay").textContent=o;
  // current answer of the selected question: below the video, not over it
  if(sel!=null){const q=shown[sel],a=answerAt(q,f);$("now").innerHTML=`<b>${esc(q.question)}</b> → ${a==null?"<span class=muted>? (not answerable yet)</span>":`<span class=yes>${esc(a)}</span>`}`}
  else $("now").innerHTML=`<span class=muted>select a question to see its answer at the current frame</span>`}
function setPlay(p){playing=p;$("play").textContent=p?"⏸ Pause":"▶ Play";clearInterval(timer);vid.pause();
  if(!p||!cur)return;const sp=+$("speed").value;
  if(!FPS()){vid.playbackRate=Math.max(.0625,sp);vid.play();timer=setInterval(()=>{t=vid.currentTime;update();if(vid.ended)setPlay(false)},40);return}
  timer=setInterval(()=>{if(t+step()>=D())return setPlay(false);seek(t+step())},1000*step()/sp)}
$("play").onclick=()=>setPlay(!playing);$("speed").onchange=()=>playing&&setPlay(true);
$("fps").onchange=()=>{const p=playing;setPlay(false);seek(t);if(p)setPlay(true)};
$("timeline").onmousedown=e=>{if(!cur)return;const r=e.currentTarget.getBoundingClientRect();
  const go=ev=>seek((ev.clientX-r.left)/r.width*D());go(e);const mv=ev=>go(ev);addEventListener("mousemove",mv);addEventListener("mouseup",()=>removeEventListener("mousemove",mv),{once:true})};
addEventListener("keydown",e=>{if(["SELECT","INPUT"].includes(e.target.tagName))return;
  if(e.key===" "){e.preventDefault();setPlay(!playing);return}
  if(e.key==="["||e.key==="]"){nav(e.key==="]"?1:-1);return}
  if(e.key==="r"||e.key==="R"){rand();return}
  if(e.key==="1"||e.key==="2"){if(sel==null)return;const q=shown[sel];MARKS[q.id]={ok:e.key==="1",note:MARKS[q.id]?.note||"",family:q.family,question:q.question};saveMarks();fill();return}
  if(playing)setPlay(false);
  if(e.key==="ArrowRight")seek(t+step()+1e-6);else if(e.key==="ArrowLeft")seek(t-step());
  else if(e.key==="j"||e.key==="k"){if(!shown.length)return;sel=sel==null?0:e.key==="k"?Math.min(shown.length-1,sel+1):Math.max(0,sel-1);seek(shown[sel].answers[0].start_frame/FPS_V);draw()}});
$("export").onclick=()=>{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(MARKS,null,1)],{type:"application/json"}));a.download=`clevrer_marks_${split()}.json`;a.click()};
$("prev").onclick=()=>nav(-1);$("next").onclick=()=>nav(1);$("rand").onclick=rand;
$("num").onchange=openNum;
$("fgroup").onchange=()=>{fill();info()};$("ffam").onchange=()=>{fill();info()};
init();
</script></body></html>"""

if __name__ == "__main__":
    main()
