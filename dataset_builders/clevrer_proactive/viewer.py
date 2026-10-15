"""Build <DATA_ROOT>/clevrer_proactive_viewer.html: review generated proactive questions (generate.py) on the video.

For the first --scenes scenes of a generated file the videos are fetched (one by one from the remote zip, see download.py)
into <RAW>/videos/. Per scene: the events (enter / exit / collisions / start / stop, BASE definitions), and the
questions: group 1 = answer from f* on, group 2 = an answer per interval; the overlay shows each question's current
answer ('?' = not answerable yet). Every question can be marked ok / wrong with a note (kept in localStorage);
"export" downloads the marks as JSON.

    python -m dataset_builders.clevrer_proactive.viewer [--file validation_balanced.json] [--scenes 20]
    python -m http.server 8000 -d data      # then open http://localhost:8000/clevrer_proactive_viewer.html
"""
import argparse
import json
import zipfile
from pathlib import Path

import fsspec

from .executor import build_cache
from .generate import BASE, OUT, State
from .paths import DATA_ROOT, RAW

BASE_URL = "https://data.csail.mit.edu/clevrer/"


def videos_url():
    """<RAW>/videos relative to DATA_ROOT, the directory served by `python -m http.server -d <DATA_ROOT>`."""
    try:
        return (RAW / "videos").resolve().relative_to(DATA_ROOT.resolve()).as_posix()
    except ValueError:
        raise SystemExit(f"{RAW / 'videos'} is outside {DATA_ROOT}: the page could not load the videos over http.server")


def fetch_videos(split, names):
    dst = RAW / "videos"
    dst.mkdir(parents=True, exist_ok=True)
    todo = [n for n in names if not (dst / n).exists()]
    if not todo:
        return
    zf = zipfile.ZipFile(fsspec.open(BASE_URL + f"videos/{split}/video_{split}.zip", "rb", block_size=2**20).open())
    members = {Path(i.filename).name: i for i in zf.infolist()}
    for n in todo:
        (dst / n).write_bytes(zf.read(members[n]))
        print(f"[ok  ] {n}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="validation_500.json")
    ap.add_argument("--scenes", type=int, default=20)
    args = ap.parse_args()

    qs = json.loads((OUT / args.file).read_text(encoding="utf-8"))
    split = qs[0]["split"]
    sids = list(dict.fromkeys(q["scene_index"] for q in qs))[:args.scenes]
    fetch_videos(split, [f"video_{s:05d}.mp4" for s in sids])
    cache = build_cache(split)
    scenes = []
    for sid in sids:
        st = State(cache[sid], BASE)
        ev = [{"kind": k, "frame": f, "objects": o} for k, lst in (("enter", st.ins), ("exit", st.outs), ("collision", st.cols),
                                                                    ("start", st.starts), ("stop", st.stops)) for f, o in lst]
        scenes.append({"scene_index": sid, "video": f"video_{sid:05d}.mp4", "objects": cache[sid]["objs"],
                       "visible": st.vis.tolist(), "moving": st.mov.tolist(), "events": sorted(ev, key=lambda e: e["frame"]),
                       "questions": [q for q in qs if q["scene_index"] == sid]})
    out = DATA_ROOT / "clevrer_proactive_viewer.html"
    out.write_text(TEMPLATE.replace("/*ITEMS*/", json.dumps(scenes, ensure_ascii=False)).replace("/*FILE*/", args.file).replace("/*VIDEOS*/", videos_url()), encoding="utf-8")
    print(f"wrote {out}: {len(scenes)} scenes, {sum(len(s['questions']) for s in scenes)} questions")


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
</style></head><body>
<div id="left">
  <div class="bar">
    <select id="scene"></select>
    <label>FPS <select id="fps"><option value="0">native</option><option>1</option><option>2</option><option selected>4</option><option>8</option></select></label>
    <button id="play">▶ Play</button>
    <label>speed <select id="speed"><option>0.1</option><option selected>0.25</option><option>0.5</option><option>1</option></select>×</label>
    <span id="time" style="font-family:ui-monospace,monospace"></span>
    <span style="color:var(--muted);font-size:12px"><kbd>Space</kbd> play · <kbd>←/→</kbd> ±1 frame · <kbd>J/K</kbd> question · <kbd>1</kbd>/<kbd>2</kbd> mark ok/wrong</span>
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
const S=/*ITEMS*/, FILE="/*FILE*/";
const $=id=>document.getElementById(id),vid=$("vid");
const FPS_V=25,NF=128;
let cur=null,t=0,sel=null,playing=false,timer=null,blobUrl=null,token=0,shown=[];
const esc=s=>String(s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const COL={gray:"#9aa0a6",red:"#e05252",blue:"#4f7de0",green:"#4caf50",brown:"#9c6b3c",purple:"#a05fd0",cyan:"#3fc6d0",yellow:"#e8c63a"};
const D=()=>NF/FPS_V,X=f=>f/NF*100,fr=()=>Math.min(NF-1,Math.floor(t*FPS_V+1e-6));
const FPS=()=>+$("fps").value,step=()=>FPS()?1/FPS():1/FPS_V;
const snap=s=>FPS()?Math.floor(s*FPS()+1e-6)/FPS():Math.floor(s*FPS_V+1e-6)/FPS_V;
const oname=i=>{const o=cur.objects[i];return `${o.color} ${o.material} ${o.shape}`};
// marks: localStorage, keyed by question id
let MARKS={};try{MARKS=JSON.parse(localStorage.getItem("clevrer_marks_"+FILE)||"{}")}catch(e){}
const saveMarks=()=>{try{localStorage.setItem("clevrer_marks_"+FILE,JSON.stringify(MARKS))}catch(e){};stat()};
function stat(){const v=Object.values(MARKS);$("stat").textContent=`marked ${v.length} · ok ${v.filter(m=>m.ok).length} · wrong ${v.filter(m=>!m.ok).length}`}
const answerAt=(q,f)=>{for(const a of q.answers)if(a.start_frame<=f&&f<=a.end_frame)return a.answer;return null};

S.forEach((s,i)=>$("scene").add(new Option(`scene ${s.scene_index} · ${s.questions.length} questions`,i)));
[...new Set(S.flatMap(s=>s.questions.map(q=>q.family)))].sort().forEach(f=>$("ffam").add(new Option(f,f)));

async function load(){cur=S[+$("scene").value];t=0;sel=null;setPlay(false);const my=++token;
  $("head").innerHTML=`<b>scene ${cur.scene_index}</b> · ${cur.video}<div style="margin-top:4px">${cur.objects.map((o,i)=>`<span class=dot style="background:${COL[o.color]}"></span>${i}: ${oname(i)}`).join(" &nbsp; ")}</div>`+
    `<div class=muted style="margin-top:2px">${cur.events.map(e=>`${e.frame} ${e.kind} ${e.objects.join("↔")}`).join(" · ")}</div>`;
  fill();
  vid.removeAttribute("src");vid.load();
  const b=await (await fetch(`/*VIDEOS*//${cur.video}`)).blob();
  if(my!==token)return;if(blobUrl)URL.revokeObjectURL(blobUrl);blobUrl=URL.createObjectURL(b);vid.src=blobUrl;vid.onloadeddata=()=>seek(0)}

function fill(){const g=$("fgroup").value,f=$("ffam").value;
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
function update(){const f=fr();
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
$("timeline").onmousedown=e=>{const r=e.currentTarget.getBoundingClientRect();
  const go=ev=>seek((ev.clientX-r.left)/r.width*D());go(e);const mv=ev=>go(ev);addEventListener("mousemove",mv);addEventListener("mouseup",()=>removeEventListener("mousemove",mv),{once:true})};
addEventListener("keydown",e=>{if(["SELECT","INPUT"].includes(e.target.tagName))return;
  if(e.key===" "){e.preventDefault();setPlay(!playing);return}
  if(e.key==="1"||e.key==="2"){if(sel==null)return;const q=shown[sel];MARKS[q.id]={ok:e.key==="1",note:MARKS[q.id]?.note||"",family:q.family,question:q.question};saveMarks();fill();return}
  if(playing)setPlay(false);
  if(e.key==="ArrowRight")seek(t+step()+1e-6);else if(e.key==="ArrowLeft")seek(t-step());
  else if(e.key==="j"||e.key==="k"){if(!shown.length)return;sel=sel==null?0:e.key==="k"?Math.min(shown.length-1,sel+1):Math.max(0,sel-1);seek(shown[sel].answers[0].start_frame/FPS_V);draw()}});
$("export").onclick=()=>{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(MARKS,null,1)],{type:"application/json"}));a.download="clevrer_marks.json";a.click()};
$("scene").onchange=load;$("fgroup").onchange=fill;$("ffam").onchange=fill;load();
</script></body></html>"""

if __name__ == "__main__":
    main()
