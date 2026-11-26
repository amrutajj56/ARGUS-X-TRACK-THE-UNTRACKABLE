import React, { useEffect, useRef, useState } from "react";
import {
  Activity, Camera, ChevronRight, Database, Download, Fingerprint,
  Layers3, Play, RefreshCw, Shield, Target, UploadCloud, Video, Wifi,
  Square, Users, FileSpreadsheet, CircleDot
} from "lucide-react";

const API = "http://127.0.0.1:8000";

export default function App() {
  const [health, setHealth] = useState(false);
  const [file, setFile] = useState(null);
  const [job, setJob] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [tab, setTab] = useState("overview");
  const [webcam, setWebcam] = useState(false);
  const [events, setEvents] = useState([]);
  const [identities, setIdentities] = useState([]);
  const picker = useRef(null);

  useEffect(() => {
    let alive = true;
    const check = async () => {
      try {
        const r = await fetch(`${API}/api/health`);
        if (alive) setHealth(r.ok);
      } catch { if (alive) setHealth(false); }
    };
    check();
    const timer = setInterval(check, 4000);
    return () => { alive = false; clearInterval(timer); };
  }, []);

  const choose = (f) => {
    if (!f) return;
    const allowed = /\.(mp4|avi|mov|mkv|m4v|webm)$/i.test(f.name);
    if (!allowed) { setError("Please choose an MP4, AVI, MOV, MKV, M4V or WEBM video."); return; }
    setFile(f); setJob(null); setEvents([]); setIdentities([]); setError("");
  };

  async function runAnalysis() {
    if (!file || busy) return;
    setBusy(true); setError(""); setJob(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const upload = await fetch(`${API}/api/upload`, { method: "POST", body: form });
      const up = await upload.json();
      if (!upload.ok) throw new Error(up.detail || "Video upload failed.");
      const start = await fetch(`${API}/api/process/${up.job_id}`, { method: "POST" });
      const started = await start.json();
      if (!start.ok) throw new Error(started.detail || "Could not start analysis.");
      let result = null;
      for (let i = 0; i < 3600; i++) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        const response = await fetch(`${API}/api/jobs/${up.job_id}`);
        result = await response.json();
        setJob({ ...result, filename: file.name });
        if (result.status === "completed") break;
        if (result.status === "failed") throw new Error(result.error || "Analysis failed.");
      }
      if (result?.status !== "completed") throw new Error("Analysis is taking longer than expected. Check the backend terminal.");
      const [ev, ids] = await Promise.all([
        fetch(`${API}/api/events/${up.job_id}`).then(r => r.json()),
        fetch(`${API}/api/identities/${up.job_id}`).then(r => r.json())
      ]);
      setEvents(ev.events || []);
      setIdentities(ids.identities || []);
    } catch (e) {
      setError(e.message || "Something went wrong.");
    } finally { setBusy(false); }
  }

  async function startWebcam() {
    setError("");
    try {
      const r = await fetch(`${API}/api/webcam/start`, { method: "POST" });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || "Could not start webcam.");
      setWebcam(true);
    } catch (e) { setError(e.message); }
  }
  async function stopWebcam() {
    try { await fetch(`${API}/api/webcam/stop`, { method: "POST" }); }
    finally { setWebcam(false); }
  }

  const stats = job?.stats;
  const activeCount = stats?.unique_global_ids ?? identities.length;
  const download = (path) => `${API}${path}`;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><div className="brand-icon"><Shield size={22}/></div><div><b>ARGUS<span>-X</span></b><small>VISION INTELLIGENCE</small></div></div>
        <div className="workspace-pill"><i/> LOCAL INSTANCE <ChevronRight size={13}/></div>
        <div className="nav-label">WORKSPACE</div>
        <button className={tab==="overview"?"nav active":"nav"} onClick={()=>setTab("overview")}><Layers3 size={17}/> Command Center</button>
        <button className={tab==="events"?"nav active":"nav"} onClick={()=>setTab("events")}><Activity size={17}/> Tracking Events</button>
        <button className={tab==="identities"?"nav active":"nav"} onClick={()=>setTab("identities")}><Fingerprint size={17}/> Identity Database</button>
        <div className="sidebar-bottom"><div className="avatar">AX</div><div><b>ARGUS Operator</b><small>Local Administrator</small></div><i className="online-dot"/></div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div className="breadcrumb">PLATFORM <ChevronRight size={13}/> <b>{tab==="overview"?"COMMAND CENTER":tab==="events"?"TRACKING EVENTS":"IDENTITY DATABASE"}</b></div>
          <div className="system-pill"><i className={health?"online-dot":"offline-dot"}/> SYSTEM {health?"OPERATIONAL":"OFFLINE"}</div>
        </header>

        <div className="content">
          <section className="hero">
            <div><div className="eyebrow"><i/> ADVANCED COMPUTER VISION PLATFORM</div><h1>{tab==="overview"?"Command Center":tab==="events"?"Tracking Events":"Identity Database"}</h1><p>Long-horizon multi-object tracking and identity intelligence.</p></div>
            <div className="hero-stamp"><span>PLATFORM STATUS</span><b>{health?"OPERATIONAL":"CONNECTING"}</b><small>Private local workspace</small></div>
          </section>

          {tab==="overview" && <>
            <section className="metrics">
              <Metric icon={<Fingerprint/>} label="GLOBAL IDENTITIES" value={stats?stats.unique_global_ids:"—"} hint="Persistent identity records"/>
              <Metric icon={<Video/>} label="FRAMES PROCESSED" value={stats?stats.frames_processed.toLocaleString():"—"} hint="Analyzed video frames"/>
              <Metric icon={<Target/>} label="DETECTIONS" value={stats?stats.track_observations.toLocaleString():"—"} hint="Object observations"/>
              <Metric icon={<Users/>} label="LOCAL TRACKS" value={stats?stats.unique_local_ids:"—"} hint="Temporary track fragments"/>
            </section>

            <div className="workspace-grid">
              <section className="panel">
                <div className="panel-heading"><div><span>01 / INPUT SOURCE</span><h2>Tracking Workspace</h2></div><span className="subtle-tag">VIDEO ANALYSIS</span></div>
                <div className="drop-area" onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();choose(e.dataTransfer.files?.[0]);}}>
                  <div className="upload-symbol"><UploadCloud size={26}/></div>
                  <b>{file?file.name:"Upload video footage"}</b>
                  <small>{file?`${(file.size/1048576).toFixed(1)} MB · Ready for analysis`:"MP4, AVI, MOV, MKV and WEBM formats"}</small>
                  <input ref={picker} type="file" hidden accept="video/*,.mkv,.webm" onChange={e=>{choose(e.target.files?.[0]);e.target.value="";}}/>
                  <button className="secondary-button" type="button" onClick={()=>picker.current?.click()}>Browse Files</button>
                </div>
                <button className="primary-button" disabled={!file||busy||!health} onClick={runAnalysis}>
                  {busy?<><RefreshCw className="spin" size={16}/> Analyzing footage…</>:<><Play size={16} fill="currentColor"/> Run Tracking Analysis</>}
                </button>
                <div className="or-divider"><span/> OR USE LIVE INPUT <span/></div>
                <button className={webcam?"stop-button":"webcam-button"} onClick={webcam?stopWebcam:startWebcam} disabled={!health}>
                  {webcam?<><Square size={15}/> Stop Live Webcam</>:<><Camera size={16}/> Start Live Webcam</>}
                </button>
                <div className="small-note"><Shield size={14}/> Your footage is processed locally.</div>
                {error&&<div className="error-box">{error}</div>}
                {busy&&<div className="progress-track"><i style={{width:`${job?.progress||3}%`}}/></div>}
                {busy&&<div className="progress-label">{job?.message||"Preparing analysis"} · {job?.progress||0}%</div>}
              </section>

              <section className="panel">
                <div className="panel-heading"><div><span>02 / VISUAL OUTPUT</span><h2>Live Tracking Preview</h2></div><span className="subtle-tag">{webcam?"LIVE":"MONITOR"}</span></div>
                {webcam?<img className="video-preview" src={`${API}/api/webcam/stream`} alt="Live ARGUS-X webcam tracking"/>:
                  job?.video_url?<video className="video-preview" controls src={download(job.video_url)}/>:
                  <div className="preview-placeholder"><div className="radar"><span/><span/><span/></div><b>{busy?"Processing video sequence":"Awaiting Video Input"}</b><small>Upload footage or start your webcam to begin object tracking.</small></div>}
                <div className="preview-footer"><span><i/> {webcam?"LIVE CAMERA STREAM":job?.video_url?"ANNOTATED VIDEO READY":"WAITING FOR INPUT"}</span>{job?.video_url&&<a href={download(job.video_url)} download><Download size={14}/> Download Video</a>}</div>
              </section>
            </div>

            <section className="panel report-panel">
              <div className="panel-heading"><div><span>03 / INTELLIGENCE SUMMARY</span><h2>Analysis Results</h2></div>{job?.csv_url&&<a className="export-button" href={download(job.csv_url)}><FileSpreadsheet size={15}/> Export CSV Report</a>}</div>
              {stats?<div className="report-grid">
                <Report label="Frames analyzed" value={stats.frames_processed.toLocaleString()}/>
                <Report label="Global identities" value={stats.unique_global_ids}/>
                <Report label="Track observations" value={stats.track_observations.toLocaleString()}/>
                <Report label="Video resolution" value={stats.resolution}/>
              </div>:<div className="empty-state"><Database size={22}/><span>No analysis results yet. Upload a video and run tracking analysis to generate reports.</span></div>}
              {stats&&<div className="class-summary"><b>DETECTION DISTRIBUTION</b>{Object.entries(stats.classes||{}).map(([name,value])=><span key={name}>{name}<strong>{value}</strong></span>)}</div>}
            </section>
          </>}

          {tab==="events" && <section className="panel table-panel"><div className="panel-heading"><div><span>IDENTITY OBSERVATIONS</span><h2>Tracking Events</h2></div>{job?.csv_url&&<a className="export-button" href={download(job.csv_url)}><Download size={15}/> Export CSV</a>}</div>
            {events.length?<div className="table-wrap"><table><thead><tr><th>FRAME</th><th>LOCAL ID</th><th>GLOBAL ID</th><th>CLASS</th><th>CONFIDENCE</th><th>IDENTITY STATUS</th></tr></thead><tbody>{events.slice(-250).reverse().map((e,i)=><tr key={`${e.frame}-${i}`}><td>{e.frame}</td><td>{e.local_track_id??"—"}</td><td><b className="gid">GID {e.global_id}</b></td><td>{e.class_name}</td><td>{e.confidence}</td><td>{e.identity_match_status}</td></tr>)}</tbody></table></div>:<div className="empty-state"><Activity size={22}/>Run a video analysis to populate tracking events.</div>}
          </section>}

          {tab==="identities" && <section className="panel table-panel"><div className="panel-heading"><div><span>PERSISTENT IDENTITY MEMORY</span><h2>Identity Database</h2></div><span className="subtle-tag">{identities.length} IDENTITIES</span></div>
            {identities.length?<div className="identity-grid">{identities.map(person=><div className="identity-card" key={person.global_id}><div className="identity-icon"><Fingerprint size={21}/></div><div><b>GLOBAL ID {person.global_id}</b><small>{person.class_name}</small></div><span>{person.observations} observations</span><small>Last seen · frame {person.last_seen_frame}</small></div>)}</div>:<div className="empty-state"><Fingerprint size={22}/>Identity records will appear after analysis.</div>}
          </section>}

          <footer><span>ARGUS-X <b>·</b> TRACK THE UNTRACKABLE</span><span>VISION INTELLIGENCE PLATFORM <b>·</b> LOCAL PROCESSING</span></footer>
        </div>
      </main>
    </div>
  );
}

function Metric({icon,label,value,hint}){return <div className="metric-card"><div className="metric-top"><span>{label}</span><i>{icon}</i></div><strong>{value}</strong><small>{hint}</small></div>}
function Report({label,value}){return <div className="report-item"><span>{label}</span><b>{value}</b></div>}
