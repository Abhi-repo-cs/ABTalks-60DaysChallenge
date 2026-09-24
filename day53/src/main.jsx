import React,{useMemo,useState} from "react";
import {createRoot} from "react-dom/client";
import {Activity,AlertTriangle,BrainCircuit,CheckCircle2,Clock3,Gauge,RefreshCw,Send,ShieldAlert,Sparkles,Users} from "lucide-react";
import "./styles.css";

const initial={tenure:12,monthlySpend:85,orders:4,complaints:0,supportTickets:1,daysSinceLastOrder:12,satisfaction:7,preferredChannel:"Mobile",contract:"Monthly",paymentMethod:"Card"};
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
function predict(f){
 let s=.18;
 s+=f.contract==="Monthly"?.18:f.contract==="Annual"?-.10:-.04;
 s+=f.daysSinceLastOrder>30?.20:f.daysSinceLastOrder>14?.09:-.03;
 s+=f.complaints*.055+f.supportTickets*.035;
 s+=f.satisfaction<=4?.16:f.satisfaction<=6?.07:-.04;
 s+=f.orders<=1?.10:f.orders>=6?-.06:0;
 s+=f.monthlySpend<40?.08:f.monthlySpend>150?-.04:0;
 s+=f.tenure<6?.08:f.tenure>24?-.05:0;
 s+=f.paymentMethod==="Bank Transfer"?.03:0;
 s=clamp(s,.02,.97);
 const risk=s>=.70?"High":s>=.45?"Medium":"Low";
 const confidence=clamp(.72+Math.abs(s-.5)*.42,.72,.96);
 const drivers=[
  ["Days since last order",f.daysSinceLastOrder>14?"Elevated":"Healthy",f.daysSinceLastOrder>14?.23:-.08],
  ["Contract type",f.contract==="Monthly"?"Higher risk":"Protective",f.contract==="Monthly"?.18:-.10],
  ["Satisfaction",f.satisfaction<=6?"Needs attention":"Healthy",f.satisfaction<=6?.15:-.05],
  ["Complaints",f.complaints>1?"Elevated":"Low",f.complaints>1?.12:-.03],
  ["Order frequency",f.orders<=1?"Low":"Healthy",f.orders<=1?.10:-.04]
 ].sort((a,b)=>Math.abs(b[2])-Math.abs(a[2])).slice(0,3);
 return {score:s,risk,confidence,drivers,timestamp:new Date().toLocaleTimeString()};
}
function Field({label,children}){return <label className="field"><span>{label}</span>{children}</label>}

function App(){
 const [form,setForm]=useState(initial),[prediction,setPrediction]=useState(()=>predict(initial));
 const [events,setEvents]=useState([{time:"Now",text:"Realtime prediction engine initialized",type:"system"}]);
 const [tab,setTab]=useState("Predict"),[customerId,setCustomerId]=useState("CUST-1053");
 const update=(k,v)=>setForm(f=>({...f,[k]:v}));
 const run=()=>{const p=predict(form);setPrediction(p);setEvents(e=>[{time:p.timestamp,text:`${customerId} scored ${Math.round(p.score*100)}% churn risk (${p.risk})`,type:p.risk==="High"?"alert":"success"},...e].slice(0,7))};
 const k=useMemo(()=>({score:Math.round(prediction.score*100),confidence:Math.round(prediction.confidence*100)}),[prediction]);
 return <div className="app">
  <header><div className="brand"><div className="logo"><BrainCircuit size={22}/></div><div><b>Customer Intelligence</b><small>Real-Time Analytics</small></div></div><div className="live"><i/> LIVE ENGINE</div></header>
  <main>
   <section className="hero"><div><div className="eyebrow"><Sparkles size={14}/> DAY 53 CAPSTONE</div><h1>Real-Time Customer Risk</h1><p>Submit live customer signals and receive an instant churn-risk prediction that flows into the analytics dashboard.</p></div><div className="status"><Activity size={18}/><div><small>Pipeline status</small><b>Operational</b></div><CheckCircle2 size={18}/></div></section>
   <nav>{["Predict","Dashboard","Workflow"].map(x=><button className={tab===x?"active":""} onClick={()=>setTab(x)} key={x}>{x}</button>)}</nav>

   {tab==="Predict"&&<div className="grid">
    <section className="panel"><div className="head"><div><div className="eyebrow">LIVE INPUT</div><h2>Customer prediction form</h2></div><span className="tag"><RefreshCw size={13}/> instant scoring</span></div>
     <div className="formgrid">
      <Field label="Customer ID"><input value={customerId} onChange={e=>setCustomerId(e.target.value)}/></Field>
      <Field label="Tenure (months)"><input type="number" min="0" value={form.tenure} onChange={e=>update("tenure",+e.target.value)}/></Field>
      <Field label="Monthly spend"><input type="number" min="0" value={form.monthlySpend} onChange={e=>update("monthlySpend",+e.target.value)}/></Field>
      <Field label="Orders / month"><input type="number" min="0" value={form.orders} onChange={e=>update("orders",+e.target.value)}/></Field>
      <Field label="Complaints"><input type="number" min="0" value={form.complaints} onChange={e=>update("complaints",+e.target.value)}/></Field>
      <Field label="Support tickets"><input type="number" min="0" value={form.supportTickets} onChange={e=>update("supportTickets",+e.target.value)}/></Field>
      <Field label="Days since last order"><input type="number" min="0" value={form.daysSinceLastOrder} onChange={e=>update("daysSinceLastOrder",+e.target.value)}/></Field>
      <Field label="Satisfaction (1–10)"><input type="number" min="1" max="10" value={form.satisfaction} onChange={e=>update("satisfaction",+e.target.value)}/></Field>
      <Field label="Preferred channel"><select value={form.preferredChannel} onChange={e=>update("preferredChannel",e.target.value)}><option>Mobile</option><option>Web</option><option>Store</option></select></Field>
      <Field label="Contract"><select value={form.contract} onChange={e=>update("contract",e.target.value)}><option>Monthly</option><option>Quarterly</option><option>Annual</option></select></Field>
      <Field label="Payment method"><select value={form.paymentMethod} onChange={e=>update("paymentMethod",e.target.value)}><option>Card</option><option>UPI</option><option>Bank Transfer</option></select></Field>
     </div>
     <button className="primary" onClick={run}><Send size={16}/> Run live prediction</button>
     <div className="note"><Clock3 size={13}/> Demo scorer simulates a low-latency prediction API.</div>
    </section>

    <section className="panel"><div className="head"><div><div className="eyebrow">REAL-TIME OUTPUT</div><h2>Prediction result</h2></div><span className="tag"><Gauge size={13}/> &lt;100 ms demo</span></div>
     <div className={"risk "+prediction.risk.toLowerCase()}><div className="riskicon">{prediction.risk==="High"?<ShieldAlert/>:prediction.risk==="Medium"?<AlertTriangle/>:<CheckCircle2/>}</div><div><small>Churn risk</small><strong>{prediction.risk}</strong><span>{k.score}% probability</span></div></div>
     <div className="metrics"><div><small>Model confidence</small><b>{k.confidence}%</b></div><div><small>Signals analyzed</small><b>{prediction.drivers.length}</b></div></div>
     <h3>Top decision signals</h3>{prediction.drivers.map(([n,s,v])=><div className="driver" key={n}><div><span>{n}</span><b className={v>0?"danger":"safe"}>{s}</b></div><div className="bar"><i style={{width:`${Math.min(100,Math.abs(v)*300)}%`}}/></div></div>)}
     <div className="note"><Clock3 size={13}/> Updated at {prediction.timestamp}</div>
    </section>
   </div>}

   {tab==="Dashboard"&&<div className="dash">
    <div className="kpi"><Users/><small>Customer scored</small><b>{customerId}</b><span>latest request</span></div>
    <div className="kpi"><ShieldAlert/><small>Live risk</small><b>{k.score}%</b><span>{prediction.risk} churn probability</span></div>
    <div className="kpi"><Gauge/><small>Confidence</small><b>{k.confidence}%</b><span>prediction confidence</span></div>
    <div className="kpi"><Activity/><small>Engine status</small><b>Live</b><span>ready for next event</span></div>
    <section className="panel events"><div className="head"><div><div className="eyebrow">STREAM</div><h2>Prediction events</h2></div></div>{events.map((e,i)=><div className="event" key={i}><i className={e.type}/><div><b>{e.text}</b><small>{e.time}</small></div></div>)}</section>
   </div>}

   {tab==="Workflow"&&<section className="workflow">{[
    ["01","Collect live input","User submits behavior, engagement and service signals."],
    ["02","Validate & transform","Inputs are checked and prepared for the model feature schema."],
    ["03","Run prediction","The model returns probability, risk band and decision signals."],
    ["04","Update dashboard","KPIs, event history and customer insights update immediately."]
   ].map((x,i)=><React.Fragment key={x[0]}><div className="step"><b>{x[0]}</b><div><strong>{x[1]}</strong><p>{x[2]}</p></div></div>{i<3&&<div className="arrow">↓</div>}</React.Fragment>)}</section>}
  </main>
  <footer>Day 53 • Customer Intelligence Platform • Real-Time Analytics Prototype</footer>
 </div>
}
createRoot(document.getElementById("root")).render(<App/>);
