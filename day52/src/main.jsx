import React, {useMemo, useState} from "react";
import {createRoot} from "react-dom/client";
import {
  LayoutDashboard, Users, BrainCircuit, BarChart3, Settings,
  Search, Bell, ChevronRight, TrendingUp, AlertTriangle,
  UserCheck, Activity, SlidersHorizontal, X
} from "lucide-react";
import {customers} from "./data";
import "./styles.css";

const money = n => new Intl.NumberFormat("en-IN", {
  style:"currency", currency:"INR", maximumFractionDigits:0
}).format(n);

function App(){
  const [section,setSection]=useState("Overview");
  const [city,setCity]=useState("All");
  const [segment,setSegment]=useState("All");
  const [query,setQuery]=useState("");
  const [selected,setSelected]=useState(null);

  const filtered=useMemo(()=>customers.filter(c=>
    (city==="All"||c.city===city) &&
    (segment==="All"||c.segment===segment) &&
    (c.id.toLowerCase().includes(query.toLowerCase()) ||
     c.name.toLowerCase().includes(query.toLowerCase()) ||
     c.city.toLowerCase().includes(query.toLowerCase()))
  ),[city,segment,query]);

  const avgRisk=filtered.reduce((a,c)=>a+c.churn,0)/(filtered.length||1);
  const revenue=filtered.reduce((a,c)=>a+c.spend,0);
  const atRisk=filtered.filter(c=>c.churn>=.65).length;

  return <div className="app">
    <aside className="sidebar">
      <div className="brand"><div className="logo">CI</div><div><b>CustomerIQ</b><span>Intelligence Platform</span></div></div>
      <nav>
        {[
          [LayoutDashboard,"Overview"],
          [Users,"Customers"],
          [BrainCircuit,"Predictions"],
          [BarChart3,"Analytics"],
          [Settings,"Settings"]
        ].map(([Icon,label])=>
          <button className={section===label?"nav active":"nav"} onClick={()=>setSection(label)} key={label}>
            <Icon size={18}/>{label}
          </button>
        )}
      </nav>
      <div className="sidebar-foot"><div className="status-dot"></div><span>Model online</span><small>Updated today</small></div>
    </aside>

    <main>
      <header>
        <div>
          <div className="eyebrow">CUSTOMER INTELLIGENCE PLATFORM</div>
          <h1>{section}</h1>
          <p>Turn customer signals into clear, actionable decisions.</p>
        </div>
        <div className="header-actions">
          <button className="icon-btn"><Bell size={18}/></button>
          <div className="avatar">A</div>
        </div>
      </header>

      <div className="toolbar">
        <div className="search"><Search size={17}/><input value={query} onChange={e=>setQuery(e.target.value)} placeholder="Search customers..." /></div>
        <div className="select-wrap"><SlidersHorizontal size={16}/><select value={city} onChange={e=>setCity(e.target.value)}><option>All</option><option>Chennai</option><option>Bengaluru</option><option>Hyderabad</option><option>Mumbai</option><option>Delhi</option></select></div>
        <select value={segment} onChange={e=>setSegment(e.target.value)}><option>All</option><option>At Risk</option><option>Watch</option><option>Loyal</option></select>
        {(city!=="All"||segment!=="All"||query) && <button className="clear" onClick={()=>{setCity("All");setSegment("All");setQuery("")}}>Clear</button>}
      </div>

      <section className="kpis">
        <Kpi icon={<Users/>} label="Customers" value={filtered.length} note="Active records" />
        <Kpi icon={<AlertTriangle/>} label="At-risk customers" value={atRisk} note="High churn probability" />
        <Kpi icon={<Activity/>} label="Avg. churn risk" value={`${(avgRisk*100).toFixed(1)}%`} note="Model probability" />
        <Kpi icon={<TrendingUp/>} label="Customer value" value={money(revenue)} note="Tracked spend" />
      </section>

      {section==="Overview" && <Overview filtered={filtered} setSelected={setSelected}/>}
      {section==="Customers" && <CustomerTable filtered={filtered} setSelected={setSelected}/>}
      {section==="Predictions" && <PredictionView filtered={filtered} setSelected={setSelected}/>}
      {section==="Analytics" && <Analytics filtered={filtered}/>}
      {section==="Settings" && <SettingsView/>}

      {selected && <Drawer customer={selected} close={()=>setSelected(null)}/>}
    </main>
  </div>
}

function Kpi({icon,label,value,note}){return <div className="kpi"><div className="kpi-icon">{icon}</div><div><span>{label}</span><strong>{value}</strong><small>{note}</small></div></div>}

function Overview({filtered,setSelected}){
  const segments=["At Risk","Watch","Loyal"];
  return <div className="grid">
    <div className="card wide">
      <div className="card-head"><div><h2>Customer health</h2><p>Current risk distribution</p></div><span className="live"><i/> Live</span></div>
      <div className="health">
        {segments.map(s=>{const count=filtered.filter(c=>c.segment===s).length; return <div className="health-row" key={s}><div><b>{s}</b><span>{count} customers</span></div><div className="bar"><i style={{width:`${filtered.length?count/filtered.length*100:0}%`}}/></div><strong>{filtered.length?Math.round(count/filtered.length*100):0}%</strong></div>})}
      </div>
    </div>
    <div className="card">
      <div className="card-head"><div><h2>Top risk signals</h2><p>Model-level drivers</p></div></div>
      <div className="signals">
        {["Total spend","Tenure","Orders","Complaints"].map((s,i)=><div className="signal" key={s}><span>{i+1}</span><div><b>{s}</b><small>{["Strong","Strong","Moderate","Moderate"][i]} influence</small></div><ChevronRight size={16}/></div>)}
      </div>
    </div>
    <div className="card wide">
      <div className="card-head"><div><h2>Priority customers</h2><p>Customers requiring attention</p></div><button className="link" onClick={()=>{}}>View all <ChevronRight size={15}/></button></div>
      <CustomerTable filtered={filtered.filter(c=>c.churn>=.55).slice(0,5)} setSelected={setSelected} compact/>
    </div>
  </div>
}

function CustomerTable({filtered,setSelected,compact=false}){
  return <div className="table-wrap"><table><thead><tr><th>Customer</th><th>Location</th><th>Segment</th><th>Orders</th><th>Spend</th><th>Churn risk</th><th></th></tr></thead>
  <tbody>{filtered.slice(0,compact?5:20).map(c=><tr key={c.id} onClick={()=>setSelected(c)}><td><div className="customer"><div className="mini-avatar">{c.id.slice(-2)}</div><div><b>{c.name}</b><small>{c.id}</small></div></div></td><td>{c.city}</td><td><Badge value={c.segment}/></td><td>{c.orders}</td><td>{money(c.spend)}</td><td><div className="risk"><span>{Math.round(c.churn*100)}%</span><div><i style={{width:`${c.churn*100}%`}}/></div></div></td><td><ChevronRight size={16}/></td></tr>)}</tbody></table>
  {filtered.length===0&&<div className="empty">No customers match the selected filters.</div>}</div>
}

function Badge({value}){return <span className={`badge ${value.toLowerCase().replace(" ","-")}`}>{value}</span>}

function PredictionView({filtered,setSelected}){
  return <div className="card"><div className="card-head"><div><h2>Churn predictions</h2><p>Model-generated probability for each customer</p></div><span className="model-pill"><BrainCircuit size={15}/> Random Forest</span></div><CustomerTable filtered={[...filtered].sort((a,b)=>b.churn-a.churn)} setSelected={setSelected}/></div>
}

function Analytics({filtered}){
  const avg=filtered.reduce((a,c)=>a+c.spend,0)/(filtered.length||1);
  return <div className="grid">
    <div className="card wide"><div className="card-head"><div><h2>Spend distribution</h2><p>Customer value across the selected population</p></div></div>
      <div className="chart">{[20,42,58,38,72,54,84,65,92,78].map((h,i)=><div className="bar-col" key={i}><i style={{height:`${h}%`}}/><span>{i+1}</span></div>)}</div>
    </div>
    <div className="card"><div className="card-head"><div><h2>Portfolio snapshot</h2><p>Current filtered population</p></div></div>
      <div className="snapshot"><div><span>Avg. customer spend</span><b>{money(avg)}</b></div><div><span>Highest spend</span><b>{money(Math.max(...filtered.map(c=>c.spend),0))}</b></div><div><span>Avg. tenure</span><b>{(filtered.reduce((a,c)=>a+c.tenure,0)/(filtered.length||1)).toFixed(1)} mo</b></div></div>
    </div>
  </div>
}

function SettingsView(){return <div className="card settings"><h2>Dashboard preferences</h2><p>UX decisions are documented in <code>reports/ux_design_report.md</code>.</p><div className="setting-row"><span>Explainability</span><b>Enabled</b></div><div className="setting-row"><span>Prediction model</span><b>Random Forest</b></div><div className="setting-row"><span>Refresh cadence</span><b>Daily</b></div></div>}

function Drawer({customer,close}){
  return <div className="overlay" onClick={close}><aside className="drawer" onClick={e=>e.stopPropagation()}>
    <div className="drawer-head"><div><span className="eyebrow">CUSTOMER PROFILE</span><h2>{customer.name}</h2><p>{customer.id} · {customer.city}</p></div><button className="icon-btn" onClick={close}><X/></button></div>
    <div className="risk-card"><div><span>Predicted churn risk</span><strong>{Math.round(customer.churn*100)}%</strong></div><div className="risk-track"><i style={{width:`${customer.churn*100}%`}}/></div><Badge value={customer.segment}/></div>
    <h3>Customer signals</h3><div className="detail-grid"><div><span>Tenure</span><b>{customer.tenure} months</b></div><div><span>Orders</span><b>{customer.orders}</b></div><div><span>Total spend</span><b>{money(customer.spend)}</b></div><div><span>Complaints</span><b>{customer.complaints}</b></div></div>
    <h3>Why this prediction?</h3><div className="explain"><p><b>Higher risk signals</b> include shorter tenure, lower order activity, and complaint history.</p><div className="ex-row"><span>Tenure</span><b>Strong</b></div><div className="ex-row"><span>Orders</span><b>Moderate</b></div><div className="ex-row"><span>Complaints</span><b>Moderate</b></div></div>
    <button className="primary">Create retention action</button>
  </aside></div>
}

createRoot(document.getElementById("root")).render(<App/>);
