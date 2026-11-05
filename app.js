'use strict';
window.APP={
  id:'dw',name:'DriftWatch',
  logo:'<span class="mk"><svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 15c3 0 3-7 6-7s3 7 6 7 3-5 6-5"/><path d="M3 20h18"/></svg></span>',
  nav:[['index.html','home','Home'],['monitors.html','grid','Monitors'],['alerts.html','bell','Alerts'],['playground.html','flask','Playground'],['settings.html','settings','Settings']],
  accents:[['#2ee6c5','#59ff9e','#26c6ff'],['#26c6ff','#2ee6c5','#59ff9e'],['#59ff9e','#2ee6c5','#26c6ff'],['#b06bff','#2ee6c5','#26c6ff']],
  cta:['monitors.html','grid','Open monitors'],
  icons:{
    grid:'M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z',
    bell:'M18 8a6 6 0 10-12 0c0 7-3 9-3 9h18s-3-2-3-9M13.7 21a2 2 0 01-3.4 0',
    flask:'M9 3h6M10 3v6l-5 9a2 2 0 002 3h10a2 2 0 002-3l-5-9V3M7 15h10',
    model:'M12 2l9 5-9 5-9-5zM3 12l9 5 9-5M3 17l9 5 9-5',
    pulse:'M3 12h4l2-7 4 14 2-7h6'
  },
  ensureSeed(){}
};
/* ---------------- seeded RNG + stats ---------------- */
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
function gauss(rng,mean,sd){let u=0,v=0;while(!u)u=rng();while(!v)v=rng();return mean+sd*Math.sqrt(-2*Math.log(u))*Math.cos(6.2831853*v);}
function sample(rng,mean,sd,n){const a=[];for(let i=0;i<n;i++)a.push(gauss(rng,mean,sd));return a;}
function histogram(data,lo,hi,bins){const h=new Array(bins).fill(0);const w=(hi-lo)/bins;data.forEach(x=>{let b=Math.floor((x-lo)/w);if(b<0)b=0;if(b>=bins)b=bins-1;h[b]++;});return h;}
function psi(expected,actual,bins){bins=bins||10;const all=expected.concat(actual);const lo=Math.min(...all),hi=Math.max(...all)+1e-9;
  const e=histogram(expected,lo,hi,bins),a=histogram(actual,lo,hi,bins);const es=expected.length,as=actual.length;let p=0;
  for(let i=0;i<bins;i++){let ep=(e[i]/es)||1e-4,ap=(a[i]/as)||1e-4;ep=Math.max(ep,1e-4);ap=Math.max(ap,1e-4);p+=(ap-ep)*Math.log(ap/ep);}return p;}
function ksStat(a,b){const sa=a.slice().sort((x,y)=>x-y),sb=b.slice().sort((x,y)=>x-y);const all=sa.concat(sb).sort((x,y)=>x-y);let d=0,i=0,j=0;
  for(const v of all){while(i<sa.length&&sa[i]<=v)i++;while(j<sb.length&&sb[j]<=v)j++;d=Math.max(d,Math.abs(i/sa.length-j/sb.length));}return d;}
function psiLabel(p){return p<0.1?'stable':p<0.25?'warning':'drift';}
function psiClass(p){return p<0.1?'ok':p<0.25?'warn':'bad';}
/* ---------------- seeded model universe ---------------- */
const FEATURE_POOL={
 'churn-predictor':['tenure_months','monthly_spend','support_tickets','login_freq','plan_tier','nps_score','days_since_login','feature_adoption'],
 'fraud-scorer':['txn_amount','txn_velocity','device_risk','geo_distance','acct_age_days','merchant_risk','hour_of_day','card_present'],
 'loan-risk':['dti_ratio','credit_score','income','loan_amount','employ_years','num_accounts','utilization','delinquencies'],
 'ctr-ranker':['ad_relevance','user_ctr_7d','bid_amount','time_on_site','session_depth','recency_days','device_type','daypart'],
 'demand-forecast':['price','promo_flag','temperature','day_of_week','stock_level','competitor_price','holiday_flag','trend_7d'],
 'support-router':['msg_length','sentiment','keyword_urgency','prior_tickets','acct_value','channel','lang_conf','response_sla']
};
const TASKS={'churn-predictor':'Binary classification','fraud-scorer':'Binary classification','loan-risk':'Regression','ctr-ranker':'Ranking','demand-forecast':'Regression','support-router':'Multiclass'};
const ENVS=['production','production','production','staging'];
function buildModels(){
  const defs=[
    ['churn-predictor','Churn Predictor v4',0.73],
    ['fraud-scorer','Fraud Scorer',0.06],
    ['loan-risk','Loan Risk Model',0.41],
    ['ctr-ranker','Ad CTR Ranker',0.52],
    ['demand-forecast','Demand Forecaster',0.19],
    ['support-router','Support Router',0.33]
  ];
  return defs.map((d,mi)=>{
    const [key,name,driftSeed]=d;const rng=mulberry32(1000+mi*97);
    const feats=FEATURE_POOL[key].map((fn,fi)=>{
      const baseMean=gauss(rng,0,1)*2,baseSd=0.6+rng()*0.8;
      // drifted features shift more for higher-drift models
      const shift=(rng()<driftSeed+0.1? (0.3+rng()*1.4) : rng()*0.25)*(0.5+driftSeed);
      const sdScale=1+(rng()<driftSeed?rng()*0.6:rng()*0.12);
      const base=sample(mulberry32(5000+mi*31+fi),baseMean,baseSd,600);
      const cur=sample(mulberry32(9000+mi*41+fi),baseMean+shift*baseSd,baseSd*sdScale,600);
      const p=psi(base,cur),k=ksStat(base,cur);
      return {name:fn,type:fi%4===2?'categorical':'numeric',base,cur,baseMean,baseSd,mean2:baseMean+shift*baseSd,sd2:baseSd*sdScale,psi:p,ks:k};
    });
    const overall=feats.reduce((s,f)=>s+f.psi,0)/feats.length;
    const drift=Math.min(0.99,overall/0.4*0.7+driftSeed*0.3);
    const baseAcc=0.82+ (mi%3)*0.04;const acc=Math.max(0.5,baseAcc-drift*0.22);
    // 30-day histories (seeded, drift ramps)
    const dh=[],ph=[],vol=[];const rh=mulberry32(2000+mi*13);
    for(let t=0;t<30;t++){const ramp=Math.pow(t/29,1.4);
      dh.push(Math.max(0,overall*(0.25+ramp*0.95)+ (rh()-0.5)*0.03));
      ph.push(Math.max(0.5,baseAcc-overall*ramp*0.5-(rh()-0.5)*0.01));
      vol.push(Math.round(8000+Math.sin(t/4)*1500+rh()*2500));
    }
    return {id:key,name,task:TASKS[key],env:ENVS[mi%ENVS.length],features:feats,psi:overall,drift,acc,baseAcc,
      preds:vol[vol.length-1],lastTrained:['6 days ago','3 weeks ago','2 months ago','11 days ago','5 weeks ago','8 days ago'][mi],
      driftHist:dh,perfHist:ph,volHist:vol,status:psiClass(overall)};
  });
}
let MODELS=buildModels();
function getModel(id){return MODELS.find(m=>m.id===id)||MODELS[0];}
/* ---------------- canvas line chart ---------------- */
function lineChart(cv,series,opts){opts=opts||{};const DPR=Math.min(2,devicePixelRatio||1);const rect=cv.getBoundingClientRect();
  const w=cv.width=(rect.width||600)*DPR,h=cv.height=(opts.h||180)*DPR;const ctx=cv.getContext('2d');ctx.clearRect(0,0,w,h);
  const pad={l:38*DPR,r:10*DPR,t:14*DPR,b:20*DPR};const iw=w-pad.l-pad.r,ih=h-pad.t-pad.b;
  const cs=getComputedStyle(document.documentElement);const col=v=>cs.getPropertyValue(v).trim()||'#2ee6c5';
  let lo=opts.min!=null?opts.min:Math.min(...series.flatMap(s=>s.data)),hi=opts.max!=null?opts.max:Math.max(...series.flatMap(s=>s.data));if(hi===lo)hi=lo+1;
  const X=i=>pad.l+iw*(i/(series[0].data.length-1)),Y=v=>pad.t+ih*(1-(v-lo)/(hi-lo));
  // grid
  ctx.strokeStyle=col('--hair');ctx.lineWidth=1*DPR;ctx.fillStyle=col('--mut');ctx.font=(10*DPR)+'px IBM Plex Mono';
  for(let g=0;g<=4;g++){const y=pad.t+ih*g/4;ctx.globalAlpha=.5;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(w-pad.r,y);ctx.stroke();ctx.globalAlpha=1;
    const val=hi-(hi-lo)*g/4;ctx.fillText(opts.fmt?opts.fmt(val):val.toFixed(2),4*DPR,y+3*DPR);}
  if(opts.threshold!=null){const ty=Y(opts.threshold);ctx.strokeStyle=col('--rose');ctx.setLineDash([5*DPR,5*DPR]);ctx.globalAlpha=.7;ctx.beginPath();ctx.moveTo(pad.l,ty);ctx.lineTo(w-pad.r,ty);ctx.stroke();ctx.setLineDash([]);ctx.globalAlpha=1;}
  series.forEach(s=>{const c=col(s.color||'--acc');
    // area
    if(s.fill){const grd=ctx.createLinearGradient(0,pad.t,0,h);grd.addColorStop(0,c+'66');grd.addColorStop(1,c+'00');ctx.fillStyle=grd;ctx.beginPath();ctx.moveTo(X(0),Y(s.data[0]));s.data.forEach((v,i)=>ctx.lineTo(X(i),Y(v)));ctx.lineTo(X(s.data.length-1),h-pad.b);ctx.lineTo(X(0),h-pad.b);ctx.closePath();ctx.fill();}
    ctx.strokeStyle=c;ctx.lineWidth=2*DPR;ctx.lineJoin='round';ctx.beginPath();s.data.forEach((v,i)=>i?ctx.lineTo(X(i),Y(v)):ctx.moveTo(X(i),Y(v)));ctx.stroke();
    const lx=X(s.data.length-1),ly=Y(s.data[s.data.length-1]);ctx.fillStyle=c;ctx.beginPath();ctx.arc(lx,ly,3*DPR,0,6.29);ctx.fill();
  });
}
/* ---------------- custom 3D hero: distribution field ---------------- */
function mountDriftHero(id){if(typeof THREE==='undefined')return;if(matchMedia('(prefers-reduced-motion: reduce)').matches)return;
  const cv=document.getElementById(id);if(!cv)return;const r=new THREE.WebGLRenderer({canvas:cv,antialias:true,alpha:true});
  const b=cv.getBoundingClientRect();let w=b.width||480,h=b.height||480;r.setPixelRatio(Math.min(2,devicePixelRatio));r.setSize(w,h,false);
  const sc=new THREE.Scene();const cam=new THREE.PerspectiveCamera(52,w/h,.1,100);cam.position.set(0,2.4,5.2);
  const cs=getComputedStyle(document.documentElement);const C=v=>new THREE.Color(cs.getPropertyValue(v).trim()||'#2ee6c5');
  const acc=C('--acc'),acc2=C('--acc2'),amb=C('--amb');
  const grp=new THREE.Group();grp.rotation.x=-0.55;sc.add(grp);
  const NX=56,NZ=30,SX=7,SZ=4;const geo=new THREE.PlaneGeometry(SX,SZ,NX-1,NZ-1);const pos=geo.attributes.position;const base=pos.array.slice();
  const cols=[];for(let i=0;i<pos.count;i++)cols.push(0,0,0);geo.setAttribute('color',new THREE.Float32BufferAttribute(cols,3));
  const wire=new THREE.LineSegments(new THREE.WireframeGeometry(geo),new THREE.LineBasicMaterial({color:acc,transparent:true,opacity:.22,blending:THREE.AdditiveBlending}));grp.add(wire);
  const pts=new THREE.Points(geo,new THREE.PointsMaterial({size:.06,vertexColors:true,transparent:true,opacity:.95,blending:THREE.AdditiveBlending,depthWrite:false}));grp.add(pts);
  function bell(x,mu,s,a){const d=(x-mu)/s;return a*Math.exp(-0.5*d*d);}
  let mX=0,mY=0;cv.addEventListener('pointermove',e=>{const r2=cv.getBoundingClientRect();mX=(e.clientX-r2.left)/r2.width-.5;mY=(e.clientY-r2.top)/r2.height-.5;});
  const col=new THREE.Color();const t0=performance.now();
  (function loop(now){const t=(now-t0)/1000;const drift=Math.sin(t*0.5)*1.8;
    for(let i=0;i<pos.count;i++){const x=base[i*3];const z=base[i*3+1];
      const baseline=bell(x,-1.3,0.9,1.5);const current=bell(x,-1.3+drift*0.0+1.1,0.9+Math.abs(Math.sin(t*0.5))*0.5,1.5);
      const mixed=bell(x,-1.0+ Math.sin(t*0.5)*1.4, 0.95, 1.6);
      const y=mixed*Math.cos(z*0.5)+baseline*0.2;pos.setZ(i,y);
      const drift01=Math.min(1,Math.abs(Math.sin(t*0.5)*1.4+1.0-(-1.3))/3);
      col.copy(acc).lerp(acc2,(y+0.3)/2).lerp(amb,Math.max(0,Math.sin(t*0.5))*0.3);
      geo.attributes.color.setXYZ(i,col.r,col.g,col.b);
    }
    pos.needsUpdate=true;geo.attributes.color.needsUpdate=true;wire.geometry.dispose();wire.geometry=new THREE.WireframeGeometry(geo);
    grp.rotation.z=mX*0.25;grp.rotation.x=-0.55+mY*0.15;grp.position.y=-0.6;
    r.render(sc,cam);requestAnimationFrame(loop);})(t0);
  addEventListener('resize',()=>{const b2=cv.getBoundingClientRect();w=b2.width;h=b2.height;cam.aspect=w/h;cam.updateProjectionMatrix();r.setSize(w,h,false);});
}
/* ---------------- helpers ---------------- */
function statRow(c){return c.map(([l,n])=>'<div class="bstat"><div class="n">'+n+'</div><div class="l">'+l+'</div></div>').join('');}
function marq(t){const s='<span>'+t.map(x=>'<b>'+x+'</b>').join(' · ')+' · </span>';return s+s;}
function featGrid(items){return items.map(([h,ic,t,d],i)=>'<a class="fcell" href="'+h+'"><div class="fi">'+icon(ic,22)+'</div><div class="fx">0'+(i+1)+'</div><h3>'+t+'</h3><p>'+d+'</p><div class="go">Open '+icon('chevron',13)+'</div></a>').join('');}
function driftColor(d){return d<0.1?'var(--grn)':d<0.25?'var(--gold)':'var(--rose)';}
function pct(x){return Math.round(x*100)+'%';}
const FEATURES="live model monitoring · data drift (PSI) · KS distance per feature · 6 seeded production models · per-feature drift breakdown · numeric & categorical features · 30-day drift history charts · performance decay tracking · prediction-volume trends · overall drift score · stable/warning/drift verdicts · model detail pages · baseline vs current histograms · drift playground · adjustable mean shift · adjustable variance · adjustable sample size · real PSI computed live · real KS computed live · threshold alerting · alert feed · severity levels · alert rules per model · training staleness flags · environment tags (prod/staging) · task-type labels · canvas line charts · gradient area fills · threshold guide lines · sortable monitor grid · search models · health summary · 3D distribution-field hero · cursor-reactive hero · ambient motion · scroll reveals · custom cursor · light & dark · 4 accents · keyboard theme toggle · JSON export · fully offline · no data leaves device · responsive · award-grade UI";
/* ---------------- PAGES ---------------- */
window.PAGES={
 index(){
   const drifting=MODELS.filter(m=>m.status==='bad').length,warning=MODELS.filter(m=>m.status==='warn').length;
   $('meta').innerHTML=statRow([['models',MODELS.length],['drifting',drifting],['<span style="color:var(--gold)">warnings</span>',warning],['offline','100%']]);
   $('marq').innerHTML=marq(['data drift','PSI','KS distance','model decay','threshold alerts','feature breakdown','fully offline']);
   $('fgrid').innerHTML=featGrid([
     ['monitors.html','grid','Monitors','Every model at a glance — drift score, health and prediction volume.'],
     ['monitors.html','model','Model detail','Per-feature PSI & KS, drift history and performance decay.'],
     ['alerts.html','bell','Alerts','Threshold-based alerting with severity and per-model rules.'],
     ['playground.html','flask','Drift playground','Shift a distribution and watch PSI & KS move in real time.'],
     ['monitors.html','pulse','Decay tracking','See accuracy slide as production data drifts from training.'],
     ['settings.html','settings','Settings','Theme, accents and a private JSON export of everything.']
   ]);
   $('featlist').textContent='50+ features — '+FEATURES+'.';
   mountDriftHero('hero3d');
 },
 monitors(){
   let q='',sort='drift';
   function draw(){
     let list=MODELS.filter(m=>m.name.toLowerCase().includes(q)||m.id.includes(q));
     list=list.slice().sort((a,b)=>sort==='drift'?b.drift-a.drift:sort==='acc'?a.acc-b.acc:a.name.localeCompare(b.name));
     $('grid').innerHTML=list.map(m=>{const c=driftColor(m.psi);
       return '<a class="mcard" href="model.html?m='+m.id+'"><div class="top"><span class="sdot '+m.status+'"></span><span class="nm">'+esc(m.name)+'</span></div>'+
       '<div class="sub">'+m.task+' · '+m.env+'</div>'+
       '<div class="meter"><i style="width:'+Math.min(100,m.psi/0.4*100)+'%;background:'+c+'"></i></div>'+
       '<div class="mrow"><span>PSI <b style="color:'+c+'">'+m.psi.toFixed(3)+'</b> · '+psiLabel(m.psi)+'</span><span>acc <b>'+pct(m.acc)+'</b></span></div>'+
       '<div class="mrow" style="margin-top:6px"><span>'+m.features.filter(f=>f.psi>=0.25).length+' / '+m.features.length+' features drifting</span><span>'+(m.preds/1000).toFixed(1)+'k/day</span></div></a>';}).join('');
   }
   $('q').oninput=()=>{q=$('q').value.toLowerCase().trim();draw();};
   $('sortseg').querySelectorAll('button').forEach(b=>b.onclick=()=>{sort=b.dataset.s;$('sortseg').querySelectorAll('button').forEach(x=>x.classList.toggle('on',x===b));draw();});
   const healthy=MODELS.filter(m=>m.status==='ok').length;
   $('summary').innerHTML=statRow([['healthy',healthy],['<span style="color:var(--gold)">warning</span>',MODELS.filter(m=>m.status==='warn').length],['<span style="color:var(--rose)">drifting</span>',MODELS.filter(m=>m.status==='bad').length],['avg PSI',(MODELS.reduce((s,m)=>s+m.psi,0)/MODELS.length).toFixed(3)]]);
   draw();
 },
 model(){
   const id=new URLSearchParams(location.search).get('m');const m=getModel(id);
   $('mname').textContent=m.name;$('mtask').textContent=m.task+' · '+m.env+' · retrained '+m.lastTrained;
   $('mtag').innerHTML='<span class="tag '+m.status+'">'+psiLabel(m.psi)+'</span>';
   $('mstats').innerHTML=statRow([['overall PSI',m.psi.toFixed(3)],['accuracy',pct(m.acc)],['<span>decay</span>','-'+pct(m.baseAcc-m.acc)],['preds/day',(m.preds/1000).toFixed(1)+'k']]);
   // charts
   lineChart($('driftChart'),[{data:m.driftHist,color:'--acc',fill:true}],{h:190,min:0,threshold:0.25,fmt:v=>v.toFixed(2)});
   lineChart($('perfChart'),[{data:m.perfHist,color:'--blue',fill:true}],{h:190,fmt:v=>(v*100).toFixed(0)+'%'});
   lineChart($('volChart'),[{data:m.volHist,color:'--acc2',fill:true}],{h:150,fmt:v=>(v/1000).toFixed(0)+'k'});
   // feature rows (sorted by drift)
   const feats=m.features.slice().sort((a,b)=>b.psi-a.psi);
   $('feats').innerHTML=feats.map(f=>{const c=psiClass(f.psi);const col=f.psi<0.1?'var(--grn)':f.psi<0.25?'var(--gold)':'var(--rose)';
     return '<div class="frow"><div><div class="fn">'+esc(f.name)+'</div><div class="ft">'+f.type+'</div></div>'+
     '<div class="fmeter"><i style="width:'+Math.min(100,f.psi/0.5*100)+'%;background:'+col+'"></i></div>'+
     '<div class="fv" style="color:'+col+'">'+f.psi.toFixed(3)+'<div class="ft">PSI</div></div>'+
     '<div class="fv">'+f.ks.toFixed(3)+'<div class="ft">KS</div></div></div>';}).join('');
   // top drifted feature histogram
   const top=feats[0];drawHist($('histTop'),top);$('histName').textContent=top.name+' — baseline vs last 7 days';
   $('featSel').innerHTML=feats.map(f=>'<option value="'+f.name+'">'+f.name+' (PSI '+f.psi.toFixed(2)+')</option>').join('');
   $('featSel').onchange=()=>{const f=m.features.find(x=>x.name===$('featSel').value);drawHist($('histTop'),f);$('histName').textContent=f.name+' — baseline vs last 7 days';};
 },
 alerts(){
   const rules=JSON.parse(localStorage.getItem('dw_thresh')||'0.25');
   const th=typeof rules==='number'?rules:0.25;
   $('thSlide').value=th;$('thVal').textContent=th.toFixed(2);
   function build(){const t=+$('thSlide').value;$('thVal').textContent=t.toFixed(2);localStorage.setItem('dw_thresh',t);
     const als=[];MODELS.forEach(m=>{
       if(m.psi>=t){als.push({sev:m.psi>=0.25?'bad':'warn',t:m.name+' — dataset drift detected',d:'Overall PSI '+m.psi.toFixed(3)+' ≥ threshold '+t.toFixed(2)+' · '+m.env,ago:(Math.round(m.psi*40))+'m ago'});}
       m.features.filter(f=>f.psi>=t).slice(0,2).forEach(f=>als.push({sev:f.psi>=0.25?'bad':'warn',t:m.name+' · '+f.name,d:'Feature PSI '+f.psi.toFixed(3)+' · KS '+f.ks.toFixed(3),ago:(Math.round(f.psi*55))+'m ago'}));
       if(m.baseAcc-m.acc>0.1)als.push({sev:'bad',t:m.name+' — performance decay',d:'Accuracy down '+pct(m.baseAcc-m.acc)+' from baseline',ago:'1h ago'});
     });
     als.sort((a,b)=>(a.sev==='bad'?0:1)-(b.sev==='bad'?0:1));
     $('alist').innerHTML=als.length?als.map(a=>'<div class="alert '+a.sev+'"><span class="sdot '+a.sev+'" style="margin-top:5px"></span><div class="ab"><div class="at">'+esc(a.t)+'</div><div class="ad">'+esc(a.d)+'</div></div><span class="ax">'+a.ago+'</span></div>').join(''):'<div class="empty">No alerts above this threshold. Lower it to be more sensitive.</div>';
     $('acount').innerHTML=statRow([['<span style="color:var(--rose)">critical</span>',als.filter(a=>a.sev==='bad').length],['<span style="color:var(--gold)">warnings</span>',als.filter(a=>a.sev==='warn').length],['threshold',t.toFixed(2)],['models',MODELS.length]]);
   }
   $('thSlide').oninput=build;build();
 },
 playground(){
   const rng=mulberry32(42);const baseline=sample(mulberry32(7),0,1,800);
   function run(){const shift=+$('pShift').value,scale=+$('pScale').value,n=+$('pN').value;
     $('pShiftV').textContent=shift.toFixed(2);$('pScaleV').textContent=scale.toFixed(2);$('pNV').textContent=n;
     const base=baseline.slice(0,n);const cur=sample(mulberry32(99),shift,scale,n);
     const p=psi(base,cur),k=ksStat(base,cur);const lbl=psiLabel(p),cls=psiClass(p);
     // histogram
     const all=base.concat(cur);const lo=Math.min(...all),hi=Math.max(...all);const bins=22;
     const hb=histogram(base,lo,hi,bins),hc=histogram(cur,lo,hi,bins);const mx=Math.max(...hb,...hc);
     let html='';for(let i=0;i<bins;i++){html+='<div class="hbar" title="bin '+i+'"><div class="b1" style="height:'+(hb[i]/mx*130)+'px"></div><div class="b2" style="height:'+(hc[i]/mx*130)+'px"></div></div>';}
     $('phist').innerHTML=html;
     const th=+( localStorage.getItem('dw_thresh')||0.25);
     $('pverdict').className='verdict '+cls;
     $('pverdict').innerHTML='<div class="vc" style="color:'+(cls==='ok'?'var(--grn)':cls==='warn'?'var(--gold)':'var(--rose)')+'">'+lbl.toUpperCase()+'</div><div><div style="font-weight:600">PSI '+p.toFixed(3)+' · KS '+k.toFixed(3)+'</div><div style="color:var(--mut);font-size:12.5px;margin-top:2px">'+(p>=th?'⚠ would fire an alert at threshold '+th.toFixed(2):'within tolerance — no alert at '+th.toFixed(2))+'</div></div>';
     $('pkv').innerHTML=[['Baseline μ / σ','0.00 / 1.00'],['Current μ / σ',shift.toFixed(2)+' / '+scale.toFixed(2)],['Sample size',n+' × 2'],['PSI',p.toFixed(4)],['KS statistic',k.toFixed(4)],['Interpretation',p<0.1?'No significant shift':p<0.25?'Moderate — investigate':'Major shift — retrain']].map(r=>'<div class="kv"><span class="k">'+r[0]+'</span><span class="v">'+r[1]+'</span></div>').join('');
   }
   ['pShift','pScale','pN'].forEach(id=>$(id).oninput=run);
   $('preset1').onclick=()=>{$('pShift').value=0;$('pScale').value=1;$('pN').value=800;run();};
   $('preset2').onclick=()=>{$('pShift').value=0.6;$('pScale').value=1.2;$('pN').value=800;run();};
   $('preset3').onclick=()=>{$('pShift').value=1.5;$('pScale').value=1.8;$('pN').value=800;run();};
   run();
 },
 settings(){
   $('themeToggle').classList.toggle('on',theme()==='light');$('themeToggle').onclick=()=>{toggleTheme();$('themeToggle').classList.toggle('on',theme()==='light');};
   $('accents2').innerHTML=APP.accents.map((a,i)=>'<button class="ad'+(i===accentIdx()?' on':'')+'" data-i="'+i+'" data-press style="width:24px;height:24px;background:linear-gradient(135deg,'+a[0]+','+a[1]+')"></button>').join('');
   $('accents2').querySelectorAll('.ad').forEach(b=>b.onclick=()=>setAccent(+b.dataset.i));
   $('exp').onclick=()=>dl('driftwatch-export.json','application/json',JSON.stringify(MODELS.map(m=>({id:m.id,psi:m.psi,acc:m.acc,status:m.status,features:m.features.map(f=>({name:f.name,psi:f.psi,ks:f.ks}))})),null,2));
   $('featlist').textContent=FEATURES+'.';
 }
};
function drawHist(cv,f){const all=f.base.concat(f.cur);const lo=Math.min(...all),hi=Math.max(...all);const bins=24;
  const hb=histogram(f.base,lo,hi,bins),hc=histogram(f.cur,lo,hi,bins);const mx=Math.max(...hb,...hc);
  if(cv.tagName==='CANVAS'){const DPR=Math.min(2,devicePixelRatio||1);const rect=cv.getBoundingClientRect();const w=cv.width=(rect.width||500)*DPR,h=cv.height=160*DPR;const ctx=cv.getContext('2d');ctx.clearRect(0,0,w,h);
    const cs=getComputedStyle(document.documentElement);const bw=w/bins;
    for(let i=0;i<bins;i++){const x=i*bw;
      ctx.fillStyle=(cs.getPropertyValue('--blue').trim()||'#26c6ff')+'66';const h1=hb[i]/mx*(h-10*DPR);ctx.fillRect(x+1*DPR,h-h1,bw-2*DPR,h1);
      const grd=ctx.createLinearGradient(0,0,0,h);grd.addColorStop(0,cs.getPropertyValue('--acc').trim()||'#2ee6c5');grd.addColorStop(1,(cs.getPropertyValue('--acc2').trim()||'#59ff9e')+'88');ctx.fillStyle=grd;ctx.globalAlpha=.7;const h2=hc[i]/mx*(h-10*DPR);ctx.fillRect(x+1*DPR,h-h2,bw-2*DPR,h2);ctx.globalAlpha=1;}
  }
}
