import { createRequire } from 'node:module';
import path from 'node:path';
const require = createRequire(path.join(process.env.VALIDATION_NODE_MODULES || process.cwd() + '/node_modules', '..', 'package.json'));
const { chromium } = require('playwright');
import fs from 'node:fs/promises';
const root=path.resolve(process.env.VALIDATION_OUTPUT || '/tmp/simlab-browser-validation');
await fs.mkdir(root,{recursive:true});
const baseUrl=process.env.VALIDATION_GUI_URL || 'http://127.0.0.1:4179';
const now=new Date().toISOString();
const host={id:'host-validation',name:'Validation host',type:'remote',enabled:true,status:'online',access:{address:'validation.local',port:22,username:'simlab',auth_method:'key',secret:{has_secret:true},host_key_fingerprint:'SHA256:validation',key_fingerprint:null},container_access:{username:'root',port:22,address_mode:'explicit',auth_method:'password',secret:{has_secret:false}},discovery:{mode:'manual',endpoint:'',label_filter:'',name_prefix:null,last_discovery:null},origin:'manual',last_check:now,last_check_message:'SSH ready',consecutive_failures:0,check_pending:false,discovery_pending:false,created_time:now,updated_time:now};
let worker={id:'worker-validation',host_id:host.id,host_name:host.name,name:'Validation Cooja',status:'available',enabled:true,container:{id:'',name:'',image:'',state:'missing',checked_at:null},endpoint:{address:'127.0.0.1',port:2222},connection:{port:2222},remote_dir:'/opt/contiki-ng/tools/cooja',probe:{ssh_ok:true,java_ok:true,java_version:'openjdk 21',cooja_jar_sha256:'1234567890abcdef',checked_at:now},current_simulation_id:null,remote_pid:null,reserved_at:null,started_at:null,origin:'manual',last_check:now,last_check_message:'Ready',docker_applies:false};
const occupancy={id:host.id,name:host.name,type:'remote',status:'online',enabled:true,schedulable:true,resource_total:1,free:1,active:0,unavailable:0,disabled:0,last_check:now,last_check_message:'Ready'};
const overview={totals:{hosts:1,hosts_operational:1,resources:1,free:1,active:0,unavailable:0,disabled:0},hosts:[occupancy],heterogeneous:false,cooja_builds:['1234567890abcdef']};
const writes=[],errors=[],audits=[];
const browser=await chromium.launch({headless:true});
const context=await browser.newContext({viewport:{width:1440,height:1000}});
const page=await context.newPage();
page.on('pageerror',error=>errors.push(error.message));
await page.route('**/api/**',async route=>{
 if(!new URL(route.request().url()).pathname.startsWith('/api/'))return route.continue();
 const req=route.request(),path=new URL(req.url()).pathname.replace('/api','');
 let status=200,data={};
 if(req.method()!=='GET'){
  writes.push({method:req.method(),path,body:req.postDataJSON()});
  if(path.endsWith('/connection')){const p=req.postDataJSON();worker={...worker,...p,connection:{...worker.connection,...p.connection}};if(!p.connection?.port)delete worker.connection.port;data=worker;}
  else if(path.endsWith('/check')){status=202;data={accepted:true};}
  else {status=409;data={detail:'Validation conflict: recover the retained result before removal'};}
 } else if(path==='/infra/hosts')data=[host];
 else if(path==='/infra/resources')data=[worker];
 else if(path==='/infra/overview')data=overview;
 else if(path==='/infra/crypto/public-key'){status=503;data={detail:'Validation: sealing key not configured'};}
 else data=[];
 await route.fulfill({status,contentType:'application/json',body:JSON.stringify(data)});
});
async function audit(name){
 await page.evaluate(()=>Promise.all(document.getAnimations().filter(a=>a.effect?.getTiming().iterations!==Infinity).map(a=>a.finished.catch(()=>{}))));
 await page.addScriptTag({path:require.resolve('axe-core/axe.min.js')});
 const result=await page.evaluate(async()=>await window.axe.run(document,{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21aa']}}));
 audits.push({name,violations:result.violations.map(v=>({id:v.id,impact:v.impact,nodes:v.nodes.map(n=>({target:n.target,summary:n.failureSummary}))}))});
}
try{
 await page.goto(baseUrl+'/infrastructure');
 await page.getByRole('heading',{name:'Infrastructure',exact:true}).waitFor();
 await page.getByText('Validation host',{exact:false}).first().waitFor();
 const details=page.locator('details.containers');
 if(!(await details.evaluate(el=>el.open)))await details.locator('summary').click();
 await page.getByRole('button',{name:'Edit',exact:true}).last().waitFor();
 await page.screenshot({path:root+'/infra-desktop.png',fullPage:true});
 await audit('infrastructure-desktop');
 const edit=page.getByRole('button',{name:'Edit',exact:true}).last();
 await edit.click();
 const dialog=page.getByRole('dialog');
 await dialog.waitFor();
 await page.getByLabel('SSH port',{exact:true}).fill('2223');
 await audit('worker-edit');
 await page.screenshot({path:root+'/worker-edit-desktop.png',fullPage:true});
 const buttons=await dialog.getByRole('button').allTextContents();
 console.log('worker buttons',buttons);
 await dialog.getByRole('button',{name:/Save/}).click();
 await dialog.waitFor({state:'hidden'});
 if(writes[0]?.body?.endpoint?.port!==2223)throw Error('worker port not submitted');
 await edit.click();await dialog.waitFor();
 await page.keyboard.press('Escape');
 await dialog.waitFor({state:'hidden'});
 const focusRestored=await edit.evaluate(el=>el===document.activeElement);
 await page.getByRole('button',{name:'Register host',exact:true}).click();
 await dialog.waitFor();
 await audit('host-register');
 await page.screenshot({path:root+'/host-register-desktop.png',fullPage:true});
 await page.keyboard.press('Escape');
 await page.setViewportSize({width:390,height:844});
 await page.screenshot({path:root+'/infra-mobile.png',fullPage:true});
 const overflow=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth}));
 await page.getByRole('button',{name:'Add worker',exact:true}).click();
 await dialog.waitFor();
 await audit('worker-mobile');
 await page.screenshot({path:root+'/worker-mobile.png',fullPage:true});
 for(let i=0;i<18;i++){await page.keyboard.press('Tab');if(!(await dialog.evaluate(el=>el.contains(document.activeElement) || document.activeElement === document.body)))throw Error('focus escaped modal');}
 await page.keyboard.press('Escape');
 await page.setViewportSize({width:1440,height:1000});
 await page.getByRole('button',{name:/Dark mode/}).click();
 await audit('infrastructure-dark');
 await page.getByRole('button',{name:'Add worker',exact:true}).click();
 await dialog.waitFor();
 await audit('worker-dark');
 await page.screenshot({path:root+'/worker-dark.png',fullPage:true});
 await page.keyboard.press('Escape');
 await page.getByRole('button',{name:/Light mode/}).click();
 await page.getByRole('button',{name:'Remove',exact:true}).last().click();
 await page.getByRole('alertdialog').getByRole('button',{name:'Remove',exact:true}).click();
 await page.getByText('Validation conflict: recover the retained result before removal',{exact:false}).waitFor();
 const conflictVisible=true;
 // Render the real individual panel with controlled evaluation records. This
 // isolates the UI contract from experiment creation and mutation.
 await page.route('**/validation-panel',route=>route.fulfill({contentType:'text/html',body:`
 <!doctype html><html lang="en"><head><title>Validation panel</title>
 <link rel="stylesheet" href="/src/style.css"></head><body><main id="app"></main>
 <script type="module">
 import {createApp,h,ref} from '/node_modules/.vite/deps/vue.js';
 import Panel from '/src/components/detail/IndividualDetailPanel.vue';
 const record=ref({state:'partial',observed:1,expected:2,missing_seeds:[2],absent:1,absent_seeds:[2],per_metric:{latency:1},objective_used:false,reason:'One planned replica was never created'});
 window.setValidationRecord=value=>record.value=value;
 createApp({setup(){return()=>h(Panel,{individual:{individual_id:'validation-genome',objectives:[1],chromosome:{gene:1},completeness:record.value},experimentId:'validation-experiment',objectiveNames:['latency'],metricColumns:['latency']})}}).mount('#app');
 </script></body></html>`}));
 await page.goto(baseUrl+'/validation-panel');
 await page.getByText('incomplete (1/2 replicas)',{exact:true}).waitFor();
 await page.getByText('1 replica(s) were never created (seeds 2)',{exact:true}).waitFor();
 await page.getByText('the algorithm was given the worst objective for this individual',{exact:true}).waitFor();
 await page.screenshot({path:root+'/individual-incomplete.png',fullPage:true});
 await page.evaluate(()=>window.setValidationRecord({state:'complete',observed:2,expected:2,reused:true,source:'genome_cache',objective_used:true,per_metric:{latency:2}}));
 await page.getByText('complete (2/2 replicas)',{exact:true}).waitFor();
 await page.getByText('carried over from an earlier evaluation of the same genome',{exact:true}).waitFor();
 await page.evaluate(()=>window.setValidationRecord(null));
 await page.getByText('not recorded',{exact:true}).waitFor();
 const result={browserVersion:browser.version(),writes,errors,audits,focusRestored,overflow,conflictVisible,completenessStates:['partial','reused complete','not recorded']};
 await fs.writeFile(root+'/gui-result.json',JSON.stringify(result,null,2));
 console.log(JSON.stringify(result,null,2));
 if(errors.length || !focusRestored || overflow.scroll>overflow.width || audits.some(a=>a.violations.length))throw Error('browser validation failed; see gui-result.json');
}catch(error){await fs.writeFile(root+'/gui-result.json',JSON.stringify({writes,errors,audits},null,2));await page.screenshot({path:root+'/ui-failure.png',fullPage:true});console.log({errors,html:await page.locator('body').innerText()});throw error;}finally{await browser.close();}
