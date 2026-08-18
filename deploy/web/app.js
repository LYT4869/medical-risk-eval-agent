const state={token:null,user:null,predictions:[]};
const $=id=>document.getElementById(id);
const randomKey=()=>crypto.randomUUID().replaceAll('-','');
function trace(response){$('trace').textContent=`request: ${response.headers.get('X-Request-Id')||'-'}\ntrace: ${response.headers.get('X-Trace-Id')||'-'}`}
async function api(path,{method='GET',body,auth=true,idempotent=false}={}){
  const headers={'Content-Type':'application/json'};
  if(auth&&state.token)headers.Authorization=`Bearer ${state.token}`;
  if(idempotent)headers['Idempotency-Key']=randomKey();
  const response=await fetch(path,{method,headers,credentials:'include',body:body?JSON.stringify(body):undefined});trace(response);
  const text=await response.text();let data={};try{data=text?JSON.parse(text):{}}catch{data={message:text}}
  if(!response.ok)throw new Error(data.message||data.error||`HTTP ${response.status}`);return data;
}
function identity(){ $('identity').textContent=state.user?`${state.user.display_name} · ${state.user.role}`:'尚未登录' }
function showError(error){$('details').textContent=`错误：${error.message}`;$('details').classList.add('error')}
async function authenticate(path,body){const data=await api(path,{method:'POST',body,auth:false});state.token=data.access_token;state.user=data.user;identity();return data}
$('register').onclick=()=>authenticate('/api/v1/auth/register',{email:$('email').value,password:$('password').value,display_name:$('displayName').value}).catch(showError);
$('login').onclick=()=>authenticate('/api/v1/auth/login',{email:$('email').value,password:$('password').value}).catch(showError);
$('predict').onclick=async()=>{try{const data=await api('/api/v1/predictions',{method:'POST',body:{sample_index:Number($('sampleIndex').value)}});state.predictions.unshift(data.prediction_id);$('prediction').innerHTML=`<b>${data.prediction_id}</b><br>label ${data.prediction.label} · probability ${data.prediction.positive_probability.toFixed(4)}<br>${data.model_version} · ${data.serving_backend}`;$('details').textContent=JSON.stringify(data,null,2)}catch(e){showError(e)}};
$('history').onclick=async()=>{try{const data=await api('/api/v1/sessions/current/history');state.predictions=data.items.map(x=>x.prediction_id);$('historyList').innerHTML=data.items.map(x=>`<div class="list-item"><span>${x.prediction_id}</span><span>${Number(x.positive_probability).toFixed(3)}</span></div>`).join('')||'暂无记录'}catch(e){showError(e)}};
$('compare').onclick=async()=>{try{if(state.predictions.length<2)throw new Error('至少需要两条预测');const data=await api('/api/v1/comparisons',{method:'POST',body:{prediction_id_a:state.predictions[1],prediction_id_b:state.predictions[0]}});$('details').textContent=JSON.stringify(data,null,2)}catch(e){showError(e)}};
$('explain').onclick=async()=>{try{if(!state.predictions.length)throw new Error('请先运行或刷新预测');const data=await api(`/api/v1/predictions/${state.predictions[0]}/explanation`);$('details').textContent=JSON.stringify(data,null,2)}catch(e){showError(e)}};
function chat(role,text,citations=[]){const node=document.createElement('div');node.className=`message ${role}`;node.textContent=text;if(citations.length){const refs=document.createElement('div');refs.className='citations';refs.textContent='引用：'+citations.map(x=>`${x.citation_id} · ${x.title}`).join(' | ');node.append(refs)}$('chatLog').append(node);$('chatLog').scrollTop=$('chatLog').scrollHeight}
$('send').onclick=async()=>{const message=$('message').value.trim();if(!message)return;chat('user',message);$('message').value='';try{const data=await api('/api/v1/chat',{method:'POST',body:{message},idempotent:true});chat('assistant',data.answer,data.citations||[])}catch(e){chat('assistant',`暂时无法完成：${e.message}`)}};
$('doctorPredict').onclick=async()=>{try{const patient=$('patientId').value.trim();const data=await api(`/api/v1/doctor/patients/${patient}/predictions`,{method:'POST',body:{sample_index:Number($('sampleIndex').value)}});state.predictions.unshift(data.prediction_id);$('details').textContent=JSON.stringify(data,null,2)}catch(e){showError(e)}};
$('doctorChat').onclick=async()=>{try{const patient=$('patientId').value.trim();const message=$('message').value.trim()||'解释当前预测';const data=await api(`/api/v1/doctor/patients/${patient}/chat`,{method:'POST',body:{message},idempotent:true});chat('assistant',data.answer,data.citations||[])}catch(e){showError(e)}};
$('doctorFeedback').onclick=async()=>{try{if(!state.predictions.length)throw new Error('请先选择或创建一条预测');const data=await api(`/api/v1/predictions/${state.predictions[0]}/feedback`,{method:'POST',body:{assessment:'agree'},idempotent:true});$('details').textContent=JSON.stringify(data,null,2)}catch(e){showError(e)}};
identity();
