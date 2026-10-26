/* User portal: durable server jobs, with polling independent of upload lifetime. */
(() => {
  const $ = id => document.getElementById(id);
  const API = window.USER_BASE + '/api/user';
  const labels = {queued_ocr:'รอ OCR',ocr:'กำลัง OCR',queued_extraction:'รอ Extraction',
    extraction:'กำลังดึงข้อมูล',completed:'เสร็จแล้ว',partial:'เสร็จบางส่วน',failed:'ไม่สำเร็จ',interrupted:'ถูกขัดจังหวะ'};
  const active = status => ['queued_ocr','ocr','queued_extraction','extraction'].includes(status);
  const fieldLabels = {document_type:'ประเภทเอกสาร',document_number:'เลขที่เอกสาร',issue_date:'วันที่เอกสาร',
    seller_tax_id:'เลขประจำตัวผู้เสียภาษีผู้ขาย',seller_branch:'สาขาที่ผู้ขาย',seller_branch_location:'ชื่อ/ตำแหน่งสาขาผู้ขาย',
    buyer_tax_id:'เลขประจำตัวผู้เสียภาษีผู้ซื้อ',buyer_branch:'สาขาที่ผู้ซื้อ',buyer_branch_location:'ชื่อ/ตำแหน่งสาขาผู้ซื้อ',
    seller_name:'ชื่อผู้ขาย',buyer_name:'ชื่อผู้ซื้อ',subtotal:'ยอดก่อน VAT',vat_total:'ภาษีมูลค่าเพิ่ม',
    amount_incl_vat:'ยอดรวม VAT',currency:'สกุลเงิน',payment_date:'วันที่ชำระเงิน',cheque_number:'เลขที่เช็ค',
    po_gr_rtv_number:'เลขที่ PO / GR / RTV',inv_rtv_cnr_number:'เลขที่เอกสารอ้างอิง INV / RTV / CNR',
    remaining_amount:'ยอดคงค้าง',remaining_vat_amount:'ภาษีคงค้าง',payer_name:'ชื่อผู้จ่ายเงิน',payee_name:'ชื่อผู้รับเงิน',
    payer_tax_id:'เลขประจำตัวผู้เสียภาษีผู้จ่ายเงิน',payee_tax_id:'เลขประจำตัวผู้เสียภาษีผู้รับเงิน',
    payer_address:'ที่อยู่ผู้จ่ายเงิน',payee_address:'ที่อยู่ผู้รับเงิน',payer_branch:'สาขาที่ผู้จ่ายเงิน',payee_branch:'สาขาที่ผู้รับเงิน',
    payer_branch_location:'ชื่อ/ตำแหน่งสาขาผู้จ่ายเงิน',payee_branch_location:'ชื่อ/ตำแหน่งสาขาผู้รับเงิน',
    income_items:'รายการเงินได้',line_items:'รายการสินค้า',other_fields:'ข้อมูลอื่น ๆ'};
  const dates = value => new Date(value).toLocaleString('th-TH');
  let files = [], settings = null, sending = false, jobs = [], batchJobs = [], selected = null;
  let historyTimer, detailTimer, batchTimer, historyLoading = false, batchLoading = false, more = false;
  $('homeLink').href = window.USER_BASE + '/user'; $('benchmarkLink').href = window.USER_BASE + '/';
  function node(tag, text, className) {
    const element = document.createElement(tag); if (text !== undefined) element.textContent = text;
    if (className) element.className = className; return element;
  }
  function message(id, text, error = false) {
    $(id).textContent = text || ''; $(id).hidden = !text; $(id).className = 'message' + (error ? ' error' : '');
  }
  async function api(path, options = {}) {
    const response = await fetch(API + path, options);
    let body; try { body = await response.json(); } catch (_) { throw new Error('Server ไม่ตอบกลับเป็น JSON (' + response.status + ')'); }
    if (!response.ok) throw new Error(body.error || 'ดำเนินการไม่สำเร็จ (' + response.status + ')');
    return body;
  }
  async function loadSettings() {
    try {
      settings = (await api('/settings')).settings;
      $('uploadLimits').textContent = 'สูงสุด ' + settings.max_upload_mb + ' MB ต่อไฟล์ · ' +
        (settings.user_file_max ? 'เลือกได้ ' + settings.user_file_max + ' ไฟล์ต่อครั้ง' : 'ไม่จำกัดจำนวนไฟล์ต่อครั้ง');
      paintFiles();
    } catch (err) { message('uploadMessage', err.message + ' · โหลดหน้าใหม่เพื่อลองอีกครั้ง', true); }
  }
  function paintFiles() {
    $('selectedFiles').replaceChildren(...files.map((file, index) => {
      const row = node('li'); row.append(node('span', file.name), node('span', (file.size/1024/1024).toFixed(2)+' MB','muted'));
      const remove = node('button','×','subtle'); remove.type='button'; remove.disabled=sending;
      remove.setAttribute('aria-label','นำ '+file.name+' ออก');
      remove.addEventListener('click',()=>{files.splice(index,1);paintFiles();});row.append(remove);return row;
    }));
    $('runButton').disabled = sending || !settings || !files.length;
    $('clearFiles').disabled = sending; $('fileInput').disabled = sending;
  }
  function choose(list) {
    if (sending) return;
    files = [...list]; paintFiles(); message('uploadMessage','');
  }
  $('fileInput').addEventListener('change', event => choose(event.target.files));
  $('clearFiles').addEventListener('click',()=>{files=[];$('fileInput').value='';paintFiles();});
  $('dropzone').addEventListener('keydown', event => {if(event.key==='Enter'||event.key===' '){event.preventDefault();if(!sending)$('fileInput').click();}});
  for (const name of ['dragenter','dragover']) $('dropzone').addEventListener(name,event=>{event.preventDefault();$('dropzone').classList.add('drag');});
  $('dropzone').addEventListener('dragleave',()=>$('dropzone').classList.remove('drag'));
  $('dropzone').addEventListener('drop',event=>{event.preventDefault();$('dropzone').classList.remove('drag');choose(event.dataTransfer.files);});
  $('uploadForm').addEventListener('submit',async event=>{
    event.preventDefault();if(sending||!files.length||!settings)return;
    if(settings.user_file_max && files.length>settings.user_file_max){message('uploadMessage','เลือกได้สูงสุด '+settings.user_file_max+' ไฟล์ต่อครั้ง',true);return;}
    const tooLarge=files.find(file=>file.size>settings.max_upload_mb*1024*1024);
    if(tooLarge){message('uploadMessage',tooLarge.name+' ใหญ่เกิน '+settings.max_upload_mb+' MB',true);return;}
    sending=true;paintFiles();clearTimeout(batchTimer);clearTimeout(detailTimer);
    selected=null;batchJobs=[];paintBatch();
    let received=0;const failed=[];
    try {
      const batch=await api('/batches',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({file_count:files.length})});
      // One multipart request per file: the size limit never applies to the sum.
      // Keep three uploads in flight; model concurrency is controlled by the server.
      let next=0;
      async function uploadWorker(){while(next<files.length){
        const index=next++,file=files[index],data=new FormData();data.append('file',file);data.append('batch_id',batch.batch_id);
        data.append('upload_id','file_'+index+'_'+batch.batch_id);
        try{const {job}=await api('/jobs',{method:'POST',body:data});received++;batchJobs.push(job);paintBatch();
          if(!selected)openJob(job.id);pollBatch();
        }catch(err){failed.push(file.name+': '+err.message);}
        message('uploadMessage','ส่งถึง server แล้ว '+received+'/'+files.length+' ไฟล์'+(failed.length?' · ส่งไม่สำเร็จ '+failed.length+' ไฟล์':''));
      }}
      await Promise.all(Array.from({length:Math.min(3,files.length)},uploadWorker));
      message('uploadMessage','ส่งถึง server แล้ว '+received+'/'+files.length+' ไฟล์'+(failed.length?'\n'+failed.join('\n'): ' · เปิดดูสถานะและประวัติได้เลย'),!!failed.length);
      loadHistory();
    }catch(err){message('uploadMessage',err.message,true);}
    finally{sending=false;paintFiles();pollBatch();}
  });
  function paintBatch(){
    $('batchJobs').replaceChildren(...batchJobs.map(job=>{
      const button=node('button',undefined,'batch-row'+(job.id===selected?' current':''));button.type='button';
      button.append(node('span',job.filename),node('span',labels[job.status]||job.status,'badge '+job.status));
      button.addEventListener('click',()=>openJob(job.id));return button;
    }));
  }
  async function pollBatch(){
    if(batchLoading)return;
    clearTimeout(batchTimer);
    const waiting=batchJobs.filter(job=>active(job.status));
    if(!waiting.length)return;
    batchLoading=true;
    // Fetch updates serially to avoid a large client-side burst after a big batch.
    try{for(const item of waiting){const {job}=await api('/jobs/'+item.id);Object.assign(item,job);}paintBatch();}
    catch(_){/* Poll again; already accepted jobs keep running on the server. */}
    finally{batchLoading=false;}
    if(batchJobs.some(job=>active(job.status)))batchTimer=setTimeout(pollBatch,2000);
  }
  function view(history){$('uploadView').hidden=history;$('historyView').hidden=!history;
    $('navUpload').classList.toggle('selected',!history);$('navHistory').classList.toggle('selected',history);}
  $('navUpload').addEventListener('click',()=>view(false));
  $('navHistory').addEventListener('click',()=>{view(true);loadHistory();});
  function valueText(value){return typeof value==='object'&&value!==null?JSON.stringify(value,null,2):String(value??'');}
  function renderFields(extracted){
    const target=$('fieldsResult');target.replaceChildren();
    if(!extracted){target.append(node('p','ข้อมูลจะแสดงเมื่อ Extraction เสร็จ','muted'));return;}
    for(const [index,doc] of (extracted.documents||[extracted]).entries()){
      const section=node('section',undefined,'document'),heading=node('h3','เอกสาร '+(index+1));
      heading.append(node('small',(doc.pages?.length?'หน้า '+doc.pages.join(', ')+' · ':'')+(doc.doc_types||[]).join(' / ')));
      section.append(heading);if(doc.error)section.append(node('p',doc.error,'message error'));
      for(const failed of [...(doc.steps_failed||[]),...(doc.steps_extras||[])])
        section.append(node('p',(failed.title||failed.id)+': '+failed.error,'message error'));
      const table=node('div',undefined,'field-table');
      for(const [key,value]of Object.entries(doc.fields||{})){
        const row=node('div',undefined,'field-row'),text=valueText(value);
        const label=key==='subtotal' && doc.doc_types?.length===1 && doc.doc_types[0]==='BILLING_NOTE'
          ? 'ยอดวางบิล' : fieldLabels[key] || key.replace(/_/g,' ');
        row.append(node('div',label,'field-key'),node('div',text||'—','field-value'+(text?'':' empty-value')));table.append(row);
      }
      if(!table.children.length)section.append(node('p','ไม่พบข้อมูลที่ดึงได้','muted'));else section.append(table);
      if(doc.item_table){const detail=node('details');detail.append(node('summary','รายการในตาราง'),node('pre',valueText(doc.item_table),'transcript'));section.append(detail);}
      target.append(section);
    }
  }
  function paintJob(job){
    $('emptyResult').hidden=true;$('jobResult').hidden=false;$('resultStatus').hidden=false;
    $('resultFile').textContent=job.filename;$('resultDate').textContent=dates(job.created_at);
    $('resultStatus').textContent=labels[job.status]||job.status;$('resultStatus').className='badge '+job.status;
    $('progress').textContent=job.progress;const result=job.result||{},timing=result.timing||{};
    $('timing').textContent=[['ocr','OCR'],['extraction','Extraction']].filter(([key])=>timing[key]!==undefined).map(([key,name])=>name+' '+timing[key].toFixed(1)+' วินาที').join(' · ');
    $('ocrStep').className=job.status==='ocr'?'current':result.pages?.length?'done':'';
    $('extractionStep').className=job.status==='extraction'?'current':result.extracted?'done':'';
    $('finishedStep').className=['completed','partial'].includes(job.status)?'done':'';
    message('jobError',job.error,true);$('warnings').replaceChildren(...(result.warnings||[]).map(text=>node('p',text,'message')));
    $('originalLink').href=API+'/jobs/'+job.id+'/file';$('exportLink').href=API+'/jobs/'+job.id+'/export';
    $('textResult').textContent=result.text||'ยังไม่มีข้อความ OCR';renderFields(result.extracted);
    $('jobSettings').textContent=JSON.stringify(job.configuration,null,2);paintBatch();
  }
  async function openJob(id,show=true){
    clearTimeout(detailTimer);selected=id;if(show)view(false);
    try{const {job}=await api('/jobs/'+id);if(selected!==id)return;paintJob(job);
      if(show)$('resultTitle').scrollIntoView?.({behavior:'smooth',block:'start'});
      if(active(job.status))detailTimer=setTimeout(()=>openJob(id,false),1200);
    }catch(err){message('jobError',err.message,true);detailTimer=setTimeout(()=>{if(selected===id)openJob(id,false);},4000);}
  }
  async function loadHistory(append=false){
    if(historyLoading)return;historyLoading=true;clearTimeout(historyTimer);
    try{
      const body=await api('/jobs?offset='+(append?jobs.length:0));jobs=append?[...jobs,...body.jobs]:body.jobs;more=body.more;
      $('historyRows').replaceChildren(...jobs.map(job=>{const row=node('tr'),status=node('td'),action=node('td');
        status.append(node('span',labels[job.status]||job.status,'badge '+job.status));
        const button=node('button','เปิดผลลัพธ์ →','subtle');button.addEventListener('click',()=>openJob(job.id));action.append(button);
        row.append(node('td',job.filename),node('td',dates(job.created_at)),status,action);return row;}));
      $('emptyHistory').hidden=!!jobs.length;$('moreHistory').hidden=!more;
      const counts=body.counts||{};$('queueSummary').textContent='OCR '+(counts.ocr||0)+' ไฟล์ · Extraction '+(counts.extraction||0)+' ไฟล์ · รอ '+((counts.queued_ocr||0)+(counts.queued_extraction||0))+' ไฟล์';
      message('historyError','');
    }catch(err){message('historyError',err.message,true);}
    finally{historyLoading=false;if(jobs.length<=50)historyTimer=setTimeout(()=>loadHistory(),5000);}
  }
  $('refreshHistory').addEventListener('click',()=>loadHistory());$('moreHistory').addEventListener('click',()=>loadHistory(true));
  function tab(fields){$('fieldsResult').hidden=!fields;$('textResult').hidden=fields;$('fieldsTab').classList.toggle('selected',fields);$('textTab').classList.toggle('selected',!fields);}
  $('fieldsTab').addEventListener('click',()=>tab(true));$('textTab').addEventListener('click',()=>tab(false));
  window.addEventListener('beforeunload',event=>{if(sending){event.preventDefault();event.returnValue='';}});
  window.addEventListener('pagehide',()=>{clearTimeout(historyTimer);clearTimeout(detailTimer);clearTimeout(batchTimer);});
  loadSettings();loadHistory();
})();
