const scenes={street:'street scene',shop:'shop display',posters:'poster-covered building',map:'map display'};
const resultImage=document.getElementById('gan-image');
const placeholder=document.getElementById('result-placeholder');
const spinner=document.getElementById('result-spinner');
const hint=document.getElementById('result-hint');
const revealIcon=document.getElementById('reveal-icon');
const revealButton=document.getElementById('reveal-result');
const frame=document.getElementById('result-frame');
const status=document.getElementById('reveal-status');
const generatedScenes=new Set();
const pendingGenerations=new Map();
let activeScene='street';
function renderResult(){
  const ready=generatedScenes.has(activeScene);
  const waiting=pendingGenerations.has(activeScene);
  resultImage.hidden=!ready;
  placeholder.hidden=ready;
  spinner.hidden=!waiting;
  hint.hidden=waiting;
  revealIcon.toggleAttribute('hidden',waiting);
  hint.textContent='Ready to see the difference?';
  revealButton.disabled=ready||waiting;
  revealButton.textContent=waiting?'Generating…':'Generate';
  revealButton.setAttribute('aria-busy',String(waiting));
  frame.setAttribute('aria-busy',String(waiting));
  status.textContent=ready?'DarkIR-GAN result displayed.':waiting?'Loading result.':'';
}
document.querySelectorAll('[data-scene]').forEach(button=>button.addEventListener('click',()=>{
  activeScene=button.dataset.scene;
  for(const kind of ['input','gan']){
    const img=document.getElementById(`${kind}-image`);
    img.src=`assets/${activeScene}-${kind}.webp`;
    img.alt=kind==='input'?`Dark, motion-blurred ${scenes[activeScene]}`:`${scenes[activeScene]} restored by DarkIR-GAN`;
  }
  document.querySelectorAll('[data-scene]').forEach(item=>item.setAttribute('aria-pressed',String(item===button)));
  renderResult();
}));
revealButton.addEventListener('click',async()=>{
  const scene=activeScene;
  if(generatedScenes.has(scene)||pendingGenerations.has(scene))return;
  const image=new Image();
  image.src=`assets/${scene}-gan.webp`;
  const generation=Promise.all([new Promise(resolve=>setTimeout(resolve,300)),image.decode()]);
  pendingGenerations.set(scene,generation);
  renderResult();
  let failed=false;
  try{
    await generation;
    generatedScenes.add(scene);
  }catch{
    failed=true;
  }finally{
    pendingGenerations.delete(scene);
    if(activeScene===scene){
      renderResult();
      if(failed){
        hint.textContent='Image could not load. Please try again.';
        status.textContent='Result could not load. Please try again.';
      }
    }
  }
});
document.getElementById('copy-citation').addEventListener('click',async()=>{
  const copyStatus=document.getElementById('copy-status');
  try{
    await navigator.clipboard.writeText(document.getElementById('bibtex').textContent);
    copyStatus.textContent='Citation copied to clipboard.';document.getElementById('copy-citation').textContent='Copied!';
    setTimeout(()=>document.getElementById('copy-citation').textContent='Copy BibTeX',2000);
  }catch{
    const selection=window.getSelection();const range=document.createRange();
    range.selectNodeContents(document.getElementById('bibtex'));selection.removeAllRanges();selection.addRange(range);
    copyStatus.textContent='Citation selected. Use your copy shortcut to copy it.';document.getElementById('copy-citation').textContent='Selected — copy text';
  }
});
