const scenes={street:'street scene',shop:'shop display',posters:'poster-covered building',map:'map display'};
const resultImage=document.getElementById('gan-image');
const placeholder=document.getElementById('result-placeholder');
const spinner=document.getElementById('result-spinner');
const hint=document.getElementById('result-hint');
const revealIcon=document.getElementById('reveal-icon');
const revealButton=document.getElementById('reveal-result');
const frame=document.getElementById('result-frame');
const status=document.getElementById('reveal-status');
let request=0;
function resetResult(){
  request++;
  resultImage.hidden=true;placeholder.hidden=false;spinner.hidden=true;
  hint.hidden=false;revealIcon.removeAttribute('hidden');hint.textContent='Ready to see the difference?';
  revealButton.disabled=false;revealButton.innerHTML='Show result <span aria-hidden="true">→</span>';
  frame.setAttribute('aria-busy','false');status.textContent='';
}
document.querySelectorAll('[data-scene]').forEach(button=>button.addEventListener('click',()=>{
  const scene=button.dataset.scene;resetResult();
  for(const kind of ['input','gan']){
    const img=document.getElementById(`${kind}-image`);img.src=`assets/${scene}-${kind}.webp`;
    img.alt=kind==='input'?`Dark, motion-blurred ${scenes[scene]}`:`${scenes[scene]} restored by DarkIR-GAN`;
  }
  document.querySelectorAll('[data-scene]').forEach(item=>item.setAttribute('aria-pressed',String(item===button)));
}));
revealButton.addEventListener('click',async()=>{
  const current=++request;
  resultImage.hidden=true;placeholder.hidden=false;spinner.hidden=false;hint.hidden=true;revealIcon.setAttribute('hidden','');
  revealButton.disabled=true;revealButton.textContent='Please wait…';frame.setAttribute('aria-busy','true');status.textContent='Loading result.';
  try{
    await Promise.all([new Promise(resolve=>setTimeout(resolve,300)),resultImage.decode()]);
    if(current!==request)return;
    resultImage.hidden=false;placeholder.hidden=true;frame.setAttribute('aria-busy','false');
    revealButton.disabled=false;revealButton.textContent='Show again ↻';status.textContent='DarkIR-GAN result displayed.';
  }catch{
    if(current!==request)return;
    resetResult();hint.textContent='Image could not load. Please try again.';status.textContent='Result could not load. Please try again.';
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
