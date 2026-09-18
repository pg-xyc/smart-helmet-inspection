importScripts('./vendor/ort.wasm.min.js');
ort.env.wasm.numThreads=1;
ort.env.wasm.wasmPaths=new URL('./vendor/',self.location.href).href;
let session;
const names={0:'安全帽',6:'人员',7:'未戴安全帽'};
function iou(a,b){const w=Math.max(0,Math.min(a[2],b[2])-Math.max(a[0],b[0])),h=Math.max(0,Math.min(a[3],b[3])-Math.max(a[1],b[1]));const inter=w*h;return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter||1);}
function decode(output){const data=output.data,n=output.dims[2],candidates=[];
if(output.dims[1]!==15)throw Error('模型输出格式不匹配');
for(let i=0;i<n;i++){let cls=0,confidence=0;for(let c=0;c<11;c++){const score=data[(4+c)*n+i];if(score>confidence){confidence=score;cls=c;}}
if(confidence<.4||!(cls in names))continue;const x=data[i],y=data[n+i],w=data[2*n+i],h=data[3*n+i];candidates.push({xyxy:[x-w/2,y-h/2,x+w/2,y+h/2],cls,label:names[cls],confidence});}
candidates.sort((a,b)=>b.confidence-a.confidence);const keep=[];for(const b of candidates){if(!keep.some(a=>a.cls===b.cls&&iou(a.xyxy,b.xyxy)>.45))keep.push(b);if(keep.length>=100)break;}return keep;}
self.onmessage=async({data})=>{try{if(data.type==='init'){session=await ort.InferenceSession.create(new URL('./helmet.onnx',self.location.href).href,{executionProviders:['wasm'],graphOptimizationLevel:'all'});self.postMessage({type:'ready'});}else if(data.type==='frame'){const start=performance.now(),input=new ort.Tensor('float32',data.pixels,[1,3,416,416]);let results;try{results=await session.run({[session.inputNames[0]]:input});self.postMessage({type:'result',id:data.id,boxes:decode(results[session.outputNames[0]]),ms:Math.round(performance.now()-start)});}finally{input.dispose();if(results)Object.values(results).forEach(t=>t.dispose());}}}catch(e){self.postMessage({type:'error',id:data.id,message:e.message});}};
