import fs from 'node:fs';import * as T from 'three';import {product} from '../src/objects.js';
T.TextureLoader.prototype.load=function(){return new T.Texture();};
const mats=new Map(),M=new Proxy({},{get:(_,k)=>{if(!mats.has(k))mats.set(k,new T.MeshStandardMaterial());return mats.get(k);}});
const d=JSON.parse(fs.readFileSync(new URL('../public/data.json',import.meta.url))),errors=[];let meshes=0;
for(const p of d.catalog){try{const g=product(p,M),b=new T.Box3().setFromObject(g),s=b.getSize(new T.Vector3());if(g.userData.fallback)errors.push([p.id,'fallback',p.kind]);if([s.x,s.z,s.y].some((v,i)=>!Number.isFinite(v)||Math.abs(v-p.dims[i])>.001))errors.push([p.id,'dimensions',s.toArray(),p.dims]);g.traverse(o=>{if(o.isMesh){meshes++;const a=o.geometry.attributes.position.array;if(a.some(v=>!Number.isFinite(v)))errors.push([p.id,'nonfinite']);o.geometry.dispose();}});}catch(e){errors.push([p.id,e.stack]);}}
console.log(JSON.stringify({products:d.catalog.length,meshes,errors},null,2));fs.writeFileSync(new URL('../docs/model-audit.json',import.meta.url),JSON.stringify({products:d.catalog.length,meshes,errors},null,2));

if (errors.length) process.exitCode = 1;
