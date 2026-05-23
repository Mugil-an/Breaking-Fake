// Prefer explicit VITE_API_BASE; otherwise in dev use Vite proxy at '/api' to avoid CORS
const API_BASE = import.meta.env.VITE_API_BASE || (import.meta.env.DEV ? '/api' : 'http://localhost:8000')
const API_KEY = import.meta.env.VITE_API_KEY || ''

function _authHeader(){
  return API_KEY ? { 'Authorization': 'Bearer ' + API_KEY } : {}
}

function _backoff(n){
  const base = 500
  return Math.min(10000, base * Math.pow(2, n))
}

async function requestJson(path, options = {}, retries = 2){
  const headers = {...(options.headers||{}), ..._authHeader()}
  try{
    const res = await fetch(API_BASE + path, {...options, headers})
    if(!res.ok){
      const text = await res.text()
      throw new Error(`${res.status} ${res.statusText}: ${text}`)
    }
    return res.json()
  }catch(e){
    if(retries>0){
      await new Promise(r=>setTimeout(r, _backoff(2-retries)))
      return requestJson(path, options, retries-1)
    }
    throw e
  }
}

export function predictSingle(file, onProgress){
  // XHR for upload progress; returns parsed JSON
  return new Promise((resolve, reject)=>{
    const xhr = new XMLHttpRequest()
    const url = API_BASE + '/predict'
    xhr.open('POST', url)
    if(API_KEY) xhr.setRequestHeader('Authorization', 'Bearer ' + API_KEY)

    xhr.upload.onprogress = function(e){
      if(e.lengthComputable && onProgress) onProgress(Math.round((e.loaded / e.total) * 100))
    }

    xhr.onload = function(){
      if(xhr.status >= 200 && xhr.status < 300){
        try{ resolve(JSON.parse(xhr.responseText)) }catch(err){ reject(err) }
      } else {
        reject(new Error(`${xhr.status} ${xhr.statusText}: ${xhr.responseText}`))
      }
    }

    xhr.onerror = function(){ reject(new Error('Network error')) }

    const form = new FormData()
    form.append('file', file)
    xhr.send(form)
  })
}

export function predictBatchFile(csvFile, onProgress){
  return new Promise((resolve, reject)=>{
    const xhr = new XMLHttpRequest()
    const url = API_BASE + '/predict-batch?background=true'
    xhr.open('POST', url)
    if(API_KEY) xhr.setRequestHeader('Authorization', 'Bearer ' + API_KEY)

    xhr.upload.onprogress = function(e){
      if(e.lengthComputable && onProgress) onProgress(Math.round((e.loaded / e.total) * 100))
    }

    xhr.onload = function(){
      if(xhr.status >= 200 && xhr.status < 300){
        try{ resolve(JSON.parse(xhr.responseText)) }catch(err){ reject(err) }
      } else {
        reject(new Error(`${xhr.status} ${xhr.statusText}: ${xhr.responseText}`))
      }
    }

    xhr.onerror = function(){ reject(new Error('Network error')) }

    const form = new FormData()
    form.append('files', csvFile)
    xhr.send(form)
  })
}

export function pollJob(jobId){
  return requestJson(`/jobs/${jobId}`)
}

export default { predictSingle, predictBatchFile, pollJob }
