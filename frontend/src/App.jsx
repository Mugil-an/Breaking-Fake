import React, {useState, useEffect, useRef} from 'react'
import { predictSingle, predictBatchFile, pollJob } from './api'

export default function App(){
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [csvFile, setCsvFile] = useState(null)
  const [results, setResults] = useState(null)
  const [jobId, setJobId] = useState(null)
  const [jobStatus, setJobStatus] = useState(null)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState(null)
  const pollingRef = useRef(null)

  useEffect(()=>{
    return ()=>{ if(previewUrl) URL.revokeObjectURL(previewUrl) }
  }, [previewUrl])

  useEffect(()=>{
    if(jobId){
      pollingRef.current = setInterval(async ()=>{
        try{
          const j = await pollJob(jobId)
          setJobStatus(j.meta)
          if(j.meta && j.meta.status === 'done'){
            setResults(j.results)
            clearInterval(pollingRef.current)
            pollingRef.current = null
          }
        }catch(e){
          console.error('Polling job failed', e)
        }
      }, 2000)
    }
    return ()=>{ if(pollingRef.current) clearInterval(pollingRef.current) }
  }, [jobId])

  const onFileChange = e => {
    const f = e.target.files[0]
    setFile(f)
    setError(null)
    if(f){
      const url = URL.createObjectURL(f)
      setPreviewUrl(url)
    } else {
      setPreviewUrl(null)
    }
  }

  const onSingle = async ()=>{
    if(!file) return
    setUploading(true)
    setUploadProgress(0)
    setError(null)
    setResults(null)

    // simple retry loop
    let attempts = 0
    const maxAttempts = 3
    while(attempts < maxAttempts){
      try{
        const res = await predictSingle(file, pct=>setUploadProgress(pct))
        setResults([res])
        setUploading(false)
        return
      }catch(e){
        attempts += 1
        if(attempts >= maxAttempts){
          const hint = /network|failed to fetch/i.test(e.message) ? ' — network/CORS issue? Is backend running and CORS enabled?' : ''
          setError(e.message + hint)
          setUploading(false)
          return
        }
        // backoff
        await new Promise(r=>setTimeout(r, 500 * Math.pow(2, attempts)))
      }
    }
  }


  return (
    <div style={{padding:20,fontFamily:'Arial'}}>
      <h1>Breaking-Fake — Inference</h1>

      <section className="card" style={{marginBottom:20}}>
        <h2>Single image</h2>
        <input type="file" accept="image/*" onChange={onFileChange} />
        <button onClick={onSingle} disabled={!file || uploading} style={{marginLeft:8}}>Upload & Predict</button>
        {previewUrl && <img src={previewUrl} alt="preview" className="thumbnail" />}
        {uploading && <div className="progress"><div className="progress-bar" style={{width: `${uploadProgress}%`}}>{uploadProgress}%</div></div>}
        {error && <div className="error">{error}</div>}
        {results && results.length>0 && (
          <div className="results">
            {results.map((r, i)=> (
              <div className="result-card" key={i}>
                {previewUrl && <img src={previewUrl} className="thumb-small" />}
                <div>
                  <div><strong>{r.class_name}</strong> <span className="confidence">{(r.confidence*100).toFixed(1)}%</span></div>
                  <pre style={{marginTop:8}}>{JSON.stringify(r.probabilities, null, 2)}</pre>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
