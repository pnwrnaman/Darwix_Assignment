# Darwix Assignment — AI Engineer Submission

Voice-agent system with a grounded knowledge base, built across four layers.

## Status

| Layer | What it is | Status |
|---|---|---|
| 1 — Knowledge Base | Extract → clean → PII-mask → chunk → embed → ChromaDB | ✅ Working |
| 2 — Voice API | FastAPI + Ollama (llama3.2) RAG backend, served via Cloudflare Tunnel to Vapi | ✅ Working |
| 3 — Localization | PH (Taglish) + ID (Bahasa) voice bots | 📋 Architecture planned, not implemented |
| 4 — Live Nudges | Real-time call signal detection + nudges | 📋 Architecture planned, not implemented |

See each layer's own README for setup. Layers 3 and 4 contain a design-plan README only — see those folders for the intended architecture.

## Quick start

```bash
# Layer 1 — build the knowledge base
cd layer1_kb
pip install -r requirements.txt --break-system-packages
python extract_and_clean.py --input-dir data --output-file processed/records.json
python chunk_and_embed.py --input-file processed/records.json --persist-dir chroma_db

# Layer 2 — serve the RAG API
cd ../layer2_voice
pip install -r requirements.txt --break-system-packages
uvicorn main:app --reload

# Expose it for Vapi
cloudflared tunnel --url http://localhost:8000
```

## Structure
```
layer1_kb/            knowledge base pipeline + retrieval API
layer2_voice/          RAG backend served to Vapi
layer3_localization/   design plan (not implemented)
layer4_nudges/         design plan (not implemented)
```

## Known limitations
- Local inference (Ollama) won't hold up at production concurrency — would move to hosted GPU inference.
- Cloudflare Tunnel is demo-grade; production would use a proper deployment + domain.
- Layers 3 and 4 are documented but not built due to the assessment time window.




## HOW ITS WORKING SO FAR
[Click here to view working video](https://drive.google.com/file/d/1eG56fPuWvQrsvjj6cN2bGO0x12waLMKl/view?usp=sharing)
