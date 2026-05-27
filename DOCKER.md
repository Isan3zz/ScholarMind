# Docker Startup

This project can run as a full Docker Compose stack:

- `frontend`: Vue app served by nginx on `http://localhost:5173`
- `backend`: FastAPI service on `http://localhost:8000`
- `elasticsearch`: Elasticsearch, available inside Docker as `http://elasticsearch:9200`
- `grobid`: GROBID PDF parser, available inside Docker as `http://grobid:8070`

## Start

Make sure `backend/.env` contains your API keys, then run from the project root:

```powershell
docker compose up -d --build
```

Open:

```text
http://localhost:5173
```

## Stop

```powershell
docker compose down
```

## Reset Elasticsearch Data

The Elasticsearch index is stored in the `scholarmind-es-data` Docker volume. To delete all indexed documents:

```powershell
docker compose down -v
```

Then start again:

```powershell
docker compose up -d --build
```

## Useful Checks

```powershell
docker compose ps
docker compose logs -f backend
docker compose logs -f frontend
```

Elasticsearch and GROBID are intentionally not published to host ports. This avoids conflicts when local `9200` or `8070` services are already running.
