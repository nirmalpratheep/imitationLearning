FROM python:3.12-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev

COPY env/ env/
COPY game/ game/
COPY openenv.yaml ./

ENV SDL_VIDEODRIVER=dummy
ENV SDL_AUDIODRIVER=dummy

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
