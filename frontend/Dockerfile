# Dev-server image, matched to docker-compose's local-bring-up goal (Phase
# 8) — a hot-reloading Vite dev server, not a production build. For an
# actual deploy, the frontend is a static build served separately (see
# render.yaml at the repo root) rather than run through this image.
FROM node:22-alpine

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm install

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
