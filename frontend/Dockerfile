# Project OmniVision UI — Vite build + nginx (proxies API & SignalR to omni-api)
FROM node:22-alpine AS build
WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci

COPY index.html vite.config.ts tsconfig.json tsconfig.app.json tsconfig.node.json tailwind.config.ts postcss.config.js ./
COPY public ./public
COPY src ./src

ARG OMNI_API_BASE_URL=
ARG OMNI_SIGNALR_HUB_PATH=/hubs/omni
ENV OMNI_API_BASE_URL=$OMNI_API_BASE_URL
ENV OMNI_SIGNALR_HUB_PATH=$OMNI_SIGNALR_HUB_PATH

RUN npm run build

FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
