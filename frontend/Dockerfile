# syntax=docker/dockerfile:1.4
# Project OmniVisionX UI — npm layer cached when package*.json unchanged (DOCKER_BUILDKIT=1).
FROM node:22-alpine AS build
WORKDIR /app

COPY package.json package-lock.json* ./
RUN --mount=type=cache,target=/root/.npm \
    npm config set fetch-retries 5 \
    && npm config set fetch-retry-mintimeout 20000 \
    && npm config set fetch-retry-maxtimeout 120000 \
    && sh -c 'if [ -f package-lock.json ]; then npm ci --no-audit --no-fund; else npm install --no-audit --no-fund; fi'

COPY index.html vite.config.ts tsconfig.json tsconfig.app.json tsconfig.node.json tailwind.config.ts postcss.config.js ./
COPY public ./public
COPY src ./src

ARG OMNI_API_BASE_URL=
ARG OMNI_SIGNALR_HUB_PATH=/hubs/omni
ARG OMNI_MEDIA_BASE_URL=
ARG OMNI_MEDIA_WEBRTC_BASE_URL=
ENV OMNI_API_BASE_URL=$OMNI_API_BASE_URL
ENV OMNI_SIGNALR_HUB_PATH=$OMNI_SIGNALR_HUB_PATH
ENV OMNI_MEDIA_BASE_URL=$OMNI_MEDIA_BASE_URL
ENV OMNI_MEDIA_WEBRTC_BASE_URL=$OMNI_MEDIA_WEBRTC_BASE_URL

RUN npm run build

FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
