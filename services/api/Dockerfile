# syntax=docker/dockerfile:1
FROM node:22-alpine AS base
WORKDIR /app
COPY package.json package-lock.json ./

# --- development image: full deps, live-reload via tsx ---
FROM base AS dev
ENV NODE_ENV=development
RUN --mount=type=cache,target=/root/.npm,sharing=locked npm ci
COPY . .
EXPOSE 8080
CMD ["npm", "run", "dev"]

# --- build stage: compile TS -> dist/ ---
FROM base AS build
RUN --mount=type=cache,target=/root/.npm,sharing=locked npm ci
COPY tsconfig.json ./
COPY src ./src
RUN npm run build

# --- production image: runtime deps + compiled output only ---
FROM node:22-alpine AS prod
ENV NODE_ENV=production
WORKDIR /app
COPY package.json package-lock.json ./
# Note: no `npm cache clean --force` here - the cache mount above is ejected
# from the image entirely (BuildKit never commits it to a layer), so cleaning
# it buys nothing for image size and only risks corrupting the shared cache
# mid-build (hit this exact failure live on Render: ENOTEMPTY rmdir on
# /root/.npm/_cacache during the clean step).
RUN --mount=type=cache,target=/root/.npm,sharing=locked npm ci --omit=dev
COPY --from=build /app/dist ./dist
EXPOSE 8080
CMD ["node", "dist/index.js"]
