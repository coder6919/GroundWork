# syntax=docker/dockerfile:1
FROM node:22-alpine AS base
WORKDIR /app
COPY package.json package-lock.json ./

# --- development image: full deps, live-reload via tsx ---
FROM base AS dev
ENV NODE_ENV=development
RUN --mount=type=cache,target=/root/.npm npm ci
COPY . .
EXPOSE 8080
CMD ["npm", "run", "dev"]

# --- build stage: compile TS -> dist/ ---
FROM base AS build
RUN --mount=type=cache,target=/root/.npm npm ci
COPY tsconfig.json ./
COPY src ./src
RUN npm run build

# --- production image: runtime deps + compiled output only ---
FROM node:22-alpine AS prod
ENV NODE_ENV=production
WORKDIR /app
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci --omit=dev && npm cache clean --force
COPY --from=build /app/dist ./dist
EXPOSE 8080
CMD ["node", "dist/index.js"]
