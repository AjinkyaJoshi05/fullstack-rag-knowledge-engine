FROM node:22-alpine

WORKDIR /app

COPY package*.json ./

RUN npm ci --omit=dev --legacy-peer-deps

COPY . .

EXPOSE 5000

CMD ["node", "server.js"]