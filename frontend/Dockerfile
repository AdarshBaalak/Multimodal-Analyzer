# Use node to build the frontend
FROM node:18 as builder

WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# Use nginx to serve the built files
FROM nginx:stable-alpine
COPY --from=builder /app/build /usr/share/nginx/html

EXPOSE 7860
CMD ["nginx", "-g", "daemon off;"]
