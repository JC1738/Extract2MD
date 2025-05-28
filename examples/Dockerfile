FROM node:20-alpine

WORKDIR /app

# Copy the entire project into the Docker image
COPY . /app/extract2md

# Set working directory to the project root inside the container
WORKDIR /app/extract2md

# Install dependencies and build the project
RUN rm -f package-lock.json && npm cache clean --force && npm install
RUN npm run build

# Move the built dist directory to /app/dist for serving
RUN mv dist /app/dist

# Copy demo.html to /app for serving
COPY examples/demo.html /app/demo.html

# Expose port for the static server
EXPOSE 8080

# Serve the /app directory (which now contains demo.html and dist/)
WORKDIR /app
CMD ["npx", "serve", "--listen", "8080", "."]