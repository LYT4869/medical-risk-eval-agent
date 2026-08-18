FROM nginxinc/nginx-unprivileged:1.27.4-alpine
COPY deploy/web/nginx-main.conf /etc/nginx/nginx.conf
COPY deploy/web/nginx.conf /etc/nginx/conf.d/default.conf
COPY deploy/web/index.html deploy/web/app.js deploy/web/styles.css /usr/share/nginx/html/
EXPOSE 8080
